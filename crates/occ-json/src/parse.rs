//! A recursive-descent JSON parser with hard limits on hostile input.
//!
//! Everything the MCP layer receives arrives as bytes on a pipe. Those bytes
//! are not automatically well formed: a client can send a 400-deep array, an
//! unterminated string, or a number with ten thousand digits. A parser that
//! trusts its input either overflows the stack or allocates until the OOM
//! killer arrives, and "the analysis server died because it was asked a
//! malformed question" is not an acceptable failure mode.
//!
//! Three limits are therefore structural rather than optional:
//!
//! * nesting depth,
//! * total input length when a caller wants one,
//! * and the number of digits accepted in a single numeric literal.
//!
//! Errors carry a byte offset and a message naming what was expected. A reply
//! serialiser that never says *where* the input broke is a debugging tax paid
//! by every future caller.

use crate::value::{Map, Number, Value};
use std::fmt;

/// Maximum nesting depth accepted.
///
/// The parser recurses once per container, and a real reply nests about eight
/// levels. 128 leaves room for any legitimate structural expression while
/// keeping worst-case stack use around a few tens of kilobytes.
pub const MAX_DEPTH: usize = 128;

/// Maximum digits in one numeric literal.
///
/// Beyond this the value cannot be represented by any of the three variants,
/// and continuing to accumulate would turn a malformed input into a denial of
/// service with no upside.
pub const MAX_NUMBER_DIGITS: usize = 64;

/// Why parsing failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseKind {
    /// Input ended in the middle of a value.
    UnexpectedEnd,
    /// A byte appeared where it cannot legally appear.
    UnexpectedByte(char),
    /// A literal (`true`, `false`, `null`) was misspelled.
    BadLiteral,
    /// A number did not follow the RFC 8259 grammar.
    BadNumber,
    /// A string escape was not a valid escape sequence.
    BadEscape,
    /// A `\u` escape was not four hex digits.
    BadUnicodeEscape,
    /// A `\u` escape introduced a lone surrogate.
    LoneSurrogate,
    /// Input was valid UTF-8 but not valid JSON at this position.
    Expected(&'static str),
    /// Nesting exceeded [`MAX_DEPTH`].
    TooDeep,
    /// Input was not valid UTF-8.
    NotUtf8,
    /// Trailing bytes after the top-level value.
    TrailingData,
}

impl fmt::Display for ParseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseKind::UnexpectedEnd => f.write_str("unexpected end of input"),
            ParseKind::UnexpectedByte(c) => write!(f, "unexpected byte {c:?}"),
            ParseKind::BadLiteral => f.write_str("malformed literal"),
            ParseKind::BadNumber => f.write_str("malformed number"),
            ParseKind::BadEscape => f.write_str("invalid escape sequence"),
            ParseKind::BadUnicodeEscape => f.write_str("`\\u` needs four hex digits"),
            ParseKind::LoneSurrogate => f.write_str("lone surrogate in `\\u` escape"),
            ParseKind::Expected(what) => write!(f, "expected {what}"),
            ParseKind::TooDeep => write!(f, "nesting deeper than {MAX_DEPTH} levels"),
            ParseKind::NotUtf8 => f.write_str("input is not valid UTF-8"),
            ParseKind::TrailingData => f.write_str("trailing data after top-level value"),
        }
    }
}

/// A parse failure with the byte offset at which it occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Byte offset into the input.
    pub offset: usize,
    /// What went wrong.
    pub kind: ParseKind,
}

impl ParseError {
    fn new(offset: usize, kind: ParseKind) -> Self {
        ParseError { offset, kind }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.kind, self.offset)
    }
}

impl std::error::Error for ParseError {}

/// Parse one JSON value from `input`.
///
/// The whole input must be consumed: trailing whitespace is allowed, trailing
/// data is not. A stream of concatenated values is a different protocol and
/// silently accepting the first one would desynchronise it.
pub fn parse(input: &str) -> Result<Value, ParseError> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
        depth: 0,
    };
    p.skip_ws();
    let value = p.value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(ParseError::new(p.pos, ParseKind::TrailingData));
    }
    Ok(value)
}

/// Parse from raw bytes, rejecting invalid UTF-8.
///
/// JSON text is UTF-8 by definition, and `&str` cannot hold anything else. The
/// check happens once here rather than per-string so the rest of the parser can
/// slice freely.
pub fn parse_bytes(input: &[u8]) -> Result<Value, ParseError> {
    let text = std::str::from_utf8(input)
        .map_err(|e| ParseError::new(e.valid_up_to(), ParseKind::NotUtf8))?;
    parse(text)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn err(&self, kind: ParseKind) -> ParseError {
        ParseError::new(self.pos, kind)
    }

    fn value(&mut self) -> Result<Value, ParseError> {
        let Some(b) = self.peek() else {
            return Err(self.err(ParseKind::UnexpectedEnd));
        };
        match b {
            b'n' => {
                self.literal(b"null")?;
                Ok(Value::Null)
            }
            b't' => {
                self.literal(b"true")?;
                Ok(Value::Bool(true))
            }
            b'f' => {
                self.literal(b"false")?;
                Ok(Value::Bool(false))
            }
            b'"' => {
                self.pos += 1;
                Ok(Value::String(self.string_body()?))
            }
            b'[' => self.array(),
            b'{' => self.object(),
            b'-' | b'0'..=b'9' => self.number(),
            other => Err(self.err(ParseKind::UnexpectedByte(other as char))),
        }
    }

    fn literal(&mut self, want: &[u8]) -> Result<(), ParseError> {
        if self.bytes[self.pos..].starts_with(want) {
            self.pos += want.len();
            Ok(())
        } else {
            Err(self.err(ParseKind::BadLiteral))
        }
    }

    fn array(&mut self) -> Result<Value, ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err(ParseKind::TooDeep));
        }
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value()?);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => break,
                Some(other) => return Err(self.err(ParseKind::UnexpectedByte(other as char))),
                None => return Err(self.err(ParseKind::UnexpectedEnd)),
            }
        }
        self.depth -= 1;
        Ok(Value::Array(items))
    }

    fn object(&mut self) -> Result<Value, ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err(ParseKind::TooDeep));
        }
        self.pos += 1; // '{'
        let mut map = Map::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err(ParseKind::Expected("an object key")));
            }
            self.pos += 1;
            let key = self.string_body()?;
            self.skip_ws();
            if self.bump() != Some(b':') {
                return Err(self.err(ParseKind::Expected("`:`")));
            }
            self.skip_ws();
            let value = self.value()?;
            // Last duplicate wins, which is what every mainstream parser does
            // and what RFC 8259 permits.
            map.insert(key, value);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => break,
                Some(other) => return Err(self.err(ParseKind::UnexpectedByte(other as char))),
                None => return Err(self.err(ParseKind::UnexpectedEnd)),
            }
        }
        self.depth -= 1;
        Ok(Value::Object(map))
    }

    /// Parse the body of a string. The opening quote has already been consumed.
    fn string_body(&mut self) -> Result<String, ParseError> {
        let start = self.pos;
        // Fast path: no escapes, no control characters. Strings dominate reply
        // size, so this path is worth taking before allocating anything.
        loop {
            match self.peek() {
                Some(b'"') => {
                    let slice = &self.bytes[start..self.pos];
                    self.pos += 1;
                    // SAFETY-free justification: `bytes` came from a `&str`, and
                    // slicing on ASCII delimiters keeps the boundaries on char
                    // boundaries unless an unescaped control byte intervened,
                    // which this loop excludes.
                    return std::str::from_utf8(slice)
                        .map(str::to_owned)
                        .map_err(|_| ParseError::new(start, ParseKind::NotUtf8));
                }
                Some(b'\\') | Some(0x00..=0x1f) => break,
                Some(_) => self.pos += 1,
                None => return Err(self.err(ParseKind::UnexpectedEnd)),
            }
        }

        let mut out = String::with_capacity(self.pos - start);
        out.push_str(
            std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| ParseError::new(start, ParseKind::NotUtf8))?,
        );
        loop {
            let Some(b) = self.bump() else {
                return Err(self.err(ParseKind::UnexpectedEnd));
            };
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(esc) = self.bump() else {
                        return Err(self.err(ParseKind::UnexpectedEnd));
                    };
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let cp = self.hex4()?;
                            if (0xd800..0xdc00).contains(&cp) {
                                // High surrogate: a low surrogate must follow,
                                // or the pair does not name a scalar value.
                                if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                                    return Err(self.err(ParseKind::LoneSurrogate));
                                }
                                let lo = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&lo) {
                                    return Err(self.err(ParseKind::LoneSurrogate));
                                }
                                let combined = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                                out.push(
                                    char::from_u32(combined)
                                        .ok_or_else(|| self.err(ParseKind::LoneSurrogate))?,
                                );
                            } else if (0xdc00..0xe000).contains(&cp) {
                                return Err(self.err(ParseKind::LoneSurrogate));
                            } else {
                                out.push(
                                    char::from_u32(cp)
                                        .ok_or_else(|| self.err(ParseKind::LoneSurrogate))?,
                                );
                            }
                        }
                        _ => return Err(self.err(ParseKind::BadEscape)),
                    }
                }
                0x00..=0x1f => {
                    // RFC 8259 requires these to be escaped. Accepting them raw
                    // produces strings that cannot be re-serialised into valid
                    // JSON, so the round trip would break silently.
                    return Err(self.err(ParseKind::BadEscape));
                }
                _ => {
                    // Multi-byte UTF-8 stays intact because the bytes were
                    // already validated as `&str` and this arm only pushes whole
                    // sequences: continuation bytes are not ASCII delimiters.
                    let start = self.pos - 1;
                    let len = utf8_len(b);
                    let end = start + len;
                    if end > self.bytes.len() {
                        return Err(self.err(ParseKind::UnexpectedEnd));
                    }
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..end])
                            .map_err(|_| ParseError::new(start, ParseKind::NotUtf8))?,
                    );
                    self.pos = end;
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let Some(b) = self.bump() else {
                return Err(self.err(ParseKind::UnexpectedEnd));
            };
            let d = match b {
                b'0'..=b'9' => u32::from(b - b'0'),
                b'a'..=b'f' => u32::from(b - b'a') + 10,
                b'A'..=b'F' => u32::from(b - b'A') + 10,
                _ => return Err(self.err(ParseKind::BadUnicodeEscape)),
            };
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.pos += 1;
        }

        // Integer part. RFC 8259 forbids leading zeros, and accepting them
        // would make `01` and `1` two texts for one value.
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.err(ParseKind::BadNumber));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                    if self.pos - start > MAX_NUMBER_DIGITS {
                        return Err(self.err(ParseKind::BadNumber));
                    }
                }
            }
            _ => return Err(self.err(ParseKind::BadNumber)),
        }

        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err(ParseKind::BadNumber));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
                if self.pos - start > MAX_NUMBER_DIGITS {
                    return Err(self.err(ParseKind::BadNumber));
                }
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err(ParseKind::BadNumber));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
                if self.pos - start > MAX_NUMBER_DIGITS {
                    return Err(self.err(ParseKind::BadNumber));
                }
            }
        }

        let text = &self.bytes[start..self.pos];
        // `from_utf8` cannot fail here: every byte consumed is ASCII.
        let text = std::str::from_utf8(text).map_err(|_| self.err(ParseKind::BadNumber))?;

        if is_float {
            let v: f64 = text.parse().map_err(|_| self.err(ParseKind::BadNumber))?;
            if v.is_infinite() {
                // `1e999` overflows to infinity. Refusing is better than
                // silently substituting a sentinel that then serialises to null.
                return Err(self.err(ParseKind::BadNumber));
            }
            return Ok(Value::Number(Number::Float(v)));
        }

        if negative {
            match text.parse::<i64>() {
                Ok(v) => Ok(Value::Number(Number::NegInt(v))),
                // Below i64::MIN: the value is representable as f64 with loss.
                // Rejecting outright would be defensible too, but a caller who
                // wrote `-99999999999999999999` asked for a number, and the
                // error would surface as a schema violation far from here.
                Err(_) => text
                    .parse::<f64>()
                    .map(|v| Value::Number(Number::Float(v)))
                    .map_err(|_| self.err(ParseKind::BadNumber)),
            }
        } else {
            match text.parse::<u64>() {
                Ok(v) => Ok(Value::Number(Number::PosInt(v))),
                Err(_) => text
                    .parse::<f64>()
                    .map(|v| Value::Number(Number::Float(v)))
                    .map_err(|_| self.err(ParseKind::BadNumber)),
            }
        }
    }
}

/// Length in bytes of a UTF-8 sequence started by `lead`.
fn utf8_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead >> 5 == 0b110 {
        2
    } else if lead >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::to_string;

    #[test]
    fn parses_every_scalar() {
        assert!(parse("null").unwrap().is_null());
        assert_eq!(parse("true").unwrap().as_bool(), Some(true));
        assert_eq!(parse("false").unwrap().as_bool(), Some(false));
        assert_eq!(parse("42").unwrap().as_u64(), Some(42));
        assert_eq!(parse("-7").unwrap().as_i64(), Some(-7));
        assert_eq!(parse("1.5").unwrap().as_f64(), Some(1.5));
        assert_eq!(parse("\"hi\"").unwrap().as_str(), Some("hi"));
    }

    #[test]
    fn rejects_leading_zero() {
        assert!(parse("01").is_err());
        assert_eq!(parse("0").unwrap().as_u64(), Some(0));
        assert_eq!(parse("0.5").unwrap().as_f64(), Some(0.5));
    }

    #[test]
    fn deep_nesting_is_refused_not_crashed() {
        let deep = "[".repeat(MAX_DEPTH + 8);
        let err = parse(&deep).unwrap_err();
        assert_eq!(err.kind, ParseKind::TooDeep);
    }

    #[test]
    fn exactly_at_the_limit_is_accepted() {
        let ok = format!("{}1{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse(&ok).is_ok(), "depth {} must be accepted", MAX_DEPTH);
    }

    #[test]
    fn surrogate_pairs_combine() {
        // U+1F600 GRINNING FACE, encoded as a surrogate pair.
        let v = parse(r#""\ud83d\ude00""#).unwrap();
        assert_eq!(v.as_str(), Some("\u{1f600}"));
    }

    #[test]
    fn lone_surrogates_are_refused() {
        assert_eq!(
            parse(r#""\ud83d""#).unwrap_err().kind,
            ParseKind::LoneSurrogate
        );
        assert_eq!(
            parse(r#""\ude00""#).unwrap_err().kind,
            ParseKind::LoneSurrogate
        );
    }

    #[test]
    fn raw_control_characters_are_refused() {
        // A literal newline inside a string produces JSON that cannot be
        // re-serialised, so the round trip would silently break.
        assert!(parse("\"a\nb\"").is_err());
    }

    #[test]
    fn escapes_round_trip() {
        let original = "quote\" backslash\\ slash/ tab\t bell\u{08}";
        let encoded = to_string(&Value::from(original));
        assert_eq!(parse(&encoded).unwrap().as_str(), Some(original));
    }

    #[test]
    fn duplicate_keys_keep_the_last_value() {
        let v = parse(r#"{"a":1,"b":2,"a":3}"#).unwrap();
        assert_eq!(v.get("a").and_then(Value::as_u64), Some(3));
        assert_eq!(v.as_object().unwrap().len(), 2);
    }

    #[test]
    fn trailing_data_is_refused() {
        assert_eq!(parse("1 2").unwrap_err().kind, ParseKind::TrailingData);
        assert!(parse("1 ").is_ok(), "trailing whitespace is fine");
    }

    #[test]
    fn errors_carry_a_position() {
        let err = parse(r#"{"a": }"#).unwrap_err();
        assert!(err.offset > 0);
        assert!(err.to_string().contains("byte"));
    }

    #[test]
    fn multibyte_strings_survive_intact() {
        let v = parse("\"日本語テキスト\"").unwrap();
        assert_eq!(v.as_str(), Some("日本語テキスト"));
        assert_eq!(to_string(&v), "\"日本語テキスト\"");
    }

    #[test]
    fn number_digit_bomb_is_refused() {
        let bomb = "1".repeat(MAX_NUMBER_DIGITS + 2);
        assert!(parse(&bomb).is_err());
    }

    #[test]
    fn exponent_overflow_is_refused() {
        assert!(parse("1e999").is_err());
        assert!(parse("1e308").is_ok());
    }

    #[test]
    fn integer_beyond_u64_becomes_float_not_error() {
        let v = parse("99999999999999999999999").unwrap();
        assert!(v.as_f64().is_some());
    }
}
