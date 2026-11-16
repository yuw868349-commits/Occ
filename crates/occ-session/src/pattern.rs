//! A pattern scanner with a YARA-shaped rule language.
//!
//! The language is deliberately a subset: text strings (with `nocase` and
//! `wide`), hex strings with `??` wildcards and `[n-m]` gaps, and a condition
//! grammar of `and`/`or`/`not`, parentheses, `N of them`, `at <addr>` and
//! `in <section>`. That covers the rules analysts actually write for triage;
//! the parts left out (modules, regular expressions, offset arithmetic) are
//! the parts that would need a full expression engine, and a half-implemented
//! one is worse than an honest parse error.
//!
//! Rules are matched against the file image with section-relative addresses,
//! so a hit reports the virtual address an analyst can jump to.

use std::collections::HashMap;

use occ_core::Addr;

/// One string definition inside a rule.
#[derive(Clone, Debug)]
pub struct PatternString {
    /// Identifier including the `$`, e.g. `$a`.
    pub id: String,
    /// Compiled matcher.
    pub kind: StringKind,
}

/// How a string is matched.
#[derive(Clone, Debug)]
pub enum StringKind {
    /// Literal bytes with optional case folding and UTF-16 widening.
    Text {
        /// Bytes to find.
        bytes: Vec<u8>,
        /// ASCII case-insensitive.
        nocase: bool,
        /// Also try the UTF-16LE form.
        wide: bool,
    },
    /// Byte pattern with wildcards and gaps.
    Hex(Vec<HexToken>),
}

/// One element of a hex pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HexToken {
    /// A byte with a bit mask; `0xff, 0xff` is exact, `0x00, 0x00` is `??`.
    Byte {
        /// Expected value.
        value: u8,
        /// Bits that must match.
        mask: u8,
    },
    /// A gap of `min..=max` arbitrary bytes.
    Jump {
        /// Minimum length.
        min: usize,
        /// Maximum length.
        max: usize,
    },
}

/// Condition expression.
#[derive(Clone, Debug)]
pub enum Condition {
    /// Always true.
    True,
    /// A specific string matched.
    Str(String),
    /// A string matched at an exact address.
    At(String, u64),
    /// A string matched inside a named section.
    In(String, String),
    /// Both sides.
    And(Box<Condition>, Box<Condition>),
    /// Either side.
    Or(Box<Condition>, Box<Condition>),
    /// Negation.
    Not(Box<Condition>),
    /// At least one string matched.
    AnyOfThem,
    /// Every declared string matched.
    AllOfThem,
    /// At least `n` strings matched.
    NOfThem(usize),
}

/// A parsed rule.
#[derive(Clone, Debug)]
pub struct Rule {
    /// Rule name.
    pub name: String,
    /// Declared strings.
    pub strings: Vec<PatternString>,
    /// Match condition.
    pub condition: Condition,
}

/// One rule that matched.
#[derive(Clone, Debug)]
pub struct RuleMatch {
    /// Rule name.
    pub rule: String,
    /// Every address the rule matched at, ascending and de-duplicated.
    ///
    /// A rule that anchors three times is three findings, not one: collapsing
    /// them to the lowest address hides the other two and makes the count
    /// wrong.
    pub addrs: Vec<Addr>,
    /// Identifiers of the strings that matched, each with every address it was
    /// found at.
    pub strings: Vec<(String, Vec<Addr>)>,
}

/// Parse a rule file.
pub fn parse(text: &str) -> Result<Vec<Rule>, String> {
    let mut rules = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let line = strip_comment(lines[i]).trim().to_string();
        i += 1;
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("rule ") {
            let name = rest
                .split('{')
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if name.is_empty() {
                return Err(format!("line {}: rule without a name", i));
            }
            let mut strings = Vec::new();
            let mut condition_text = String::new();
            let mut mode = "";
            loop {
                if i >= lines.len() {
                    return Err(format!("rule {name}: unterminated"));
                }
                let raw = strip_comment(lines[i]).trim().to_string();
                i += 1;
                if raw.is_empty() {
                    continue;
                }
                if raw == "}" {
                    break;
                }
                if raw == "strings:" {
                    mode = "strings";
                    continue;
                }
                if raw == "condition:" {
                    mode = "condition";
                    continue;
                }
                match mode {
                    "strings" => {
                        // A hex string may span lines; keep reading until the
                        // closing brace.
                        let mut def = raw;
                        while def.contains('{') && !def.contains('}') {
                            if i >= lines.len() {
                                return Err(format!("rule {name}: unterminated hex string"));
                            }
                            def.push(' ');
                            def.push_str(strip_comment(lines[i]).trim());
                            i += 1;
                        }
                        strings.push(parse_string(&def).map_err(|e| format!("rule {name}: {e}"))?);
                    }
                    "condition" => {
                        condition_text.push(' ');
                        condition_text.push_str(&raw);
                    }
                    _ => return Err(format!("rule {name}: unexpected line {raw:?}")),
                }
            }
            let condition = parse_condition(&condition_text)
                .map_err(|e| format!("rule {name}: condition: {e}"))?;
            rules.push(Rule {
                name,
                strings,
                condition,
            });
            continue;
        }
        return Err(format!("line {i}: expected `rule <name> {{`, found {line:?}"));
    }
    Ok(rules)
}

fn strip_comment(line: &str) -> &str {
    // A `//` inside a quoted string is not a comment.
    let mut in_quote = false;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'"' {
            in_quote = !in_quote;
        }
        if !in_quote && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            return &line[..i];
        }
        i += 1;
    }
    line
}

fn parse_string(def: &str) -> Result<PatternString, String> {
    let (id, rest) = def
        .split_once('=')
        .ok_or_else(|| format!("expected `$id = ...`, found {def:?}"))?;
    let id = id.trim().to_string();
    if !id.starts_with('$') {
        return Err(format!("string identifier must start with `$`: {id}"));
    }
    let rest = rest.trim();
    if rest.starts_with('"') {
        let (bytes, flags) = parse_text(rest)?;
        return Ok(PatternString {
            id,
            kind: StringKind::Text {
                bytes,
                nocase: flags.contains(&"nocase"),
                wide: flags.contains(&"wide"),
            },
        });
    }
    if rest.starts_with('{') {
        let inner = rest
            .trim_start_matches('{')
            .trim_end_matches('}')
            .trim();
        return Ok(PatternString {
            id,
            kind: StringKind::Hex(parse_hex(inner)?),
        });
    }
    Err(format!("unsupported string syntax: {rest}"))
}

fn parse_text(s: &str) -> Result<(Vec<u8>, Vec<&str>), String> {
    let mut out = Vec::new();
    let mut chars = s.char_indices();
    let start = s.find('"').ok_or("missing opening quote")?;
    let mut end = None;
    for (i, c) in chars.by_ref() {
        if i <= start {
            continue;
        }
        if c == '\\' {
            continue;
        }
        if c == '"' {
            end = Some(i);
            break;
        }
    }
    let end = end.ok_or("missing closing quote")?;
    let body = &s[start + 1..end];
    let mut it = body.chars();
    while let Some(c) = it.next() {
        match c {
            '\\' => match it.next() {
                Some('n') => out.push(b'\n'),
                Some('t') => out.push(b'\t'),
                Some('r') => out.push(b'\r'),
                Some('0') => out.push(0),
                Some('\\') => out.push(b'\\'),
                Some('"') => out.push(b'"'),
                Some('x') => {
                    let hi = it.next().and_then(|c| c.to_digit(16)).ok_or("bad \\x escape")?;
                    let lo = it.next().and_then(|c| c.to_digit(16)).ok_or("bad \\x escape")?;
                    out.push((hi * 16 + lo) as u8);
                }
                Some(other) => return Err(format!("unknown escape \\{other}")),
                None => return Err("trailing backslash".into()),
            },
            c if c.is_ascii() => out.push(c as u8),
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    let flags: Vec<&str> = s[end + 1..].split_whitespace().collect();
    Ok((out, flags))
}

fn parse_hex(s: &str) -> Result<Vec<HexToken>, String> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        if tok.starts_with('[') {
            let inner = tok.trim_start_matches('[').trim_end_matches(']');
            let (min, max) = match inner.split_once('-') {
                Some((a, b)) => (
                    a.parse::<usize>().map_err(|_| format!("bad jump {tok}"))?,
                    b.parse::<usize>().map_err(|_| format!("bad jump {tok}"))?,
                ),
                None => {
                    let n = inner.parse::<usize>().map_err(|_| format!("bad jump {tok}"))?;
                    (n, n)
                }
            };
            out.push(HexToken::Jump { min, max });
            continue;
        }
        if tok.len() != 2 {
            return Err(format!("hex token must be two characters: {tok}"));
        }
        let b = tok.as_bytes();
        let (value, mask) = match (nibble(b[0]), nibble(b[1])) {
            (Some(hi), Some(lo)) => ((hi << 4) | lo, 0xff),
            (Some(hi), None) if b[1] == b'?' => (hi << 4, 0xf0),
            (None, Some(lo)) if b[0] == b'?' => (lo, 0x0f),
            (None, None) if b[0] == b'?' && b[1] == b'?' => (0, 0),
            _ => return Err(format!("invalid hex token {tok}")),
        };
        out.push(HexToken::Byte { value, mask });
    }
    if out.is_empty() {
        return Err("empty hex string".into());
    }
    Ok(out)
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Word(String),
    Num(usize),
    LParen,
    RParen,
}

fn tokenize_condition(s: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<Tok>| {
        if cur.is_empty() {
            return;
        }
        if let Ok(n) = cur.parse::<usize>() {
            out.push(Tok::Num(n));
        } else {
            out.push(Tok::Word(cur.clone()));
        }
        cur.clear();
    };
    for c in s.chars() {
        match c {
            '(' => {
                flush(&mut cur, &mut out);
                out.push(Tok::LParen);
            }
            ')' => {
                flush(&mut cur, &mut out);
                out.push(Tok::RParen);
            }
            c if c.is_whitespace() => flush(&mut cur, &mut out),
            c => cur.push(c),
        }
    }
    flush(&mut cur, &mut out);
    out
}

fn parse_condition(s: &str) -> Result<Condition, String> {
    let toks = tokenize_condition(s);
    let mut p = CondParser { toks, pos: 0 };
    let c = p.parse_or()?;
    if p.pos != p.toks.len() {
        return Err(format!("unexpected token at {}", p.pos));
    }
    Ok(c)
}

struct CondParser {
    toks: Vec<Tok>,
    pos: usize,
}

impl CondParser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat_word(&mut self, w: &str) -> bool {
        if let Some(Tok::Word(x)) = self.peek()
            && x == w
        {
            self.pos += 1;
            return true;
        }
        false
    }

    fn parse_or(&mut self) -> Result<Condition, String> {
        let mut left = self.parse_and()?;
        while self.eat_word("or") {
            let right = self.parse_and()?;
            left = Condition::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Condition, String> {
        let mut left = self.parse_unary()?;
        while self.eat_word("and") {
            let right = self.parse_unary()?;
            left = Condition::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Condition, String> {
        if self.eat_word("not") {
            return Ok(Condition::Not(Box::new(self.parse_unary()?)));
        }
        self.parse_term()
    }

    fn parse_term(&mut self) -> Result<Condition, String> {
        match self.peek().cloned() {
            Some(Tok::LParen) => {
                self.pos += 1;
                let inner = self.parse_or()?;
                match self.peek() {
                    Some(Tok::RParen) => {
                        self.pos += 1;
                        Ok(inner)
                    }
                    _ => Err("missing closing parenthesis".into()),
                }
            }
            Some(Tok::Num(n)) => {
                self.pos += 1;
                if !self.eat_word("of") || !self.eat_word("them") {
                    return Err("expected `of them` after a count".into());
                }
                Ok(Condition::NOfThem(n))
            }
            Some(Tok::Word(w)) => {
                self.pos += 1;
                match w.as_str() {
                    "any" => {
                        if self.eat_word("of") && self.eat_word("them") {
                            Ok(Condition::AnyOfThem)
                        } else {
                            Err("expected `any of them`".into())
                        }
                    }
                    "all" => {
                        if self.eat_word("of") && self.eat_word("them") {
                            Ok(Condition::AllOfThem)
                        } else {
                            Err("expected `all of them`".into())
                        }
                    }
                    "true" => Ok(Condition::True),
                    "false" => Ok(Condition::Not(Box::new(Condition::True))),
                    other if other.starts_with('$') => {
                        if self.eat_word("at") {
                            match self.peek().cloned() {
                                Some(Tok::Num(n)) => {
                                    self.pos += 1;
                                    Ok(Condition::At(other.to_string(), n as u64))
                                }
                                Some(Tok::Word(hex)) => {
                                    self.pos += 1;
                                    let v = occ_core::addr::parse_addr(&hex)
                                        .ok_or_else(|| format!("bad address {hex}"))?;
                                    Ok(Condition::At(other.to_string(), v.get()))
                                }
                                _ => Err("expected an address after `at`".into()),
                            }
                        } else if self.eat_word("in") {
                            match self.peek().cloned() {
                                Some(Tok::Word(sec)) => {
                                    self.pos += 1;
                                    Ok(Condition::In(other.to_string(), sec))
                                }
                                _ => Err("expected a section name after `in`".into()),
                            }
                        } else {
                            Ok(Condition::Str(other.to_string()))
                        }
                    }
                    other => Err(format!("unknown term {other:?}")),
                }
            }
            _ => Err("unexpected end of condition".into()),
        }
    }
}

/// A section the scanner can test `in <section>` against.
#[derive(Clone, Debug)]
pub struct SectionRange {
    /// Name.
    pub name: String,
    /// Start address.
    pub start: u64,
    /// Size.
    pub size: u64,
}

/// Scan `data` (mapped at `base`) with `rules`.
pub fn scan(
    data: &[u8],
    base: u64,
    rules: &[Rule],
    sections: &[SectionRange],
) -> Vec<RuleMatch> {
    let mut out = Vec::new();
    for rule in rules {
        let mut hits: HashMap<&str, Vec<u64>> = HashMap::new();
        for s in &rule.strings {
            let found = match &s.kind {
                StringKind::Text {
                    bytes,
                    nocase,
                    wide,
                } => find_text(data, bytes, *nocase, *wide),
                StringKind::Hex(tokens) => find_hex(data, tokens),
            };
            if !found.is_empty() {
                hits.insert(s.id.as_str(), found.into_iter().map(|o| base + o as u64).collect());
            }
        }
        if eval(&rule.condition, &hits, sections, &rule.strings) {
            let mut strings: Vec<(String, Vec<Addr>)> = hits
                .iter()
                .map(|(id, at)| (id.to_string(), at.iter().map(|a| Addr(*a)).collect()))
                .collect();
            strings.sort_by(|a, b| a.0.cmp(&b.0));
            let mut addrs: Vec<Addr> = hits
                .values()
                .flatten()
                .map(|a| Addr(*a))
                .collect::<Vec<_>>();
            addrs.sort_unstable();
            addrs.dedup();
            out.push(RuleMatch {
                rule: rule.name.clone(),
                addrs,
                strings,
            });
        }
    }
    out
}

fn eval(
    c: &Condition,
    hits: &HashMap<&str, Vec<u64>>,
    sections: &[SectionRange],
    declared: &[PatternString],
) -> bool {
    match c {
        Condition::True => true,
        Condition::Str(id) => hits.contains_key(id.as_str()),
        Condition::At(id, addr) => hits
            .get(id.as_str())
            .map(|v| v.contains(addr))
            .unwrap_or(false),
        Condition::In(id, sec) => {
            let Some(range) = sections.iter().find(|s| &s.name == sec) else {
                return false;
            };
            hits.get(id.as_str())
                .map(|v| {
                    v.iter()
                        .any(|a| *a >= range.start && *a < range.start + range.size)
                })
                .unwrap_or(false)
        }
        Condition::And(a, b) => eval(a, hits, sections, declared) && eval(b, hits, sections, declared),
        Condition::Or(a, b) => eval(a, hits, sections, declared) || eval(b, hits, sections, declared),
        Condition::Not(a) => !eval(a, hits, sections, declared),
        Condition::AnyOfThem => declared.iter().any(|s| hits.contains_key(s.id.as_str())),
        Condition::AllOfThem => declared.iter().all(|s| hits.contains_key(s.id.as_str())),
        Condition::NOfThem(n) => {
            declared
                .iter()
                .filter(|s| hits.contains_key(s.id.as_str()))
                .count()
                >= *n
        }
    }
}

fn find_text(data: &[u8], needle: &[u8], nocase: bool, wide: bool) -> Vec<usize> {
    let mut out = Vec::new();
    if needle.is_empty() || needle.len() > data.len() {
        return out;
    }
    let eq = |a: &[u8], b: &[u8]| -> bool {
        if nocase {
            a.eq_ignore_ascii_case(b)
        } else {
            a == b
        }
    };
    let mut i = 0usize;
    while i + needle.len() <= data.len() {
        if eq(&data[i..i + needle.len()], needle) {
            out.push(i);
            i += needle.len().max(1);
            continue;
        }
        i += 1;
    }
    if wide {
        let mut w: Vec<u8> = Vec::with_capacity(needle.len() * 2);
        for b in needle {
            w.push(*b);
            w.push(0);
        }
        let mut i = 0usize;
        while i + w.len() <= data.len() {
            if eq(&data[i..i + w.len()], &w) {
                out.push(i);
                i += w.len().max(1);
                continue;
            }
            i += 1;
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn find_hex(data: &[u8], tokens: &[HexToken]) -> Vec<usize> {
    let mut out = Vec::new();
    for start in 0..data.len() {
        if match_at(data, start, tokens) {
            out.push(start);
        }
    }
    out
}

fn match_at(data: &[u8], start: usize, tokens: &[HexToken]) -> bool {
    fn go(data: &[u8], pos: usize, toks: &[HexToken]) -> bool {
        match toks.first() {
            None => true,
            Some(HexToken::Byte { value, mask }) => {
                if pos >= data.len() || data[pos] & mask != value & mask {
                    return false;
                }
                go(data, pos + 1, &toks[1..])
            }
            Some(HexToken::Jump { min, max }) => {
                for len in *min..=*max {
                    if pos + len <= data.len() && go(data, pos + len, &toks[1..]) {
                        return true;
                    }
                }
                false
            }
        }
    }
    go(data, start, tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = b"....hello world....\x48\x8b\x05\x10\x00\x00\x00...MZ..";

    fn sections() -> Vec<SectionRange> {
        vec![SectionRange {
            name: ".text".into(),
            start: 0x1000,
            size: 64,
        }]
    }

    #[test]
    fn parses_and_matches_text() {
        let rules = parse(
            r#"
            rule greeting {
              strings:
                $a = "hello"
              condition:
                $a
            }
            "#,
        )
        .unwrap();
        let m = scan(SAMPLE, 0x1000, &rules, &sections());
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].rule, "greeting");
        assert_eq!(m[0].addrs, vec![Addr(0x1000 + 4)]);
    }

    #[test]
    fn every_occurrence_is_reported() {
        let data = b"AAAAxxAAAAyyAAAA";
        let rules = parse(
            r#"
            rule rep {
              strings:
                $a = "AAAA"
              condition:
                $a
            }
            "#,
        )
        .unwrap();
        let m = scan(data, 0x2000, &rules, &sections());
        assert_eq!(m.len(), 1);
        assert_eq!(
            m[0].addrs,
            vec![Addr(0x2000), Addr(0x2000 + 6), Addr(0x2000 + 12)]
        );
        assert_eq!(m[0].strings.len(), 1);
        assert_eq!(m[0].strings[0].0, "$a");
        assert_eq!(m[0].strings[0].1.len(), 3);
    }

    #[test]
    fn hex_with_wildcards_and_gap() {
        let rules = parse(
            r#"
            rule mov_rip {
              strings:
                $a = { 48 8b ?? ?? ?? ?? ?? }
              condition:
                $a
            }
            "#,
        )
        .unwrap();
        assert_eq!(scan(SAMPLE, 0, &rules, &sections()).len(), 1);
    }

    #[test]
    fn conditions_combine() {
        let rules = parse(
            r#"
            rule both {
              strings:
                $a = "hello"
                $b = "absent"
                $c = "world"
              condition:
                $a and $c and not $b
            }
            rule either {
              strings:
                $a = "hello"
                $b = "absent"
              condition:
                any of them
            }
            rule counted {
              strings:
                $a = "hello"
                $b = "world"
                $c = "absent"
              condition:
                2 of them
            }
            "#,
        )
        .unwrap();
        let names: Vec<String> = scan(SAMPLE, 0, &rules, &sections())
            .into_iter()
            .map(|m| m.rule)
            .collect();
        assert!(names.contains(&"both".to_string()));
        assert!(names.contains(&"either".to_string()));
        assert!(names.contains(&"counted".to_string()));
    }

    #[test]
    fn in_section_scopes_a_match() {
        let rules = parse(
            r#"
            rule scoped {
              strings:
                $a = "hello"
              condition:
                $a in .text
            }
            "#,
        )
        .unwrap();
        assert_eq!(scan(SAMPLE, 0x1000, &rules, &sections()).len(), 1);
        let far = scan(SAMPLE, 0x9000, &rules, &sections());
        assert!(far.is_empty());
    }

    #[test]
    fn reports_syntax_errors_with_context() {
        assert!(parse("rule x { strings: $a = 42 condition: $a }").is_err());
        assert!(parse("nonsense").is_err());
    }
}
