//! Serialisation back to JSON text.
//!
//! Two writers, one grammar. The compact form is what goes on the wire; the
//! pretty form exists for transcripts and tests, where a single 900-byte line
//! is unreadable.
//!
//! Escaping follows the minimal set RFC 8259 requires plus the two characters
//! that would otherwise break an enclosing context. Everything else is emitted
//! as-is: over-escaping inflates every reply and makes byte comparison between
//! runs harder, not easier.

use crate::value::{Map, Number, Value};

/// Serialise compactly. No whitespace except inside strings.
pub fn to_string(value: &Value) -> String {
    let mut out = String::with_capacity(128);
    write_value(&mut out, value);
    out
}

/// Serialise with two-space indentation.
pub fn to_string_pretty(value: &Value) -> String {
    let mut out = String::with_capacity(256);
    write_pretty(&mut out, value, 0);
    out
}

/// Append one value to `out`.
pub fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_value(out, v);
            }
            out.push('}');
        }
    }
}

fn write_pretty(out: &mut String, value: &Value, indent: usize) {
    match value {
        Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                push_indent(out, indent + 1);
                write_pretty(out, item, indent + 1);
            }
            out.push('\n');
            push_indent(out, indent);
            out.push(']');
        }
        Value::Object(map) if !map.is_empty() => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                push_indent(out, indent + 1);
                write_string(out, k);
                out.push_str(": ");
                write_pretty(out, v, indent + 1);
            }
            out.push('\n');
            push_indent(out, indent);
            out.push('}');
        }
        // Empty containers stay on one line: `{}` and `[]` are two characters,
        // and expanding them to three lines helps nobody.
        other => write_value(out, other),
    }
}

fn push_indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn write_number(out: &mut String, n: &Number) {
    use std::fmt::Write;
    // Ignoring the result is correct: writing to a `String` cannot fail.
    let _ = write!(out, "{n}");
}

/// Append a quoted, escaped JSON string.
pub fn write_string(out: &mut String, s: &str) {
    out.push('"');
    let bytes = s.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let escape = match b {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            b'\n' => "\\n",
            b'\r' => "\\r",
            b'\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            // Remaining C0 controls have no short form.
            0x00..=0x1f => "",
            _ => continue,
        };
        if start < i {
            out.push_str(&s[start..i]);
        }
        if escape.is_empty() {
            push_u_escape(out, b);
        } else {
            out.push_str(escape);
        }
        start = i + 1;
    }
    if start < bytes.len() {
        out.push_str(&s[start..]);
    }
    out.push('"');
}

fn push_u_escape(out: &mut String, b: u8) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push_str("\\u00");
    out.push(HEX[(b >> 4) as usize] as char);
    out.push(HEX[(b & 0xf) as usize] as char);
}

/// Escape a string into a standalone JSON string literal.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    write_string(&mut out, s);
    out
}

/// Render an object's members one per line for human-facing summaries.
///
/// This is not a general facility: it exists so `Reply::summary` can embed a
/// short aligned list without every call site re-implementing padding.
pub fn aligned_lines(map: &Map, width: usize) -> String {
    let mut out = String::new();
    for (k, v) in map.iter() {
        out.push_str("  ");
        out.push_str(k);
        for _ in k.len()..width {
            out.push(' ');
        }
        out.push(' ');
        out.push_str(&scalar_text(v));
        out.push('\n');
    }
    out
}

/// One-line rendering of a scalar, or a bracketed count for containers.
pub fn scalar_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{} items]", a.len()),
        Value::Object(m) => format!("{{{} keys}}", m.len()),
        other => to_string(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_obj;

    #[test]
    fn compact_output_has_no_spaces() {
        let v = json_obj! {
            "a" => 1u64,
            "b" => vec![Value::from(1u64), Value::from(2u64)],
        };
        assert_eq!(to_string(&v), r#"{"a":1,"b":[1,2]}"#);
    }

    #[test]
    fn pretty_output_indents_by_two() {
        let v = json_obj! { "a" => 1u64 };
        assert_eq!(to_string_pretty(&v), "{\n  \"a\": 1\n}");
    }

    #[test]
    fn empty_containers_stay_compact() {
        assert_eq!(to_string_pretty(&Value::Array(vec![])), "[]");
        assert_eq!(to_string_pretty(&Value::Object(Map::new())), "{}");
    }

    #[test]
    fn control_characters_are_escaped() {
        assert_eq!(quote("\u{01}"), r#""\u0001""#);
        assert_eq!(quote("a\tb"), r#""a\tb""#);
    }

    #[test]
    fn forward_slash_is_not_escaped() {
        // Escaping `/` is legal but pointless, and paths are everywhere in this
        // toolset's output.
        assert_eq!(quote("/usr/bin/ls"), r#""/usr/bin/ls""#);
    }

    #[test]
    fn non_ascii_is_emitted_raw() {
        assert_eq!(quote("日本"), "\"日本\"");
    }

    #[test]
    fn round_trip_through_parse_preserves_structure() {
        let original = json_obj! {
            "s" => "a\"b\\c\u{1}",
            "n" => 42u64,
            "f" => 1.5f64,
            "neg" => -7i64,
            "arr" => vec![Value::Null, Value::Bool(true)],
        };
        let text = to_string(&original);
        let back = crate::parse::parse(&text).expect("reparse");
        assert_eq!(to_string(&back), text);
    }
}
