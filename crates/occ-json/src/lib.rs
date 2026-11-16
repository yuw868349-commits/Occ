//! JSON for occ, with no external dependencies.
//!
//! This crate exists because the project ships with an empty dependency list.
//! It covers the subset of `serde_json` that the analysis engine and the MCP
//! transport actually use: a value tree with insertion-ordered objects, exact
//! 64-bit integers, a bounded recursive-descent parser, and a compact and
//! pretty writer.
//!
//! It is not a drop-in replacement for `serde`. There is no derive, and no
//! reflection: every type that needs a wire form implements it by hand. That is
//! a deliberate trade — the wire form of an analysis reply is part of the
//! product's interface, and having it written out explicitly next to the type
//! is worth more here than the boilerplate saved by a macro.

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod parse;
pub mod schema;
pub mod value;
pub mod write;

pub use parse::{MAX_DEPTH, ParseError, ParseKind, parse, parse_bytes};
pub use value::{Map, Number, Value};
pub use write::{quote, to_string, to_string_pretty};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn nested_structures_round_trip() {
        let text = r#"{
            "image": {"id": 0, "base": "0x400000", "sections": 31},
            "imports": ["puts", "malloc", "free"],
            "flags": {"pie": true, "stripped": false},
            "weights": [0.5, 0.25, 0.125],
            "nothing": null
        }"#;
        let v = parse(text).expect("parse");
        let again = to_string(&v);
        let reparsed = parse(&again).expect("reparse");
        assert_eq!(to_string(&reparsed), again);
        assert_eq!(
            v.get("image")
                .and_then(|i| i.get("base"))
                .and_then(Value::as_str),
            Some("0x400000")
        );
    }
}
