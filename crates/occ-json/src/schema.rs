//! Schema description and typed extraction from a parsed [`Value`].
//!
//! Two jobs that both belong to the request boundary:
//!
//! * [`Schema`] renders a JSON Schema fragment, which is what an MCP client
//!   reads from `tools/list` to decide how to fill in arguments.
//! * [`Obj`] reads a parsed object into typed Rust values, which is what the
//!   server does once a call arrives.
//!
//! They are deliberately separate. A schema is a contract shown to a caller; an
//! extractor is the enforcement of that contract. Letting one derive from the
//! other through reflection would hide the contract inside the type definition,
//! and the whole point of `tools/list` is that the contract is legible on its
//! own.

use crate::value::{Map, Value};
use std::fmt;

/// A JSON Schema fragment, limited to the keywords this server emits.
#[derive(Clone, Debug)]
pub enum Schema {
    /// Any JSON value.
    Any,
    /// `null`.
    Null,
    /// `true` or `false`.
    Boolean,
    /// An integer with optional bounds.
    Integer {
        /// Inclusive lower bound.
        min: Option<i64>,
        /// Inclusive upper bound.
        max: Option<i64>,
    },
    /// Any number.
    Number,
    /// A string, optionally constrained to a set of values.
    String {
        /// Allowed values, when the field is an enumeration.
        choices: Option<Vec<String>>,
        /// Advisory default.
        default: Option<String>,
    },
    /// An array of a single element type.
    Array {
        /// Element schema.
        items: Box<Schema>,
        /// Maximum element count, when bounded.
        max_items: Option<usize>,
    },
    /// An object with declared properties.
    Object {
        /// Property name, schema, requirement flag and description.
        properties: Vec<Property>,
        /// Whether undeclared properties are rejected.
        deny_unknown: bool,
    },
    /// A one-of union.
    OneOf(Vec<Schema>),
}

/// One declared property of an [`Schema::Object`].
#[derive(Clone, Debug)]
pub struct Property {
    /// Member name as it appears on the wire.
    pub name: &'static str,
    /// Value schema.
    pub schema: Schema,
    /// Whether the member must be present.
    pub required: bool,
    /// Human-facing description.
    pub description: &'static str,
}

impl Schema {
    /// Declare a required property.
    pub fn req(name: &'static str, schema: Schema, description: &'static str) -> Property {
        Property {
            name,
            schema,
            required: true,
            description,
        }
    }

    /// Declare an optional property.
    pub fn opt(name: &'static str, schema: Schema, description: &'static str) -> Property {
        Property {
            name,
            schema,
            required: false,
            description,
        }
    }

    /// Render as a JSON Schema value.
    pub fn to_json(&self) -> Value {
        use crate::json_obj;
        match self {
            Schema::Any => json_obj! { "type" => Value::Array(vec![
                Value::from("string"), Value::from("number"), Value::from("boolean"),
                Value::from("array"), Value::from("object"), Value::from("null"),
            ]) },
            Schema::Null => json_obj! { "type" => "null" },
            Schema::Boolean => json_obj! { "type" => "boolean" },
            Schema::Integer { min, max } => {
                let mut o = Map::new();
                o.insert("type", "integer");
                if let Some(m) = min {
                    o.insert("minimum", *m);
                }
                if let Some(m) = max {
                    o.insert("maximum", *m);
                }
                Value::Object(o)
            }
            Schema::Number => json_obj! { "type" => "number" },
            Schema::String { choices, default } => {
                let mut o = Map::new();
                o.insert("type", "string");
                if let Some(c) = choices {
                    o.insert(
                        "enum",
                        Value::Array(c.iter().map(|s| Value::from(s.as_str())).collect()),
                    );
                }
                if let Some(d) = default {
                    o.insert("default", d.as_str());
                }
                Value::Object(o)
            }
            Schema::Array { items, max_items } => {
                let mut o = Map::new();
                o.insert("type", "array");
                o.insert("items", items.to_json());
                if let Some(m) = max_items {
                    o.insert("maxItems", *m);
                }
                Value::Object(o)
            }
            Schema::Object {
                properties,
                deny_unknown,
            } => {
                let mut props = Map::new();
                let mut required = Vec::new();
                for p in properties {
                    let mut spec = match p.schema.to_json() {
                        Value::Object(m) => m,
                        other => {
                            let mut m = Map::new();
                            m.insert("allOf", other);
                            m
                        }
                    };
                    spec.insert("description", p.description);
                    props.insert(p.name, Value::Object(spec));
                    if p.required {
                        required.push(Value::from(p.name));
                    }
                }
                let mut o = Map::new();
                o.insert("type", "object");
                o.insert("properties", Value::Object(props));
                if !required.is_empty() {
                    o.insert("required", Value::Array(required));
                }
                o.insert("additionalProperties", !*deny_unknown);
                Value::Object(o)
            }
            Schema::OneOf(variants) => {
                let mut o = Map::new();
                o.insert(
                    "oneOf",
                    Value::Array(variants.iter().map(Schema::to_json).collect()),
                );
                Value::Object(o)
            }
        }
    }
}

/// Which field failed and why.
#[derive(Clone, Debug)]
pub struct FieldError {
    /// Dotted path to the offending member.
    pub path: String,
    /// What was expected.
    pub expected: String,
    /// What arrived, in short form.
    pub found: String,
}

impl FieldError {
    fn new(path: &str, expected: impl Into<String>, found: &Value) -> Self {
        FieldError {
            path: path.to_string(),
            expected: expected.into(),
            found: found.type_name().to_string(),
        }
    }
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: expected {}, found {}",
            self.path, self.expected, self.found
        )
    }
}

impl std::error::Error for FieldError {}

/// A borrowed object being read into typed fields.
///
/// Every accessor takes the path so the error names the member. Call sites look
/// like `p.req_str("at")?`, which is short enough that hand-writing thirty
/// parameter structs is not a burden.
pub struct Obj<'a> {
    value: &'a Value,
    path: String,
}

impl<'a> Obj<'a> {
    /// Wrap a value expected to be an object.
    pub fn new(value: &'a Value) -> Result<Self, FieldError> {
        if !matches!(value, Value::Object(_)) {
            return Err(FieldError::new("<arguments>", "an object", value));
        }
        Ok(Obj {
            value,
            path: String::new(),
        })
    }

    /// Wrap a value, prefixing error paths.
    pub fn at(value: &'a Value, path: impl Into<String>) -> Result<Self, FieldError> {
        let path = path.into();
        if !matches!(value, Value::Object(_)) {
            return Err(FieldError::new(&path, "an object", value));
        }
        Ok(Obj { value, path })
    }

    fn member(&self, key: &str) -> Option<&'a Value> {
        self.value.get(key).filter(|v| !v.is_null())
    }

    fn child_path(&self, key: &str) -> String {
        if self.path.is_empty() {
            key.to_string()
        } else {
            format!("{}.{}", self.path, key)
        }
    }

    /// True when the member is present and not null.
    pub fn has(&self, key: &str) -> bool {
        self.member(key).is_some()
    }

    /// Visit every member not in `known`.
    fn unknown_keys(&self, known: &[&str]) -> Vec<String> {
        self.value
            .as_object()
            .map(|m| {
                m.iter()
                    .filter(|(k, _)| !known.contains(k))
                    .map(|(k, _)| k.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Reject members that no declared field will read.
    ///
    /// A silently ignored typo (`statment` for `statement`) turns into a
    /// confidently wrong answer instead of an error, so unknown keys are
    /// refused rather than tolerated.
    pub fn deny_unknown(&self, known: &[&str]) -> Result<(), FieldError> {
        let extra = self.unknown_keys(known);
        if extra.is_empty() {
            return Ok(());
        }
        Err(FieldError {
            path: self.path.clone(),
            expected: format!("only the documented members ({})", known.join(", ")),
            found: format!("also got {}", extra.join(", ")),
        })
    }

    /// Required string.
    pub fn req_str(&self, key: &str) -> Result<&'a str, FieldError> {
        let path = self.child_path(key);
        let Some(v) = self.member(key) else {
            return Err(FieldError {
                path,
                expected: "a string".into(),
                found: "nothing".into(),
            });
        };
        v.as_str()
            .ok_or_else(|| FieldError::new(&path, "a string", v))
    }

    /// Optional string.
    pub fn opt_str(&self, key: &str) -> Result<Option<&'a str>, FieldError> {
        let path = self.child_path(key);
        match self.member(key) {
            None => Ok(None),
            Some(v) => v
                .as_str()
                .map(Some)
                .ok_or_else(|| FieldError::new(&path, "a string", v)),
        }
    }

    /// Required integer.
    pub fn req_u64(&self, key: &str) -> Result<u64, FieldError> {
        let path = self.child_path(key);
        let Some(v) = self.member(key) else {
            return Err(FieldError {
                path,
                expected: "an integer".into(),
                found: "nothing".into(),
            });
        };
        v.as_u64()
            .ok_or_else(|| FieldError::new(&path, "a non-negative integer", v))
    }

    /// Optional integer.
    pub fn opt_u64(&self, key: &str) -> Result<Option<u64>, FieldError> {
        let path = self.child_path(key);
        match self.member(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .map(Some)
                .ok_or_else(|| FieldError::new(&path, "a non-negative integer", v)),
        }
    }

    /// Optional integer that must fit `usize`.
    pub fn opt_usize(&self, key: &str) -> Result<Option<usize>, FieldError> {
        let path = self.child_path(key);
        match self.opt_u64(key)? {
            None => Ok(None),
            Some(n) => usize::try_from(n).map(Some).map_err(|_| FieldError {
                path,
                expected: format!("an integer below {}", usize::MAX),
                found: format!("{n}"),
            }),
        }
    }

    /// Optional boolean.
    pub fn opt_bool(&self, key: &str) -> Result<Option<bool>, FieldError> {
        let path = self.child_path(key);
        match self.member(key) {
            None => Ok(None),
            Some(v) => v
                .as_bool()
                .map(Some)
                .ok_or_else(|| FieldError::new(&path, "a boolean", v)),
        }
    }

    /// Optional array of strings.
    pub fn opt_str_vec(&self, key: &str) -> Result<Vec<String>, FieldError> {
        let path = self.child_path(key);
        let Some(v) = self.member(key) else {
            return Ok(Vec::new());
        };
        let Some(items) = v.as_array() else {
            return Err(FieldError::new(&path, "an array of strings", v));
        };
        let mut out = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let Some(s) = item.as_str() else {
                return Err(FieldError::new(&format!("{path}[{i}]"), "a string", item));
            };
            out.push(s.to_string());
        }
        Ok(out)
    }
}

/// Extract a value that names one of a fixed set of labels.
pub fn one_of<'a>(
    obj: &Obj<'_>,
    key: &str,
    allowed: &[&'a str],
) -> Result<Option<&'a str>, FieldError> {
    let Some(raw) = obj.opt_str(key)? else {
        return Ok(None);
    };
    match allowed.iter().find(|a| **a == raw) {
        Some(a) => Ok(Some(a)),
        None => Err(FieldError {
            path: key.to_string(),
            expected: format!("one of {}", allowed.join(", ")),
            found: format!("{raw:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    #[test]
    fn schema_renders_required_and_optional() {
        let s = Schema::Object {
            properties: vec![
                Schema::req(
                    "path",
                    Schema::String {
                        choices: None,
                        default: None,
                    },
                    "file to open",
                ),
                Schema::opt(
                    "label",
                    Schema::String {
                        choices: None,
                        default: None,
                    },
                    "display name",
                ),
            ],
            deny_unknown: true,
        };
        let j = s.to_json();
        assert_eq!(
            j.get("required").and_then(Value::as_array).map(Vec::len),
            Some(1)
        );
        assert_eq!(
            j.get("additionalProperties").and_then(Value::as_bool),
            Some(false)
        );
        let props = j.get("properties").and_then(Value::as_object).unwrap();
        assert_eq!(props.len(), 2);
        assert_eq!(
            props
                .get("path")
                .and_then(|p| p.get("description"))
                .and_then(Value::as_str),
            Some("file to open")
        );
    }

    #[test]
    fn missing_required_field_names_the_path() {
        let v = parse(r#"{}"#).unwrap();
        let obj = Obj::new(&v).unwrap();
        let err = obj.req_str("path").unwrap_err();
        assert_eq!(err.path, "path");
        assert_eq!(err.found, "nothing");
    }

    #[test]
    fn wrong_type_reports_what_arrived() {
        let v = parse(r#"{"len": "eight"}"#).unwrap();
        let obj = Obj::new(&v).unwrap();
        let err = obj.opt_u64("len").unwrap_err();
        assert_eq!(err.found, "string");
        assert!(err.to_string().contains("len"));
    }

    #[test]
    fn null_is_treated_as_absent() {
        // A client that serialises an unset optional as `null` should get the
        // same behaviour as one that omits it. Rejecting `null` for an optional
        // field is a compatibility bug waiting to happen.
        let v = parse(r#"{"len": null}"#).unwrap();
        let obj = Obj::new(&v).unwrap();
        assert_eq!(obj.opt_u64("len").unwrap(), None);
        assert!(!obj.has("len"));
    }

    #[test]
    fn unknown_members_are_refused_with_a_list() {
        let v = parse(r#"{"path":"/bin/ls","lable":"x"}"#).unwrap();
        let obj = Obj::new(&v).unwrap();
        let err = obj.deny_unknown(&["path", "label"]).unwrap_err();
        assert!(err.found.contains("lable"), "{err}");
    }

    #[test]
    fn nested_arrays_report_the_index() {
        // Second element is a number, not a string: the error must name the
        // index, because "expected a string" without a position is useless in
        // a twenty-element breakpoint list.
        let v = parse(r#"{"b":["0x1",42]}"#).unwrap();
        let obj = Obj::new(&v).unwrap();
        let err = obj.opt_str_vec("b").unwrap_err();
        assert_eq!(err.path, "b[1]");
        assert_eq!(err.found, "number");
    }

    #[test]
    fn one_of_accepts_only_listed_labels() {
        let v = parse(r#"{"depth":"summary"}"#).unwrap();
        let obj = Obj::new(&v).unwrap();
        assert_eq!(
            one_of(&obj, "depth", &["summary", "detail"]).unwrap(),
            Some("summary")
        );

        let v = parse(r#"{"depth":"deep"}"#).unwrap();
        let obj = Obj::new(&v).unwrap();
        let err = one_of(&obj, "depth", &["summary", "detail"]).unwrap_err();
        assert!(err.expected.contains("summary"));
    }
}
