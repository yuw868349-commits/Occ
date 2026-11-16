//! JSON values, ordered maps and numeric precision rules.
//!
//! The data model follows RFC 8259 with two deliberate departures from the
//! obvious implementation, both of which matter for this project:
//!
//! Object member order is preserved. Analysis replies are read by language
//! models and by humans skimming a transcript; a map that reorders itself
//! between two identical calls makes byte-level comparison of replies
//! impossible and turns every diff into noise.
//!
//! Integers never pass through `f64`. Addresses in this toolset routinely
//! exceed 2^53, and `9007199254740993` round-tripping to
//! `9007199254740992` would corrupt exactly the values the whole design is
//! built to keep exact.

use std::fmt;

/// A JSON number.
///
/// The three variants are not an optimisation: they are the three cases that
/// must be told apart on the wire. A `u64` field that arrives as `-1` is an
/// error, not a float, and a address that arrives as `1.8e19` must not be
/// silently truncated into an integer.
#[derive(Clone, Copy, Debug)]
pub enum Number {
    /// Non-negative integer that fits `u64`.
    PosInt(u64),
    /// Negative integer that fits `i64`.
    NegInt(i64),
    /// Anything with a fractional part or an exponent.
    Float(f64),
}

impl Number {
    /// Integer value, if this number is integral and fits `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Number::PosInt(u) => i64::try_from(u).ok(),
            Number::NegInt(i) => Some(i),
            Number::Float(f) => {
                if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                    Some(f as i64)
                } else {
                    None
                }
            }
        }
    }

    /// Integer value, if this number is integral and fits `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Number::PosInt(u) => Some(u),
            Number::NegInt(i) => u64::try_from(i).ok(),
            Number::Float(f) => {
                if f.fract() == 0.0 && f >= 0.0 && f <= u64::MAX as f64 {
                    Some(f as u64)
                } else {
                    None
                }
            }
        }
    }

    /// Value as `f64`, with the usual precision loss above 2^53.
    pub fn as_f64(&self) -> f64 {
        match *self {
            Number::PosInt(u) => u as f64,
            Number::NegInt(i) => i as f64,
            Number::Float(f) => f,
        }
    }

    /// True for the two integer variants.
    pub fn is_integer(&self) -> bool {
        matches!(self, Number::PosInt(_) | Number::NegInt(_))
    }
}

impl From<u64> for Number {
    fn from(v: u64) -> Self {
        Number::PosInt(v)
    }
}

impl From<u32> for Number {
    fn from(v: u32) -> Self {
        Number::PosInt(u64::from(v))
    }
}

impl From<usize> for Number {
    fn from(v: usize) -> Self {
        Number::PosInt(v as u64)
    }
}

impl From<i64> for Number {
    fn from(v: i64) -> Self {
        if v < 0 {
            Number::NegInt(v)
        } else {
            Number::PosInt(v as u64)
        }
    }
}

impl From<i32> for Number {
    fn from(v: i32) -> Self {
        Number::from(i64::from(v))
    }
}

impl From<f64> for Number {
    fn from(v: f64) -> Self {
        // A float that happens to be integral stays a float: `1.0` and `1` are
        // different JSON texts, and a caller that sent `1.0` for a float field
        // should get `1.0` back.
        Number::Float(v)
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Number::PosInt(u) => write!(f, "{u}"),
            Number::NegInt(i) => write!(f, "{i}"),
            Number::Float(v) => write_float(f, v),
        }
    }
}

/// Write a float so the output parses back to the same value.
///
/// `f64`'s `Display` already produces the shortest round-tripping decimal, so
/// the only work left is JSON's requirement that an integral value carry a
/// fractional part — `1` reads as an integer, `1.0` reads as a float, and a
/// reader that dispatches on that distinction needs the difference kept.
fn write_float(f: &mut fmt::Formatter<'_>, v: f64) -> fmt::Result {
    if v.is_nan() || v.is_infinite() {
        // RFC 8259 has no representation for these. `null` is the least
        // surprising substitute and is what a JSON reader will expect if it
        // sees the field at all.
        return f.write_str("null");
    }
    let s = format!("{v}");
    f.write_str(&s)?;
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        f.write_str(".0")?;
    }
    Ok(())
}

/// An insertion-ordered JSON object.
///
/// Backed by a `Vec` because replies are small (tens of keys, not thousands),
/// and because stable key order is worth more here than O(1) lookup. Duplicate
/// keys on parse keep the last value, matching every mainstream parser.
#[derive(Clone, Debug, Default)]
pub struct Map {
    entries: Vec<(String, Value)>,
}

impl Map {
    /// Empty object.
    pub fn new() -> Self {
        Map {
            entries: Vec::new(),
        }
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when there are no members.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a member.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Look up a member mutably.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Insert or replace a member, preserving the position of an existing key.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        let key = key.into();
        let value = value.into();
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
            return;
        }
        self.entries.push((key, value));
    }

    /// Remove a member, returning its value.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let idx = self.entries.iter().position(|(k, _)| k == key)?;
        Some(self.entries.remove(idx).1)
    }

    /// Iterate over members in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Iterate mutably over member values.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Value> {
        self.entries.iter_mut().map(|(_, v)| v)
    }
}

impl FromIterator<(String, Value)> for Map {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(iter: I) -> Self {
        let mut map = Map::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

/// A JSON value.
#[derive(Clone, Debug)]
pub enum Value {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number.
    Number(Number),
    /// A string. Valid UTF-8 by construction.
    String(String),
    /// An array.
    Array(Vec<Value>),
    /// An object.
    Object(Map),
}

impl Value {
    /// The `null` value.
    pub const NULL: Value = Value::Null;

    /// True when this is `Null`.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Borrow as `str` when this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as `bool` when this is a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Borrow as a number.
    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Value::Number(n) => Some(n),
            _ => None,
        }
    }

    /// Integral value as `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        self.as_number().and_then(Number::as_i64)
    }

    /// Integral value as `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        self.as_number().and_then(Number::as_u64)
    }

    /// Value as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        self.as_number().map(Number::as_f64)
    }

    /// Borrow as an array.
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Borrow as an array mutably.
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Borrow as an object.
    pub fn as_object(&self) -> Option<&Map> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow as an object mutably.
    pub fn as_object_mut(&mut self) -> Option<&mut Map> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Member lookup. Returns `None` for non-objects.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object().and_then(|m| m.get(key))
    }

    /// Member lookup mutably.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.as_object_mut().and_then(|m| m.get_mut(key))
    }

    /// Index into an array, or look up a member of an object.
    pub fn at(&self, index: usize) -> Option<&Value> {
        self.as_array().and_then(|a| a.get(index))
    }

    /// Short name of this value's type, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::Number(Number::PosInt(v))
    }
}

impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::Number(Number::PosInt(u64::from(v)))
    }
}

impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::Number(Number::PosInt(v as u64))
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Number(Number::from(v))
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Number(Number::from(v))
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Number(Number::Float(v))
    }
}

impl From<Number> for Value {
    fn from(v: Number) -> Self {
        Value::Number(v)
    }
}

impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::Array(v)
    }
}

impl From<Map> for Value {
    fn from(v: Map) -> Self {
        Value::Object(v)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(v) => v.into(),
            None => Value::Null,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&crate::write::to_string(self))
    }
}

/// Build an object from `key => value` pairs.
///
/// Keys accept anything `Into<String>`; values anything `Into<Value>`. Written
/// as a macro rather than a builder method because reply assembly is the single
/// most common operation in the MCP layer, and a chain of `.insert(..)` calls
/// buries the shape of the reply.
#[macro_export]
macro_rules! json_obj {
    () => { $crate::Value::Object($crate::Map::new()) };
    ($($key:expr => $value:expr),+ $(,)?) => {{
        let mut map = $crate::Map::new();
        $( map.insert($key, $crate::Value::from($value)); )+
        $crate::Value::Object(map)
    }};
}

/// Build an array.
#[macro_export]
macro_rules! json_arr {
    () => { $crate::Value::Array(Vec::new()) };
    ($($value:expr),+ $(,)?) => {
        $crate::Value::Array(vec![$($crate::Value::from($value)),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_integers_survive_exactly() {
        // 2^53 + 1 is the first integer an f64 cannot hold. A parser that goes
        // through `f64` returns 9007199254740992 here and every address above
        // 8 PiB silently shifts.
        let v: Value = Value::Number(Number::PosInt(9007199254740993));
        assert_eq!(crate::write::to_string(&v), "9007199254740993");
        assert_eq!(v.as_u64(), Some(9007199254740993));
    }

    #[test]
    fn negative_and_positive_are_distinct_variants() {
        assert!(matches!(Number::from(-1i64), Number::NegInt(-1)));
        assert!(matches!(Number::from(1i64), Number::PosInt(1)));
        assert_eq!(Number::from(0i64).as_u64(), Some(0));
    }

    #[test]
    fn object_order_is_insertion_order() {
        let mut m = Map::new();
        m.insert("z", 1u64);
        m.insert("a", 2u64);
        m.insert("m", 3u64);
        let keys: Vec<&str> = m.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, ["z", "a", "m"]);
    }

    #[test]
    fn inserting_an_existing_key_keeps_its_position() {
        let mut m = Map::new();
        m.insert("a", 1u64);
        m.insert("b", 2u64);
        m.insert("a", 9u64);
        let keys: Vec<&str> = m.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, ["a", "b"]);
        assert_eq!(m.get("a").and_then(Value::as_u64), Some(9));
    }

    #[test]
    fn integral_floats_keep_their_fractional_part() {
        let v = Value::from(1.0f64);
        assert_eq!(crate::write::to_string(&v), "1.0");
    }

    #[test]
    fn non_finite_floats_become_null() {
        assert_eq!(crate::write::to_string(&Value::from(f64::NAN)), "null");
        assert_eq!(crate::write::to_string(&Value::from(f64::INFINITY)), "null");
    }
}
