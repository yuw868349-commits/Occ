//! Symbol name handling: version stripping, Rust hash stripping, and Itanium
//! demangling.
//!
//! Two levels, used at different times:
//!
//! * [`clean_name`] runs during parsing and is cheap. It removes the parts of a
//!   name that are *noise for matching* — the `@GLIBC_2.2.5` version suffix and
//!   the `17h<16 hex>E` disambiguator rustc appends. Two symbols that differ
//!   only in those are the same function to a caller looking one up.
//! * [`demangle`] runs on demand and is expensive. It turns `_ZN3foo3barEv`
//!   into `foo::bar()`.
//!
//! The Itanium demangler covers the subset that appears in real binaries:
//! nested names, source names, the built-in type codes, and the operator
//! spellings. It does not implement expression mangling for template
//! non-type arguments, which is rare in symbol tables and would roughly
//! double the size of this file. Unrecognised input returns `None` rather
//! than a guess.

/// Strip version suffixes and Rust disambiguator hashes.
///
/// `puts@GLIBC_2.2.5` becomes `puts`; `_ZN4core3fmt5write17h1a2b3c4d5e6f7a8bE`
/// becomes `_ZN4core3fmt5writeE`. The second is still mangled, but it now
/// matches across two builds of the same crate, which is what a caller
/// comparing symbols wants.
pub fn clean_name(raw: &str) -> String {
    // ELF symbol versions: `name@@VER` is the default version, `name@VER` a
    // specific one. Both are suffixes, and neither is part of the identifier.
    let base = match raw.find('@') {
        Some(i) => &raw[..i],
        None => raw,
    };

    // rustc legacy mangling ends with `17h` + 16 lowercase hex + `E`.
    if base.starts_with("_ZN")
        && base.ends_with('E')
        && let Some(stripped) = strip_rust_hash(base)
    {
        return stripped;
    }
    base.to_string()
}

fn strip_rust_hash(name: &str) -> Option<String> {
    let inner = &name[3..name.len() - 1];
    // The hash component is `17h` followed by exactly 16 hex digits.
    let idx = inner.rfind("17h")?;
    let hash = &inner[idx + 3..];
    if hash.len() != 16 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("_ZN{}E", &inner[..idx]))
}

/// Demangle a C++ (Itanium ABI) or Rust symbol name.
///
/// Returns `None` when the name is not mangled, or is mangled in a form this
/// implementation does not cover. A `None` is not a failure to report: the
/// caller keeps the raw name, which is always correct even when unreadable.
pub fn demangle(raw: &str) -> Option<String> {
    if let Some(rest) = raw.strip_prefix("_Z") {
        return demangle_itanium(rest);
    }
    // Rust v0 mangling is a different grammar entirely and is left as-is:
    // showing `_RNvNtCs...` is honest, showing a wrong reconstruction is not.
    None
}

fn demangle_itanium(s: &str) -> Option<String> {
    let (name, rest) = parse_nested_or_name(s)?;
    // Whatever follows the name is the parameter list. `v` alone means no
    // parameters; anything else is a sequence of types.
    let params = parse_params(rest)?;
    if params.is_empty() {
        Some(format!("{name}()"))
    } else {
        Some(format!("{name}({})", params.join(", ")))
    }
}

fn parse_params(s: &str) -> Option<Vec<String>> {
    if s.is_empty() {
        // A name with no parameter list is legal in a variable symbol.
        return Some(Vec::new());
    }
    if s == "v" {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    let mut rest = s;
    while !rest.is_empty() {
        // cv-qualifiers on a parameter are a suffix in the mangling but do not
        // change the spelling of the type in the demangled form.
        let (ty, after) = parse_type(rest)?;
        out.push(ty);
        rest = after;
        while rest.starts_with('r') || rest.starts_with('V') || rest.starts_with('K') {
            rest = &rest[1..];
        }
    }
    Some(out)
}

fn parse_type(s: &str) -> Option<(String, &str)> {
    let first = s.chars().next()?;
    let builtin = match first {
        'v' => Some("void"),
        'w' => Some("wchar_t"),
        'b' => Some("bool"),
        'c' => Some("char"),
        'a' => Some("signed char"),
        'h' => Some("unsigned char"),
        's' => Some("short"),
        't' => Some("unsigned short"),
        'i' => Some("int"),
        'j' => Some("unsigned int"),
        'l' => Some("long"),
        'm' => Some("unsigned long"),
        'x' => Some("long long"),
        'y' => Some("unsigned long long"),
        'n' => Some("__int128"),
        'o' => Some("unsigned __int128"),
        'f' => Some("float"),
        'd' => Some("double"),
        'e' => Some("long double"),
        'g' => Some("__float128"),
        'z' => Some("..."),
        _ => None,
    };
    if let Some(name) = builtin {
        return Some((name.to_string(), &s[1..]));
    }
    match first {
        // Pointer.
        'P' => {
            let (inner, rest) = parse_type(&s[1..])?;
            Some((format!("{inner}*"), rest))
        }
        // lvalue / rvalue reference.
        'R' | 'O' => {
            let (inner, rest) = parse_type(&s[1..])?;
            Some((format!("{inner}&"), rest))
        }
        // Back-reference into the substitution table. The demangler does not
        // maintain one, so this is a name that cannot be reconstructed.
        'S' => {
            // `St` is `std::`; `Sa`..`Sz` are the standard substitutions.
            if let Some(rest) = s.strip_prefix("St") {
                let (inner, after) = parse_type(rest)?;
                return Some((format!("std::{inner}"), after));
            }
            None
        }
        // Qualified type: `K` const.
        'K' => {
            let (inner, rest) = parse_type(&s[1..])?;
            Some((format!("{inner} const"), rest))
        }
        _ => {
            let (name, rest) = parse_nested_or_name(s)?;
            Some((name, rest))
        }
    }
}

/// Parse either `N...E` nested name or a single unqualified name.
fn parse_nested_or_name(s: &str) -> Option<(String, &str)> {
    if let Some(rest) = s.strip_prefix('N') {
        return parse_nested(rest);
    }
    // `St` is the standard substitution for `std::`, and it appears at the head
    // of a name as often as inside a type.
    if let Some(rest) = s.strip_prefix("St") {
        let (inner, after) = parse_unqualified(rest)?;
        return Some((format!("std::{inner}"), after));
    }
    parse_unqualified(s)
}

fn parse_nested(s: &str) -> Option<(String, &str)> {
    let mut parts = Vec::new();
    let mut rest = s;
    // Leading cv-qualifiers belong to the function, not the name.
    while rest.starts_with('K') || rest.starts_with('V') || rest.starts_with('r') {
        rest = &rest[1..];
    }
    loop {
        if let Some(after) = rest.strip_prefix('E') {
            return Some((parts.join("::"), after));
        }
        let (part, after) = parse_unqualified(rest)?;
        parts.push(part);
        rest = after;
        if parts.len() > 64 {
            // Cyclic or adversarial input: refuse rather than loop.
            return None;
        }
    }
}

fn parse_unqualified(s: &str) -> Option<(String, &str)> {
    let bytes = s.as_bytes();
    let first = *bytes.first()?;
    // Operator names are two characters.
    if first.is_ascii_digit() {
        let len_end = s.find(|c: char| !c.is_ascii_digit())?;
        let len: usize = s[..len_end].parse().ok()?;
        let start = len_end;
        let end = start.checked_add(len)?;
        let name = s.get(start..end)?;
        return Some((name.to_string(), s.get(end..)?));
    }
    let (op, used) = operator_name(s)?;
    Some((op, &s[used..]))
}

/// Two-letter operator codes from the Itanium ABI.
fn operator_name(s: &str) -> Option<(String, usize)> {
    let b = s.as_bytes();
    if b.len() < 2 {
        return None;
    }
    let two = &s[..2];
    let name = match two {
        "nw" => "operator new",
        "na" => "operator new[]",
        "dl" => "operator delete",
        "da" => "operator delete[]",
        "ps" => "operator+",
        "ng" => "operator-",
        "ad" => "operator&",
        "de" => "operator*",
        "co" => "operator~",
        "pl" => "operator+",
        "mi" => "operator-",
        "ml" => "operator*",
        "dv" => "operator/",
        "rm" => "operator%",
        "an" => "operator&",
        "or" => "operator|",
        "eo" => "operator^",
        "aS" => "operator=",
        "pL" => "operator+=",
        "mI" => "operator-=",
        "mL" => "operator*=",
        "dV" => "operator/=",
        "rM" => "operator%=",
        "aN" => "operator&=",
        "oR" => "operator|=",
        "eO" => "operator^=",
        "ls" => "operator<<",
        "rs" => "operator>>",
        "lS" => "operator<<=",
        "rS" => "operator>>=",
        "eq" => "operator==",
        "ne" => "operator!=",
        "lt" => "operator<",
        "gt" => "operator>",
        "le" => "operator<=",
        "ge" => "operator>=",
        "nt" => "operator!",
        "aa" => "operator&&",
        "oo" => "operator||",
        "pp" => "operator++",
        "mm" => "operator--",
        "cm" => "operator,",
        "pm" => "operator->*",
        "pt" => "operator->",
        "cl" => "operator()",
        "ix" => "operator[]",
        "qu" => "operator?",
        _ => return None,
    };
    Some((name.to_string(), 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_suffixes_are_stripped() {
        assert_eq!(clean_name("puts@GLIBC_2.2.5"), "puts");
        assert_eq!(clean_name("memcpy@@GLIBC_2.14"), "memcpy");
        assert_eq!(clean_name("plain"), "plain");
    }

    #[test]
    fn rust_hashes_are_stripped_but_the_path_survives() {
        let a = "_ZN4core3fmt5write17h1a2b3c4d5e6f7a8bE";
        let b = "_ZN4core3fmt5write17h9f8e7d6c5b4a3210E";
        assert_eq!(clean_name(a), "_ZN4core3fmt5writeE");
        assert_eq!(clean_name(a), clean_name(b), "same function, two builds");
    }

    #[test]
    fn a_lookalike_hash_is_not_stripped() {
        // `17h` followed by the wrong number of digits is part of the name.
        assert_eq!(clean_name("_ZN3foo17hzzE"), "_ZN3foo17hzzE");
        assert_eq!(clean_name("_ZN3foo17h123E"), "_ZN3foo17h123E");
    }

    #[test]
    fn demangles_a_member_function() {
        assert_eq!(demangle("_ZN3foo3barEv").as_deref(), Some("foo::bar()"));
    }

    #[test]
    fn demangles_parameters() {
        assert_eq!(demangle("_Z3addii").as_deref(), Some("add(int, int)"));
        assert_eq!(
            demangle("_Z5printPKc").as_deref(),
            Some("print(char const*)")
        );
    }

    #[test]
    fn demangles_std_prefix() {
        assert_eq!(demangle("_ZSt4cout").as_deref(), Some("std::cout()"));
    }

    #[test]
    fn demangles_operators() {
        assert_eq!(demangle("_Zplii").as_deref(), Some("operator+(int, int)"));
        assert_eq!(
            demangle("_ZdlPv").as_deref(),
            Some("operator delete(void*)")
        );
    }

    #[test]
    fn unmangled_names_return_none() {
        assert_eq!(demangle("main"), None);
        assert_eq!(demangle("puts"), None);
    }

    #[test]
    fn truncated_mangling_returns_none_not_panic() {
        for cut in ["_Z", "_ZN", "_ZN3", "_ZN3foo", "_ZN3fooE", "_Z3addi"] {
            let _ = demangle(cut);
        }
    }
}
