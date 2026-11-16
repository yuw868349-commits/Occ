//! Virtual addresses.

use std::fmt;

/// A virtual address inside a loaded image.
///
/// This is a distinct type rather than a bare `u64` on purpose. Mixing file
/// offsets and virtual addresses is the single most common source of wrong
/// answers in binary tooling, and a newtype makes the mistake a compile error
/// instead of a plausible-looking wrong number.
///
/// # Serialisation
///
/// Renders as a hex string (`"0x401000"`), never as a JSON number. JavaScript
/// clients parse JSON numbers as IEEE-754 doubles and lose precision above
/// 2^53; a 64-bit image base plus a large offset can exceed that. Deserialising
/// accepts a hex string, a decimal string, or a JSON integer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Addr(pub u64);

impl Addr {
    /// Address zero, used as a null sentinel.
    pub const NULL: Addr = Addr(0);

    /// Construct from a raw value.
    pub const fn new(value: u64) -> Self {
        Addr(value)
    }

    /// Raw value.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Add a signed delta, saturating at zero and at `u64::MAX`.
    ///
    /// Saturation rather than wrapping is deliberate: a backward jump that
    /// underflows below the image base is a corrupted control-flow edge, and
    /// clamping to zero makes that visible instead of silently landing in
    /// valid-looking high memory.
    pub fn offset(self, delta: i64) -> Addr {
        if delta >= 0 {
            Addr(self.0.saturating_add(delta as u64))
        } else {
            Addr(self.0.saturating_sub(delta.unsigned_abs()))
        }
    }

    /// Forward by `n` bytes.
    pub const fn add(self, n: u64) -> Addr {
        Addr(self.0.saturating_add(n))
    }

    /// True when inside `[start, start + len)`.
    pub const fn within(self, start: Addr, len: u64) -> bool {
        self.0 >= start.0 && self.0 - start.0 < len
    }

    /// True when zero.
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

impl fmt::Debug for Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

impl From<u64> for Addr {
    fn from(value: u64) -> Self {
        Addr(value)
    }
}

impl From<Addr> for u64 {
    fn from(value: Addr) -> Self {
        value.0
    }
}

/// Parse `0x`-prefixed hex, bare hex with an `h` suffix, or decimal.
///
/// Accepting several spellings is intentional: an LLM reading `objdump` output
/// will produce `401000`, a human will write `0x401000`, and a Windows-oriented
/// tool may emit `401000h`. Rejecting any of them would turn a formatting
/// difference into an analysis failure.
pub fn parse_addr(input: &str) -> Option<Addr> {
    let s = input.trim();
    if s.is_empty() {
        return None;
    }
    let (digits, radix) = if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))
    {
        (rest, 16)
    } else if let Some(rest) = s.strip_suffix('h').or_else(|| s.strip_suffix('H')) {
        (rest, 16)
    } else {
        (s, 10)
    };
    let cleaned: String = digits.chars().filter(|c| *c != '_').collect();
    if cleaned.is_empty() {
        return None;
    }
    u64::from_str_radix(&cleaned, radix).ok().map(Addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documented_spellings() {
        assert_eq!(parse_addr("0x401000"), Some(Addr(0x401000)));
        assert_eq!(parse_addr("401000h"), Some(Addr(0x401000)));
        assert_eq!(parse_addr("4198400"), Some(Addr(0x401000)));
        assert_eq!(parse_addr("0x40_10_00"), Some(Addr(0x401000)));
        assert_eq!(parse_addr("  0X10  "), Some(Addr(16)));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_addr(""), None);
        assert_eq!(parse_addr("0x"), None);
        assert_eq!(parse_addr("zz"), None);
        assert_eq!(parse_addr("0xgg"), None);
    }

    #[test]
    fn renders_as_hex_string_not_number() {
        // On the wire an address is always a string. A JSON number above 2^53
        // comes back mangled from any JavaScript client, and the mangling looks
        // like a wrong address rather than a transport bug.
        assert_eq!(Addr(0x401000).to_string(), "0x401000");
        assert_eq!(Addr(0).to_string(), "0x0");
    }

    #[test]
    fn round_trips_large_address() {
        // 2^53 + 1: the first integer a double cannot represent.
        let big = Addr((1u64 << 53) + 1);
        assert_eq!(parse_addr(&big.to_string()), Some(big));
    }

    #[test]
    fn accepts_decimal_input() {
        assert_eq!(parse_addr("4096"), Some(Addr(4096)));
    }

    #[test]
    fn offset_saturates_instead_of_wrapping() {
        assert_eq!(Addr(16).offset(-32), Addr(0));
        assert_eq!(Addr(16).offset(-8), Addr(8));
        assert_eq!(Addr(u64::MAX).add(1), Addr(u64::MAX));
    }

    #[test]
    fn within_is_half_open() {
        let start = Addr(0x1000);
        assert!(Addr(0x1000).within(start, 4));
        assert!(Addr(0x1003).within(start, 4));
        assert!(!Addr(0x1004).within(start, 4));
        assert!(!Addr(0x0fff).within(start, 4));
    }
}
