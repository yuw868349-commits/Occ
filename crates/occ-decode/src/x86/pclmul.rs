//! The carry-less multiplication, named after the quadwords it works on.
//!
//! `pclmulqdq` multiplies two quadwords of a 128-bit register and writes
//! the 128-bit result to the low half of the destination. Which
//! quadwords it multiplies is decided by the immediate byte, and the
//! instruction has a name for each of the four choices, so a listing that
//! calls all of them `pclmulqdq` has thrown away the only thing that
//! tells them apart.
//!
//! The immediate byte carries three fields:
//!
//! * bit 0 selects the quadword of the first source,
//! * bit 1 selects the quadword of the destination,
//! * bit 4 selects the quadword of the second source.
//!
//! The legacy form has only one source besides the destination, so bit 4
//! has no meaning there. The names below follow the rules that GNU's
//! disassembler uses, which were read out of its source and then checked
//! against a second disassembler byte for byte.
//!
//! Two of the immediate values name the same instruction as two others:
//! `0x10` and `0x11` select the same quadwords as `0x02` and `0x03`,
//! because the extra bit picks the third source and the name does not
//! say which one. Every other value is a reserved extension, and the
//! instruction keeps its plain name with the byte printed as an
//! immediate.
//!
//! The names disagree with the other disassembler, which spells every
//! one of them `pclmulqdq` and leaves the byte in the operand list. The
//! four names here are the ones the manual gives, so they are the ones
//! used.

/// The name of the legacy instruction for each value of the immediate byte.
///
/// The four names cover the values `0x00` through `0x03`.
const NAMES: [&str; 4] = [
    "pclmullqlqdq",
    "pclmulhqlqdq",
    "pclmullqhqdq",
    "pclmulhqhqdq",
];

/// The vector forms of the same names.
const VECTOR_NAMES: [&str; 4] = [
    "vpclmullqlqdq",
    "vpclmulhqlqdq",
    "vpclmullqhqdq",
    "vpclmulhqhqdq",
];

/// The name of the carry-less multiplication for the given immediate byte.
///
/// Returns `None` when the byte names no choice of quadwords, in which
/// case the instruction keeps its plain name and the byte is printed as
/// an immediate operand.
pub fn name(vector: bool, imm: u8) -> Option<&'static str> {
    // The bit that selects the third source does not change the name, so
    // `0x10` and `0x11` name the same instructions as `0x02` and `0x03`.
    let index = match imm {
        0x10 => 2usize,
        0x11 => 3,
        other => usize::from(other),
    };
    let table = if vector { &VECTOR_NAMES } else { &NAMES };
    table.get(index).copied()
}
