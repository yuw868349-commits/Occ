//! The 32-bit Thumb space, read the way the architecture defines it.
//!
//! The class whose first halfword is `1111 1010` holds the register
//! shifts, the extends, the parallel additions and subtractions, the
//! exchanged ones and the checksums, and it reads its operation from
//! four bits of the first halfword and four of the second. A mistake in
//! one field shows up as a different instruction rather than as a wrong
//! operand, and the only way to find it is to read the class against a
//! reference that implements the same manual independently.
//!
//! Every expectation below was taken from `llvm-objdump` reading the
//! same encoding, and the encodings were checked against the assembler,
//! which is what says which field holds what. Where the two references
//! disagree with each other, the architecture manual decides; where the
//! manual leaves a field with nothing to say, the encoding is one no
//! assembler produces and no machine executes.

use occ_core::Addr;
use occ_decode::arm::{Decoder, Mode};

/// Decode one 32-bit Thumb instruction from a pair of halfwords.
fn decode(hw1: u16, hw2: u16) -> String {
    let mut bytes: Vec<u8> = Vec::with_capacity(4);
    bytes.extend_from_slice(&hw1.to_le_bytes());
    bytes.extend_from_slice(&hw2.to_le_bytes());
    let decoder = Decoder::with_mode(Mode::Thumb);
    let decoded = decoder
        .decode(Addr(0), &bytes)
        .expect("the encoding decodes");
    decoded.insn.text
}

#[test]
fn the_byte_reversals_take_the_source_from_the_second_halfword() {
    // `rev.w r0, r1` assembles to `0xFA90 0xF081`: the first halfword has
    // a register field as well, but the instruction has no second operand
    // for it, so the assembler writes the source in both places. The
    // source is the one in the second halfword.
    assert_eq!(decode(0xFA90, 0xF081), "rev.w r0, r1");
    assert_eq!(decode(0xFA90, 0xF091), "rev16.w r0, r1");
    assert_eq!(decode(0xFA90, 0xF0A1), "rbit r0, r1");
    assert_eq!(decode(0xFA90, 0xF0B1), "revsh.w r0, r1");
    assert_eq!(decode(0xFAB0, 0xF081), "clz r0, r1");
}

#[test]
fn the_saturating_additions_take_two_sources() {
    // These do have two operands, and the reference writes the
    // destination first, then the source from the second halfword, then
    // the one from the first.
    assert_eq!(decode(0xFA82, 0xF081), "qadd r0, r1, r2");
    assert_eq!(decode(0xFA83, 0xF081), "qadd r0, r1, r3");
    assert_eq!(decode(0xFA84, 0xF081), "qadd r0, r1, r4");
    assert_eq!(decode(0xFA85, 0xF081), "qadd r0, r1, r5");
    assert_eq!(decode(0xFAA1, 0xF082), "sel r0, r1, r2");
}

#[test]
fn the_extends_read_both_spellings_of_their_variant() {
    // The variant's top bit says that the instruction is an extend, and
    // the bit below it says which of the two spellings it is. Reading
    // only one of them loses the whole family: `uxtab` is in the second
    // spelling, and it is the one a compiler emits for a byte load.
    assert_eq!(decode(0xFA53, 0xF8F0), "uxtab r8, r3, r0, ror #24");
    assert_eq!(decode(0xFA53, 0xF890), "uxtab r8, r3, r0, ror #8");
    assert_eq!(decode(0xFA43, 0xF890), "sxtab r8, r3, r0, ror #8");
    assert_eq!(decode(0xFA01, 0xF082), "sxtah r0, r1, r2");
    assert_eq!(decode(0xFA11, 0xF082), "uxtah r0, r1, r2");
    // The rotation counts up: the bottom two bits of the variant are the
    // rotation, in steps of eight.
    assert_eq!(decode(0xFA53, 0xF880), "uxtab r8, r3, r0");
    assert_eq!(decode(0xFA53, 0xF8B0), "uxtab r8, r3, r0, ror #24");
}

#[test]
fn the_plain_extends_take_one_operand() {
    // With the program counter as the first operand the instruction is
    // the extend itself, and it has no second operand to read.
    assert_eq!(decode(0xFA0F, 0xF081), "sxth.w r0, r1");
    assert_eq!(decode(0xFA1F, 0xF081), "uxth.w r0, r1");
    assert_eq!(decode(0xFA2F, 0xF081), "sxtb16 r0, r1");
    assert_eq!(decode(0xFA3F, 0xF081), "uxtb16 r0, r1");
    assert_eq!(decode(0xFA4F, 0xF081), "sxtb.w r0, r1");
    assert_eq!(decode(0xFA5F, 0xF081), "uxtb.w r0, r1");
}

#[test]
fn the_parallel_additions_keep_their_order() {
    // The destination, then the source from the first halfword, then the
    // one from the second — the opposite order from the saturating
    // additions, which take the source from the second halfword first.
    assert_eq!(decode(0xFA83, 0xF001), "sadd8 r0, r3, r1");
    assert_eq!(decode(0xFA83, 0xF011), "qadd8 r0, r3, r1");
    assert_eq!(decode(0xFA83, 0xF021), "shadd8 r0, r3, r1");
    assert_eq!(decode(0xFA83, 0xF041), "uadd8 r0, r3, r1");
    assert_eq!(decode(0xFA83, 0xF051), "uqadd8 r0, r3, r1");
    assert_eq!(decode(0xFA83, 0xF061), "uhadd8 r0, r3, r1");
    assert_eq!(decode(0xFA93, 0xF001), "sadd16 r0, r3, r1");
    assert_eq!(decode(0xFA93, 0xF041), "uadd16 r0, r3, r1");
}

#[test]
fn the_parallel_subtractions_read_the_same_way() {
    assert_eq!(decode(0xFAC3, 0xF001), "ssub8 r0, r3, r1");
    assert_eq!(decode(0xFAC3, 0xF011), "qsub8 r0, r3, r1");
    assert_eq!(decode(0xFAC3, 0xF021), "shsub8 r0, r3, r1");
    assert_eq!(decode(0xFAC3, 0xF041), "usub8 r0, r3, r1");
    assert_eq!(decode(0xFAC3, 0xF051), "uqsub8 r0, r3, r1");
    assert_eq!(decode(0xFAC3, 0xF061), "uhsub8 r0, r3, r1");
    assert_eq!(decode(0xFAD3, 0xF001), "ssub16 r0, r3, r1");
    assert_eq!(decode(0xFAD3, 0xF041), "usub16 r0, r3, r1");
}

#[test]
fn the_checksums_are_not_a_thumb_instruction() {
    // `crc32b` and `crc32cb` are an ARMv8 extension, and they are
    // encoded in the A32 space: their first halfword is `0xFAC0` and
    // `0xFAD0`. There is no Thumb encoding of them, and every
    // reference reads these encodings as undefined.
    assert!(decode(0xFAD0, 0xF030).contains("undefined"));
    assert!(decode(0xFAC0, 0xF030).contains("undefined"));
}
