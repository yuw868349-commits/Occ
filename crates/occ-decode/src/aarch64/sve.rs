//! SVE decoding.
//!
//! SVE lives in the A64 encoding space whose top-level `op0` (`bits 28..25`)
//! is `0b0010`. Within it the space splits on `bits 28..24` into two main
//! groups — `0b00100` (integer arithmetic, permuting, and every load and
//! store) and `0b00101` (immediates, predicate operations, FP arithmetic) —
//! with `bits 31..29` free to pick element widths and separate sub-groups.
//!
//! Every family below was laid out from an exhaustive grid comparison with
//! the reference disassembler (`llvm-objdump` over all 16384 words of top
//! byte `0x04` and beyond), and an encoding that no family claims is
//! reported as `.word` with the family named, never as a plausible-looking
//! instruction.

use occ_core::Addr;

use super::{R, bits, undefined};

/// Element letters, indexed by `size`.
const SIZES: [char; 4] = ['b', 'h', 's', 'd'];

/// NEON element suffixes for the `*qv` reduction destinations.
const VFORMS: [&str; 4] = ["16b", "8h", "4s", "2d"];

/// `z<n>`, no element size.
fn z(n: u32) -> String {
    format!("z{}", n & 31)
}

/// `z<n>.<t>`.
fn zt(n: u32, t: char) -> String {
    format!("z{}.{}", n & 31, t)
}

/// `p<n>`, optionally with an element size.
fn p(n: u32) -> String {
    format!("p{}", n & 15)
}

/// `p<n>.<t>`.
fn pt(n: u32, t: char) -> String {
    format!("p{}.{}", n & 15, t)
}

fn scalar(t: char, n: u32) -> String {
    format!("{t}{}", n & 31)
}

/// The base register of an SVE memory access.  The architecture allows the
/// stack pointer there, so register 31 prints as `sp` rather than `x31`.
fn sp_reg(n: u32) -> String {
    if n & 31 == 31 {
        "sp".to_string()
    } else {
        format!("x{}", n & 31)
    }
}

/// `W<x>`/`WZR` for the 32-bit general-register `index` operands.
fn wreg(n: u32) -> String {
    if n & 31 == 31 {
        "wzr".to_string()
    } else {
        format!("w{}", n & 31)
    }
}

/// `X<x>`/`XZR` for the 64-bit `index` operands.
fn xreg(n: u32) -> String {
    if n & 31 == 31 {
        "xzr".to_string()
    } else {
        format!("x{}", n & 31)
    }
}

/// `X<x>`/`XZR` for the operands that read the general register as the
/// zero register in its thirty-first encoding.
fn zreg(n: u32, wide: bool) -> String {
    if wide {
        if n & 31 == 31 {
            "xzr".to_string()
        } else {
            format!("x{}", n & 31)
        }
    } else if n & 31 == 31 {
        "wzr".to_string()
    } else {
        format!("w{}", n & 31)
    }
}

/// `index`'s general-register operands: 64-bit elements use X names, the
/// rest use W names.
fn gpr(n: u32, wide: bool) -> String {
    if wide { xreg(n) } else { wreg(n) }
}

/// A general register of an encoding that also allows the stack pointer.
/// Register 31 then prints as `sp` or `wsp` rather than `xzr` or `wzr`.
fn gpr_sp(n: u32, wide: bool) -> String {
    if n & 31 == 31 {
        if wide {
            "sp".to_string()
        } else {
            "wsp".to_string()
        }
    } else if wide {
        format!("x{}", n & 31)
    } else {
        format!("w{}", n & 31)
    }
}

/// A signed immediate printed the way the assembler prints it (`#0x2`,
/// `#-0x1e`), taken from a `width`-bit two's complement field.
fn simm(field: u32, width: u32) -> String {
    let mask = (1 << width) - 1;
    let v = field & mask;
    if v & (1 << (width - 1)) != 0 {
        format!("#-0x{:x}", (1 << width) - v)
    } else {
        format!("#0x{:x}", v)
    }
}

/// Decode one word of the SVE groups (`op0` = `0b0010`).
pub(super) fn decode(origin: Addr, word: u32) -> R {
    match bits(word, 28, 24) {
        0b00100 => group_a(origin, word),
        0b00101 => group_b(origin, word),
        _ => undefined(origin, word, "SVE (unallocated)"),
    }
}

/// The `bits(28:24) = 0b00100` group: integer arithmetic, permuting
/// operations, and every SVE load and store.
fn group_a(origin: Addr, word: u32) -> R {
    match bits(word, 31, 24) {
        0x04 => integer_04(origin, word),
        0x44 => super::sve_tab44::decode_44(origin, word),
        0x24 => compare_24(origin, word),
        0x64 => float_64_65(origin, word),
        0x84 => gather_84_85(origin, word),
        0xA4 => cont_load_a45(origin, word),
        0xC4 => gather_c45(origin, word),
        0xE4 => store_e45(origin, word),
        _ => undefined(origin, word, "SVE (unimplemented family)"),
    }
}

/// The `bits(28:24) = 0b00101` group: immediates, predicate operations,
/// floating-point arithmetic.
fn group_b(origin: Addr, word: u32) -> R {
    match bits(word, 31, 24) {
        0x05 => permute_05(origin, word),
        0x45 => super::sve_tab45::decode_45(origin, word),
        0x25 => integer_25(origin, word),
        0x65 => float_64_65(origin, word),
        0x85 => gather_84_85(origin, word),
        0xA5 => cont_load_a45(origin, word),
        0xC5 => gather_c45(origin, word),
        0xE5 => store_e45(origin, word),
        _ => undefined(origin, word, "SVE (unimplemented family)"),
    }
}

// === top byte 0x04: SVE integer core =======================================
//
// The space splits on bit 21 and `bits(15:13)` into sixteen families, each
// with its own opcode/field layout. Everything below mirrors that split.

fn integer_04(origin: Addr, word: u32) -> R {
    let sz = bits(word, 23, 22) as usize;
    let bit21 = bits(word, 21, 21) == 1;
    match (bit21, bits(word, 15, 13)) {
        (false, 0b000) => pred_binary_04(origin, word, sz),
        (false, 0b001) => reduce_04(origin, word, sz),
        (false, 0b010) => mla_mls_04(word, sz, "mla"),
        (false, 0b011) => mla_mls_04(word, sz, "mls"),
        (false, 0b100) => pred_shift_04(origin, word, sz),
        (false, 0b101) => unary_pred_04(origin, word, sz),
        (false, 0b110) => mad_msb_04(word, sz, "mad"),
        (false, 0b111) => mad_msb_04(word, sz, "msb"),
        (true, 0b000) => unpred_arith_04(origin, word, sz),
        (true, 0b001) => unpred_logic_04(origin, word, sz),
        (true, 0b010) => index_vl_04(origin, word, sz),
        (true, 0b011) => unpred_mul_04(origin, word, sz),
        (true, 0b100) => unpred_shift_04(origin, word, sz),
        (true, 0b101) => scalar_misc_04(origin, word, sz),
        (true, 0b110) => vector_incdec_04(origin, word, sz),
        (true, 0b111) => scalar_incdec_04(origin, word, sz),
        _ => undefined(origin, word, "SVE integer (unimplemented family)"),
    }
}

/// Predicated binary arithmetic and logic: `Zdn, Pg/m, Zdn, Zm`.
/// Divide is 32/64-bit only.
const PRED_BINARY: &[(u8, &str)] = &[
    (0b00000, "add"),
    (0b00001, "sub"),
    (0b00011, "subr"),
    (0b01000, "smax"),
    (0b01001, "umax"),
    (0b01010, "smin"),
    (0b01011, "umin"),
    (0b01100, "sabd"),
    (0b01101, "uabd"),
    (0b10000, "mul"),
    (0b10010, "smulh"),
    (0b10011, "umulh"),
    (0b10100, "sdiv"),
    (0b10101, "udiv"),
    (0b10110, "sdivr"),
    (0b10111, "udivr"),
    (0b11000, "orr"),
    (0b11001, "eor"),
    (0b11010, "and"),
    (0b11011, "bic"),
];

fn pred_binary_04(origin: Addr, word: u32, sz: usize) -> R {
    let Some((_, mn)) = PRED_BINARY
        .iter()
        .find(|(k, _)| *k == bits(word, 20, 16) as u8)
    else {
        return undefined(origin, word, "SVE integer predicated (unallocated)");
    };
    // sdiv/udiv/sdivr/udivr exist only for 32- and 64-bit elements.
    if matches!(*mn, "sdiv" | "udiv" | "sdivr" | "udivr") && sz < 2 {
        return undefined(origin, word, "SVE integer predicated (unallocated)");
    }
    let t = SIZES[sz];
    let zdn = bits(word, 4, 0);
    let zm = bits(word, 9, 5);
    R::new(
        mn,
        format!(
            "{mn} {}, {}/m, {}, {}",
            zt(zdn, t),
            p(bits(word, 12, 10)),
            zt(zdn, t),
            zt(zm, t)
        ),
    )
}

/// Reductions to a scalar (or `movprfx`, which shares the encoding space).
/// `(opc, size-mask, mnemonic, destination kind)` where the destination is
/// 0: an X/D scalar, 1: a scalar of the element size, 2: a NEON view
/// (`v1.16b` and friends), 3/4: movprfx with `/z` or `/m`.
const REDUCE: &[(u8, u8, &str, u8)] = &[
    (0b00000, 0b0111, "saddv", 0),
    (0b00001, 0b1111, "uaddv", 0),
    (0b00101, 0b1111, "addqv", 2),
    (0b01000, 0b1111, "smaxv", 1),
    (0b01001, 0b1111, "umaxv", 1),
    (0b01010, 0b1111, "sminv", 1),
    (0b01011, 0b1111, "uminv", 1),
    (0b01100, 0b1111, "smaxqv", 2),
    (0b01101, 0b1111, "umaxqv", 2),
    (0b01110, 0b1111, "sminqv", 2),
    (0b01111, 0b1111, "uminqv", 2),
    (0b10000, 0b1111, "movprfx", 3),
    (0b10001, 0b1111, "movprfx", 4),
    (0b11000, 0b1111, "orv", 1),
    (0b11001, 0b1111, "eorv", 1),
    (0b11010, 0b1111, "andv", 1),
    (0b11100, 0b1111, "orqv", 2),
    (0b11101, 0b1111, "eorqv", 2),
    (0b11110, 0b1111, "andqv", 2),
];

fn reduce_04(origin: Addr, word: u32, sz: usize) -> R {
    let Some((_, mask, mn, kind)) = REDUCE.iter().find(|(k, ..)| *k == bits(word, 20, 16) as u8)
    else {
        return undefined(origin, word, "SVE integer reduce (unallocated)");
    };
    if sz > 3 || (*mask & (1 << sz)) == 0 {
        return undefined(origin, word, "SVE integer reduce (unallocated)");
    }
    let t = SIZES[sz];
    let rd = bits(word, 4, 0);
    let zn = bits(word, 9, 5);
    let pg = p(bits(word, 12, 10));
    match kind {
        // Destination fixed at 64 bits regardless of the size field.
        0 => R::new(
            mn,
            format!("{mn} {}, {}, {}", scalar('d', rd), pg, zt(zn, t)),
        ),
        // Destination spelled with the element letter.
        1 => R::new(mn, format!("{mn} {}, {}, {}", scalar(t, rd), pg, zt(zn, t))),
        // Destination is the NEON view of a Q register, sized like the source.
        2 => R::new(
            mn,
            format!("{mn} v{}.{}, {}, {}", rd, VFORMS[sz], pg, zt(zn, t)),
        ),
        // movprfx, zeroing or merging.
        _ => {
            let sfx = if *kind == 3 { "z" } else { "m" };
            R::new(
                mn,
                format!("{mn} {}, {}/{}, {}", zt(rd, t), pg, sfx, zt(zn, t)),
            )
        }
    }
}

/// MLA / MLS: `Zda, Pg/m, Zn, Zm`.
fn mla_mls_04(word: u32, sz: usize, mn: &'static str) -> R {
    let t = SIZES[sz];
    R::new(
        mn,
        format!(
            "{mn} {}, {}/m, {}, {}",
            zt(bits(word, 4, 0), t),
            p(bits(word, 12, 10)),
            zt(bits(word, 9, 5), t),
            zt(bits(word, 20, 16), t)
        ),
    )
}

/// MAD / MSB: `Zdn, Pg/m, Za, Zm` — the multiply-add operands are read from
/// different fields than MLA's.
fn mad_msb_04(word: u32, sz: usize, mn: &'static str) -> R {
    let t = SIZES[sz];
    R::new(
        mn,
        format!(
            "{mn} {}, {}/m, {}, {}",
            zt(bits(word, 4, 0), t),
            p(bits(word, 12, 10)),
            zt(bits(word, 20, 16), t),
            zt(bits(word, 9, 5), t)
        ),
    )
}

/// Predicated shifts: by immediate (`opc 0-15`, 32/64-bit elements and
/// byte/half via the immediate's high bits), by vector (`opc 16-23`), and
/// by 64-bit vector (`opc 24-27`).
const PRED_SHIFT_IMM: &[(u8, &str, bool)] = &[
    (0b00000, "asr", false),
    (0b00001, "lsr", false),
    (0b00011, "lsl", true),
    (0b00100, "asrd", false),
    (0b00110, "sqshl", true),
    (0b00111, "uqshl", true),
    (0b01100, "srshr", false),
    (0b01101, "urshr", false),
    (0b01111, "sqshlu", true),
];

const PRED_SHIFT_VEC: &[(u8, &str)] = &[
    (0b10000, "asr"),
    (0b10001, "lsr"),
    (0b10011, "lsl"),
    (0b10100, "asrr"),
    (0b10101, "lsrr"),
    (0b10111, "lslr"),
];

const PRED_SHIFT_WIDE: &[(u8, &str)] = &[(0b11000, "asr"), (0b11001, "lsr"), (0b11011, "lsl")];

/// Element size and shift amount for a shift-by-immediate. The encoding
/// packs the element width into `size` plus the immediate's top bits:
/// size 0 covers byte/half (immediate 8..31), size 1 word, size 2/3 double.
fn shift_imm(sz: usize, imm: u32, left: bool) -> Option<(char, u32)> {
    match (sz, left, imm) {
        (0, true, 8..=15) => Some(('b', imm - 8)),
        (0, true, 16..=31) => Some(('h', imm - 16)),
        (0, false, 8..=15) => Some(('b', 16 - imm)),
        (0, false, 16..=31) => Some(('h', 32 - imm)),
        (1, true, _) => Some(('s', imm)),
        (1, false, _) => Some(('s', 32 - imm)),
        (2, true, _) => Some(('d', imm)),
        (2, false, _) => Some(('d', 64 - imm)),
        (3, true, _) => Some(('d', imm + 32)),
        (3, false, _) => Some(('d', 32 - imm)),
        _ => None,
    }
}

fn pred_shift_04(origin: Addr, word: u32, sz: usize) -> R {
    let opc = bits(word, 20, 16) as u8;
    let zdn = bits(word, 4, 0);
    let zm = bits(word, 9, 5);
    if let Some((_, mn, left)) = PRED_SHIFT_IMM.iter().find(|(k, ..)| *k == opc) {
        // Shifts by immediate come in 32/64-bit elements, plus byte/half
        // through the immediate's high bits.
        let Some((t, shift)) = shift_imm(sz, bits(word, 9, 5), *left) else {
            return undefined(origin, word, "SVE predicated shift (unallocated)");
        };
        let pg = format!("{}/m", p(bits(word, 12, 10)));
        return R::new(
            mn,
            format!(
                "{mn} {}, {}, {}, #0x{:x}",
                zt(zdn, t),
                pg,
                zt(zdn, t),
                shift
            ),
        );
    }
    if let Some((_, mn)) = PRED_SHIFT_VEC.iter().find(|(k, _)| *k == opc) {
        let t = SIZES[sz];
        let pg = format!("{}/m", p(bits(word, 12, 10)));
        return R::new(
            mn,
            format!("{mn} {}, {}, {}, {}", zt(zdn, t), pg, zt(zdn, t), zt(zm, t)),
        );
    }
    if let Some((_, mn)) = PRED_SHIFT_WIDE.iter().find(|(k, _)| *k == opc) {
        // Wide shifts take a 64-bit element vector as their shift amount and
        // are only defined for 8/16/32-bit elements.
        if sz > 2 {
            return undefined(origin, word, "SVE predicated shift (unallocated)");
        }
        let t = SIZES[sz];
        let pg = format!("{}/m", p(bits(word, 12, 10)));
        return R::new(
            mn,
            format!(
                "{mn} {}, {}, {}, {}",
                zt(zdn, t),
                pg,
                zt(zdn, t),
                zt(zm, 'd')
            ),
        );
    }
    undefined(origin, word, "SVE predicated shift (unallocated)")
}

/// Unary predicated operations. Opcode bit 20 picks zeroing (`/z`) or
/// merging (`/m`); widening forms only exist for larger elements.
const UNARY_PRED: &[(u8, u8, &str)] = &[
    (0x0, 0b1110, "sxtb"),
    (0x1, 0b1110, "uxtb"),
    (0x2, 0b1100, "sxth"),
    (0x3, 0b1100, "uxth"),
    (0x4, 0b1000, "sxtw"),
    (0x5, 0b1000, "uxtw"),
    (0x6, 0b1111, "abs"),
    (0x7, 0b1111, "neg"),
    (0x8, 0b1111, "cls"),
    (0x9, 0b1111, "clz"),
    (0xa, 0b1111, "cnt"),
    (0xb, 0b1111, "cnot"),
    (0xc, 0b1110, "fabs"),
    (0xd, 0b1110, "fneg"),
    (0xe, 0b1111, "not"),
];

fn unary_pred_04(origin: Addr, word: u32, sz: usize) -> R {
    let opc = bits(word, 20, 16);
    let Some((_, mask, mn)) = UNARY_PRED.iter().find(|(k, ..)| *k == (opc & 0xf) as u8) else {
        return undefined(origin, word, "SVE unary predicated (unallocated)");
    };
    if (*mask & (1 << sz)) == 0 {
        return undefined(origin, word, "SVE unary predicated (unallocated)");
    }
    let t = SIZES[sz];
    let sfx = if opc & 0x10 == 0 { "z" } else { "m" };
    R::new(
        mn,
        format!(
            "{mn} {}, {}/{}, {}",
            zt(bits(word, 4, 0), t),
            p(bits(word, 12, 10)),
            sfx,
            zt(bits(word, 9, 5), t)
        ),
    )
}

/// Unpredicated three-register arithmetic; the selector is `bits(12:10)`.
const UNPRED_ARITH: &[(u8, &str)] = &[
    (0b000, "add"),
    (0b001, "sub"),
    (0b100, "sqadd"),
    (0b101, "uqadd"),
    (0b110, "sqsub"),
    (0b111, "uqsub"),
];

fn unpred_arith_04(origin: Addr, word: u32, sz: usize) -> R {
    let Some((_, mn)) = UNPRED_ARITH
        .iter()
        .find(|(k, _)| *k == bits(word, 12, 10) as u8)
    else {
        return undefined(origin, word, "SVE integer unpredicated (unallocated)");
    };
    let t = SIZES[sz];
    R::new(
        mn,
        format!(
            "{mn} {}, {}, {}",
            zt(bits(word, 4, 0), t),
            zt(bits(word, 9, 5), t),
            zt(bits(word, 20, 16), t)
        ),
    )
}

/// Unpredicated logic on 64-bit elements, plus the SVE2 three-register
/// ternary logic that shares the space: `xar` (selector 5), `eor3`/`bcax`
/// (selector 6), `bsl`-family (selector 7).
const UNPRED_LOGIC: [&str; 4] = ["and", "orr", "eor", "bic"];

const TERNARY: &[(u8, &str)] = &[(0b00, "eor3"), (0b01, "bcax")];
const TERNARY_BSL: [&str; 4] = ["bsl", "bsl1n", "bsl2n", "nbsl"];

/// Element width and rotation for `xar`, again packed across `size` and the
/// opcode's low bits.
fn xar_shift(sz: usize, opc: u32) -> Option<(char, u32)> {
    match (sz, opc) {
        (0, 8..=15) => Some(('b', 16 - opc)),
        (0, 16..=31) => Some(('h', 32 - opc)),
        (1, _) => Some(('s', 32 - opc)),
        (2, _) => Some(('d', 64 - opc)),
        (3, _) => Some(('d', 32 - opc)),
        _ => None,
    }
}

fn unpred_logic_04(origin: Addr, word: u32, sz: usize) -> R {
    let rd = bits(word, 4, 0);
    let rn = bits(word, 9, 5);
    let rm = bits(word, 20, 16);
    match bits(word, 12, 10) {
        // and/orr/eor/bic on .d elements; `orr` with equal sources prints as
        // a plain `mov`.
        0b100 => {
            let mn = UNPRED_LOGIC[sz];
            if mn == "orr" && rn == rm {
                return R::new("mov", format!("mov {}.d, {}.d", z(rd), z(rn)));
            }
            R::new(mn, format!("{mn} {}.d, {}.d, {}.d", z(rd), z(rn), z(rm)))
        }
        0b101 => {
            let Some((t, shift)) = xar_shift(sz, rm) else {
                return undefined(origin, word, "SVE xar (unallocated)");
            };
            R::new(
                "xar",
                format!(
                    "xar {}, {}, {}, #0x{:x}",
                    zt(rd, t),
                    zt(rd, t),
                    zt(rn, t),
                    shift
                ),
            )
        }
        0b110 => {
            let Some((_, mn)) = TERNARY.iter().find(|(k, _)| *k == sz as u8) else {
                return undefined(origin, word, "SVE ternary logic (unallocated)");
            };
            R::new(
                mn,
                format!("{mn} {}.d, {}.d, {}.d, {}.d", z(rd), z(rd), z(rm), z(rn)),
            )
        }
        0b111 => {
            let mn = TERNARY_BSL[sz];
            R::new(
                mn,
                format!("{mn} {}.d, {}.d, {}.d, {}.d", z(rd), z(rd), z(rm), z(rn)),
            )
        }
        _ => undefined(origin, word, "SVE integer unpredicated (unallocated)"),
    }
}

/// `index` in its four operand forms, plus the vector-length arithmetic
/// (`addvl`/`addpl`/`addsvl`/`addspl`/`rdvl`/`rdsvl`) that shares the space.
fn index_vl_04(origin: Addr, word: u32, sz: usize) -> R {
    let t = SIZES[sz];
    let rd = bits(word, 4, 0);
    let rn = bits(word, 9, 5);
    let rm = bits(word, 20, 16);
    let wide = sz == 3;
    match bits(word, 12, 10) {
        0b000 => R::new(
            "index",
            format!("index {}, {}, {}", zt(rd, t), simm(rn, 5), simm(rm, 5)),
        ),
        0b001 => R::new(
            "index",
            format!("index {}, {}, {}", zt(rd, t), gpr(rn, wide), simm(rm, 5)),
        ),
        0b010 => R::new(
            "index",
            format!("index {}, {}, {}", zt(rd, t), simm(rn, 5), gpr(rm, wide)),
        ),
        0b011 => R::new(
            "index",
            format!("index {}, {}, {}", zt(rd, t), gpr(rn, wide), gpr(rm, wide)),
        ),
        // Vector-length arithmetic: bit 11 picks VL vs SVL, the size field
        // picks addvl (size 0), addpl (size 1) or rdvl (size 2, rm == 31).
        // The base register spells 31 as `sp`, not `xzr`.
        _ => {
            let (mn, one_reg) = match (bits(word, 11, 11), sz, rm) {
                (0, 0, _) => ("addvl", false),
                (0, 1, _) => ("addpl", false),
                (0, 2, 31) => ("rdvl", true),
                (1, 0, _) => ("addsvl", false),
                (1, 1, _) => ("addspl", false),
                (1, 2, 31) => ("rdsvl", true),
                _ => return undefined(origin, word, "SVE vector-length arithmetic"),
            };
            let imm = simm(bits(word, 10, 5), 6);
            let rn_txt = if rm & 31 == 31 {
                "sp".to_string()
            } else {
                format!("x{}", rm)
            };
            if one_reg {
                // `rdvl` reads a general register, which spells 31 as the
                // zero register, where `addvl` reads a base that spells it
                // as the stack pointer
                R::new(mn, format!("{mn} {}, {}", zreg(rd, true), imm))
            } else {
                R::new(mn, format!("{mn} {}, {}, {}", sp_reg(rd), rn_txt, imm))
            }
        }
    }
}

/// Unpredicated multiply family; `pmul` is byte-only, the high/saturating
/// multiplies exist for every element size.
const UNPRED_MUL: &[(u8, u8, &str)] = &[
    (0b000, 0b1111, "mul"),
    (0b001, 0b0001, "pmul"),
    (0b010, 0b1111, "smulh"),
    (0b011, 0b1111, "umulh"),
    (0b100, 0b1111, "sqdmulh"),
    (0b101, 0b1111, "sqrdmulh"),
];

fn unpred_mul_04(origin: Addr, word: u32, sz: usize) -> R {
    let Some((_, mask, mn)) = UNPRED_MUL
        .iter()
        .find(|(k, ..)| *k == bits(word, 12, 10) as u8)
    else {
        return undefined(origin, word, "SVE multiply (unallocated)");
    };
    if (*mask & (1 << sz)) == 0 {
        return undefined(origin, word, "SVE multiply (unallocated)");
    }
    let t = SIZES[sz];
    R::new(
        mn,
        format!(
            "{mn} {}, {}, {}",
            zt(bits(word, 4, 0), t),
            zt(bits(word, 9, 5), t),
            zt(bits(word, 20, 16), t)
        ),
    )
}

/// Unpredicated shifts: by 64-bit element (`bits(12:10)` 0/1/3) and by
/// immediate (`4/5/7`, layout like the predicated shifts).
const UNPRED_SHIFT_WIDE: &[(u8, &str)] = &[(0b000, "asr"), (0b001, "lsr"), (0b011, "lsl")];

const UNPRED_SHIFT_IMM: &[(u8, &str, bool)] = &[
    (0b100, "asr", false),
    (0b101, "lsr", false),
    (0b111, "lsl", true),
];

fn unpred_shift_04(origin: Addr, word: u32, sz: usize) -> R {
    let sel = bits(word, 12, 10);
    let rd = bits(word, 4, 0);
    let rn = bits(word, 9, 5);
    if let Some((_, mn)) = UNPRED_SHIFT_WIDE.iter().find(|(k, _)| *k == sel as u8) {
        if sz > 2 {
            return undefined(origin, word, "SVE unpredicated shift (unallocated)");
        }
        let t = SIZES[sz];
        return R::new(
            mn,
            format!(
                "{mn} {}, {}, {}",
                zt(rd, t),
                zt(rn, t),
                zt(bits(word, 20, 16), 'd')
            ),
        );
    }
    if let Some((_, mn, left)) = UNPRED_SHIFT_IMM.iter().find(|(k, ..)| *k == sel as u8) {
        let Some((t, shift)) = shift_imm(sz, bits(word, 20, 16), *left) else {
            return undefined(origin, word, "SVE unpredicated shift (unallocated)");
        };
        return R::new(
            mn,
            format!("{mn} {}, {}, #0x{:x}", zt(rd, t), zt(rn, t), shift),
        );
    }
    undefined(origin, word, "SVE unpredicated shift (unallocated)")
}

/// Vector addressing (`adr`), `ftssel`, `fexpa`, and the unpredicated
/// `movprfx` alias, all sharing `bits(12:10)` as their selector.
fn scalar_misc_04(origin: Addr, word: u32, sz: usize) -> R {
    let rd = bits(word, 4, 0);
    let rn = bits(word, 9, 5);
    let rm = bits(word, 20, 16);
    match bits(word, 12, 10) {
        // adr: the element type and the extension are both chosen by the
        // size field; 32-bit elements shift left, 64-bit ones extend.
        sel @ 0..=3 => {
            let (t, ext) = match sz {
                0 => ('d', Some("sxtw")),
                1 => ('d', Some("uxtw")),
                2 => ('s', None),
                _ => ('d', None),
            };
            let ext = match (ext, sel) {
                (Some(e), 0) => format!(", {e}"),
                (Some(e), _) => format!(", {e} #{}", sel),
                (None, 0) => String::new(),
                (None, _) => format!(", lsl #{}", sel),
            };
            R::new(
                "adr",
                format!("adr {}, [{}, {}{}]", zt(rd, t), zt(rn, t), zt(rm, t), ext),
            )
        }
        0b100 => {
            if sz == 0 {
                return undefined(origin, word, "SVE scalar misc (unallocated)");
            }
            let t = SIZES[sz];
            R::new(
                "ftssel",
                format!("ftssel {}, {}, {}", zt(rd, t), zt(rn, t), zt(rm, t)),
            )
        }
        0b110 => {
            if sz == 0 || rm != 0 {
                return undefined(origin, word, "SVE scalar misc (unallocated)");
            }
            let t = SIZES[sz];
            R::new("fexpa", format!("fexpa {}, {}", zt(rd, t), zt(rn, t)))
        }
        0b111 => {
            if sz != 0 || rm != 0 {
                return undefined(origin, word, "SVE scalar misc (unallocated)");
            }
            R::new("movprfx", format!("movprfx {}, {}", z(rd), z(rn)))
        }
        _ => undefined(origin, word, "SVE scalar misc (unallocated)"),
    }
}

/// The SVE `pattern` field spelled the way the assembler prints it. `None`
/// means the operand is dropped entirely (pattern `all` with multiplier 1).
fn pattern_text(pat: u32, mul: u32) -> Option<String> {
    Some(match pat {
        0 => "pow2".into(),
        // the pattern names the number of vector lengths the operation
        // covers: the first eight count up, the rest double from 16
        1..=8 => format!("vl{pat}"),
        9..=13 => format!("vl{}", 1 << (pat - 5)),
        14..=28 => format!("#0x{:x}", pat),
        29 => "mul4".into(),
        30 => "mul3".into(),
        _ if mul == 1 => return None,
        _ => "all".into(),
    })
}

/// Vector increment/decrement (`mid=110`): the selector picks the
/// saturating/unsigned variant and whether the operation is `inc` or `dec`.
const VEC_INCDEC: [&str; 6] = [
    "sqinc", "uqinc", "sqdec", "uqdec", // saturating, multiplier = opc + 1
    "inc", "dec", // plain, multiplier = opc - 15
];

fn vector_incdec_04(origin: Addr, word: u32, sz: usize) -> R {
    if sz == 0 {
        return undefined(origin, word, "SVE inc/dec vector (unallocated)");
    }
    let opc = bits(word, 20, 16);
    let sel = bits(word, 12, 10);
    // sel 0/1 have a plain inc/dec variant at `opc >= 16`; sel 2/3 do not.
    let idx = match (sel, opc >= 16) {
        (0b00, false) => 0,
        (0b00, true) => 4,
        (0b01, false) => 1,
        (0b01, true) => 5,
        (0b10, false) => 2,
        (0b11, false) => 3,
        _ => return undefined(origin, word, "SVE inc/dec vector (unallocated)"),
    };
    let mn: &'static str = match (VEC_INCDEC[idx], SIZES[sz]) {
        ("sqinc", 'h') => "sqinch",
        ("sqinc", 's') => "sqincw",
        ("sqinc", 'd') => "sqincd",
        ("uqinc", 'h') => "uqinch",
        ("uqinc", 's') => "uqincw",
        ("uqinc", 'd') => "uqincd",
        ("sqdec", 'h') => "sqdech",
        ("sqdec", 's') => "sqdecw",
        ("sqdec", 'd') => "sqdecd",
        ("uqdec", 'h') => "uqdech",
        ("uqdec", 's') => "uqdecw",
        ("uqdec", 'd') => "uqdecd",
        ("inc", 'h') => "inch",
        ("inc", 's') => "incw",
        ("inc", 'd') => "incd",
        ("dec", 'h') => "dech",
        ("dec", 's') => "decw",
        _ => "decd",
    };
    let mul = if idx < 4 { opc + 1 } else { opc - 15 };
    let mut text = format!("{mn} {},", zt(bits(word, 4, 0), SIZES[sz]));
    match pattern_text(bits(word, 9, 5), mul) {
        Some(pat) => text = format!("{text} {pat}"),
        None => {
            text.pop();
        } // trailing comma from the destination
    }
    if mul > 1 {
        text = format!("{text}, mul #0x{:x}", mul);
    }
    R::new(mn, text)
}

/// Scalar increment/decrement/count (`mid=111`): plain forms update one X
/// register; the `sq`/`uq` forms are destructive and print the register as
/// both an X and a W name (32-bit) or a single W name.
const SC_INCDEC: [&str; 7] = [
    "cnt", "inc", "dec", // x-register forms
    "sqinc", "uqinc", "sqdec", "uqdec", // destructive forms, multiplier = opc + 1
];

fn scalar_incdec_04(origin: Addr, word: u32, sz: usize) -> R {
    let opc = bits(word, 20, 16);
    let sel = bits(word, 12, 10);
    let rd = bits(word, 4, 0);
    let idx = match (sel, opc) {
        (0b000, 0..=15) => 0,
        (0b000, _) => 1,
        (0b001, 16..) => 2,
        (0b100, _) => 3,
        (0b101, _) => 4,
        (0b110, _) => 5,
        (0b111, _) => 6,
        _ => return undefined(origin, word, "SVE inc/dec scalar (unallocated)"),
    };
    let base = SC_INCDEC[idx];
    let mn: &'static str = match (base, SIZES[sz]) {
        ("cnt", 'b') => "cntb",
        ("cnt", 'h') => "cnth",
        ("cnt", 's') => "cntw",
        ("cnt", 'd') => "cntd",
        ("inc", 'b') => "incb",
        ("inc", 'h') => "inch",
        ("inc", 's') => "incw",
        ("inc", 'd') => "incd",
        ("dec", 'b') => "decb",
        ("dec", 'h') => "dech",
        ("dec", 's') => "decw",
        ("dec", 'd') => "decd",
        ("sqinc", 'b') => "sqincb",
        ("sqinc", 'h') => "sqinch",
        ("sqinc", 's') => "sqincw",
        ("sqinc", 'd') => "sqincd",
        ("uqinc", 'b') => "uqincb",
        ("uqinc", 'h') => "uqinch",
        ("uqinc", 's') => "uqincw",
        ("uqinc", 'd') => "uqincd",
        ("sqdec", 'b') => "sqdecb",
        ("sqdec", 'h') => "sqdech",
        ("sqdec", 's') => "sqdecw",
        ("sqdec", 'd') => "sqdecd",
        ("uqdec", 'b') => "uqdecb",
        ("uqdec", 'h') => "uqdech",
        ("uqdec", 's') => "uqdecw",
        _ => "uqdecd",
    };
    // Multiplier: `cnt` and the low-half destructive forms take `opc + 1`;
    // the plain and high-half forms take `opc - 15`.
    let mul = match (idx, opc) {
        (0, _) => opc + 1,
        (1..=2, _) => opc - 15,
        (_, 16..) => opc - 15,
        _ => opc + 1,
    };
    let dest = match sel {
        0b000 | 0b001 => zreg(rd, true),
        // sel 4-6 with opc < 16 are destructive (print X and W names); with
        // opc >= 16 they update a single X register.
        0b100 | 0b110 if opc < 16 => format!("{}, {}", zreg(rd, true), zreg(rd, false)),
        0b101 | 0b111 if opc < 16 => zreg(rd, false),
        _ => zreg(rd, true),
    };
    let mut text = format!("{mn} {dest},");
    match pattern_text(bits(word, 9, 5), mul) {
        Some(pat) => text = format!("{text} {pat}"),
        None => {
            text.pop();
        }
    }
    if mul > 1 {
        text = format!("{text}, mul #0x{:x}", mul);
    }
    R::new(mn, text)
}

// === top byte 0x24: SVE integer compare ====================================
//
// `bits(31:24) = 0b00100100` holds the SVE2 compares that write a
// predicate. Bit 21 splits the space: `bit 21 = 0` is the register form
// (`Pd, Pg/z, Zn, Zm`), `bit 21 = 1` the unsigned-immediate form
// (`Pd, Pg/z, Zn, #imm7`).
//
// In the register form the opcode is spread over bits 15, 13 and 4, and
// bit 14 says whether the second source register has the same width as
// the first (`cmp_1 = 0`) or is always 64 bits wide (`cmp_1 = 1`). Two
// opcodes are only defined in the wide shape: `cmpeq` and `cmpne` with
// `cmp_1 = 0` take a 64-bit second operand, and they — like every other
// wide form — exist for byte, half and word elements only.

/// The register-form compares with a same-width second operand.
fn cmp_same(opc: u32) -> (&'static str, bool) {
    match opc {
        0b000 => ("cmphs", false),
        0b001 => ("cmphi", false),
        0b010 => ("cmpeq", true),
        0b011 => ("cmpne", true),
        0b100 => ("cmpge", false),
        0b101 => ("cmpgt", false),
        0b110 => ("cmpeq", false),
        _ => ("cmpne", false),
    }
}

/// The register-form compares with a 64-bit second operand.
fn cmp_wide(opc: u32) -> &'static str {
    match opc {
        0b000 => "cmpge",
        0b001 => "cmpgt",
        0b010 => "cmplt",
        0b011 => "cmple",
        0b100 => "cmphs",
        0b101 => "cmphi",
        0b110 => "cmplo",
        _ => "cmpls",
    }
}

/// The unsigned-immediate compares.
fn cmp_ucmp(opc: u32) -> &'static str {
    match opc {
        0b00 => "cmphs",
        0b01 => "cmphi",
        0b10 => "cmplo",
        _ => "cmpls",
    }
}

fn compare_24(origin: Addr, word: u32) -> R {
    let sz = bits(word, 23, 22) as usize;
    let t = SIZES[sz];
    let pg = p(bits(word, 12, 10));
    let zn = bits(word, 9, 5);
    let pd = bits(word, 3, 0);

    if bits(word, 21, 21) == 1 {
        // Unsigned immediate: the seven-bit immediate spans bits 20..14
        // and the two opcode bits are 13 and 4.
        let imm7 = (bits(word, 20, 16) << 2) | bits(word, 15, 14);
        let opc = (bits(word, 13, 13) << 1) | bits(word, 4, 4);
        let mn = cmp_ucmp(opc);
        return R::new(
            mn,
            format!("{mn} {}, {}/z, {}, #0x{:x}", pt(pd, t), pg, zt(zn, t), imm7),
        );
    }

    let zm = bits(word, 20, 16);
    let opc = (bits(word, 15, 15) << 2) | (bits(word, 13, 13) << 1) | bits(word, 4, 4);
    let (mn, wide_zm) = if bits(word, 14, 14) == 1 {
        (cmp_wide(opc), true)
    } else {
        cmp_same(opc)
    };
    if wide_zm && sz == 3 {
        return undefined(origin, word, "SVE integer compare (unallocated)");
    }
    let zm_size = if wide_zm { 'd' } else { t };
    R::new(
        mn,
        format!(
            "{mn} {}, {}/z, {}, {}",
            pt(pd, t),
            pg,
            zt(zn, t),
            zt(zm, zm_size)
        ),
    )
}

// === top byte 0x05: permuting, immediates ==================================
//
// `bits(31:24) = 0b00000101` holds the SVE instructions that take an
// immediate: the logical-immediate bitwise operations, the vector DUP
// family, and the predicated copies. Bit 21 splits the space in two:
// `bit 21 = 0` holds the immediate forms, `bit 21 = 1` the permuting
// instructions.

fn permute_05(origin: Addr, word: u32) -> R {
    if bits(word, 21, 21) == 0 {
        return immediate_05(origin, word);
    }
    permuting_05(origin, word)
}

/// The immediate half of top byte `0x05`: the logical-immediate bitwise
/// operations, the vector DUP family, and the predicated copies.
fn immediate_05(origin: Addr, word: u32) -> R {
    match bits(word, 20, 20) {
        0 => {
            // The logical-immediate forms all require `bits(21:18)` to be
            // zero; anything else in this window is unallocated.
            if bits(word, 19, 18) != 0 {
                return undefined(origin, word, "SVE immediate (unallocated)");
            }
            match bits(word, 23, 22) {
                3 => dupm_05(origin, word),
                opc => logical_imm_05(origin, word, opc),
            }
        }
        _ => {
            if bits(word, 15, 13) == 0b110 {
                fcpyi_05(origin, word)
            } else if bits(word, 15, 15) == 0 {
                cpyi_05(origin, word)
            } else {
                undefined(origin, word, "SVE immediate (unallocated)")
            }
        }
    }
}

/// The permuting half of top byte `0x05`.
///
/// The space is organised by the twelve opcode bits `bits(21:10)`.
/// `bits(23:22)` carries the element size for most families, and the
/// opcode row decides which instruction the word encodes.
fn permuting_05(origin: Addr, word: u32) -> R {
    let sz = bits(word, 23, 22) as usize;
    let t = SIZES[sz];
    let l = bits(word, 20, 16);
    let m = bits(word, 15, 10);
    let g = bits(word, 23, 21);
    let zd = bits(word, 4, 0);
    let zn = bits(word, 9, 5);
    // `SEL` names its predicate with four bits, every other family with three.
    let pg = bits(word, 12, 10);
    let sel_pg = bits(word, 13, 10);
    // Only 64-bit elements take X names; the rest use W names.
    let wide = sz == 3;

    match m {
        // `EXT` and the 128-bit interleaves share `bits(15:13) = 0`.
        0..=7 => {
            let imm = (l << 3) | bits(word, 12, 10);
            match g {
                0b001 => R::new(
                    "ext",
                    format!(
                        "ext {}, {}, {}, #0x{imm:x}",
                        zt(zd, 'b'),
                        zt(zd, 'b'),
                        zt(zn, 'b')
                    ),
                ),
                0b011 => R::new(
                    "ext",
                    format!(
                        "ext {}, {{ {}, {} }}, #0x{imm:x}",
                        zt(zd, 'b'),
                        zt(zn, 'b'),
                        zt((zn + 1) & 31, 'b')
                    ),
                ),
                0b101 => {
                    // The 128-bit interleaves come in pairs picked by bit 10.
                    let second = bits(word, 10, 10) == 1;
                    let mn = match bits(word, 12, 11) {
                        0b00 if !second => "zip1",
                        0b00 => "zip2",
                        0b01 if !second => "uzp1",
                        0b01 => "uzp2",
                        0b11 if !second => "trn1",
                        0b11 => "trn2",
                        _ => return undefined(origin, word, "SVE permute (unallocated)"),
                    };
                    R::new(
                        mn,
                        format!("{mn} {}, {}, {}", zt(zd, 'q'), zt(zn, 'q'), zt(l, 'q')),
                    )
                }
                _ => undefined(origin, word, "SVE permute (unallocated)"),
            }
        }
        // `DUP` of one element.
        8 => {
            let Some((t2, n)) = tsz_match(l, true) else {
                return undefined(origin, word, "SVE DUP element (unallocated)");
            };
            let idx = (bits(word, 23, 22) << n) | tsz_index(l, n);
            if idx == 0 {
                // With a zero index the reference prefers the scalar form.
                R::new("mov", format!("mov {}, {}", zt(zd, t2), scalar(t2, zn)))
            } else {
                R::new("mov", format!("mov {}, {}[{idx}]", zt(zd, t2), zt(zn, t2)))
            }
        }
        // `DUPQ` and `EXTQ`.
        9 => {
            if g == 0b001 {
                let Some((t2, n)) = tsz_match(l, false) else {
                    return undefined(origin, word, "SVE DUPQ (unallocated)");
                };
                let idx = tsz_index(l, n);
                R::new(
                    "dupq",
                    format!("dupq {}, {}[{idx}]", zt(zd, t2), zt(zn, t2)),
                )
            } else if g == 0b011 && bits(word, 20, 20) == 0 {
                R::new(
                    "extq",
                    format!(
                        "extq {}, {}, {}, #0x{:x}",
                        zt(zd, 'b'),
                        zt(zd, 'b'),
                        zt(zn, 'b'),
                        bits(word, 19, 16)
                    ),
                )
            } else {
                undefined(origin, word, "SVE permute (unallocated)")
            }
        }
        // `TBL`, `TBX` and `TBXQ`.
        10..=13 => {
            let second = bits(word, 10, 10) == 1;
            match (bits(word, 12, 11), second) {
                (0b01, false) => R::new(
                    "tbl",
                    format!(
                        "tbl {}, {{ {}, {} }}, {}",
                        zt(zd, t),
                        zt(zn, t),
                        zt((zn + 1) & 31, t),
                        zt(l, t)
                    ),
                ),
                (0b01, true) => R::new(
                    "tbx",
                    format!("tbx {}, {}, {}", zt(zd, t), zt(zn, t), zt(l, t)),
                ),
                (0b10, false) => R::new(
                    "tbl",
                    format!("tbl {}, {{ {} }}, {}", zt(zd, t), zt(zn, t), zt(l, t)),
                ),
                (0b10, true) => R::new(
                    "tbxq",
                    format!("tbxq {}, {}, {}", zt(zd, t), zt(zn, t), zt(l, t)),
                ),
                _ => undefined(origin, word, "SVE permute (unallocated)"),
            }
        }
        // The `bits(15:13) = 0b001` row: `DUP` from a register, `INSR`, the
        // unpack instructions, the vector `REV` and `PMOV`.
        14 => match l {
            0b00000 => R::new("mov", format!("mov {}, {}", zt(zd, t), gpr_sp(zn, wide))),
            0b00100 => R::new("insr", format!("insr {}, {}", zt(zd, t), gpr(zn, wide))),
            0b10100 => R::new("insr", format!("insr {}, {}", zt(zd, t), scalar(t, zn))),
            0b10000..=0b10011 => {
                if sz == 0 {
                    return undefined(origin, word, "SVE unpack (unallocated)");
                }
                let mn = match l {
                    0b10000 => "sunpklo",
                    0b10001 => "sunpkhi",
                    0b10010 => "uunpklo",
                    _ => "uunpkhi",
                };
                R::new(mn, format!("{mn} {}, {}", zt(zd, t), zt(zn, SIZES[sz - 1])))
            }
            0b11000 => R::new("rev", format!("rev {}, {}", zt(zd, t), zt(zn, t))),
            _ => pmov_05(origin, word, l),
        },
        // The predicate interleaves, plus `PUNPK` and the predicate `REV`.
        16..=21 => {
            // Every family in this row reserves bit 9 and fixes bit 4 at
            // zero.
            if bits(word, 9, 9) != 0 || bits(word, 4, 4) != 0 {
                return undefined(origin, word, "SVE permute (unallocated)");
            }
            if m == 16 {
                if g == 0b001 && (l == 0b10000 || l == 0b10001) {
                    let mn = if l == 0b10000 { "punpklo" } else { "punpkhi" };
                    return R::new(mn, format!("{mn} {}, {}", pt(zd, 'h'), pt(zn, 'b')));
                }
                if l == 0b10100 {
                    return R::new("rev", format!("rev {}, {}", pt(zd, t), pt(zn, t)));
                }
            }
            let Some(mn) = zip_mn(bits(word, 12, 10)) else {
                return undefined(origin, word, "SVE permute (unallocated)");
            };
            if l > 0x0F {
                return undefined(origin, word, "SVE permute (unallocated)");
            }
            R::new(
                mn,
                format!("{mn} {}, {}, {}", pt(zd, t), pt(zn, t), pt(l, t)),
            )
        }
        // The vector interleaves.
        24..=29 => {
            let Some(mn) = zip_mn(bits(word, 12, 10)) else {
                return undefined(origin, word, "SVE permute (unallocated)");
            };
            R::new(
                mn,
                format!("{mn} {}, {}, {}", zt(zd, t), zt(zn, t), zt(l, t)),
            )
        }
        // The predicated permutes.
        32..=39 => match l {
            0b00000 => R::new(
                "mov",
                format!("mov {}, {}/m, {}", zt(zd, t), p(pg), scalar(t, zn)),
            ),
            0b00001 => R::new(
                "compact",
                format!("compact {}, {}, {}", zt(zd, t), p(pg), zt(zn, t)),
            ),
            0b00010 => R::new(
                "lasta",
                format!("lasta {}, {}, {}", scalar(t, zd), p(pg), zt(zn, t)),
            ),
            0b00011 => R::new(
                "lastb",
                format!("lastb {}, {}, {}", scalar(t, zd), p(pg), zt(zn, t)),
            ),
            0b00100 => rev_05(origin, word, "revb", sz, false),
            0b00101 => rev_05(origin, word, "revh", sz, false),
            0b00110 => rev_05(origin, word, "revw", sz, false),
            0b00111 => rev_05(origin, word, "rbit", sz, false),
            0b01000 => R::new(
                "clasta",
                format!(
                    "clasta {}, {}, {}, {}",
                    zt(zd, t),
                    p(pg),
                    zt(zd, t),
                    zt(zn, t)
                ),
            ),
            0b01001 => R::new(
                "clastb",
                format!(
                    "clastb {}, {}, {}, {}",
                    zt(zd, t),
                    p(pg),
                    zt(zd, t),
                    zt(zn, t)
                ),
            ),
            0b01010 => R::new(
                "clasta",
                format!(
                    "clasta {}, {}, {}, {}",
                    scalar(t, zd),
                    p(pg),
                    scalar(t, zd),
                    zt(zn, t)
                ),
            ),
            0b01011 => R::new(
                "clastb",
                format!(
                    "clastb {}, {}, {}, {}",
                    scalar(t, zd),
                    p(pg),
                    scalar(t, zd),
                    zt(zn, t)
                ),
            ),
            0b01100 => R::new(
                "splice",
                format!(
                    "splice {}, {}, {}, {}",
                    zt(zd, t),
                    p(pg),
                    zt(zd, t),
                    zt(zn, t)
                ),
            ),
            0b01101 => R::new(
                "splice",
                format!(
                    "splice {}, {}, {{ {}, {} }}",
                    zt(zd, t),
                    p(pg),
                    zt(zn, t),
                    zt((zn + 1) & 31, t)
                ),
            ),
            0b01110 => {
                if sz != 0 {
                    return undefined(origin, word, "SVE reverse (unallocated)");
                }
                R::new(
                    "revd",
                    format!("revd {}, {}/m, {}", zt(zd, 'q'), p(pg), zt(zn, 'q')),
                )
            }
            0b10001 => R::new(
                "expand",
                format!("expand {}, {}, {}", zt(zd, t), p(pg), zt(zn, t)),
            ),
            _ => undefined(origin, word, "SVE permute (unallocated)"),
        },
        // The last-element readers that write a general register, the zeroing
        // reverses and the predicated copy from a general register.
        40..=47 => match l {
            0b00000 => R::new(
                "lasta",
                format!("lasta {}, {}, {}", gpr(zd, wide), p(pg), zt(zn, t)),
            ),
            0b00001 => R::new(
                "lastb",
                format!("lastb {}, {}, {}", gpr(zd, wide), p(pg), zt(zn, t)),
            ),
            0b00100 => rev_05(origin, word, "revb", sz, true),
            0b00101 => rev_05(origin, word, "revh", sz, true),
            0b00110 => rev_05(origin, word, "revw", sz, true),
            0b00111 => rev_05(origin, word, "rbit", sz, true),
            0b01000 => R::new(
                "mov",
                format!("mov {}, {}/m, {}", zt(zd, t), p(pg), gpr_sp(zn, wide)),
            ),
            0b01110 => {
                if sz != 0 {
                    return undefined(origin, word, "SVE reverse (unallocated)");
                }
                R::new(
                    "revd",
                    format!("revd {}, {}/z, {}", zt(zd, 'q'), p(pg), zt(zn, 'q')),
                )
            }
            0b10000 => R::new(
                "clasta",
                format!(
                    "clasta {}, {}, {}, {}",
                    gpr(zd, wide),
                    p(pg),
                    gpr(zd, wide),
                    zt(zn, t)
                ),
            ),
            0b10001 => R::new(
                "clastb",
                format!(
                    "clastb {}, {}, {}, {}",
                    gpr(zd, wide),
                    p(pg),
                    gpr(zd, wide),
                    zt(zn, t)
                ),
            ),
            _ => undefined(origin, word, "SVE permute (unallocated)"),
        },
        // `SEL` picks between two vectors under a predicate.
        48..=63 => {
            if l == zd {
                // When the two source registers are the same the reference
                // prints the `MOV` alias.
                R::new(
                    "mov",
                    format!("mov {}, {}/m, {}", zt(zd, t), p(sel_pg), zt(zn, t)),
                )
            } else {
                R::new(
                    "sel",
                    format!(
                        "sel {}, {}, {}, {}",
                        zt(zd, t),
                        p(sel_pg),
                        zt(zn, t),
                        zt(l, t)
                    ),
                )
            }
        }
        _ => undefined(origin, word, "SVE permute (unallocated)"),
    }
}

/// `PMOV` moves a vector lane into a predicate or back again. The opcode
/// splits the form across `bits(23:22)` and `bits(18:17)`, and bit 16
/// picks the direction.
fn pmov_05(origin: Addr, word: u32, l: u32) -> R {
    // The opcode leaves `bits(20:19)` at `0b01`.
    if l & 0b11000 != 0b01000 {
        return undefined(origin, word, "SVE PMOV (unallocated)");
    }
    let zd = bits(word, 4, 0);
    let zn = bits(word, 9, 5);
    let hi = bits(word, 23, 22);
    let lo = bits(word, 18, 17);
    let to_vector = l & 1 == 1;
    // The predicate-to-vector form reserves bit 9 and has a four-bit
    // destination predicate; the vector-to-predicate form spends bit 9 on
    // the source register and has a five-bit vector destination.
    if to_vector && bits(word, 9, 9) != 0 {
        return undefined(origin, word, "SVE PMOV (unallocated)");
    }
    if !to_vector && bits(word, 4, 4) != 0 {
        return undefined(origin, word, "SVE PMOV (unallocated)");
    }
    match hi {
        0b00 if lo == 0b01 => {
            if to_vector {
                R::new("pmov", format!("pmov {}, {}", z(zd), pt(zn, 'b')))
            } else {
                R::new("pmov", format!("pmov {}, {}", pt(zd, 'b'), z(zn)))
            }
        }
        0b00 => {
            if lo != 0b10 && lo != 0b11 {
                return undefined(origin, word, "SVE PMOV (unallocated)");
            }
            let idx = lo & 1;
            if to_vector {
                R::new("pmov", format!("pmov {}[{idx}], {}", z(zd), pt(zn, 'h')))
            } else {
                R::new("pmov", format!("pmov {}, {}[{idx}]", pt(zd, 'h'), z(zn)))
            }
        }
        0b01 => {
            if to_vector {
                R::new("pmov", format!("pmov {}[{lo}], {}", z(zd), pt(zn, 's')))
            } else {
                R::new("pmov", format!("pmov {}, {}[{lo}]", pt(zd, 's'), z(zn)))
            }
        }
        _ => {
            // The 64-bit form takes the top index bit from bit 22.
            let idx = lo | if hi == 0b11 { 4 } else { 0 };
            if to_vector {
                R::new("pmov", format!("pmov {}[{idx}], {}", z(zd), pt(zn, 'd')))
            } else {
                R::new("pmov", format!("pmov {}, {}[{idx}]", pt(zd, 'd'), z(zn)))
            }
        }
    }
}

/// `REVB`, `REVH`, `REVW` and `RBIT` accept different element widths, and
/// come in a merging and a zeroing form.
fn rev_05(origin: Addr, word: u32, mn: &'static str, sz: usize, zero: bool) -> R {
    let accepted = match mn {
        "revb" => sz >= 1,
        "revh" => sz >= 2,
        "revw" => sz == 3,
        _ => true,
    };
    if !accepted {
        return undefined(origin, word, "SVE reverse (unallocated)");
    }
    let t = SIZES[sz];
    let zd = bits(word, 4, 0);
    let zn = bits(word, 9, 5);
    let pg = bits(word, 12, 10);
    let suffix = if zero { "/z" } else { "/m" };
    R::new(
        mn,
        format!("{mn} {}, {}{suffix}, {}", zt(zd, t), p(pg), zt(zn, t)),
    )
}

/// The `tsz` field of the `DUP` encodings: the low bits pick the element
/// size and the bits above them carry the element index.
fn tsz_match(l: u32, quad: bool) -> Option<(char, u32)> {
    for (mask, value, t, n) in [
        (0b00001u32, 0b00001u32, 'b', 4u32),
        (0b00011, 0b00010, 'h', 3),
        (0b00111, 0b00100, 's', 2),
        (0b01111, 0b01000, 'd', 1),
    ] {
        if l & mask == value {
            return Some((t, n));
        }
    }
    // Only the element `DUP` has a 128-bit form.
    if quad && l & 0b11111 == 0b10000 {
        return Some(('q', 0));
    }
    None
}

/// The element index the `tsz` field carries.
fn tsz_index(l: u32, n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        (l >> (5 - n)) & ((1 << n) - 1)
    }
}

/// The mnemonic for the interleaving opcode.
fn zip_mn(opc: u32) -> Option<&'static str> {
    Some(match opc {
        0b000 => "zip1",
        0b001 => "zip2",
        0b010 => "uzp1",
        0b011 => "uzp2",
        0b100 => "trn1",
        0b101 => "trn2",
        _ => return None,
    })
}

// --- The logical-immediate bitwise operations --------------------------------
//
// `ORR`, `EOR` and `AND` (vectors, immediate) share one encoding:
// `bits(23:22)` picks the operation, `bits(21:18)` must be zero, and
// `bits(17:5)` carries the 13-bit logical immediate.

/// `ORR`/`EOR`/`AND` with a logical immediate.
fn logical_imm_05(origin: Addr, word: u32, opc: u32) -> R {
    let mn = match opc {
        0 => "orr",
        1 => "eor",
        2 => "and",
        _ => return undefined(origin, word, "SVE logical immediate (unallocated)"),
    };
    let imm13 = bits(word, 17, 5);
    let Some(val) = decode_logical_immediate(imm13, 64) else {
        return undefined(origin, word, "SVE logical immediate (unallocated)");
    };
    let zd = bits(word, 4, 0);
    // The reference assembler prints the narrowest element width whose
    // lanes all carry the same value: 8 bits first, then 16, then 32.
    for (width, suffix) in [(8u32, 'b'), (16, 'h'), (32, 's')] {
        if !identical_elements(val, width) {
            continue;
        }
        let Some(v) = decode_logical_immediate(imm13, width) else {
            break;
        };
        return R::new(
            mn,
            format!("{mn} {}, {}, #0x{v:x}", zt(zd, suffix), zt(zd, suffix)),
        );
    }
    // No alias matched, so the instruction prints as itself over 64-bit
    // elements.
    R::new(
        mn,
        format!("{mn} {}, {}, #0x{val:x}", zt(zd, 'd'), zt(zd, 'd')),
    )
}

// --- DUP (immediate) ---------------------------------------------------------
//
// The logical-immediate DUP shares its encoding with `MOV`. The reference
// assembler prefers `MOV` whenever the immediate has no single CPY/DUP
// equivalent, and `DUP` otherwise.

/// `DUP`/`MOV` with a logical immediate.
fn dupm_05(origin: Addr, word: u32) -> R {
    let imm13 = bits(word, 17, 5);
    let Some(val) = decode_logical_immediate(imm13, 64) else {
        return undefined(origin, word, "SVE DUP immediate (unallocated)");
    };
    let zd = bits(word, 4, 0);
    // `mov` aliases come first: they are preferred over every `dupm`.
    for (width, suffix) in [(16u32, 'h'), (32, 's'), (64, 'd')] {
        if !identical_elements(val, width) || !is_sve_move_mask_preferred(val as i64) {
            continue;
        }
        let Some(v) = decode_logical_immediate(imm13, width) else {
            continue;
        };
        let (text, comment) = print_sve_logical_imm(v, width);
        let prefix = format!("mov\t{}, {text}", zt(zd, suffix));
        return R::new("mov", with_comment(&prefix, comment.as_deref()));
    }
    for (width, suffix) in [(8u32, 'b'), (16, 'h'), (32, 's')] {
        if !identical_elements(val, width) {
            continue;
        }
        let Some(v) = decode_logical_immediate(imm13, width) else {
            continue;
        };
        return R::new("dupm", format!("dupm\t{}, #0x{v:x}", zt(zd, suffix)));
    }
    R::new("dupm", format!("dupm\t{}, #0x{val:x}", zt(zd, 'd')))
}

// --- CPY (immediate) ---------------------------------------------------------
//
// `CPY` (and its `MOV` alias) copies a replicated immediate into the
// elements a predicate selects.

/// `CPY`/`MOV` of an 8-bit immediate, with an optional shift.
fn cpyi_05(origin: Addr, word: u32) -> R {
    let sz = bits(word, 23, 22) as usize;
    let sh = bits(word, 13, 13) == 1;
    let imm8 = bits(word, 12, 5) as u8;
    // An 8-bit element cannot carry the shifted form.
    if sz == 0 && sh {
        return undefined(origin, word, "SVE CPY immediate (unallocated)");
    }
    let width = 8u32 << sz;
    let value = cpy_imm_value(imm8, sh, width);
    let qual = if bits(word, 14, 14) == 1 { "/m" } else { "/z" };
    let prefix = if imm8 == 0 && sh {
        format!(
            "mov\t{}, {}{qual}, #0x0, lsl #8",
            zt(bits(word, 4, 0), SIZES[sz]),
            p(bits(word, 19, 16))
        )
    } else {
        format!(
            "mov\t{}, {}{qual}, #0x{value:x}",
            zt(bits(word, 4, 0), SIZES[sz]),
            p(bits(word, 19, 16))
        )
    };
    // The shifted spelling of a zero immediate carries no value comment.
    let comment = if imm8 == 0 && sh {
        None
    } else {
        Some(format!("{}", value as i64))
    };
    R::new("mov", with_comment(&prefix, comment.as_deref()))
}

// --- FCPY (immediate) --------------------------------------------------------
//
// `FCPY` copies a replicated 8-bit floating-point constant. The same
// 8-bit pattern means the same number in 16-, 32- and 64-bit form.

/// `FCPY`/`FMOV` of an 8-bit floating-point immediate.
fn fcpyi_05(origin: Addr, word: u32) -> R {
    let sz = bits(word, 23, 22) as usize;
    // There is no 8-bit floating-point form.
    if sz == 0 {
        return undefined(origin, word, "SVE FCPY immediate (unallocated)");
    }
    let imm8 = bits(word, 12, 5) as u8;
    let value = fp_imm_value(imm8);
    R::new(
        "fmov",
        format!(
            "fmov\t{}, {}/m, #{value:.8}",
            zt(bits(word, 4, 0), SIZES[sz]),
            p(bits(word, 19, 16))
        ),
    )
}

// === top byte 0x25: SVE2 predicates, counts and immediates =================
//
// The `bits(31:24) = 0b00100101` space splits on bit 21. `bit 21 = 0` holds
// the signed predicate compares, the predicate logic, the break family and
// the `PTRUE` group; `bit 21 = 1` holds `WHILE`, `PSEL`, the predicate
// counts and the immediate arithmetic.

/// The `bits(31:24) = 0b00100101` group.
fn integer_25(origin: Addr, word: u32) -> R {
    let sz = bits(word, 23, 22) as usize;
    let t = SIZES[sz];
    let byte = bits(word, 23, 16);
    let mid = bits(word, 15, 10);

    if bits(word, 21, 21) == 0 {
        // `PFALSE`, `WRFFR` and `SETFFR` sit in rows of their own inside
        // this half of the space.
        if byte == 0x18 && mid == 0x39 && bits(word, 9, 4) == 0 {
            return R::new("pfalse", format!("pfalse\tp{}.b", bits(word, 3, 0)));
        }
        if byte == 0x50
            && bits(word, 15, 14) == 0b11
            && bits(word, 9, 9) == 0
            && bits(word, 4, 0) == 0
        {
            return R::new(
                "ptest",
                format!(
                    "ptest\tp{}, p{}.b",
                    bits(word, 13, 10),
                    bits(word, 8, 5) & 0xF
                ),
            );
        }
        if bits(word, 14, 14) == 0 {
            return scmp_vi_25(origin, word, t);
        }
        if bits(word, 15, 15) == 0 {
            if bits(word, 20, 20) == 0 {
                return pred_log_25(origin, word);
            }
            if bits(word, 19, 19) == 0 {
                return break_25(origin, word, byte);
            }
            return brkn_25(origin, word, byte);
        }
        if bits(word, 23, 23) == 0 && bits(word, 20, 20) == 0 {
            return brkp_25(origin, word);
        }
        if bits(word, 20, 20) == 1 && bits(word, 19, 19) == 1 {
            return ptrue_25(origin, word, sz, t, byte, mid);
        }
        return undefined(origin, word, "SVE predicate (unallocated)");
    }

    // `bit 21 = 1`.
    // The conditional-terminate row shares the `WHILE` opcode row but
    // fixes the middle bits of it.
    if bits(word, 15, 10) == 0b001000 && bits(word, 23, 23) == 1 {
        return cterm_25(origin, word);
    }
    if bits(word, 15, 13) == 0b000 {
        return while_25(word, t);
    }
    if mid == 0x0C {
        return while_rr2_25(word, t);
    }
    if bits(word, 15, 14) == 0b01 {
        // the counter forms of this row carry bit 4, and the initial
        // counter sits in the mnemonic slot PSEL leaves at zero
        if bits(word, 4, 4) == 1 {
            return while_pn_25(origin, word);
        }
        return psel_25(origin, word);
    }
    match byte & 0x38 {
        0x20 => {
            if bits(word, 15, 14) == 0b10 {
                return pcount_25(origin, word, t);
            }
            arith_imm0_25(origin, word, sz, t)
        }
        0x28 => {
            if byte == 0x28 && mid == 0x24 && bits(word, 4, 0) == 0 && bits(word, 9, 9) == 0 {
                return R::new("wrffr", format!("wrffr\tp{}.b", bits(word, 8, 5) & 0xF));
            }
            if byte == 0x2C && mid == 0x24 && bits(word, 9, 0) == 0 {
                return R::new("setffr", "setffr".to_string());
            }
            if mid == 0x20 || mid == 0x21 {
                return count_v_25(origin, word, sz, t);
            }
            if mid == 0x22 || mid == 0x23 {
                return count_r_25(origin, word, t);
            }
            if bits(word, 15, 13) == 0b110 {
                return arith_imm_25(origin, word, t);
            }
            undefined(origin, word, "SVE count (unallocated)")
        }
        0x30 => {
            if bits(word, 15, 13) == 0b110 {
                return arith_imm_25(origin, word, t);
            }
            undefined(origin, word, "SVE arithmetic (unallocated)")
        }
        0x38 => {
            if byte & 0x07 == 0 {
                return dup_imm_25(origin, word, sz, t);
            }
            if byte & 0x07 == 1 {
                return dup_fpimm_25(origin, word, sz, t);
            }
            undefined(origin, word, "SVE duplicate (unallocated)")
        }
        _ => undefined(origin, word, "SVE arithmetic (unallocated)"),
    }
}

/// The signed predicate compare against a five-bit immediate.
fn scmp_vi_25(origin: Addr, word: u32, t: char) -> R {
    let opc = (bits(word, 15, 15) << 2) | (bits(word, 13, 13) << 1) | bits(word, 4, 4);
    let mn = match opc {
        0b000 => "cmpge",
        0b001 => "cmpgt",
        0b010 => "cmplt",
        0b011 => "cmple",
        0b100 => "cmpeq",
        0b101 => "cmpne",
        _ => return undefined(origin, word, "SVE integer compare (unallocated)"),
    };
    R::new(
        mn,
        format!(
            "{mn}\t{}, {}/z, {}, {}",
            pt(bits(word, 3, 0), t),
            p(bits(word, 12, 10)),
            zt(bits(word, 9, 5), t),
            simm(bits(word, 20, 16), 5)
        ),
    )
}

/// The predicate logic operations, with the aliases the reference folds
/// into them.
fn pred_log_25(origin: Addr, word: u32) -> R {
    let opc = (bits(word, 23, 22) << 2) | (bits(word, 9, 9) << 1) | bits(word, 4, 4);
    let pg = bits(word, 13, 10);
    let pn = bits(word, 8, 5) & 0xF;
    let pm = bits(word, 19, 16);
    let pd = bits(word, 3, 0);
    let mn = match opc {
        0b0000 => "and",
        0b0001 => "bic",
        0b0010 => "eor",
        0b0011 => "sel",
        0b0100 => "ands",
        0b0101 => "bics",
        0b0110 => "eors",
        0b1000 => "orr",
        0b1001 => "orn",
        0b1010 => "nor",
        0b1011 => "nand",
        0b1100 => "orrs",
        0b1101 => "orns",
        0b1110 => "nors",
        0b1111 => "nands",
        _ => return undefined(origin, word, "SVE predicate logic (unallocated)"),
    };
    let (out, text) = match opc {
        // `SEL` carries no predication qualifier, and a self-move prints as
        // the merging form of `MOV`.
        0b0011 if pm == pd => ("mov", format!("mov\tp{pd}.b, p{pg}/m, p{pn}.b")),
        0b0011 => ("sel", format!("sel\tp{pd}.b, p{pg}, p{pn}.b, p{pm}.b")),
        0b0000 if pm == pn => ("mov", format!("mov\tp{pd}.b, p{pg}/z, p{pn}.b")),
        0b0100 if pm == pn => ("movs", format!("movs\tp{pd}.b, p{pg}/z, p{pn}.b")),
        0b1000 if pg == pn && pn == pm => ("mov", format!("mov\tp{pd}.b, p{pn}.b")),
        0b1100 if pg == pn && pn == pm => ("movs", format!("movs\tp{pd}.b, p{pn}.b")),
        0b0010 if pm == pg => ("not", format!("not\tp{pd}.b, p{pg}/z, p{pn}.b")),
        0b0110 if pm == pg => ("nots", format!("nots\tp{pd}.b, p{pg}/z, p{pn}.b")),
        _ => (mn, format!("{mn}\tp{pd}.b, p{pg}/z, p{pn}.b, p{pm}.b")),
    };
    R::new(out, text)
}

/// The break-pair operations.
fn brkp_25(origin: Addr, word: u32) -> R {
    // The architecture reserves bit 9.
    if bits(word, 9, 9) != 0 {
        return undefined(origin, word, "SVE break pair (unallocated)");
    }
    let opc = (bits(word, 22, 22) << 1) | bits(word, 4, 4);
    let mn = match opc {
        0b00 => "brkpa",
        0b01 => "brkpb",
        0b10 => "brkpas",
        0b11 => "brkpbs",
        _ => return undefined(origin, word, "SVE break pair (unallocated)"),
    };
    R::new(
        mn,
        format!(
            "{mn}\t{}, {}/z, p{}.b, p{}.b",
            pt(bits(word, 3, 0), 'b'),
            p(bits(word, 13, 10)),
            bits(word, 8, 5) & 0xF,
            bits(word, 19, 16)
        ),
    )
}

/// `BRKA`, `BRKAS`, `BRKB` and `BRKBS`, in both the zeroing and the
/// destructive form.
fn break_25(origin: Addr, word: u32, byte: u32) -> R {
    if byte & 0x3F != 0x10 {
        return undefined(origin, word, "SVE break (unallocated)");
    }
    // The architecture reserves bit 9.
    if bits(word, 9, 9) != 0 {
        return undefined(origin, word, "SVE break (unallocated)");
    }
    let (mn, sett) = match (bits(word, 23, 23), bits(word, 22, 22)) {
        (0, 0) => ("brka", false),
        (0, 1) => ("brkas", true),
        (1, 0) => ("brkb", false),
        (1, 1) => ("brkbs", true),
        _ => return undefined(origin, word, "SVE break (unallocated)"),
    };
    // The sett forms only have the zeroing qualifier.
    if sett && bits(word, 4, 4) == 1 {
        return undefined(origin, word, "SVE break (unallocated)");
    }
    let qual = if bits(word, 4, 4) == 1 { 'm' } else { 'z' };
    let pd = bits(word, 3, 0);
    let pg = bits(word, 13, 10);
    let pn = bits(word, 8, 5) & 0xF;
    R::new(mn, format!("{mn}\tp{pd}.b, p{pg}/{qual}, p{pn}.b"))
}

/// `BRKN` and `BRKNS`.
fn brkn_25(origin: Addr, word: u32, byte: u32) -> R {
    if byte & 0x3F != 0x18 || bits(word, 23, 23) != 0 || bits(word, 4, 4) != 0 {
        return undefined(origin, word, "SVE break negate (unallocated)");
    }
    // The architecture reserves bit 9.
    if bits(word, 9, 9) != 0 {
        return undefined(origin, word, "SVE break negate (unallocated)");
    }
    let mn = if bits(word, 22, 22) == 1 {
        "brkns"
    } else {
        "brkn"
    };
    let pd = bits(word, 3, 0);
    R::new(
        mn,
        format!(
            "{mn}\tp{pd}.b, {}/z, p{}.b, p{pd}.b",
            p(bits(word, 13, 10)),
            bits(word, 8, 5) & 0xF
        ),
    )
}

/// `PTRUE`, `PTRUES`, `PFIRST`, `PNEXT` and `RDFFR`.
fn ptrue_25(origin: Addr, word: u32, sz: usize, t: char, byte: u32, mid: u32) -> R {
    let pd = bits(word, 3, 0);
    let pattern = bits(word, 9, 5);
    match byte & 0x3F {
        0x18 => {
            if mid == 0x38 {
                return ptrue_25_named(origin, word, t, pd, pattern, false);
            }
            if mid == 0x30 {
                // `PFIRST` is a sixteen-bit operation only.
                if sz != 1 {
                    return undefined(origin, word, "SVE PFIRST (unallocated)");
                }
                // The architecture reserves bit 9; the pattern field stops
                // at bit 5, so bit 4 is reserved as well.
                if bits(word, 9, 9) != 0 || bits(word, 4, 4) != 0 {
                    return undefined(origin, word, "SVE PFIRST (unallocated)");
                }
                return R::new(
                    "pfirst",
                    format!("pfirst\tp{pd}.b, p{}, p{pd}.b", bits(word, 8, 5) & 0xF),
                );
            }
            if mid == 0x3C && bits(word, 23, 23) == 0 {
                // The architecture reserves bit 9, and bit 4 below the
                // pattern field.
                if bits(word, 9, 9) != 0 || bits(word, 4, 4) != 0 {
                    return undefined(origin, word, "SVE RDFFR (unallocated)");
                }
                let mn = if bits(word, 22, 22) == 1 {
                    "rdffrs"
                } else {
                    "rdffr"
                };
                // Bit 16 picks the unpredicated form from the predicated one.
                if bits(word, 16, 16) == 1 {
                    return R::new(mn, format!("{mn}\tp{pd}.b"));
                }
                return R::new(
                    mn,
                    format!("{mn}\tp{pd}.b, {}/z", p(bits(word, 8, 5) & 0xF)),
                );
            }
            undefined(origin, word, "SVE predicate group (unallocated)")
        }
        0x19 => {
            if mid == 0x38 {
                return ptrue_25_named(origin, word, t, pd, pattern, true);
            }
            if mid == 0x3C {
                // The unpredicated `RDFFR`, which shares this byte with
                // `PTRUES` and differs only in bit 16.
                if bits(word, 9, 9) != 0 {
                    return undefined(origin, word, "SVE RDFFR (unallocated)");
                }
                // The unpredicated form has no predicate operand, so the
                // field that would name one is fixed at zero, and only the
                // non-setting variant has one.
                if bits(word, 8, 5) != 0
                    || bits(word, 4, 4) != 0
                    || bits(word, 22, 22) != 0
                    || bits(word, 23, 23) != 0
                {
                    return undefined(origin, word, "SVE RDFFR (unallocated)");
                }
                return R::new("rdffr", format!("rdffr\tp{pd}.b"));
            }
            if mid == 0x31 {
                // The architecture reserves bits 9 and 4; the pattern field
                // stops at bit 5 and the bit below it is not part of it.
                if bits(word, 9, 9) != 0 || bits(word, 4, 4) != 0 {
                    return undefined(origin, word, "SVE PNEXT (unallocated)");
                }
                return R::new(
                    "pnext",
                    format!("pnext\tp{pd}.{t}, p{}, p{pd}.{t}", bits(word, 8, 5) & 0xF),
                );
            }
            undefined(origin, word, "SVE predicate group (unallocated)")
        }
        _ => undefined(origin, word, "SVE predicate group (unallocated)"),
    }
}

/// `PTRUE` and its setting variant.
fn ptrue_25_named(origin: Addr, word: u32, t: char, pd: u32, pattern: u32, setting: bool) -> R {
    // The pattern field ends at bit 5; the bit below it is reserved.
    if bits(word, 4, 4) != 0 {
        return undefined(origin, word, "SVE PTRUE (unallocated)");
    }
    let opc = (bits(word, 18, 17) << 1) | bits(word, 16, 16);
    let name = match (opc, setting) {
        (0b000, false) => "ptrue",
        (0b001, true) => "ptrues",
        _ => return undefined(origin, word, "SVE PTRUE (unallocated)"),
    };
    // The all-ones pattern has no name, and the reference then leaves the
    // operand out altogether.
    let pattern = pred_pattern(pattern);
    let text = if pattern.is_empty() {
        format!("{name}\tp{pd}.{t}")
    } else {
        format!("{name}\tp{pd}.{t}, {pattern}")
    };
    R::new(name, text)
}

/// `CTERMEQ` and `CTERMNE`: the conditional terminate, which compares two
/// general registers and sets the flags.
fn cterm_25(origin: Addr, word: u32) -> R {
    // The low four bits are fixed at zero and bit 4 picks the operation.
    if bits(word, 3, 0) != 0 {
        return undefined(origin, word, "SVE conditional terminate (unallocated)");
    }
    let mn = if bits(word, 4, 4) == 0 {
        "ctermeq"
    } else {
        "ctermne"
    };
    // Bit 22 picks the width of the comparison.
    let wide = bits(word, 22, 22) == 1;
    let rn = bits(word, 9, 5);
    let rm = bits(word, 20, 16);
    R::new(
        mn,
        format!(
            "{mn}\t{}, {}",
            if wide { xreg(rn) } else { wreg(rn) },
            if wide { xreg(rm) } else { wreg(rm) }
        ),
    )
}

/// The `WHILE` family, in its thirty-two and sixty-four bit forms.
fn while_25(word: u32, t: char) -> R {
    let opc = (bits(word, 12, 12) << 3)
        | (bits(word, 11, 11) << 2)
        | (bits(word, 10, 10) << 1)
        | bits(word, 4, 4);
    let mn = match opc & 0b111 {
        0b000 => "whilege",
        0b001 => "whilegt",
        0b010 => "whilelt",
        0b011 => "whilele",
        0b100 => "whilehs",
        0b101 => "whilehi",
        0b110 => "whilelo",
        0b111 => "whilels",
        _ => unreachable!(),
    };
    let (rn, rm) = if bits(word, 12, 12) == 1 {
        (xreg(bits(word, 9, 5)), xreg(bits(word, 20, 16)))
    } else {
        (wreg(bits(word, 9, 5)), wreg(bits(word, 20, 16)))
    };
    R::new(mn, format!("{mn}\tp{}.{t}, {rn}, {rm}", bits(word, 3, 0)))
}

/// The `WHILE` names, in the order the opcode selects them: the first of
/// each pair is the signed form and the second the one that excludes the
/// equal case.
const WHILE_NAMES: [&str; 8] = [
    "whilege", "whilegt", "whilelt", "whilele", "whilehs", "whilehi", "whilelo", "whilels",
];

/// `WHILE` with a counter destination, the pair form of it, and `PEXT`.
///
/// The row fixes bit 4 and moves the destination into the low bits: a
/// counter reads one of the eight predicate-as-counter registers, a pair
/// reads two consecutive predicates, and `PEXT` extracts one lane of a
/// counter into a predicate.
fn while_pn_25(origin: Addr, word: u32) -> R {
    let sz = bits(word, 23, 22) as usize;
    let t = SIZES[sz];
    let base = bits(word, 11, 10) as usize;
    let rn = zreg(bits(word, 9, 5), true);
    let rm = zreg(bits(word, 20, 16), true);
    if bits(word, 12, 12) == 0 {
        let mn = WHILE_NAMES[base * 2 + bits(word, 3, 3) as usize];
        let vl = if bits(word, 13, 13) == 1 {
            "vlx4"
        } else {
            "vlx2"
        };
        return R::new(
            mn,
            format!("{mn}	pn{}.{t}, {rn}, {rm}, {vl}", bits(word, 2, 0) + 8),
        );
    }
    if bits(word, 13, 13) == 0 {
        let mn = WHILE_NAMES[base * 2 + bits(word, 0, 0) as usize];
        let lo = bits(word, 3, 1) * 2;
        return R::new(
            mn,
            format!("{mn}	{{ p{lo}.{t}, p{}.{t} }}, {rn}, {rm}", lo + 1),
        );
    }
    // The rest of the row is `PEXT` and the counter form of `PTRUE`.  The
    // extraction reads a lane out of a counter into a predicate, either
    // one predicate or two consecutive ones.
    let pn = bits(word, 7, 5) + 8;
    // the counter forms read their counter out of the low bits only
    if base < 3 && bits(word, 20, 16) != 0 {
        return undefined(origin, word, "SVE PEXT (unallocated)");
    }
    match base {
        0 => R::new(
            "pext",
            format!(
                "pext\tp{}.{t}, pn{pn}[{}]",
                bits(word, 3, 0),
                bits(word, 9, 8)
            ),
        ),
        1 => {
            // the pair form only names the first two lanes of a counter
            if bits(word, 9, 8) > 1 {
                return undefined(origin, word, "SVE PEXT (unallocated)");
            }
            let n = bits(word, 3, 0);
            R::new(
                "pext",
                format!(
                    "pext\t{{ p{n}.{t}, p{}.{t} }}, pn{pn}[{}]",
                    (n + 1) & 15,
                    bits(word, 9, 8)
                ),
            )
        }
        2 => {
            // the initial counter: every parameter of the pattern is zero
            if bits(word, 9, 5) != 0 || bits(word, 3, 3) != 0 {
                return undefined(origin, word, "SVE PTRUE (unallocated)");
            }
            R::new("ptrue", format!("ptrue\tpn{}.{t}", bits(word, 2, 0) + 8))
        }
        _ => undefined(origin, word, "SVE PEXT (unallocated)"),
    }
}

/// `WHILEWR` and `WHILERW`.
fn while_rr2_25(word: u32, t: char) -> R {
    let mn = if bits(word, 4, 4) == 0 {
        "whilewr"
    } else {
        "whilerw"
    };
    R::new(
        mn,
        format!(
            "{mn}\tp{}.{t}, {}, {}",
            bits(word, 3, 0),
            xreg(bits(word, 9, 5)),
            xreg(bits(word, 20, 16))
        ),
    )
}

/// `PSEL`, the SME predicate select.
fn psel_25(origin: Addr, word: u32) -> R {
    // The architecture reserves bit 9 and fixes bit 4 at zero.
    if bits(word, 9, 9) != 0 || bits(word, 4, 4) != 0 {
        return undefined(origin, word, "SVE PSEL (unallocated)");
    }
    let (t, imm) = match bits(word, 20, 18) {
        0b001 | 0b011 | 0b101 | 0b111 => ('b', (bits(word, 23, 22) << 2) | bits(word, 20, 19)),
        0b010 | 0b110 => ('h', (bits(word, 23, 22) << 1) | bits(word, 20, 20)),
        0b100 => ('s', bits(word, 23, 22)),
        0b000 => {
            if bits(word, 22, 22) != 1 {
                return undefined(origin, word, "SVE PSEL (unallocated)");
            }
            ('d', bits(word, 23, 23))
        }
        _ => return undefined(origin, word, "SVE PSEL (unallocated)"),
    };
    R::new(
        "psel",
        format!(
            "psel\tp{}, p{}, p{}.{t}[w{}, {imm}]",
            bits(word, 3, 0),
            bits(word, 13, 10),
            bits(word, 8, 5) & 0xF,
            bits(word, 17, 16) + 12
        ),
    )
}

/// The predicate population counts.
fn pcount_25(origin: Addr, word: u32, t: char) -> R {
    // Bit 9 picks between the count over a predicate and the count over a
    // predicate-as-counter and a vector length.
    if bits(word, 9, 9) == 1 {
        // The vector-length form fixes the middle bits of the opcode and
        // leaves the rest of the row alone.
        if bits(word, 15, 11) != 0b10000 || bits(word, 20, 16) != 0 {
            return undefined(origin, word, "SVE predicate count (unallocated)");
        }
        let vl = if bits(word, 10, 10) == 1 {
            "vlx4"
        } else {
            "vlx2"
        };
        // The counter is a predicate-as-counter, numbered 0 to 15.
        let pn = bits(word, 9, 5) - 16;
        return R::new(
            "cntp",
            format!("cntp\t{}, pn{pn}.{t}, {vl}", zreg(bits(word, 4, 0), true)),
        );
    }
    let mn = match bits(word, 18, 16) {
        0b000 => "cntp",
        0b001 => "firstp",
        0b010 => "lastp",
        _ => return undefined(origin, word, "SVE predicate count (unallocated)"),
    };
    R::new(
        mn,
        format!(
            "{mn}\t{}, p{}, p{}.{t}",
            zreg(bits(word, 4, 0), true),
            bits(word, 13, 10),
            bits(word, 8, 5) & 0xF
        ),
    )
}

/// The vector saturating counts.
fn count_v_25(origin: Addr, word: u32, sz: usize, t: char) -> R {
    if sz == 0 {
        return undefined(origin, word, "SVE count (unallocated)");
    }
    let opc = ((bits(word, 18, 16)) << 2) | bits(word, 10, 9);
    let mn = match opc {
        0b00000 => "sqincp",
        0b00100 => "uqincp",
        0b01000 => "sqdecp",
        0b01100 => "uqdecp",
        0b10000 => "incp",
        0b10100 => "decp",
        _ => return undefined(origin, word, "SVE count (unallocated)"),
    };
    R::new(
        mn,
        format!(
            "{mn}\tz{}.{t}, p{}.{t}",
            bits(word, 4, 0),
            bits(word, 8, 5) & 0xF
        ),
    )
}

/// The scalar saturating counts.
fn count_r_25(origin: Addr, word: u32, t: char) -> R {
    let opc = ((bits(word, 18, 16)) << 2) | bits(word, 10, 9);
    let (mn, reg, wide) = match opc {
        0b00000 => ("sqincp", 'x', true),
        0b00010 => ("sqincp", 'x', false),
        0b00100 => ("uqincp", 'w', false),
        0b00110 => ("uqincp", 'x', false),
        0b01000 => ("sqdecp", 'x', true),
        0b01010 => ("sqdecp", 'x', false),
        0b01100 => ("uqdecp", 'w', false),
        0b01110 => ("uqdecp", 'x', false),
        0b10000 => ("incp", 'x', false),
        0b10100 => ("decp", 'x', false),
        _ => return undefined(origin, word, "SVE count (unallocated)"),
    };
    let rd = bits(word, 4, 0);
    let text = if wide {
        format!(
            "{mn}\t{}, p{}.{t}, {}",
            zreg(rd, true),
            bits(word, 8, 5) & 0xF,
            zreg(rd, false)
        )
    } else {
        format!(
            "{mn}\t{}, p{}.{t}",
            zreg(rd, reg == 'x'),
            bits(word, 8, 5) & 0xF
        )
    };
    R::new(mn, text)
}

/// The add, subtract and saturating immediate forms.
fn arith_imm0_25(origin: Addr, word: u32, sz: usize, t: char) -> R {
    if bits(word, 15, 14) != 0b11 {
        return undefined(origin, word, "SVE arithmetic immediate (unallocated)");
    }
    let mn = match bits(word, 18, 16) {
        0b000 => "add",
        0b001 => "sub",
        0b011 => "subr",
        0b100 => "sqadd",
        0b101 => "uqadd",
        0b110 => "sqsub",
        0b111 => "uqsub",
        _ => return undefined(origin, word, "SVE arithmetic immediate (unallocated)"),
    };
    let sh = bits(word, 13, 13) == 1;
    if sz == 0 && sh {
        return undefined(origin, word, "SVE arithmetic immediate (unallocated)");
    }
    let width = 8u32 << sz;
    // The add and subtract immediate is an unsigned eight-bit value.
    let imm8 = bits(word, 12, 5);
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let value = ((imm8 as u64) << (if sh { 8 } else { 0 })) & mask;
    let prefix = if sh && value == 0 {
        // A zero shifted immediate has to keep its shift, or the encoding
        // would read back differently.
        format!(
            "{mn}\tz{}.{t}, z{}.{t}, #0x0, lsl #8",
            bits(word, 4, 0),
            bits(word, 4, 0)
        )
    } else {
        format!(
            "{mn}\tz{}.{t}, z{}.{t}, #0x{value:x}",
            bits(word, 4, 0),
            bits(word, 4, 0)
        )
    };
    let comment = if sh && value == 0 {
        None
    } else {
        Some(format!("{}", value as i64))
    };
    R::new(mn, with_comment(&prefix, comment.as_deref()))
}

/// The minimum, maximum and multiply immediate forms.
fn arith_imm_25(origin: Addr, word: u32, t: char) -> R {
    let mn = match bits(word, 21, 16) {
        0b101000 => "smax",
        0b101001 => "umax",
        0b101010 => "smin",
        0b101011 => "umin",
        0b110000 => "mul",
        _ => return undefined(origin, word, "SVE arithmetic immediate (unallocated)"),
    };
    let imm8 = bits(word, 12, 5);
    let imm = match mn {
        "smax" | "smin" | "mul" => simm(imm8, 8),
        _ => format!("#0x{imm8:x}"),
    };
    R::new(
        mn,
        format!(
            "{mn}\tz{}.{t}, z{}.{t}, {imm}",
            bits(word, 4, 0),
            bits(word, 4, 0)
        ),
    )
}

/// `MOV` of a replicated immediate.
fn dup_imm_25(origin: Addr, word: u32, sz: usize, t: char) -> R {
    if bits(word, 15, 14) != 0b11 {
        return undefined(origin, word, "SVE duplicate immediate (unallocated)");
    }
    let sh = bits(word, 13, 13) == 1;
    if sz == 0 && sh {
        return undefined(origin, word, "SVE duplicate immediate (unallocated)");
    }
    let width = 8u32 << sz;
    // The replicated immediate is a signed eight-bit value.
    let imm8 = bits(word, 12, 5) as u8;
    let value = cpy_imm_value(imm8, sh, width);
    let prefix = if sh && value == 0 {
        // A zero shifted immediate has to keep its shift, or the encoding
        // would read back differently.
        format!("mov\tz{}.{t}, #0x0, lsl #8", bits(word, 4, 0))
    } else {
        format!("mov\tz{}.{t}, #0x{value:x}", bits(word, 4, 0))
    };
    let comment = if sh && value == 0 {
        None
    } else {
        Some(format!("{}", value as i64))
    };
    R::new("mov", with_comment(&prefix, comment.as_deref()))
}

/// `FMOV` of a replicated floating-point immediate.
fn dup_fpimm_25(origin: Addr, word: u32, sz: usize, t: char) -> R {
    if bits(word, 15, 14) != 0b11 || sz == 0 || bits(word, 13, 13) != 0 {
        return undefined(origin, word, "SVE duplicate immediate (unallocated)");
    }
    let value = fp_imm_value(bits(word, 12, 5) as u8);
    R::new(
        "fmov",
        format!("fmov\tz{}.{t}, #{value:.8}", bits(word, 4, 0)),
    )
}

// === top bytes 0xE4 / 0xE5: SVE stores =====================================
//
// The store space splits on `bits(15:13)` into eight families:
//
//   000  128-bit contiguous store (0xE4) / predicate spill (0xE5)
//   001  scatter non-temporal store / 128-bit scatter store
//   010  contiguous store with a register offset / vector spill (0xE5)
//   011  non-temporal or multi-register store with a register offset
//   100  scatter store, unsigned index extension
//   101  scatter store, scaled vector index or immediate index
//   110  scatter store, signed index extension
//   111  contiguous store with a scaled immediate offset
//
// Bit 24 separates the byte and halfword forms (0xE4) from the word and
// doubleword forms (0xE5); it is the top bit of the size field in every
// family that has one.

/// Decode one word of the SVE store space.
fn store_e45(origin: Addr, word: u32) -> R {
    match store_e45_text(word) {
        Ok((mn, text)) => R::new(mn, text),
        Err(family) => undefined(origin, word, family),
    }
}

/// The printed form of a store: the mnemonic and the whole instruction.
type StoreText = Result<(&'static str, String), &'static str>;

fn store_e45_text(word: u32) -> StoreText {
    match bits(word, 15, 13) {
        0b000 => store_128_e45(word),
        0b001 => store_scatter_nt_e45(word),
        0b010 => store_cont_reg_e45(word),
        0b011 => store_multi_reg_e45(word),
        0b100 | 0b110 => store_scatter_vec_e45(word),
        0b101 => store_scatter_idx_e45(word),
        _ => store_imm_e45(word),
    }
}

/// The 128-bit contiguous store, and the predicate spill of the same
/// encoding space.
fn store_128_e45(word: u32) -> StoreText {
    let zt = bits(word, 4, 0);
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let base = sp_reg(rn);
    if bits(word, 24, 24) == 0 {
        let nregs = bits(word, 23, 22);
        let (mn, scale) = match nregs {
            1 => ("st2q", 2i64),
            2 => ("st3q", 3),
            3 => ("st4q", 4),
            _ => return Err("SVE 128-bit store (unallocated)"),
        };
        let mem = if bits(word, 21, 21) == 0 {
            if bits(word, 20, 20) != 0 {
                return Err("SVE 128-bit store (unallocated)");
            }
            let v = simm_value(bits(word, 19, 16), 4) * scale;
            if v == 0 {
                format!("[{base}]")
            } else {
                format!("[{base}, {}, mul vl]", imm_text(v))
            }
        } else {
            let rm = bits(word, 20, 16);
            if rm == 31 {
                return Err("SVE 128-bit store (unallocated)");
            }
            format!("[{base}, x{rm}, lsl #4]")
        };
        Ok((
            mn,
            format!("{mn}\t{{ {} }}, p{pg}, {mem}", zlist(zt, nregs + 1, 'q')),
        ))
    } else if bits(word, 23, 23) == 1 && bits(word, 22, 22) == 0 && bits(word, 4, 4) == 0 {
        // The predicate spill: `Inst{4}` is fixed at zero and `bits(3:0)`
        // name the predicate register.
        let v = simm_value((bits(word, 21, 16) << 3) | bits(word, 12, 10), 9);
        let mem = if v == 0 {
            format!("[{base}]")
        } else {
            format!("[{base}, {}, mul vl]", imm_text(v))
        };
        Ok(("str", format!("str\tp{}, {mem}", bits(word, 3, 0))))
    } else {
        Err("SVE store (unallocated)")
    }
}

/// The scatter non-temporal store, and the 128-bit scatter store that
/// shares its encoding.
fn store_scatter_nt_e45(word: u32) -> StoreText {
    let zt = bits(word, 4, 0);
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let rm = bits(word, 20, 16);
    if bits(word, 24, 24) == 0
        && bits(word, 23, 23) == 0
        && bits(word, 22, 22) == 0
        && bits(word, 21, 21) == 1
    {
        let mem = if rm == 31 {
            format!("[z{rn}.d]")
        } else {
            format!("[z{rn}.d, x{rm}]")
        };
        return Ok(("st1q", format!("st1q\t{{ z{zt}.q }}, p{pg}, {mem}")));
    }
    if bits(word, 21, 21) != 0 {
        return Err("SVE scatter non-temporal store (unallocated)");
    }
    let (mn, elem) = match bits(word, 24, 22) {
        0 => ("stnt1b", 'd'),
        1 => ("stnt1b", 's'),
        2 => ("stnt1h", 'd'),
        3 => ("stnt1h", 's'),
        4 => ("stnt1w", 'd'),
        5 => ("stnt1w", 's'),
        6 => ("stnt1d", 'd'),
        _ => return Err("SVE scatter non-temporal store (unallocated)"),
    };
    let mem = if rm == 31 {
        format!("[z{rn}.{elem}]")
    } else {
        format!("[z{rn}.{elem}, x{rm}]")
    };
    Ok((mn, format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, {mem}")))
}

/// The contiguous store with a register offset, and the vector spill.
fn store_cont_reg_e45(word: u32) -> StoreText {
    let zt = bits(word, 4, 0);
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let base = sp_reg(rn);
    if bits(word, 24, 24) == 1 && bits(word, 23, 23) == 1 && bits(word, 22, 22) == 0 {
        let v = simm_value((bits(word, 21, 16) << 3) | bits(word, 12, 10), 9);
        let mem = if v == 0 {
            format!("[{base}]")
        } else {
            format!("[{base}, {}, mul vl]", imm_text(v))
        };
        return Ok(("str", format!("str\tz{zt}, {mem}")));
    }
    let (mn, elem, shift) = match bits(word, 24, 21) {
        0 => ("st1b", 'b', 0),
        1 => ("st1b", 'h', 0),
        2 => ("st1b", 's', 0),
        3 => ("st1b", 'd', 0),
        5 => ("st1h", 'h', 1),
        6 => ("st1h", 's', 1),
        7 => ("st1h", 'd', 1),
        8 => ("st1w", 'q', 2),
        10 => ("st1w", 's', 2),
        11 => ("st1w", 'd', 2),
        14 => ("st1d", 'q', 3),
        15 => ("st1d", 'd', 3),
        _ => return Err("SVE contiguous store (unallocated)"),
    };
    let rm = bits(word, 20, 16);
    if rm == 31 {
        return Err("SVE contiguous store (unallocated)");
    }
    let mem = if shift == 0 {
        format!("[{base}, x{rm}]")
    } else {
        format!("[{base}, x{rm}, lsl #{shift}]")
    };
    Ok((mn, format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, {mem}")))
}

/// The non-temporal store and the multi-register store, both with a
/// register offset.
fn store_multi_reg_e45(word: u32) -> StoreText {
    let zt = bits(word, 4, 0);
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let base = sp_reg(rn);
    let rm = bits(word, 20, 16);
    if rm == 31 {
        return Err("SVE store with register offset (unallocated)");
    }
    if bits(word, 22, 21) == 0 {
        let (mn, elem, shift) = match bits(word, 24, 23) {
            0 => ("stnt1b", 'b', 0),
            1 => ("stnt1h", 'h', 1),
            2 => ("stnt1w", 's', 2),
            3 => ("stnt1d", 'd', 3),
            _ => return Err("SVE non-temporal store (unallocated)"),
        };
        let mem = if shift == 0 {
            format!("[{base}, x{rm}]")
        } else {
            format!("[{base}, x{rm}, lsl #{shift}]")
        };
        return Ok((mn, format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, {mem}")));
    }
    let nregs = bits(word, 22, 21);
    let (mn, elem, shift) = match (bits(word, 24, 23), nregs) {
        (0, 1) => ("st2b", 'b', 0),
        (0, 2) => ("st3b", 'b', 0),
        (0, 3) => ("st4b", 'b', 0),
        (1, 1) => ("st2h", 'h', 1),
        (1, 2) => ("st3h", 'h', 1),
        (1, 3) => ("st4h", 'h', 1),
        (2, 1) => ("st2w", 's', 2),
        (2, 2) => ("st3w", 's', 2),
        (2, 3) => ("st4w", 's', 2),
        (3, 1) => ("st2d", 'd', 3),
        (3, 2) => ("st3d", 'd', 3),
        (3, 3) => ("st4d", 'd', 3),
        _ => return Err("SVE multi-register store (unallocated)"),
    };
    let mem = if shift == 0 {
        format!("[{base}, x{rm}]")
    } else {
        format!("[{base}, x{rm}, lsl #{shift}]")
    };
    Ok((
        mn,
        format!("{mn}\t{{ {} }}, p{pg}, {mem}", zlist(zt, nregs + 1, elem)),
    ))
}

/// The scatter store with a vector index. `bits(15:14)` picks between the
/// unsigned and the signed index extension.
fn store_scatter_vec_e45(word: u32) -> StoreText {
    let zt = bits(word, 4, 0);
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let base = sp_reg(rn);
    let rm = bits(word, 20, 16);
    let scaled = bits(word, 21, 21) == 1;
    let (mn, elem, shift) = match bits(word, 24, 22) {
        0 => ("st1b", 'd', 0),
        1 => ("st1b", 's', 0),
        2 => ("st1h", 'd', 1),
        3 => ("st1h", 's', 1),
        4 => ("st1w", 'd', 2),
        5 => ("st1w", 's', 2),
        6 => ("st1d", 'd', 3),
        _ => return Err("SVE scatter store (unallocated)"),
    };
    if scaled && shift == 0 {
        return Err("SVE scatter store (unallocated)");
    }
    let ext = if bits(word, 14, 14) == 0 {
        "uxtw"
    } else {
        "sxtw"
    };
    let ext = if scaled {
        format!("{ext} #{shift}")
    } else {
        ext.to_string()
    };
    Ok((
        mn,
        format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, [{base}, z{rm}.{elem}, {ext}]"),
    ))
}

/// The scatter store with a scaled vector index, and the one with an
/// immediate index.
fn store_scatter_idx_e45(word: u32) -> StoreText {
    let zt = bits(word, 4, 0);
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let base = sp_reg(rn);
    let rm = bits(word, 20, 16);
    if bits(word, 22, 22) == 0 {
        let (mn, elem, shift) = match bits(word, 24, 23) {
            0 => ("st1b", 'd', 0),
            1 => ("st1h", 'd', 1),
            2 => ("st1w", 'd', 2),
            3 => ("st1d", 'd', 3),
            _ => return Err("SVE scatter store (unallocated)"),
        };
        if bits(word, 21, 21) == 1 {
            if shift == 0 {
                return Err("SVE scatter store (unallocated)");
            }
            return Ok((
                mn,
                format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, [{base}, z{rm}.{elem}, lsl #{shift}]"),
            ));
        }
        return Ok((
            mn,
            format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, [{base}, z{rm}.{elem}]"),
        ));
    }
    let opc = bits(word, 24, 23) * 2 + bits(word, 21, 21);
    let (mn, elem, scale) = match opc {
        0 => ("st1b", 'd', 1i64),
        1 => ("st1b", 's', 1),
        2 => ("st1h", 'd', 2),
        3 => ("st1h", 's', 2),
        4 => ("st1w", 'd', 4),
        5 => ("st1w", 's', 4),
        6 => ("st1d", 'd', 8),
        _ => return Err("SVE scatter store (unallocated)"),
    };
    let v = rm as i64 * scale;
    let mem = if v == 0 {
        format!("[z{rn}.{elem}]")
    } else {
        format!("[z{rn}.{elem}, {}]", imm_text(v))
    };
    Ok((mn, format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, {mem}")))
}

/// The contiguous store with a scaled immediate offset, and the two
/// multi-register forms that share it.
fn store_imm_e45(word: u32) -> StoreText {
    let zt = bits(word, 4, 0);
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let base = sp_reg(rn);
    let imm4 = bits(word, 19, 16);
    if bits(word, 20, 20) == 0 {
        let (mn, elem) = match (bits(word, 24, 23), bits(word, 22, 21)) {
            (0, 0) => ("st1b", 'b'),
            (0, 1) => ("st1b", 'h'),
            (0, 2) => ("st1b", 's'),
            (0, 3) => ("st1b", 'd'),
            (1, 1) => ("st1h", 'h'),
            (1, 2) => ("st1h", 's'),
            (1, 3) => ("st1h", 'd'),
            (2, 0) => ("st1w", 'q'),
            (2, 2) => ("st1w", 's'),
            (2, 3) => ("st1w", 'd'),
            (3, 2) => ("st1d", 'q'),
            (3, 3) => ("st1d", 'd'),
            _ => return Err("SVE contiguous store (unallocated)"),
        };
        let v = simm_value(imm4, 4);
        let mem = if v == 0 {
            format!("[{base}]")
        } else {
            format!("[{base}, {}, mul vl]", imm_text(v))
        };
        return Ok((mn, format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, {mem}")));
    }
    if bits(word, 22, 21) == 0 {
        let (mn, elem) = match bits(word, 24, 23) {
            0 => ("stnt1b", 'b'),
            1 => ("stnt1h", 'h'),
            2 => ("stnt1w", 's'),
            3 => ("stnt1d", 'd'),
            _ => return Err("SVE non-temporal store (unallocated)"),
        };
        let v = simm_value(imm4, 4);
        let mem = if v == 0 {
            format!("[{base}]")
        } else {
            format!("[{base}, {}, mul vl]", imm_text(v))
        };
        return Ok((mn, format!("{mn}\t{{ z{zt}.{elem} }}, p{pg}, {mem}")));
    }
    let nregs = bits(word, 22, 21);
    let (mn, elem) = match (bits(word, 24, 23), nregs) {
        (0, 1) => ("st2b", 'b'),
        (0, 2) => ("st3b", 'b'),
        (0, 3) => ("st4b", 'b'),
        (1, 1) => ("st2h", 'h'),
        (1, 2) => ("st3h", 'h'),
        (1, 3) => ("st4h", 'h'),
        (2, 1) => ("st2w", 's'),
        (2, 2) => ("st3w", 's'),
        (2, 3) => ("st4w", 's'),
        (3, 1) => ("st2d", 'd'),
        (3, 2) => ("st3d", 'd'),
        (3, 3) => ("st4d", 'd'),
        _ => return Err("SVE multi-register store (unallocated)"),
    };
    let v = simm_value(imm4, 4) * (nregs as i64 + 1);
    let mem = if v == 0 {
        format!("[{base}]")
    } else {
        format!("[{base}, {}, mul vl]", imm_text(v))
    };
    Ok((
        mn,
        format!("{mn}\t{{ {} }}, p{pg}, {mem}", zlist(zt, nregs + 1, elem)),
    ))
}

/// The SVE register list of a store, printed the way the reference
/// prints a `VecList`: one register on its own, two separated by a
/// comma, three or more as a range.  A run that would wrap around the
/// register file cannot be written as a range, so it is listed out.
fn zlist(zt: u32, n: u32, elem: char) -> String {
    match n {
        0 => String::new(),
        1 => format!("z{}.{}", zt & 31, elem),
        2 => format!("z{}.{}, z{}.{}", zt & 31, elem, (zt + 1) & 31, elem),
        _ if (zt & 31) + n <= 32 => {
            format!("z{}.{} - z{}.{}", zt & 31, elem, (zt + n - 1) & 31, elem)
        }
        _ => (0..n)
            .map(|i| format!("z{}.{}", (zt + i) & 31, elem))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

// === top bytes 0xC4/0xC5: SVE 64-bit gather loads and prefetch ===========
//
// The space is laid out by the architecture as
//
//   Inst{31-25} = 1100010   Inst{24-23} = opc{3-2}
//   Inst{22}    = xs         Inst{21}    = scaled
//   Inst{20-16} = Zm        Inst{15}    = lsl
//   Inst{14-13} = opc{1-0}  Inst{12-10} = Pg
//   Inst{9-5}   = Rn/Zn      Inst{4-0}   = Zt
//
// with a few fields that differ from family to family.  Because
// `Inst{24-23} = opc{3-2}` hands out its most significant bit first, `opc{3}`
// lands on bit 24 and `opc{2}` on bit 23.  Bit 24 is the only bit that
// separates top byte 0xC4 from 0xC5, so one opcode table shows up shifted by
// one place in the two.
//
// Six families share the space:
//
//   sve_mem_64b_gld_sv             base register plus a vector of offsets
//   sve_mem_64b_gld_vi             vector of pointers plus an immediate index
//   sve_mem_64b_prfm_sv            prefetch, base register plus vector offsets
//   sve_mem_64b_prfm_vi            prefetch, vector of pointers plus an index
//   sve2_mem_gldnt_vs_base         SVE2 gather, vector of pointers plus scalar
//   sve_mem_128b_gld_64_unscaled   the 128-bit unscaled gather (ld1q)

/// Decode one word of the SVE 64-bit gather and prefetch space.
fn gather_c45(origin: Addr, word: u32) -> R {
    match gather_c45_text(word) {
        Ok((mn, text)) => R::new(mn, text),
        Err(family) => undefined(origin, word, family),
    }
}

/// The printed form of a gather: the mnemonic and the whole instruction.
type GatherText = Result<(&'static str, String), &'static str>;

fn gather_c45_text(word: u32) -> GatherText {
    // Each family names the bits it fixes and the space is partitioned
    // between them, so at most one of them can claim a word.
    if let Some(r) = gld_sv_text(word) {
        return r;
    }
    if let Some(r) = gld_vi_text(word) {
        return r;
    }
    if let Some(r) = gldnt_text(word) {
        return r;
    }
    if let Some(r) = ld1q_text(word) {
        return r;
    }
    if let Some(r) = prfm_sv_text(word) {
        return r;
    }
    if let Some(r) = prfm_vi_text(word) {
        return r;
    }
    Err("SVE 64-bit gather (unallocated)")
}

/// The 4-bit opcode of the 64-bit gather loads: the mnemonic and the size of
/// one element in bytes.  `0b1100` and `0b1101` are unallocated.
fn gld4_opc(opc: u32) -> Option<(&'static str, u32)> {
    Some(match opc {
        0b0000 => ("ld1sb", 1),
        0b0001 => ("ldff1sb", 1),
        0b0010 => ("ld1b", 1),
        0b0011 => ("ldff1b", 1),
        0b0100 => ("ld1sh", 2),
        0b0101 => ("ldff1sh", 2),
        0b0110 => ("ld1h", 2),
        0b0111 => ("ldff1h", 2),
        0b1000 => ("ld1sw", 4),
        0b1001 => ("ldff1sw", 4),
        0b1010 => ("ld1w", 4),
        0b1011 => ("ldff1w", 4),
        0b1110 => ("ld1d", 8),
        0b1111 => ("ldff1d", 8),
        _ => return None,
    })
}

/// The 5-bit opcode of the SVE2 gather that takes a vector of pointers and a
/// scalar offset: the mnemonic, the size of one element in bytes, and the
/// width of the vector the encoding names.  The low half of the table is the
/// 32-bit form, the high half the 64-bit one.
fn gldnt_opc(opc: u32) -> Option<(&'static str, u32, char)> {
    Some(match opc {
        0b00000 => ("ldnt1sb", 1, 's'),
        0b00001 => ("ldnt1b", 1, 's'),
        0b00100 => ("ldnt1sh", 2, 's'),
        0b00101 => ("ldnt1h", 2, 's'),
        0b01001 => ("ldnt1w", 4, 's'),
        0b10000 => ("ldnt1sb", 1, 'd'),
        0b10010 => ("ldnt1b", 1, 'd'),
        0b10100 => ("ldnt1sh", 2, 'd'),
        0b10110 => ("ldnt1h", 2, 'd'),
        0b11000 => ("ldnt1sw", 4, 'd'),
        0b11010 => ("ldnt1w", 4, 'd'),
        0b11110 => ("ldnt1d", 8, 'd'),
        _ => return None,
    })
}

/// `sve_mem_64b_gld_sv`: a gather from a base register, with the offsets
/// held in a vector.
fn gld_sv_text(word: u32) -> Option<GatherText> {
    if bits(word, 31, 25) != 0b1100010 {
        return None;
    }
    let opc = (bits(word, 24, 24) << 3) | (bits(word, 23, 23) << 2) | bits(word, 14, 13);
    let (mn, esz) = gld4_opc(opc)?;
    let xs = bits(word, 22, 22) == 1;
    let scaled = bits(word, 21, 21) == 1;
    let lsl = bits(word, 15, 15) == 1;
    // The offset operand takes one of four shapes.  The offsets are either
    // extended to 64 bits (`uxtw` or `sxtw`, optionally scaled by the element
    // size) or are already 64-bit wide and shifted instead.
    let off: Option<String> = match (xs, scaled, lsl) {
        (false, false, false) | (true, false, false) => {
            Some(if xs { "sxtw" } else { "uxtw" }.to_string())
        }
        (false, true, false) | (true, true, false) => {
            if esz == 1 {
                return None; // byte elements have no scaled form
            }
            let ext = if xs { "sxtw" } else { "uxtw" };
            Some(format!("{ext} #{}", esz.trailing_zeros()))
        }
        (true, false, true) => None, // 64-bit offsets, no shift
        (true, true, true) => {
            if esz == 1 {
                return None; // byte elements have no scaled form
            }
            Some(format!("lsl #{}", esz.trailing_zeros()))
        }
        // The unsigned 64-bit offsets do not exist.
        _ => return None,
    };
    let rn = bits(word, 9, 5);
    let base = sp_reg(rn);
    let zm = bits(word, 20, 16);
    let mem = match &off {
        None => format!("[{base}, z{zm}.d]"),
        Some(o) => format!("[{base}, z{zm}.d, {o}]"),
    };
    Some(Ok((
        mn,
        format!(
            "{mn}\t{{ z{}.d }}, p{}/z, {mem}",
            bits(word, 4, 0),
            bits(word, 12, 10)
        ),
    )))
}

/// `sve_mem_64b_gld_vi`: a gather from a vector of pointers, indexed by an
/// immediate.
fn gld_vi_text(word: u32) -> Option<GatherText> {
    if bits(word, 31, 25) != 0b1100010 {
        return None;
    }
    if bits(word, 22, 21) != 0b01 {
        return None;
    }
    if bits(word, 15, 15) != 1 {
        return None;
    }
    let opc = (bits(word, 24, 24) << 3) | (bits(word, 23, 23) << 2) | bits(word, 14, 13);
    let (mn, esz) = gld4_opc(opc)?;
    let imm5 = bits(word, 20, 16);
    let zn = bits(word, 9, 5);
    let mem = if imm5 == 0 {
        format!("[z{zn}.d]")
    } else {
        format!("[z{zn}.d, {}]", imm_text(imm5 as i64 * esz as i64))
    };
    Some(Ok((
        mn,
        format!(
            "{mn}\t{{ z{}.d }}, p{}/z, {mem}",
            bits(word, 4, 0),
            bits(word, 12, 10)
        ),
    )))
}

/// `sve2_mem_gldnt_vs_base`: the SVE2 gather that takes a vector of pointers
/// and a scalar offset.
fn gldnt_text(word: u32) -> Option<GatherText> {
    if bits(word, 31, 31) != 1 {
        return None;
    }
    if bits(word, 29, 25) != 0b00010 {
        return None;
    }
    if bits(word, 22, 21) != 0b00 {
        return None;
    }
    if bits(word, 15, 15) != 1 {
        return None;
    }
    let opc = (bits(word, 30, 30) << 4)
        | (bits(word, 24, 24) << 3)
        | (bits(word, 23, 23) << 2)
        | bits(word, 14, 13);
    let (mn, _esz, elem) = gldnt_opc(opc)?;
    let rm = bits(word, 20, 16);
    let zn = bits(word, 9, 5);
    let mem = if rm == 31 {
        format!("[z{zn}.{elem}]")
    } else {
        format!("[z{zn}.{elem}, x{rm}]")
    };
    Some(Ok((
        mn,
        format!(
            "{mn}\t{{ z{}.{elem} }}, p{}/z, {mem}",
            bits(word, 4, 0),
            bits(word, 12, 10)
        ),
    )))
}

/// `sve_mem_128b_gld_64_unscaled`: the 128-bit unscaled gather.
fn ld1q_text(word: u32) -> Option<GatherText> {
    if bits(word, 31, 21) != 0b11000100000 {
        return None;
    }
    if bits(word, 15, 13) != 0b101 {
        return None;
    }
    let rm = bits(word, 20, 16);
    let zn = bits(word, 9, 5);
    let mem = if rm == 31 {
        format!("[z{zn}.d]")
    } else {
        format!("[z{zn}.d, x{rm}]")
    };
    Some(Ok((
        "ld1q",
        format!(
            "ld1q\t{{ z{}.q }}, p{}/z, {mem}",
            bits(word, 4, 0),
            bits(word, 12, 10)
        ),
    )))
}

/// `sve_mem_64b_prfm_sv`: a prefetch from a base register, with the offsets
/// held in a vector.
fn prfm_sv_text(word: u32) -> Option<GatherText> {
    if bits(word, 31, 23) != 0b110001000 {
        return None;
    }
    if bits(word, 21, 21) != 1 {
        return None;
    }
    if bits(word, 4, 4) != 0 {
        return None;
    }
    let msz = bits(word, 14, 13);
    let xs = bits(word, 22, 22) == 1;
    let lsl = bits(word, 15, 15) == 1;
    let off: Option<String> = if lsl {
        // Only the signed 64-bit offsets exist in the shifted form.
        if !xs {
            return None;
        }
        if msz == 0 {
            None
        } else {
            Some(format!("lsl #{msz}"))
        }
    } else {
        let ext = if xs { "sxtw" } else { "uxtw" };
        if msz == 0 {
            Some(ext.to_string())
        } else {
            Some(format!("{ext} #{msz}"))
        }
    };
    let rn = bits(word, 9, 5);
    let base = sp_reg(rn);
    let zm = bits(word, 20, 16);
    let mem = match &off {
        None => format!("[{base}, z{zm}.d]"),
        Some(o) => format!("[{base}, z{zm}.d, {o}]"),
    };
    let prf = prf_mnemonic(msz);
    Some(Ok((
        prf,
        format!(
            "{prf}\t{}, p{}, {mem}",
            prfop_text(bits(word, 3, 0)),
            bits(word, 12, 10)
        ),
    )))
}

/// `sve_mem_64b_prfm_vi`: a prefetch from a vector of pointers, indexed by an
/// immediate.
fn prfm_vi_text(word: u32) -> Option<GatherText> {
    if bits(word, 31, 25) != 0b1100010 {
        return None;
    }
    if bits(word, 22, 21) != 0b00 {
        return None;
    }
    if bits(word, 15, 13) != 0b111 {
        return None;
    }
    if bits(word, 4, 4) != 0 {
        return None;
    }
    let msz = bits(word, 24, 23);
    let imm5 = bits(word, 20, 16);
    let zn = bits(word, 9, 5);
    let mem = if imm5 == 0 {
        format!("[z{zn}.d]")
    } else {
        format!("[z{zn}.d, {}]", imm_text(imm5 as i64 * (1i64 << msz)))
    };
    let prf = prf_mnemonic(msz);
    Some(Ok((
        prf,
        format!(
            "{prf}\t{}, p{}, {mem}",
            prfop_text(bits(word, 3, 0)),
            bits(word, 12, 10)
        ),
    )))
}

/// The prefetch mnemonic for a memory size field.
fn prf_mnemonic(msz: u32) -> &'static str {
    match msz {
        0 => "prfb",
        1 => "prfh",
        2 => "prfw",
        _ => "prfd",
    }
}

/// The prefetch operation the reference prints.  The architecture names the
/// first six values and the next six; the rest print as the raw four-bit
/// field.
fn prfop_text(prfop: u32) -> String {
    match prfop {
        0 => "pldl1keep".to_string(),
        1 => "pldl1strm".to_string(),
        2 => "pldl2keep".to_string(),
        3 => "pldl2strm".to_string(),
        4 => "pldl3keep".to_string(),
        5 => "pldl3strm".to_string(),
        8 => "pstl1keep".to_string(),
        9 => "pstl1strm".to_string(),
        10 => "pstl2keep".to_string(),
        11 => "pstl2strm".to_string(),
        12 => "pstl3keep".to_string(),
        13 => "pstl3strm".to_string(),
        v => format!("#0x{v:x}"),
    }
}

// === top bytes 0x84/0x85: SVE 32-bit gathers, prefetches and replicates ====
//
// Both top bytes sit in the same encoding space, `Inst{31-25} = 0b1000010`,
// and hold ten instruction classes.  Each class fixes enough bits that no
// two of them can claim the same word:
//
//   sve_mem_32b_gld_sv        scalar base plus a vector of offsets
//   sve_mem_32b_prfm_sv       the same shape, as a prefetch
//   sve_mem_32b_gld_vi        a vector of pointers plus an immediate index
//   sve_mem_ld_dup            LD1R, a replicated load with an immediate
//   sve2_mem_gldnt_vs_base    LDNT1x, a vector of pointers plus a scalar
//   sve_mem_prfm_ss           prefetch, scalar base plus scalar offset
//   sve_mem_32b_prfm_vi       prefetch, a vector of pointers plus an index
//   sve_mem_prfm_si           prefetch, predicate, immediate scaled by VL
//   sve_mem_z_fill            LDR (vector, immediate) scaled by VL
//   sve_mem_p_fill            LDR (predicate, immediate) scaled by VL
//
// The two top bytes differ only in bit 24, which is the most significant
// bit of the four-bit opcode of the gather loads and one half of the
// dtype field of the replicate loads.  A zero in bit 24 selects the
// 8-bit and 16-bit accesses, a one selects the 32-bit and 64-bit ones.

/// Decode one word of the SVE 32-bit gather space.
fn gather_84_85(origin: Addr, word: u32) -> R {
    match gather_84_85_text(word) {
        Ok((mn, text)) => R::new(mn, text),
        Err(family) => undefined(origin, word, family),
    }
}

/// The printed form of one word of the 32-bit gather space.
fn gather_84_85_text(word: u32) -> GatherText {
    if bits(word, 31, 25) != 0b1000010 {
        return Err("SVE 32-bit gather (unallocated)");
    }
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let rt = bits(word, 4, 0);
    // The four-bit opcode of the 32-bit gather loads: bit 24 comes first,
    // then bit 23, then the two bits below the predicate field.
    let opc = (bits(word, 24, 24) << 3) | (bits(word, 23, 23) << 2) | bits(word, 14, 13);

    // `sve_mem_z_fill` and `sve_mem_p_fill`: a whole vector or predicate
    // loaded from memory, with the offset scaled by the vector length.
    let kind = bits(word, 15, 13);
    if bits(word, 31, 22) == 0b1000010110 && matches!(kind, 0b010 | 0b000) {
        let imm9 = (bits(word, 21, 16) << 3) | bits(word, 12, 10);
        let v = simm_value(imm9, 9);
        let dst = if kind == 0b010 {
            format!("z{rt}")
        } else {
            // A predicate register only names p0 to p15.
            if rt > 15 {
                return Err("SVE 32-bit gather (unallocated)");
            }
            format!("p{rt}")
        };
        let mem = if v == 0 {
            format!("[{}]", sp_reg(rn))
        } else {
            format!("[{}, {}, mul vl]", sp_reg(rn), imm_text(v))
        };
        return Ok(("ldr", format!("ldr\t{dst}, {mem}")));
    }

    if bits(word, 15, 15) == 0 {
        // `sve_mem_prfm_si`: a prefetch of a whole vector, with the offset
        // scaled by the vector length.
        if bits(word, 31, 22) == 0b1000010111 {
            if bits(word, 4, 4) != 0 {
                return Err("SVE 32-bit gather (unallocated)");
            }
            let msz = bits(word, 14, 13);
            let v = simm_value(bits(word, 21, 16), 6);
            let mem = if v == 0 {
                format!("[{}]", sp_reg(rn))
            } else {
                format!("[{}, {}, mul vl]", sp_reg(rn), imm_text(v))
            };
            let prf = prf_mnemonic(msz);
            return Ok((
                prf,
                format!("{prf}\t{}, p{pg}, {mem}", prfop_text(bits(word, 3, 0))),
            ));
        }

        // `sve_mem_32b_prfm_sv`: a prefetch from a base register with the
        // offsets held in a vector.
        if bits(word, 24, 23) == 0b00 && bits(word, 21, 21) == 1 {
            if bits(word, 4, 4) != 0 {
                return Err("SVE 32-bit gather (unallocated)");
            }
            let msz = bits(word, 14, 13);
            let xs = bits(word, 22, 22) == 1;
            let mut ext = if xs { "sxtw" } else { "uxtw" }.to_string();
            if msz != 0 {
                ext.push_str(&format!(" #{msz}"));
            }
            let zm = bits(word, 20, 16);
            let prf = prf_mnemonic(msz);
            return Ok((
                prf,
                format!(
                    "{prf}\t{}, p{pg}, [{}, z{zm}.s, {ext}]",
                    prfop_text(bits(word, 3, 0)),
                    sp_reg(rn)
                ),
            ));
        }

        // `sve_mem_32b_gld_sv`: a gather from a base register with the
        // offsets held in a vector.
        let Some((mn, esz)) = gld32_opc(opc) else {
            return Err("SVE 32-bit gather (unallocated)");
        };
        let xs = bits(word, 22, 22) == 1;
        let scaled = bits(word, 21, 21) == 1;
        // Byte accesses have no scaled form.
        if esz == 1 && scaled {
            return Err("SVE 32-bit gather (unallocated)");
        }
        let mut ext = if xs { "sxtw" } else { "uxtw" }.to_string();
        if scaled {
            ext.push_str(&format!(" #{}", esz.trailing_zeros()));
        }
        let zm = bits(word, 20, 16);
        return Ok((
            mn,
            format!(
                "{mn}\t{{ z{rt}.s }}, p{pg}/z, [{}, z{zm}.s, {ext}]",
                sp_reg(rn)
            ),
        ));
    }

    // `sve_mem_32b_gld_vi`: a gather from a vector of pointers, indexed by
    // an immediate.
    if bits(word, 22, 21) == 0b01 {
        let Some((mn, esz)) = gld32_opc(opc) else {
            return Err("SVE 32-bit gather (unallocated)");
        };
        let imm = bits(word, 20, 16) * esz;
        let mem = if imm == 0 {
            format!("[z{rn}.s]")
        } else {
            format!("[z{rn}.s, {}]", imm_text(imm as i64))
        };
        return Ok((mn, format!("{mn}\t{{ z{rt}.s }}, p{pg}/z, {mem}")));
    }

    // `sve_mem_ld_dup`: the replicated loads, LD1R and its relatives.
    if bits(word, 22, 22) == 1 {
        let Some((mn, elem, acc)) = lddup_opc(bits(word, 24, 23), bits(word, 14, 13)) else {
            return Err("SVE 32-bit gather (unallocated)");
        };
        let imm = bits(word, 21, 16) * acc;
        let mem = if imm == 0 {
            format!("[{}]", sp_reg(rn))
        } else {
            format!("[{}, {}]", sp_reg(rn), imm_text(imm as i64))
        };
        return Ok((mn, format!("{mn}\t{{ z{rt}.{elem} }}, p{pg}/z, {mem}")));
    }

    // The remaining classes all live in `Inst{22-21} = 0b00` and are told
    // apart by the two bits below the predicate field: bit 14 alone picks
    // the LDNT1x gather, and bit 13 then picks the prefetch shape.
    if bits(word, 14, 14) == 0 {
        // `sve2_mem_gldnt_vs_base`: a vector of pointers plus a scalar
        // offset.
        let opc5 = (bits(word, 30, 30) << 4) | opc;
        let Some(mn) = gldnt32_opc(opc5) else {
            return Err("SVE 32-bit gather (unallocated)");
        };
        let rm = bits(word, 20, 16);
        let mem = if rm == 31 {
            format!("[z{rn}.s]")
        } else {
            format!("[z{rn}.s, x{rm}]")
        };
        return Ok((mn, format!("{mn}\t{{ z{rt}.s }}, p{pg}/z, {mem}")));
    }
    if bits(word, 13, 13) == 0 {
        // `sve_mem_prfm_ss`: a prefetch from a base register with a scalar
        // offset.
        if bits(word, 4, 4) != 0 {
            return Err("SVE 32-bit gather (unallocated)");
        }
        let opc3 = (bits(word, 24, 24) << 2) | (bits(word, 23, 23) << 1) | bits(word, 14, 14);
        if !matches!(opc3, 0b001 | 0b011 | 0b101 | 0b111) {
            return Err("SVE 32-bit gather (unallocated)");
        }
        let rm = bits(word, 20, 16);
        // The offset register may not be the zero register.
        if rm == 31 {
            return Err("SVE 32-bit gather (unallocated)");
        }
        let k = opc3 >> 1;
        let off = if k == 0 {
            format!("x{rm}")
        } else {
            format!("x{rm}, lsl #{k}")
        };
        let prf = prf_mnemonic(k);
        return Ok((
            prf,
            format!(
                "{prf}\t{}, p{pg}, [{}, {off}]",
                prfop_text(bits(word, 3, 0)),
                sp_reg(rn)
            ),
        ));
    }
    // `sve_mem_32b_prfm_vi`: a prefetch from a vector of pointers, indexed
    // by an immediate.
    if bits(word, 4, 4) != 0 {
        return Err("SVE 32-bit gather (unallocated)");
    }
    let msz = bits(word, 24, 23);
    let imm = bits(word, 20, 16) * (1 << msz);
    let mem = if imm == 0 {
        format!("[z{rn}.s]")
    } else {
        format!("[z{rn}.s, {}]", imm_text(imm as i64))
    };
    let prf = prf_mnemonic(msz);
    Ok((
        prf,
        format!("{prf}\t{}, p{pg}, {mem}", prfop_text(bits(word, 3, 0))),
    ))
}

/// The four-bit opcode of the 32-bit gather loads: the mnemonic and the
/// size of one element in bytes.
fn gld32_opc(opc: u32) -> Option<(&'static str, u32)> {
    Some(match opc {
        0b0000 => ("ld1sb", 1),
        0b0001 => ("ldff1sb", 1),
        0b0010 => ("ld1b", 1),
        0b0011 => ("ldff1b", 1),
        0b0100 => ("ld1sh", 2),
        0b0101 => ("ldff1sh", 2),
        0b0110 => ("ld1h", 2),
        0b0111 => ("ldff1h", 2),
        0b1010 => ("ld1w", 4),
        0b1011 => ("ldff1w", 4),
        _ => return None,
    })
}

/// The five-bit opcode of the SVE2 gather that takes a vector of 32-bit
/// pointers and a scalar offset.
fn gldnt32_opc(opc: u32) -> Option<&'static str> {
    Some(match opc {
        0b00000 => "ldnt1sb",
        0b00001 => "ldnt1b",
        0b00100 => "ldnt1sh",
        0b00101 => "ldnt1h",
        0b01001 => "ldnt1w",
        _ => return None,
    })
}

/// The replicated loads: the two dtype fields give the mnemonic, the
/// element the destination register holds and the size of the access in
/// bytes, which is what the immediate is scaled by.
fn lddup_opc(dtypeh: u32, dtypel: u32) -> Option<(&'static str, char, u32)> {
    Some(match (dtypeh, dtypel) {
        (0b00, 0b00) => ("ld1rb", 'b', 1),
        (0b00, 0b01) => ("ld1rb", 'h', 1),
        (0b00, 0b10) => ("ld1rb", 's', 1),
        (0b00, 0b11) => ("ld1rb", 'd', 1),
        (0b01, 0b00) => ("ld1rsw", 'd', 4),
        (0b01, 0b01) => ("ld1rh", 'h', 2),
        (0b01, 0b10) => ("ld1rh", 's', 2),
        (0b01, 0b11) => ("ld1rh", 'd', 2),
        (0b10, 0b00) => ("ld1rsh", 'd', 2),
        (0b10, 0b01) => ("ld1rsh", 's', 2),
        (0b10, 0b10) => ("ld1rw", 's', 4),
        (0b10, 0b11) => ("ld1rw", 'd', 4),
        (0b11, 0b00) => ("ld1rsb", 'd', 1),
        (0b11, 0b01) => ("ld1rsb", 's', 1),
        (0b11, 0b10) => ("ld1rsb", 'h', 1),
        (0b11, 0b11) => ("ld1rd", 'd', 8),
        _ => return None,
    })
}

// === top bytes 0xA4/0xA5: SVE contiguous loads ==============================
//
// Both top bytes share one encoding space, `Inst{31-25} = 0b1010010`, which
// holds every contiguous load: the plain LD1 forms and their fault-only and
// non-temporal relatives, the replicated loads LD1RQ and LD1RO, the 128-bit
// contiguous loads, and the two-, three- and four-register interleaving
// loads.  The three bits below the predicate field pick the class:
//
//   000  LD1RQ / LD1RO, register offset
//   001  LD1RQ / LD1RO and the 128-bit loads, immediate
//   010  LD1x, register offset
//   011  LDFF1x, register offset
//   100  the 128-bit loads and LD2Q/LD3Q/LD4Q, register offset
//   101  LD1x / LDNF1x, immediate
//   110  LDNT1x and LD2/LD3/LD4, register offset
//   111  LDNT1x and LD2/LD3/LD4, immediate
//
// The two top bytes differ in bit 24, the top bit of the size and of the
// dtype field: a zero there selects the byte and halfword accesses, a one
// the word and doubleword ones.

/// Decode one word of the SVE contiguous-load space.
fn cont_load_a45(origin: Addr, word: u32) -> R {
    match cont_load_a45_text(word) {
        Ok((mn, text)) => R::new(mn, text),
        Err(family) => undefined(origin, word, family),
    }
}

/// The printed form of one word of the contiguous-load space.
fn cont_load_a45_text(word: u32) -> GatherText {
    if bits(word, 31, 25) != 0b1010010 {
        return Err("SVE contiguous load (unallocated)");
    }
    let pg = bits(word, 12, 10);
    let rn = bits(word, 9, 5);
    let rt = bits(word, 4, 0);
    let base = sp_reg(rn);
    let m3 = bits(word, 15, 13);
    let b22_21 = bits(word, 22, 21);
    let rm = bits(word, 20, 16);
    // The immediate offset is a signed four-bit field that shares its bits
    // with the register offset.
    let imm4 = simm_value(bits(word, 19, 16), 4);

    match m3 {
        // LD1RQ and LD1RO with a register offset.
        0b000 => {
            if rm == 31 {
                return Err("SVE contiguous load (unallocated)");
            }
            let sz = bits(word, 24, 23);
            let (mn, elem) = match b22_21 {
                0b00 => ld1rq_opc(sz),
                0b01 => ld1ro_opc(sz),
                _ => return Err("SVE contiguous load (unallocated)"),
            };
            let mem = format!("[{base}, {}]", reg_off(rm, sz));
            Ok((mn, format!("{mn}\t{{ z{rt}.{elem} }}, p{pg}/z, {mem}")))
        }
        // LD1RQ, LD1RO and the 128-bit contiguous loads with an immediate.
        0b001 => {
            let sz = bits(word, 24, 23);
            let (mn, elem, scale, vl) = match (b22_21, bits(word, 20, 20)) {
                // LD1RQ replicates one 128-bit group, LD1RO one 256-bit
                // group, and the offset counts groups of the element.
                (0b00, 0) => {
                    let (mn, elem) = ld1rq_opc(sz);
                    (mn, elem, 16, false)
                }
                (0b01, 0) => {
                    let (mn, elem) = ld1ro_opc(sz);
                    (mn, elem, 32, false)
                }
                // The 128-bit contiguous loads move whole quads, and their
                // offset scales with the vector length.
                (0b00, 1) => {
                    let Some((mn, elem, _shift)) = q128_opc(sz) else {
                        return Err("SVE contiguous load (unallocated)");
                    };
                    (mn, elem, 1, true)
                }
                _ => return Err("SVE contiguous load (unallocated)"),
            };
            let v = imm4 * scale;
            let mem = offset_mem(&base, v, vl);
            Ok((mn, format!("{mn}\t{{ z{rt}.{elem} }}, p{pg}/z, {mem}")))
        }
        // LD1x and LDFF1x with a register offset.
        0b010 | 0b011 => {
            let Some((mn, elem, acc)) = cld_dtype(bits(word, 24, 21)) else {
                return Err("SVE contiguous load (unallocated)");
            };
            let mn = if m3 == 0b011 { ldff1_mnemonic(mn) } else { mn };
            let mem = if rm == 31 {
                // The first-copy form has no encoding without an offset.
                if m3 == 0b010 {
                    return Err("SVE contiguous load (unallocated)");
                }
                format!("[{base}]")
            } else {
                format!("[{base}, {}]", reg_off(rm, acc.trailing_zeros()))
            };
            Ok((mn, format!("{mn}\t{{ z{rt}.{elem} }}, p{pg}/z, {mem}")))
        }
        // The 128-bit contiguous loads with a register offset.
        0b100 => match b22_21 {
            0b00 => {
                let Some((mn, elem, shift)) = q128_opc(bits(word, 24, 23)) else {
                    return Err("SVE contiguous load (unallocated)");
                };
                if rm == 31 {
                    return Err("SVE contiguous load (unallocated)");
                }
                let mem = format!("[{base}, {}]", reg_off(rm, shift));
                Ok((mn, format!("{mn}\t{{ z{rt}.{elem} }}, p{pg}/z, {mem}")))
            }
            // LD2Q, LD3Q and LD4Q with a register offset.
            0b01 => {
                let sz = bits(word, 24, 23);
                if sz == 0b00 || rm == 31 {
                    return Err("SVE contiguous load (unallocated)");
                }
                let n = 1 + sz as usize;
                let mem = format!("[{base}, {}]", reg_off(rm, 4));
                let mn = ldq_mnemonic(n);
                Ok((
                    mn,
                    format!("{mn}\t{{ {} }}, p{pg}/z, {mem}", zlist(rt, n as u32, 'q')),
                ))
            }
            _ => Err("SVE contiguous load (unallocated)"),
        },
        // LD1x and LDNF1x with an immediate.
        0b101 => {
            let Some((mn, elem, _acc)) = cld_dtype(bits(word, 24, 21)) else {
                return Err("SVE contiguous load (unallocated)");
            };
            let mn = if bits(word, 20, 20) == 1 {
                ldnf1_mnemonic(mn)
            } else {
                mn
            };
            let mem = offset_mem(&base, imm4, true);
            Ok((mn, format!("{mn}\t{{ z{rt}.{elem} }}, p{pg}/z, {mem}")))
        }
        // LDNT1x with a register offset.
        0b110 if b22_21 == 0b00 => {
            let (mn, elem, acc) = ldnt1_opc(bits(word, 24, 23));
            if rm == 31 {
                return Err("SVE contiguous load (unallocated)");
            }
            let mem = format!("[{base}, {}]", reg_off(rm, acc.trailing_zeros()));
            Ok((mn, format!("{mn}\t{{ z{rt}.{elem} }}, p{pg}/z, {mem}")))
        }
        // LDNT1x with an immediate.
        0b111 if bits(word, 22, 20) == 0b000 => {
            let (mn, elem, _acc) = ldnt1_opc(bits(word, 24, 23));
            let mem = offset_mem(&base, imm4, true);
            Ok((mn, format!("{mn}\t{{ z{rt}.{elem} }}, p{pg}/z, {mem}")))
        }
        // LD2, LD3 and LD4 with a register offset.
        0b110 => {
            let n = b22_21 as usize + 1;
            let (suf, elem, acc) = eld_opc(bits(word, 24, 23));
            if rm == 31 {
                return Err("SVE contiguous load (unallocated)");
            }
            let mem = format!("[{base}, {}]", reg_off(rm, acc.trailing_zeros()));
            let mn = ld_mnemonic(n, suf);
            Ok((
                mn,
                format!("{mn}\t{{ {} }}, p{pg}/z, {mem}", zlist(rt, n as u32, elem)),
            ))
        }
        // LD2, LD3 and LD4 with an immediate.
        0b111 => {
            let nregs = (bits(word, 20, 20) << 2) | b22_21;
            let (mn, elem, n) = match nregs {
                // The quad forms move whole 128-bit groups.
                0b100 => {
                    let sz = bits(word, 24, 23);
                    if sz == 0b00 {
                        return Err("SVE contiguous load (unallocated)");
                    }
                    let n = 1 + sz as usize;
                    (ldq_mnemonic(n), 'q', n)
                }
                0b001..=0b011 => {
                    let n = nregs as usize + 1;
                    let (suf, elem, _acc) = eld_opc(bits(word, 24, 23));
                    (ld_mnemonic(n, suf), elem, n)
                }
                _ => return Err("SVE contiguous load (unallocated)"),
            };
            let mem = offset_mem(&base, imm4 * n as i64, true);
            Ok((
                mn,
                format!("{mn}\t{{ {} }}, p{pg}/z, {mem}", zlist(rt, n as u32, elem)),
            ))
        }
        _ => Err("SVE contiguous load (unallocated)"),
    }
}

/// The size field of the replicated loads: the mnemonic and the element the
/// destination register holds.
fn ld1rq_opc(sz: u32) -> (&'static str, char) {
    match sz {
        0b00 => ("ld1rqb", 'b'),
        0b01 => ("ld1rqh", 'h'),
        0b10 => ("ld1rqw", 's'),
        _ => ("ld1rqd", 'd'),
    }
}

/// The size field of the 256-bit replicated load, LD1RO.
fn ld1ro_opc(sz: u32) -> (&'static str, char) {
    match sz {
        0b00 => ("ld1rob", 'b'),
        0b01 => ("ld1roh", 'h'),
        0b10 => ("ld1row", 's'),
        _ => ("ld1rod", 'd'),
    }
}

/// The dtype field of the plain contiguous loads: the mnemonic, the element
/// the destination register holds and the size of one access in bytes.
fn cld_dtype(dtype: u32) -> Option<(&'static str, char, u32)> {
    Some(match dtype {
        0b0000 => ("ld1b", 'b', 1),
        0b0001 => ("ld1b", 'h', 1),
        0b0010 => ("ld1b", 's', 1),
        0b0011 => ("ld1b", 'd', 1),
        0b0100 => ("ld1sw", 'd', 4),
        0b0101 => ("ld1h", 'h', 2),
        0b0110 => ("ld1h", 's', 2),
        0b0111 => ("ld1h", 'd', 2),
        0b1000 => ("ld1sh", 'd', 2),
        0b1001 => ("ld1sh", 's', 2),
        0b1010 => ("ld1w", 's', 4),
        0b1011 => ("ld1w", 'd', 4),
        0b1100 => ("ld1sb", 'd', 1),
        0b1101 => ("ld1sb", 's', 1),
        0b1110 => ("ld1sb", 'h', 1),
        _ => ("ld1d", 'd', 8),
    })
}

/// The size field of the non-temporal loads.
fn ldnt1_opc(sz: u32) -> (&'static str, char, u32) {
    match sz {
        0b00 => ("ldnt1b", 'b', 1),
        0b01 => ("ldnt1h", 'h', 2),
        0b10 => ("ldnt1w", 's', 4),
        _ => ("ldnt1d", 'd', 8),
    }
}

/// The size field of the interleaving loads: the mnemonic suffix, the
/// element and the size of one access in bytes.
fn eld_opc(sz: u32) -> (&'static str, char, u32) {
    match sz {
        0b00 => ("b", 'b', 1),
        0b01 => ("h", 'h', 2),
        0b10 => ("w", 's', 4),
        _ => ("d", 'd', 8),
    }
}

/// The size field of the 128-bit contiguous loads: the mnemonic, the element
/// and the shift the register offset is scaled by.
fn q128_opc(sz: u32) -> Option<(&'static str, char, u32)> {
    Some(match sz {
        0b10 => ("ld1w", 'q', 2),
        0b11 => ("ld1d", 'q', 3),
        _ => return None,
    })
}

/// The fault-only variant of a plain contiguous load.
fn ldff1_mnemonic(mn: &'static str) -> &'static str {
    match mn {
        "ld1b" => "ldff1b",
        "ld1h" => "ldff1h",
        "ld1w" => "ldff1w",
        "ld1d" => "ldff1d",
        "ld1sb" => "ldff1sb",
        "ld1sh" => "ldff1sh",
        "ld1sw" => "ldff1sw",
        _ => "ldff1d",
    }
}

/// The non-fault-only variant of a plain contiguous load.
fn ldnf1_mnemonic(mn: &'static str) -> &'static str {
    match mn {
        "ld1b" => "ldnf1b",
        "ld1h" => "ldnf1h",
        "ld1w" => "ldnf1w",
        "ld1d" => "ldnf1d",
        "ld1sb" => "ldnf1sb",
        "ld1sh" => "ldnf1sh",
        "ld1sw" => "ldnf1sw",
        _ => "ldnf1d",
    }
}

/// The mnemonic of an interleaving load of `n` registers.
fn ld_mnemonic(n: usize, suffix: &str) -> &'static str {
    match (n, suffix) {
        (2, "b") => "ld2b",
        (3, "b") => "ld3b",
        (4, "b") => "ld4b",
        (2, "h") => "ld2h",
        (3, "h") => "ld3h",
        (4, "h") => "ld4h",
        (2, "w") => "ld2w",
        (3, "w") => "ld3w",
        (4, "w") => "ld4w",
        (2, "d") => "ld2d",
        (3, "d") => "ld3d",
        _ => "ld4d",
    }
}

/// The mnemonic of an interleaving load of whole quads.
fn ldq_mnemonic(n: usize) -> &'static str {
    match n {
        2 => "ld2q",
        3 => "ld3q",
        _ => "ld4q",
    }
}

/// A register offset operand, shifted when the access size asks for it.
fn reg_off(rm: u32, shift: u32) -> String {
    if shift == 0 {
        format!("x{rm}")
    } else {
        format!("x{rm}, lsl #{shift}")
    }
}

/// A memory operand with a byte offset, which is scaled by the vector
/// length in the families that move whole vectors.
fn offset_mem(base: &str, v: i64, vl: bool) -> String {
    if v == 0 {
        format!("[{base}]")
    } else if vl {
        format!("[{base}, {}, mul vl]", imm_text(v))
    } else {
        format!("[{base}, {}]", imm_text(v))
    }
}

// === top bytes 0x64/0x65: SVE floating-point ================================
//
// The two top bytes hold the floating-point half of the SVE encoding
// space: the arithmetic, comparison, conversion and reduction families,
// in their predicated, unpredicated and indexed forms.  Each family fixes
// a set of bits and leaves the rest to the operands; the tables below
// list every family the reference disassembler accepts, with the bits it
// fixes and the operands it prints.
//
// The tables were derived from the reference disassembler by comparing the
// printed form of every word of the space, so they describe what the
// architecture really decodes rather than what a manual says it should.

/// What an operand of a floating-point instruction is.
#[derive(Clone, Copy)]
enum FpOpKind {
    /// A vector register: `z<n>.<t>`.
    Z,
    /// A vector register with an element index: `z<n>.<t>[i]`.
    ZIdx,
    /// A governing or merging predicate: `p<n>`.
    P,
    /// The destination predicate of a comparison: `p<n>.<t>`.
    Pd,
    /// The vector register of a reduction: `v<n>.<k><t>`.
    Simd,
    /// The scalar register of a reduction: `<t><n>`.
    Scalar,
    /// The rotation of a complex multiply, two bits: 0, 90, 180 or 270.
    Rot2,
    /// The rotation of a complex add, one bit: 90 or 270.
    Rot1,
    /// A floating-point immediate of half and one.
    FpImm,
    /// A floating-point zero.
    FZero,
    /// A hexadecimal immediate.
    IHex,
}

/// One operand of a floating-point instruction.
#[derive(Clone, Copy)]
struct FpOp {
    /// What the operand is.
    kind: FpOpKind,
    /// The element size letter, or none.
    elem: char,
    /// The register field.
    hi: u8,
    lo: u8,
    /// The number of lanes of a reduced result.
    aux: u32,
    /// The bits that hold an element index.
    ibits: u32,
    /// The text of a floating-point immediate.
    text: &'static str,
}

/// One family of the floating-point space.
struct FpClass {
    /// The bits the family fixes.
    mask: u32,
    /// The value those bits take.
    val: u32,
    /// The mnemonic.
    mn: &'static str,
    /// The operands, in the order they are printed.
    ops: &'static [FpOp],
}

/// The instruction classes of the top byte 0x64.
static FP64: &[FpClass] = &[
    FpClass {
        mask: 0xffa0fc00,
        val: 0x64200000,
        mn: "fmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x580000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffa0fc00,
        val: 0x64200400,
        mn: "fmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x580000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffa0fc00,
        val: 0x64200800,
        mn: "bfmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x580000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffa0fc00,
        val: 0x64200c00,
        mn: "bfmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x580000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffa0fc00,
        val: 0x64202000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x580000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffa0fc00,
        val: 0x64202800,
        mn: "bfmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x580000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe08000,
        val: 0x64400000,
        mn: "fcmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Rot2,
                elem: '\0',
                hi: 14,
                lo: 13,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe08000,
        val: 0x64800000,
        mn: "fcmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Rot2,
                elem: '\0',
                hi: 14,
                lo: 13,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe08000,
        val: 0x64c00000,
        mn: "fcmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Rot2,
                elem: '\0',
                hi: 14,
                lo: 13,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f000,
        val: 0x64a01000,
        mn: "fcmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180000,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Rot2,
                elem: '\0',
                hi: 11,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f000,
        val: 0x64e01000,
        mn: "fcmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 's',
                hi: 19,
                lo: 16,
                aux: 0,
                ibits: 0x100000,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Rot2,
                elem: '\0',
                hi: 11,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f400,
        val: 0x64a04000,
        mn: "fmlalb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180800,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f400,
        val: 0x64a04400,
        mn: "fmlalt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180800,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f400,
        val: 0x64a06000,
        mn: "fmlslb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180800,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f400,
        val: 0x64a06400,
        mn: "fmlslt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180800,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f400,
        val: 0x64e04000,
        mn: "bfmlalb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180800,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f400,
        val: 0x64e04400,
        mn: "bfmlalt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180800,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f400,
        val: 0x64e06000,
        mn: "bfmlslb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180800,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0f400,
        val: 0x64e06400,
        mn: "bfmlslt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180800,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64202400,
        mn: "bfclamp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64204000,
        mn: "fdot",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64208000,
        mn: "fdot",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64602400,
        mn: "fclamp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64604000,
        mn: "bfdot",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'h',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64608000,
        mn: "bfdot",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x6460e400,
        mn: "bfmmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a00000,
        mn: "fmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 's',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a00400,
        mn: "fmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 's',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a02000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 's',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x180000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a02400,
        mn: "fclamp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a08000,
        mn: "fmlalb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a08400,
        mn: "fmlalt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a0a000,
        mn: "fmlslb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a0a400,
        mn: "fmlslt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64a0e400,
        mn: "fmmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e00000,
        mn: "fmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'd',
                hi: 19,
                lo: 16,
                aux: 0,
                ibits: 0x100000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e00400,
        mn: "fmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'd',
                hi: 19,
                lo: 16,
                aux: 0,
                ibits: 0x100000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e02000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::ZIdx,
                elem: 'd',
                hi: 19,
                lo: 16,
                aux: 0,
                ibits: 0x100000,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e02400,
        mn: "fclamp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e08000,
        mn: "bfmlalb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e08400,
        mn: "bfmlalt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e0a000,
        mn: "bfmlslb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e0a400,
        mn: "bfmlslt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x64e0e400,
        mn: "fmmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfffee000,
        val: 0x64408000,
        mn: "fcadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Rot1,
                elem: '\0',
                hi: 16,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfffee000,
        val: 0x64808000,
        mn: "fcadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Rot1,
                elem: '\0',
                hi: 16,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfffee000,
        val: 0x64c08000,
        mn: "fcadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Rot1,
                elem: '\0',
                hi: 16,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6402a000,
        mn: "fcvtxnt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x640aa000,
        mn: "fcvtxnt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641ac000,
        mn: "fcvtx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641c8000,
        mn: "frint32z",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641ca000,
        mn: "frint32x",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641cc000,
        mn: "frint32z",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641ce000,
        mn: "frint32x",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641d8000,
        mn: "frint64z",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641da000,
        mn: "frint64x",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641dc000,
        mn: "frint64z",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641de000,
        mn: "frint64x",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641ea000,
        mn: "flogb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641ec000,
        mn: "flogb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x641ee000,
        mn: "flogb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64508000,
        mn: "faddp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6450a000,
        mn: "faddqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 8,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64548000,
        mn: "fmaxnmp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6454a000,
        mn: "fmaxnmqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 8,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64558000,
        mn: "fminnmp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6455a000,
        mn: "fminnmqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 8,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64568000,
        mn: "fmaxp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6456a000,
        mn: "fmaxqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 8,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64578000,
        mn: "fminp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6457a000,
        mn: "fminqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 8,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64588000,
        mn: "frintn",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6458a000,
        mn: "frintp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6458c000,
        mn: "frintm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6458e000,
        mn: "frintz",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64598000,
        mn: "frinta",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6459c000,
        mn: "frintx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6459e000,
        mn: "frinti",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645b8000,
        mn: "frecpx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645ba000,
        mn: "fsqrt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645cc000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645ce000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645d8000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645da000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645dc000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645de000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645ec000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645ee000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645f8000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645fa000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645fc000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x645fe000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6480a000,
        mn: "fcvtnt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6481a000,
        mn: "fcvtlt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6482a000,
        mn: "bfcvtnt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6488a000,
        mn: "fcvtnt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6489a000,
        mn: "fcvtlt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x648aa000,
        mn: "bfcvtnt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64908000,
        mn: "faddp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6490a000,
        mn: "faddqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 4,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64948000,
        mn: "fmaxnmp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6494a000,
        mn: "fmaxnmqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 4,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64958000,
        mn: "fminnmp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6495a000,
        mn: "fminnmqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 4,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64968000,
        mn: "fmaxp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6496a000,
        mn: "fmaxqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 4,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64978000,
        mn: "fminp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6497a000,
        mn: "fminqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 4,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64988000,
        mn: "frintn",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6498a000,
        mn: "frintp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6498c000,
        mn: "frintm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6498e000,
        mn: "frintz",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64998000,
        mn: "frinta",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6499c000,
        mn: "frintx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6499e000,
        mn: "frinti",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649a8000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649aa000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649ac000,
        mn: "bfcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649b8000,
        mn: "frecpx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649ba000,
        mn: "fsqrt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649d8000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649da000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649f8000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x649fa000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64c2a000,
        mn: "fcvtnt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64c3a000,
        mn: "fcvtlt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64caa000,
        mn: "fcvtnt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64cba000,
        mn: "fcvtlt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d08000,
        mn: "faddp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d0a000,
        mn: "faddqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 2,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d48000,
        mn: "fmaxnmp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d4a000,
        mn: "fmaxnmqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 2,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d58000,
        mn: "fminnmp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d5a000,
        mn: "fminnmqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 2,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d68000,
        mn: "fmaxp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d6a000,
        mn: "fmaxqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 2,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d78000,
        mn: "fminp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d7a000,
        mn: "fminqv",
        ops: &[
            FpOp {
                kind: FpOpKind::Simd,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 2,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d88000,
        mn: "frintn",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d8a000,
        mn: "frintp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d8c000,
        mn: "frintm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d8e000,
        mn: "frintz",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d98000,
        mn: "frinta",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d9c000,
        mn: "frintx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64d9e000,
        mn: "frinti",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64da8000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64daa000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dac000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dae000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64db8000,
        mn: "frecpx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dba000,
        mn: "fsqrt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dc8000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dca000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dd8000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dda000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64ddc000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dde000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64de8000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dea000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64df8000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dfa000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dfc000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x64dfe000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
];

/// The instruction classes of the top byte 0x65.
static FP65: &[FpClass] = &[
    FpClass {
        mask: 0xffe0e000,
        val: 0x65200000,
        mn: "bfmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65202000,
        mn: "bfmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65600000,
        mn: "fmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65602000,
        mn: "fmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65604000,
        mn: "fnmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65606000,
        mn: "fnmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65608000,
        mn: "fmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x6560a000,
        mn: "fmsb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x6560c000,
        mn: "fnmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x6560e000,
        mn: "fnmsb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65a00000,
        mn: "fmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65a02000,
        mn: "fmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65a04000,
        mn: "fnmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65a06000,
        mn: "fnmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65a08000,
        mn: "fmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65a0a000,
        mn: "fmsb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65a0c000,
        mn: "fnmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65a0e000,
        mn: "fnmsb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65e00000,
        mn: "fmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65e02000,
        mn: "fmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65e04000,
        mn: "fnmla",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65e06000,
        mn: "fnmls",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65e08000,
        mn: "fmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65e0a000,
        mn: "fmsb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65e0c000,
        mn: "fnmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e000,
        val: 0x65e0e000,
        mn: "fnmsb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65404000,
        mn: "fcmge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65404010,
        mn: "fcmgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65406000,
        mn: "fcmeq",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65406010,
        mn: "fcmne",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x6540c000,
        mn: "fcmuo",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x6540c010,
        mn: "facge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x6540e010,
        mn: "facgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65804000,
        mn: "fcmge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65804010,
        mn: "fcmgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65806000,
        mn: "fcmeq",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65806010,
        mn: "fcmne",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x6580c000,
        mn: "fcmuo",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x6580c010,
        mn: "facge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x6580e010,
        mn: "facgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65c04000,
        mn: "fcmge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65c04010,
        mn: "fcmgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65c06000,
        mn: "fcmeq",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65c06010,
        mn: "fcmne",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65c0c000,
        mn: "fcmuo",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65c0c010,
        mn: "facge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0e010,
        val: 0x65c0e010,
        mn: "facgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65000000,
        mn: "bfadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65000400,
        mn: "bfsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65000800,
        mn: "bfmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65400000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65400400,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65400800,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65400c00,
        mn: "ftsmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65401800,
        mn: "frecps",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65401c00,
        mn: "frsqrts",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65800000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65800400,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65800800,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65800c00,
        mn: "ftsmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65801800,
        mn: "frecps",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65801c00,
        mn: "frsqrts",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65c00000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65c00400,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65c00800,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65c00c00,
        mn: "ftsmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65c01800,
        mn: "frecps",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffe0fc00,
        val: 0x65c01c00,
        mn: "frsqrts",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 20,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfff8fc00,
        val: 0x65508000,
        mn: "ftmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::IHex,
                elem: '\0',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfff8fc00,
        val: 0x65908000,
        mn: "ftmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::IHex,
                elem: '\0',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfff8fc00,
        val: 0x65d08000,
        mn: "ftmad",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::IHex,
                elem: '\0',
                hi: 18,
                lo: 16,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65008000,
        mn: "bfadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65018000,
        mn: "bfsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65028000,
        mn: "bfmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65048000,
        mn: "bfmaxnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65058000,
        mn: "bfminnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65068000,
        mn: "bfmax",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65078000,
        mn: "bfmin",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x650aa000,
        mn: "fcvtx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6510a000,
        mn: "frint32z",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6511a000,
        mn: "frint32x",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6512a000,
        mn: "frint32z",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6513a000,
        mn: "frint32x",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6514a000,
        mn: "frint64z",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6515a000,
        mn: "frint64x",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6516a000,
        mn: "frint64z",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6517a000,
        mn: "frint64x",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x651aa000,
        mn: "flogb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x651ca000,
        mn: "flogb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x651ea000,
        mn: "flogb",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65402000,
        mn: "faddv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65408000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6540a000,
        mn: "frintn",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65418000,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6541a000,
        mn: "frintp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65428000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6542a000,
        mn: "frintm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65438000,
        mn: "fsubr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6543a000,
        mn: "frintz",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65442000,
        mn: "fmaxnmv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65448000,
        mn: "fmaxnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6544a000,
        mn: "frinta",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65452000,
        mn: "fminnmv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65458000,
        mn: "fminnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65462000,
        mn: "fmaxv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65468000,
        mn: "fmax",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6546a000,
        mn: "frintx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65472000,
        mn: "fminv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65478000,
        mn: "fmin",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6547a000,
        mn: "frinti",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65488000,
        mn: "fabd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65498000,
        mn: "fscale",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x654a8000,
        mn: "fmulx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x654c8000,
        mn: "fdivr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x654ca000,
        mn: "frecpx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x654d8000,
        mn: "fdiv",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x654da000,
        mn: "fsqrt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6552a000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6553a000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6554a000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6555a000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6556a000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6557a000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65582000,
        mn: "fadda",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x655aa000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x655ba000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x655ca000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x655da000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x655ea000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x655fa000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65802000,
        mn: "faddv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65808000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6580a000,
        mn: "frintn",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65818000,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6581a000,
        mn: "frintp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65828000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6582a000,
        mn: "frintm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65838000,
        mn: "fsubr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6583a000,
        mn: "frintz",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65842000,
        mn: "fmaxnmv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65848000,
        mn: "fmaxnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6584a000,
        mn: "frinta",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65852000,
        mn: "fminnmv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65858000,
        mn: "fminnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65862000,
        mn: "fmaxv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65868000,
        mn: "fmax",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6586a000,
        mn: "frintx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65872000,
        mn: "fminv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65878000,
        mn: "fmin",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6587a000,
        mn: "frinti",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65888000,
        mn: "fabd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6588a000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65898000,
        mn: "fscale",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6589a000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x658a8000,
        mn: "fmulx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x658aa000,
        mn: "bfcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x658c8000,
        mn: "fdivr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x658ca000,
        mn: "frecpx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x658d8000,
        mn: "fdiv",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x658da000,
        mn: "fsqrt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6594a000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x6595a000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65982000,
        mn: "fadda",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x659ca000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x659da000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c02000,
        mn: "faddv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c08000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c0a000,
        mn: "frintn",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c18000,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c1a000,
        mn: "frintp",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c28000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c2a000,
        mn: "frintm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c38000,
        mn: "fsubr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c3a000,
        mn: "frintz",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c42000,
        mn: "fmaxnmv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c48000,
        mn: "fmaxnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c4a000,
        mn: "frinta",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c52000,
        mn: "fminnmv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c58000,
        mn: "fminnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c62000,
        mn: "fmaxv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c68000,
        mn: "fmax",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c6a000,
        mn: "frintx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c72000,
        mn: "fminv",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c78000,
        mn: "fmin",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c7a000,
        mn: "frinti",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c88000,
        mn: "fabd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c8a000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c98000,
        mn: "fscale",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65c9a000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65ca8000,
        mn: "fmulx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65caa000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65cba000,
        mn: "fcvt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65cc8000,
        mn: "fdivr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65cca000,
        mn: "frecpx",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65cd8000,
        mn: "fdiv",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65cda000,
        mn: "fsqrt",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d0a000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d1a000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d4a000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d5a000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d6a000,
        mn: "scvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d7a000,
        mn: "ucvtf",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d82000,
        mn: "fadda",
        ops: &[
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: '\0',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Scalar,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d8a000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65d9a000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65dca000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65dda000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65dea000,
        mn: "fcvtzs",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe000,
        val: 0x65dfa000,
        mn: "fcvtzu",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65502000,
        mn: "fcmge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65502010,
        mn: "fcmgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65512000,
        mn: "fcmlt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65512010,
        mn: "fcmle",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65522000,
        mn: "fcmeq",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65532000,
        mn: "fcmne",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'h',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65902000,
        mn: "fcmge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65902010,
        mn: "fcmgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65912000,
        mn: "fcmlt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65912010,
        mn: "fcmle",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65922000,
        mn: "fcmeq",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65932000,
        mn: "fcmne",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 's',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65d02000,
        mn: "fcmge",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65d02010,
        mn: "fcmgt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65d12000,
        mn: "fcmlt",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65d12010,
        mn: "fcmle",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65d22000,
        mn: "fcmeq",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe010,
        val: 0x65d32000,
        mn: "fcmne",
        ops: &[
            FpOp {
                kind: FpOpKind::Pd,
                elem: 'd',
                hi: 3,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'z',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FZero,
                elem: '\0',
                hi: 0,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65588000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65598000,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x655a8000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x655b8000,
        mn: "fsubr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x655c8000,
        mn: "fmaxnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x655d8000,
        mn: "fminnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x655e8000,
        mn: "fmax",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x655f8000,
        mn: "fmin",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65988000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65998000,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x659a8000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x659b8000,
        mn: "fsubr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x659c8000,
        mn: "fmaxnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x659d8000,
        mn: "fminnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x659e8000,
        mn: "fmax",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x659f8000,
        mn: "fmin",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65d88000,
        mn: "fadd",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65d98000,
        mn: "fsub",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65da8000,
        mn: "fmul",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65db8000,
        mn: "fsubr",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.5",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65dc8000,
        mn: "fmaxnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65dd8000,
        mn: "fminnm",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65de8000,
        mn: "fmax",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xffffe3c0,
        val: 0x65df8000,
        mn: "fmin",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::P,
                elem: 'm',
                hi: 12,
                lo: 10,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::FpImm,
                elem: '\0',
                hi: 5,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "0.0",
            },
        ],
    },
    FpClass {
        mask: 0xfffffc00,
        val: 0x654e3000,
        mn: "frecpe",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfffffc00,
        val: 0x654f3000,
        mn: "frsqrte",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'h',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfffffc00,
        val: 0x658e3000,
        mn: "frecpe",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfffffc00,
        val: 0x658f3000,
        mn: "frsqrte",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 's',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfffffc00,
        val: 0x65ce3000,
        mn: "frecpe",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
    FpClass {
        mask: 0xfffffc00,
        val: 0x65cf3000,
        mn: "frsqrte",
        ops: &[
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 4,
                lo: 0,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
            FpOp {
                kind: FpOpKind::Z,
                elem: 'd',
                hi: 9,
                lo: 5,
                aux: 0,
                ibits: 0x0,
                text: "",
            },
        ],
    },
];
/// Decode one word of the SVE floating-point space.
fn float_64_65(origin: Addr, word: u32) -> R {
    let table = if bits(word, 31, 24) == 0x64 {
        FP64
    } else {
        FP65
    };
    for cls in table {
        if word & cls.mask == cls.val {
            return R::new(cls.mn, fp_text(word, cls));
        }
    }
    undefined(origin, word, "SVE floating-point (unallocated)")
}

/// The printed form of a floating-point instruction.
fn fp_text(word: u32, cls: &FpClass) -> String {
    let mut out = String::from(cls.mn);
    out.push('\t');
    for (i, op) in cls.ops.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&fp_operand(word, op, cls.mn));
    }
    out
}

/// The value of a register field.
fn fp_field(word: u32, hi: u8, lo: u8) -> u32 {
    (word >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

/// The printed form of one operand.
fn fp_operand(word: u32, op: &FpOp, mn: &str) -> String {
    match op.kind {
        FpOpKind::Z => {
            format!("z{}.{}", fp_field(word, op.hi, op.lo) & 31, op.elem)
        }
        FpOpKind::ZIdx => format!(
            "z{}.{}[{}]",
            fp_field(word, op.hi, op.lo) & 31,
            op.elem,
            fp_index(word, op),
        ),
        FpOpKind::P => {
            let p = fp_field(word, op.hi, op.lo) & 15;
            if op.elem == '\0' {
                format!("p{p}")
            } else {
                format!("p{p}/{}", op.elem)
            }
        }
        FpOpKind::Pd => {
            format!("p{}.{}", fp_field(word, op.hi, op.lo) & 15, op.elem)
        }
        FpOpKind::Simd => format!(
            "v{}.{}{}",
            fp_field(word, op.hi, op.lo) & 31,
            op.aux,
            op.elem
        ),
        FpOpKind::Scalar => {
            format!("{}{}", op.elem, fp_field(word, op.hi, op.lo) & 31)
        }
        FpOpKind::Rot2 => format!("#{}", fp_field(word, op.hi, op.lo) * 90),
        FpOpKind::Rot1 => format!("#{}", 90 + fp_field(word, op.hi, op.lo) * 180),
        FpOpKind::FpImm => {
            // the immediate is one of two values, and which pair it comes
            // from is a property of the mnemonic: the multiply scales by
            // two where the others add one
            if fp_field(word, op.hi, op.lo) == 0 {
                format!("#{}", op.text)
            } else if mn == "fmul" {
                "#2.0".to_string()
            } else {
                "#1.0".to_string()
            }
        }
        FpOpKind::FZero => "#0.0".to_string(),
        FpOpKind::IHex => format!("#0x{:x}", fp_field(word, op.hi, op.lo)),
    }
}

/// The element index of an indexed operand.  The index and the register
/// share their bits, so the index takes the bits the register leaves over,
/// from the lowest upwards.
fn fp_index(word: u32, op: &FpOp) -> u32 {
    let mut value = 0;
    let mut pos = 0;
    let mut bits = op.ibits;
    while bits != 0 {
        let b = bits.trailing_zeros();
        if word >> b & 1 != 0 {
            value |= 1 << pos;
        }
        bits &= bits - 1;
        pos += 1;
    }
    value
}

// --- Shared helpers ----------------------------------------------------------

/// A signed immediate read as a value, so that a caller can leave a zero
/// offset out of the printed form.
fn simm_value(field: u32, width: u32) -> i64 {
    let mask = (1u64 << width) - 1;
    let v = (field as u64) & mask;
    if v & (1 << (width - 1)) != 0 {
        v as i64 - (1i64 << width)
    } else {
        v as i64
    }
}

/// A signed immediate printed the way the assembler prints it.
fn imm_text(v: i64) -> String {
    if v < 0 {
        format!("#-0x{:x}", -v)
    } else {
        format!("#0x{:x}", v)
    }
}

/// The value a CPY immediate stands for: the 8-bit pattern is read as a
/// signed byte and then scaled by the shift.
fn cpy_imm_value(imm8: u8, sh: bool, width: u32) -> u64 {
    let signed = (imm8 as i8) as i64;
    let scaled = signed << if sh { 8 } else { 0 };
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    (scaled as u64) & mask
}

/// The number an 8-bit floating-point immediate stands for: a sign, a
/// power of two and four mantissa bits.
fn fp_imm_value(imm8: u8) -> f64 {
    let sign = if imm8 & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = imm8 >> 4 & 3;
    let bias = if imm8 & 0x40 != 0 { 4 } else { 0 };
    let scale = 2f64.powi(1 + exponent as i32 - bias);
    sign * scale * (1.0 + (imm8 & 0xf) as f64 / 16.0)
}

/// Attach the `// =value` comment the reference printer adds, aligned to
/// its own column.
///
/// The reference puts the comment a fixed distance past the start of the
/// mnemonic, but never less than one space after the instruction text.
fn with_comment(prefix: &str, comment: Option<&str>) -> String {
    let Some(comment) = comment else {
        return prefix.to_string();
    };
    let mn_len = prefix.find([' ', '\t']).unwrap_or(prefix.len());
    let col = (mn_len + 25).max(prefix.len() + 1);
    format!("{prefix}{}// ={comment}", " ".repeat(col - prefix.len()))
}

/// The name the reference printer gives to a `PTRUE` pattern.
fn pred_pattern(v: u32) -> String {
    match v {
        0 => "pow2".to_string(),
        1..=8 => format!("vl{v}"),
        9..=13 => format!("vl{}", 1u32 << (v - 5)),
        29 => "mul4".to_string(),
        30 => "mul3".to_string(),
        31 => String::new(),
        _ => format!("#0x{v:x}"),
    }
}

/// Decode a 13-bit logical immediate into the value it denotes, or `None`
/// when the encoding is one the architecture leaves undefined.
fn decode_logical_immediate(val: u32, regsize: u32) -> Option<u64> {
    let n = (val >> 12) & 1;
    let immr = (val >> 6) & 0x3f;
    let imms = val & 0x3f;
    let word = (n << 6) | (!imms & 0x3f);
    let len = 31i32 - word.leading_zeros() as i32;
    if len < 0 {
        return None;
    }
    let mut size = 1u64 << len;
    let r = immr as u64 & (size - 1);
    let s = imms as u64 & (size - 1);
    if s == size - 1 {
        return None;
    }
    let mut pattern = (1u64 << (s + 1)) - 1;
    for _ in 0..r {
        pattern = (pattern >> 1) | (pattern << (size - 1));
        pattern &= u64::MAX >> (64 - size);
    }
    // Replicate the pattern across the register.
    while size != regsize as u64 {
        if size >= regsize as u64 {
            // The reference decoder would read past the element here; no
            // alias ever selects such a width, so refuse the encoding.
            return None;
        }
        pattern |= pattern << size;
        size *= 2;
    }
    Some(pattern)
}

/// Whether every lane of `width` bits in `val` carries the same value.
fn identical_elements(val: u64, width: u32) -> bool {
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let first = val & mask;
    let mut at = width;
    while at < 64 {
        if (val >> at) & mask != first {
            return false;
        }
        at += width;
    }
    true
}

/// Whether `val` is a constant a CPY or DUP can hold in `width`-bit lanes.
fn is_sve_cpy_imm(val: i64, width: u32) -> bool {
    let mask = if width >= 64 {
        0
    } else {
        !((1i64 << width) - 1)
    };
    let high = val & mask;
    if high != 0 && high != mask {
        return false;
    }
    if val & 0xff != 0 {
        return val as i8 as i64 == truncate_signed(val, width);
    }
    if val & 0xff00 != 0 {
        return val as i16 as i64 == truncate_signed(val, width);
    }
    val == 0
}

/// Whether `val` has no single CPY/DUP equivalent and should therefore
/// print as `MOV` rather than `DUP`.
fn is_sve_move_mask_preferred(val: i64) -> bool {
    if is_sve_cpy_imm(val, 64) {
        return false;
    }
    let raw = val as u64;
    if identical_elements(raw, 32) && is_sve_cpy_imm(truncate_signed(val, 32), 32) {
        return false;
    }
    if identical_elements(raw, 16) && is_sve_cpy_imm(truncate_signed(val, 16), 16) {
        return false;
    }
    if identical_elements(raw, 8) && is_sve_cpy_imm(truncate_signed(val, 8), 8) {
        return false;
    }
    is_logical_immediate(raw, 64)
}

/// Truncate `val` to `width` bits, keeping the sign.
fn truncate_signed(val: i64, width: u32) -> i64 {
    if width >= 64 {
        val
    } else {
        (val << (64 - width)) >> (64 - width)
    }
}

/// Whether `val` can be written as a logical immediate of `regsize` bits.
fn is_logical_immediate(val: u64, regsize: u32) -> bool {
    if val == 0 || val == u64::MAX {
        return false;
    }
    if regsize != 64 && (val >> regsize != 0 || val == (u64::MAX >> (64 - regsize))) {
        return false;
    }
    // Narrow to the smallest element whose halves agree.
    let mut size = regsize;
    loop {
        size /= 2;
        let mask = (1u64 << size) - 1;
        if (val & mask) != ((val >> size) & mask) {
            size *= 2;
            break;
        }
        if size <= 2 {
            break;
        }
    }
    let mask = u64::MAX >> (64 - size);
    let element = val & mask;
    if is_shifted_mask(element) {
        return true;
    }
    let inverted = element | !mask;
    if !is_shifted_mask(!inverted) {
        return false;
    }
    let leading = (!inverted).leading_zeros();
    leading + (!inverted).trailing_zeros() >= 64 - size
}

/// Whether `val` is a run of one bits.
fn is_shifted_mask(val: u64) -> bool {
    if val == 0 {
        return false;
    }
    let mut spread = val;
    spread |= spread >> 1;
    spread |= spread >> 2;
    spread |= spread >> 4;
    spread |= spread >> 8;
    spread |= spread >> 16;
    spread |= spread >> 32;
    spread != 0
}

/// Print a logical immediate the way the reference printer does: a
/// hexadecimal operand, and a decimal comment when the value fits the
/// signed range the operand width describes.
fn print_sve_logical_imm(val: u64, width: u32) -> (String, Option<String>) {
    let masked = match width {
        16 => val & 0xffff,
        32 => val & 0xffff_ffff,
        _ => val,
    };
    let signed_ok = match width {
        16 => true,
        32 => (masked as i32) == (masked as i16 as i32),
        _ => (masked as i64) == (masked as i16 as i64),
    };
    let fits_16 = masked <= 0xffff;
    if signed_ok || fits_16 {
        (format!("#0x{masked:x}"), Some(format!("{}", masked as i64)))
    } else {
        (format!("#0x{masked:x}"), None)
    }
}
