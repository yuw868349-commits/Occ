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

use super::{bits, undefined, R};

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

/// `index`'s general-register operands: 64-bit elements use X names, the
/// rest use W names.
fn gpr(n: u32, wide: bool) -> String {
    if wide {
        xreg(n)
    } else {
        wreg(n)
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
        _ => undefined(origin, word, "SVE (unimplemented family)"),
    }
}

/// The `bits(28:24) = 0b00101` group: immediates, predicate operations,
/// floating-point arithmetic.
fn group_b(origin: Addr, word: u32) -> R {
    match bits(word, 31, 24) {
        0x05 => undefined(origin, word, "SVE permute (unimplemented)"),
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
    let Some((_, mn)) = PRED_BINARY.iter().find(|(k, _)| *k == bits(word, 20, 16) as u8)
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
        format!("{mn} {}, {}/m, {}, {}", zt(zdn, t), p(bits(word, 12, 10)), zt(zdn, t), zt(zm, t)),
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
        0 => R::new(mn, format!("{mn} {}, {}, {}", scalar('d', rd), pg, zt(zn, t))),
        // Destination spelled with the element letter.
        1 => R::new(mn, format!("{mn} {}, {}, {}", scalar(t, rd), pg, zt(zn, t))),
        // Destination is the NEON view of a Q register, sized like the source.
        2 => R::new(mn, format!("{mn} v{}.{}, {}, {}", rd, VFORMS[sz], pg, zt(zn, t))),
        // movprfx, zeroing or merging.
        _ => {
            let sfx = if *kind == 3 { "z" } else { "m" };
            R::new(mn, format!("{mn} {}, {}/{}, {}", zt(rd, t), pg, sfx, zt(zn, t)))
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
            format!("{mn} {}, {}, {}, #0x{:x}", zt(zdn, t), pg, zt(zdn, t), shift),
        );
    }
    if let Some((_, mn)) = PRED_SHIFT_VEC.iter().find(|(k, _)| *k == opc) {
        let t = SIZES[sz];
        let pg = format!("{}/m", p(bits(word, 12, 10)));
        return R::new(mn, format!("{mn} {}, {}, {}, {}", zt(zdn, t), pg, zt(zdn, t), zt(zm, t)));
    }
    if let Some((_, mn)) = PRED_SHIFT_WIDE.iter().find(|(k, _)| *k == opc) {
        // Wide shifts take a 64-bit element vector as their shift amount and
        // are only defined for 8/16/32-bit elements.
        if sz > 2 {
            return undefined(origin, word, "SVE predicated shift (unallocated)");
        }
        let t = SIZES[sz];
        let pg = format!("{}/m", p(bits(word, 12, 10)));
        return R::new(mn, format!("{mn} {}, {}, {}, {}", zt(zdn, t), pg, zt(zdn, t), zt(zm, 'd')));
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
    let Some((_, mn)) = UNPRED_ARITH.iter().find(|(k, _)| *k == bits(word, 12, 10) as u8) else {
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
                format!("xar {}, {}, {}, #0x{:x}", zt(rd, t), zt(rd, t), zt(rn, t), shift),
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
            let rn_txt = if rm & 31 == 31 { "sp".to_string() } else { format!("x{}", rm) };
            if one_reg {
                R::new(mn, format!("{mn} {}, {}", scalar('x', rd), imm))
            } else {
                R::new(mn, format!("{mn} {}, {}, {}", scalar('x', rd), rn_txt, imm))
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
    let Some((_, mask, mn)) = UNPRED_MUL.iter().find(|(k, ..)| *k == bits(word, 12, 10) as u8)
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
        return R::new(mn, format!("{mn} {}, {}, {}", zt(rd, t), zt(rn, t), zt(bits(word, 20, 16), 'd')));
    }
    if let Some((_, mn, left)) = UNPRED_SHIFT_IMM.iter().find(|(k, ..)| *k == sel as u8) {
        let Some((t, shift)) = shift_imm(sz, bits(word, 20, 16), *left) else {
            return undefined(origin, word, "SVE unpredicated shift (unallocated)");
        };
        return R::new(mn, format!("{mn} {}, {}, #0x{:x}", zt(rd, t), zt(rn, t), shift));
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
            R::new("adr", format!("adr {}, [{}, {}{}]", zt(rd, t), zt(rn, t), zt(rm, t), ext))
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
        1..=13 => format!("vl{}", 1 << (pat - 1)),
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
        0b000 | 0b001 => format!("x{}", rd),
        // sel 4-6 with opc < 16 are destructive (print X and W names); with
        // opc >= 16 they update a single X register.
        0b100 | 0b110 if opc < 16 => format!("x{rd}, w{rd}"),
        0b101 | 0b111 if opc < 16 => format!("w{rd}"),
        _ => format!("x{rd}"),
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

// === top byte 0x05: permuting, immediates ==================================

fn permute_05(origin: Addr, word: u32) -> R {
    let _ = origin;
    let _ = word;
    undefined(origin, word, "SVE permute (unimplemented)")
}
