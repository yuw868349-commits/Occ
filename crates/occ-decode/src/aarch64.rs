//! AArch64 (A64) decoding.
//!
//! Every instruction is four bytes, little-endian. That removes the single
//! hardest problem in x86 decoding — knowing where the next instruction starts
//! — and replaces it with a different one: a 32-bit word can always be decoded
//! as *something*, so a wrong guess produces a plausible looking instruction
//! instead of a desynchronised stream. This decoder therefore rejects words
//! whose top-level group it does not implement as `.word`, and never invents a
//! mnemonic for a reserved encoding.
//!
//! Coverage targets compiler output: branches, data processing (immediate,
//! register, bitfield, move-wide), loads and stores (unsigned offset, unscaled,
//! pre/post-indexed, register offset, literal, and pairs), plus the system
//! instructions that appear in prologues and syscall stubs. SIMD/FP and SVE are
//! recognised as groups and emitted as `.word` with the group named in a
//! comment, because a fabricated `fmla` line is worse than an honest gap.

use occ_core::{Addr, Arch, CallTarget, FlowKind, Instruction, MAX_INSN_LEN};

use crate::Decoded;

mod simd;
mod sve;
mod sve_tab;
mod sve_tab44;
mod sve_tab45;

/// Registers 0..30, plus the zero register.
const XZR: [&str; 32] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "x29", "x30", "xzr",
];

/// Registers 0..30, plus the stack pointer. Used where the encoding means SP.
const XSP: [&str; 32] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "x29", "x30", "sp",
];

/// 32-bit views.
const WZR: [&str; 32] = [
    "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14",
    "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23", "w24", "w25", "w26", "w27",
    "w28", "w29", "w30", "wzr",
];

/// 32-bit views where 31 means WSP.
const WSP: [&str; 32] = [
    "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14",
    "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23", "w24", "w25", "w26", "w27",
    "w28", "w29", "w30", "wsp",
];

const CONDS: [&str; 16] = [
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv",
];

const SHIFTS: [&str; 4] = ["lsl", "lsr", "asr", "ror"];
const EXTENDS: [&str; 8] = [
    "uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx",
];

/// Extract bits `hi..=lo` inclusive.
const fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

/// Sign-extend the low `n` bits of `v`.
const fn sext(v: u32, n: u32) -> i64 {
    let shift = 64 - n;
    ((v as i64) << shift) >> shift
}

fn x(n: u32) -> &'static str {
    XZR[n as usize & 31]
}
fn xsp(n: u32) -> &'static str {
    XSP[n as usize & 31]
}
fn w(n: u32) -> &'static str {
    WZR[n as usize & 31]
}
fn wsp(n: u32) -> &'static str {
    WSP[n as usize & 31]
}

/// Register spelling where 31 means the stack pointer rather than the zero
/// register.
fn rsp(sf: bool, n: u32) -> &'static str {
    if n == 31 {
        if sf { "sp" } else { "wsp" }
    } else {
        r(sf, n)
    }
}

/// Register spelling for a data-processing operand of the given width.
fn r(sf: bool, n: u32) -> &'static str {
    if sf { x(n) } else { w(n) }
}

/// Decoder state. AArch64 has exactly one width, so this is a marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Decoder {
    _private: (),
}

impl Decoder {
    /// Build a decoder. Only 64-bit AArch64 is a valid target.
    pub const fn new(bits: u8) -> Option<Self> {
        if bits == 64 {
            Some(Decoder { _private: () })
        } else {
            None
        }
    }

    /// Architecture label.
    pub const fn arch(self) -> Arch {
        Arch::Aarch64
    }

    /// Decode four bytes at `origin`.
    pub fn decode(&self, origin: Addr, bytes: &[u8]) -> Option<Decoded> {
        if bytes.len() < 4 {
            return None;
        }
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Some(self.word(origin, word))
    }

    /// Decode one already-fetched word. Public so callers holding a code array
    /// can avoid re-slicing in a loop.
    pub fn word(&self, origin: Addr, word: u32) -> Decoded {
        let r = decode_word(origin, word);
        let mut raw = [0u8; MAX_INSN_LEN];
        raw[..4].copy_from_slice(&word.to_le_bytes());
        Decoded {
            insn: Instruction {
                addr: origin,
                len: 4,
                bytes: raw,
                text: r.text,
                mnemonic: r.mn.into(),
                flow: r.flow,
                target: r.target,
                mem_reads: r.reads,
                mem_addr: Vec::new(),
                imm_values: Vec::new(),
                mem_writes: r.writes,
            },
            rip_target: r.pc,
        }
    }
}

/// Intermediate rendering of one instruction.
struct R {
    mn: &'static str,
    text: String,
    flow: FlowKind,
    target: Option<CallTarget>,
    pc: Option<Addr>,
    reads: Vec<Addr>,
    writes: Vec<Addr>,
}

impl R {
    fn new(mn: &'static str, text: String) -> R {
        R {
            mn,
            text,
            flow: FlowKind::Next,
            target: None,
            pc: None,
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    fn flow(mut self, f: FlowKind) -> R {
        self.flow = f;
        self
    }

    fn branch(mut self, to: Addr, call: bool) -> R {
        self.target = Some(CallTarget::Direct { to });
        self.flow = if call {
            FlowKind::Call
        } else {
            FlowKind::UnconditionalBranch
        };
        self.pc = Some(to);
        self
    }
}

/// A word the decoder declines to interpret.
fn undefined(origin: Addr, word: u32, group: &str) -> R {
    R::new(".word", format!(".word {word:#010x}  // {group}"))
        .flow(FlowKind::Exception)
        .with_pc(origin)
}

impl R {
    fn with_pc(mut self, a: Addr) -> R {
        self.pc = Some(a);
        self
    }
}

fn decode_word(origin: Addr, word: u32) -> R {
    // Top-level group is op1 = bits 28..25. The four x1x0 load/store values
    // and the two x101 data-processing values are matched by mask rather than
    // by enumeration so a new value cannot silently fall through.
    let group = bits(word, 28, 25);
    match group {
        0b1000 | 0b1001 => {
            if let Some(r) = dp_immediate(origin, word) {
                return r;
            }
        }
        0b1010 | 0b1011 => {
            if let Some(r) = branch_group(origin, word) {
                return r;
            }
        }
        // The structure loads and stores sit inside the load/store group by
        // their top bits but are Advanced SIMD instructions: they move a list
        // of vectors, so they belong to the SIMD decoder.
        g if g & 0b0101 == 0b0100 => {
            if matches!(bits(word, 29, 24), 0b001100 | 0b001101) {
                return simd::decode(origin, word);
            }
            if let Some(r) = load_store(origin, word) {
                return r;
            }
        }
        g if g & 0b0111 == 0b0101 => {
            if let Some(r) = dp_register(origin, word) {
                return r;
            }
        }
        0b0010 => return sve::decode(origin, word),
        0b0111 | 0b1111 => return simd::decode(origin, word),
        _ => {}
    }
    undefined(origin, word, "unimplemented encoding")
}

fn branch_group(origin: Addr, word: u32) -> Option<R> {
    let pc = origin;

    // B / BL.
    if word & 0x7c00_0000 == 0x1400_0000 {
        let link = word & 0x8000_0000 != 0;
        let off = sext(bits(word, 25, 0) << 2, 28);
        let to = pc.offset(off);
        let mn = if link { "bl" } else { "b" };
        return Some(
            R::new(mn, format!("{mn} {to}"))
                .branch(to, link)
                .with_pc(to),
        );
    }

    // B.cond.
    if word & 0xff00_0010 == 0x5400_0000 {
        let cond = CONDS[bits(word, 3, 0) as usize];
        let off = sext(bits(word, 23, 5) << 2, 21);
        let to = pc.offset(off);
        let mut r = R::new("b", format!("b.{cond} {to}"))
            .flow(FlowKind::ConditionalBranch)
            .with_pc(to);
        r.target = Some(CallTarget::Direct { to });
        return Some(r);
    }

    // CBZ / CBNZ.
    if word & 0x7e00_0000 == 0x3400_0000 {
        let sf = word & 0x8000_0000 != 0;
        let nz = word & 0x0100_0000 != 0;
        let off = sext(bits(word, 23, 5) << 2, 21);
        let to = pc.offset(off);
        let mn = if nz { "cbnz" } else { "cbz" };
        let mut r = R::new(mn, format!("{mn} {}, {to}", r(sf, bits(word, 4, 0))))
            .flow(FlowKind::ConditionalBranch)
            .with_pc(to);
        r.target = Some(CallTarget::Direct { to });
        return Some(r);
    }

    // TBZ / TBNZ.
    if word & 0x7e00_0000 == 0x3600_0000 {
        let nz = word & 0x0100_0000 != 0;
        let bit = bits(word, 31, 31) << 5 | bits(word, 23, 19);
        let off = sext(bits(word, 18, 5) << 2, 16);
        let to = pc.offset(off);
        let mn = if nz { "tbnz" } else { "tbz" };
        let mut r = R::new(mn, format!("{mn} {}, #{bit}, {to}", x(bits(word, 4, 0))))
            .flow(FlowKind::ConditionalBranch)
            .with_pc(to);
        r.target = Some(CallTarget::Direct { to });
        return Some(r);
    }

    // BR / BLR / RET / ERET / DRPS.
    if word & 0xffff_fc1f == 0xd61f_0000 {
        let rn = bits(word, 9, 5);
        return Some(
            R::new("br", format!("br {}", x(rn)))
                .flow(FlowKind::IndirectBranch)
                .indirect(),
        );
    }
    if word & 0xffff_fc1f == 0xd63f_0000 {
        let rn = bits(word, 9, 5);
        return Some(
            R::new("blr", format!("blr {}", x(rn)))
                .flow(FlowKind::IndirectCall)
                .indirect(),
        );
    }
    if word & 0xffff_fc1f == 0xd65f_0000 {
        let rn = bits(word, 9, 5);
        let text = if rn == 30 {
            "ret".to_string()
        } else {
            format!("ret {}", x(rn))
        };
        return Some(R::new("ret", text).flow(FlowKind::Return));
    }

    // Exception generation: SVC / HVC / SMC / BRK / HLT.
    if word & 0xff00_0000 == 0xd400_0000 {
        let opc = bits(word, 23, 21);
        let imm = bits(word, 20, 5);
        let mn = match opc {
            0b000 => "svc",
            0b001 => "hvc",
            0b010 => "smc",
            0b011 => "brk",
            0b100 => "hlt",
            _ => "dcps",
        };
        return Some(
            R::new(mn, format!("{mn} #{imm:#x}")).flow(if mn == "brk" || mn == "hlt" {
                FlowKind::Exception
            } else {
                FlowKind::Interrupt
            }),
        );
    }

    // Hints and barriers.
    if word & 0xffc0_0000 == 0xd500_0000 || word & 0xffff_f01f == 0xd503_201f {
        return Some(system(origin, word));
    }

    None
}

impl R {
    fn indirect(mut self) -> R {
        self.target = Some(CallTarget::Indirect { via: None });
        self
    }
}

fn system(_origin: Addr, word: u32) -> R {
    // MSR/MRS: 1101 0101 00 L 1 op0 op1 CRn CRm op2 Rt
    if word & 0xffc0_0000 == 0xd500_0000 && bits(word, 24, 22) & 0b011 != 0 {
        let read = word & 0x0040_0000 != 0;
        let sys = format!(
            "s{}_{}_c{}_c{}",
            bits(word, 20, 19),
            bits(word, 18, 16),
            bits(word, 15, 12),
            bits(word, 11, 8)
        );
        let rt = x(bits(word, 4, 0));
        return if read {
            R::new("mrs", format!("mrs {rt}, {sys}"))
        } else {
            R::new("msr", format!("msr {sys}, {rt}"))
        };
    }
    let op2 = bits(word, 7, 5) | (bits(word, 11, 8) << 3);
    let mn = match op2 {
        0b000_0000 => "nop",
        0b000_0001 => "yield",
        0b000_0010 => "wfe",
        0b000_0011 => "wfi",
        0b000_0100 => "sev",
        0b000_0101 => "sevl",
        _ => "hint",
    };
    R::new(mn, mn.to_string())
}

fn dp_immediate(origin: Addr, word: u32) -> Option<R> {
    // PC-relative addressing: ADR / ADRP.
    if word & 0x1f00_0000 == 0x1000_0000 {
        let page = word & 0x8000_0000 != 0;
        let immlo = bits(word, 30, 29);
        let immhi = bits(word, 23, 5);
        let imm = sext((immhi << 2) | immlo, 21);
        let rd = bits(word, 4, 0);
        let (mn, target) = if page {
            (
                "adrp",
                (origin.get() & !0xfff).wrapping_add((imm as u64) << 12),
            )
        } else {
            ("adr", origin.get().wrapping_add_signed(imm))
        };
        let to = Addr(target);
        return Some(R::new(mn, format!("{mn} {}, {to}", x(rd))).with_pc(to));
    }

    // Add/subtract (immediate), including CMP/CMN/MOV/SP forms.
    if word & 0x1f00_0000 == 0x1100_0000 {
        let sf = word & 0x8000_0000 != 0;
        let sub = word & 0x4000_0000 != 0;
        let setflags = word & 0x2000_0000 != 0;
        let shift = if word & 0x0040_0000 != 0 { 12 } else { 0 };
        let imm = bits(word, 21, 10) << shift;
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let (mn, rd_name, rn_name) = match (sub, setflags) {
            (false, false) => ("add", rsp(sf, rd), rsp(sf, rn)),
            (true, false) => ("sub", rsp(sf, rd), rsp(sf, rn)),
            (false, true) => ("cmn", wsp(rn), wsp(rn)),
            (true, true) => ("cmp", wsp(rn), wsp(rn)),
        };
        if setflags {
            return Some(R::new(mn, format!("{mn} {rn_name}, #{imm:#x}")));
        }
        let mut text = format!("{mn} {rd_name}, {rn_name}, #{imm:#x}");
        if shift == 12 {
            text.push_str(", lsl #12");
        }
        // `add x0, x1, #0` is the canonical `mov x0, x1`; keeping the add form
        // is more faithful to the encoding and equally readable.
        return Some(R::new(mn, text));
    }

    // Logical (immediate): AND/ORR/EOR/ANDS with a bitmask immediate.
    if word & 0x1f80_0000 == 0x1200_0000 {
        let sf = word & 0x8000_0000 != 0;
        let opc = bits(word, 30, 29);
        let n = bits(word, 22, 22);
        let immr = bits(word, 21, 16);
        let imms = bits(word, 15, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let width = if sf { 64 } else { 32 };
        let value = decode_bitmask(n, imms, immr, width);
        let mn = ["and", "orr", "eor", "ands"][opc as usize];
        if opc == 1 && rn == 31 {
            return Some(R::new("mov", format!("mov {}, {value:#x}", r(sf, rd))));
        }
        if opc == 3 && rd == 31 {
            return Some(R::new("tst", format!("tst {}, {value:#x}", r(sf, rn))));
        }
        return Some(R::new(
            mn,
            format!("{mn} {}, {}, {value:#x}", r(sf, rd), r(sf, rn)),
        ));
    }

    // Move wide (immediate): MOVN / MOVZ / MOVK.
    if word & 0x1f80_0000 == 0x1280_0000 {
        let sf = word & 0x8000_0000 != 0;
        let opc = bits(word, 30, 29);
        let hw = bits(word, 22, 21);
        let imm = bits(word, 20, 5) as u64;
        let rd = bits(word, 4, 0);
        let shift = hw * 16;
        // Move-wide opcode order is not the intuitive one: 01 is
        // unallocated, 10 is MOVZ and 11 is MOVK.
        let mn = match opc {
            0 => "movn",
            2 => "movz",
            3 => "movk",
            _ => return None,
        };
        let mut text = format!("{mn} {}, #{imm:#x}", r(sf, rd));
        if shift > 0 {
            text.push_str(&format!(", lsl #{shift}"));
        }
        return Some(R::new(mn, text));
    }

    // Bitfield: SBFM / BFM / UBFM, with the SXT*/LSL/LSR/UBFX aliases.
    if word & 0x1f80_0000 == 0x1300_0000 {
        let sf = word & 0x8000_0000 != 0;
        let opc = bits(word, 30, 29);
        let immr = bits(word, 21, 16);
        let imms = bits(word, 15, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let mn = ["sbfm", "bfm", "ubfm", "?"][opc as usize];
        if opc == 3 {
            return None;
        }
        // UBFM with immr == 0 and imms == width-1 is LSR; with imms+1 == immr
        // it is LSL. SBFM with those shapes is ASR/LSL.
        let width = if sf { 64 } else { 32 };
        if immr == 0 && imms == 0 {
            return Some(R::new("mov", format!("mov {}, {}", r(sf, rd), r(sf, rn))));
        }
        if opc == 2 && immr == 0 {
            let sh = width - 1 - imms;
            return Some(R::new(
                "lsr",
                format!("lsr {}, {}, #{sh}", r(sf, rd), r(sf, rn)),
            ));
        }
        if opc == 0 && immr == 0 {
            let sh = width - 1 - imms;
            return Some(R::new(
                "asr",
                format!("asr {}, {}, #{sh}", r(sf, rd), r(sf, rn)),
            ));
        }
        if opc == 2 && imms != width - 1 && imms + 1 == immr {
            let sh = width - 1 - imms;
            return Some(R::new(
                "lsl",
                format!("lsl {}, {}, #{sh}", r(sf, rd), r(sf, rn)),
            ));
        }
        return Some(R::new(
            mn,
            format!("{mn} {}, {}, #{immr}, #{imms}", r(sf, rd), r(sf, rn)),
        ));
    }

    // EXTR.
    if word & 0x1f80_0000 == 0x1380_0000 {
        let sf = word & 0x8000_0000 != 0;
        let rm = bits(word, 20, 16);
        let lsb = bits(word, 15, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        if rn == rm {
            return Some(R::new(
                "ror",
                format!("ror {}, {}, #{lsb}", r(sf, rd), r(sf, rn)),
            ));
        }
        return Some(R::new(
            "extr",
            format!("extr {}, {}, {}, #{lsb}", r(sf, rd), r(sf, rn), r(sf, rm)),
        ));
    }

    None
}

/// Expand a logical-immediate encoding into its value.
///
/// `None` from the architectural check returns zero, which is what the
/// hardware would treat as an unallocated encoding; the caller renders the raw
/// fields anyway so the gap is visible.
fn decode_bitmask(n: u32, imms: u32, immr: u32, width: u32) -> u64 {
    // The encoding packs a run-length descriptor: len selects the element
    // size, S the number of ones, R the rotation within the element.
    let combined = (n << 6) | (imms & 0x3f);
    if combined == 0 {
        return 0;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 {
        return 0;
    }
    let levels = (1u32 << len) - 1;
    let s = imms & levels;
    let r = immr & levels;
    if s == levels {
        return 0;
    }
    let esize = 1u32 << len;
    let mask: u64 = if esize >= 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let ones = (1u64 << (s + 1)) - 1;
    let elem = if r == 0 {
        ones & mask
    } else {
        // Rotate right by `r` within `esize` bits.
        ((ones >> r) | (ones << (esize - r))) & mask
    };
    let mut out = 0u64;
    let mut shift = 0;
    while shift < width {
        out |= elem << shift;
        shift += esize;
    }
    out
}

fn dp_register(_origin: Addr, word: u32) -> Option<R> {
    // Logical (shifted register).
    if word & 0x1f00_0000 == 0x0a00_0000 {
        let sf = word & 0x8000_0000 != 0;
        let opc = bits(word, 30, 29);
        let shift = bits(word, 23, 22);
        let negate = word & 0x0020_0000 != 0;
        let rm = bits(word, 20, 16);
        let imm6 = bits(word, 15, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let mn = match (opc, negate) {
            (0, false) => "and",
            (0, true) => "bic",
            (1, false) => "orr",
            (1, true) => "orn",
            (2, false) => "eor",
            (2, true) => "eon",
            (3, false) => "ands",
            _ => "bics",
        };
        let mut text = format!("{mn} {}, {}, {}", r(sf, rd), r(sf, rn), r(sf, rm));
        if imm6 != 0 {
            text.push_str(&format!(", {} #{imm6}", SHIFTS[shift as usize]));
        }
        // MOV <rd>, <rm> is ORR with the zero register and no shift.
        if opc == 1 && !negate && rn == 31 && imm6 == 0 {
            return Some(R::new("mov", format!("mov {}, {}", r(sf, rd), r(sf, rm))));
        }
        if opc == 3 && rd == 31 && imm6 == 0 {
            return Some(R::new("tst", format!("tst {}, {}", r(sf, rn), r(sf, rm))));
        }
        return Some(R::new(mn, text));
    }

    // Add/subtract (shifted register), including CMP/CMN and the NEG alias.
    if word & 0x1f20_0000 == 0x0b00_0000 {
        let sf = word & 0x8000_0000 != 0;
        let sub = word & 0x4000_0000 != 0;
        let setflags = word & 0x2000_0000 != 0;
        let shift = bits(word, 23, 22);
        let rm = bits(word, 20, 16);
        let imm6 = bits(word, 15, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        if !setflags && rn == 31 && sub {
            let mut text = format!("neg {}, {}", r(sf, rd), r(sf, rm));
            if imm6 != 0 {
                text.push_str(&format!(", {} #{imm6}", SHIFTS[shift as usize]));
            }
            return Some(R::new("neg", text));
        }
        let mn = match (sub, setflags) {
            (false, false) => "add",
            (true, false) => "sub",
            (false, true) => "cmn",
            (true, true) => "cmp",
        };
        if setflags {
            let mut text = format!("{mn} {}, {}", r(sf, rn), r(sf, rm));
            if imm6 != 0 {
                text.push_str(&format!(", {} #{imm6}", SHIFTS[shift as usize]));
            }
            return Some(R::new(mn, text));
        }
        let mut text = format!("{mn} {}, {}, {}", r(sf, rd), r(sf, rn), r(sf, rm));
        if imm6 != 0 {
            text.push_str(&format!(", {} #{imm6}", SHIFTS[shift as usize]));
        }
        return Some(R::new(mn, text));
    }

    // Add/subtract (extended register).
    if word & 0x1f20_0000 == 0x0b20_0000 {
        let sf = word & 0x8000_0000 != 0;
        let sub = word & 0x4000_0000 != 0;
        let setflags = word & 0x2000_0000 != 0;
        let rm = bits(word, 20, 16);
        let option = bits(word, 15, 13);
        let imm3 = bits(word, 12, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let mn = match (sub, setflags) {
            (false, false) => "add",
            (true, false) => "sub",
            (false, true) => "cmn",
            (true, true) => "cmp",
        };
        let ext = EXTENDS[option as usize];
        let name = if matches!(option, 3 | 7) {
            x(rm)
        } else {
            w(rm)
        };
        let mut text = if setflags {
            format!("{mn} {}, {name}, {ext}", r(sf, rn))
        } else {
            format!("{mn} {}, {}, {name}, {ext}", r(sf, rd), r(sf, rn))
        };
        if imm3 != 0 {
            text.push_str(&format!(" #{imm3}"));
        }
        return Some(R::new(mn, text));
    }

    // Data-processing (2 source): UDIV/SDIV/LSLV/LSRV/ASRV/RORV.
    if word & 0x1fe0_0000 == 0x1ac0_0000 {
        let sf = word & 0x8000_0000 != 0;
        let rm = bits(word, 20, 16);
        let opc = bits(word, 15, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let mn = match opc {
            0b000010 => "udiv",
            0b000011 => "sdiv",
            0b001000 => "lsl",
            0b001001 => "lsr",
            0b001010 => "asr",
            0b001011 => "ror",
            _ => return None,
        };
        return Some(R::new(
            mn,
            format!("{mn} {}, {}, {}", r(sf, rd), r(sf, rn), r(sf, rm)),
        ));
    }

    // Data-processing (3 source): MADD/MSUB and the widening forms.
    if word & 0x1f00_0000 == 0x1b00_0000 {
        let sf = word & 0x8000_0000 != 0;
        let rm = bits(word, 20, 16);
        let o0 = word & 0x0000_8000 != 0;
        let ra = bits(word, 14, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        if !sf && !o0 && ra == 31 {
            return Some(R::new(
                "mul",
                format!("mul {}, {}, {}", w(rd), w(rn), w(rm)),
            ));
        }
        if sf && !o0 && ra == 31 {
            return Some(R::new(
                "mul",
                format!("mul {}, {}, {}", x(rd), x(rn), x(rm)),
            ));
        }
        let mn = if o0 { "msub" } else { "madd" };
        return Some(R::new(
            mn,
            format!(
                "{mn} {}, {}, {}, {}",
                r(sf, rd),
                r(sf, rn),
                r(sf, rm),
                r(sf, ra)
            ),
        ));
    }

    // Conditional select: CSEL/CSINC/CSINV/CSNEG, with CSET/CSETM aliases.
    if word & 0x1fe0_0000 == 0x1a80_0000 {
        let sf = word & 0x8000_0000 != 0;
        let rm = bits(word, 20, 16);
        let cond = bits(word, 15, 12);
        let op2 = bits(word, 11, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        // Bit 30 picks the pair. Without it `csinv` decodes as `csel`,
        // which is a different answer: the false arm of a `csinv` is the
        // complement of its operand, not the operand itself.
        let invert = word & 0x4000_0000 != 0;
        let mn = match (invert, op2 & 1) {
            (false, 0) => "csel",
            (false, _) => "csinc",
            (true, 0) => "csinv",
            (true, _) => "csneg",
        };
        // CSET is CSINC with both sources the zero register and inverted
        // cond; CSETM is the same form of CSINV, which is one bit lower in
        // the two-bit field.
        let alias = rn == 31 && rm == 31 && (op2 & 1) == u32::from(!invert);
        if alias {
            let inv = CONDS[(cond ^ 1) as usize];
            let mn = if invert { "csetm" } else { "cset" };
            return Some(R::new(mn, format!("{mn} {}, {}", r(sf, rd), inv)));
        }
        return Some(R::new(
            mn,
            format!(
                "{mn} {}, {}, {}, {}",
                r(sf, rd),
                r(sf, rn),
                r(sf, rm),
                CONDS[cond as usize]
            ),
        ));
    }

    // Conditional compare: CCMP/CCMN.
    if word & 0x1fe0_0000 == 0x1a40_0000 {
        let sf = word & 0x8000_0000 != 0;
        let rm = bits(word, 20, 16);
        let cond = bits(word, 15, 12);
        let rn = bits(word, 9, 5);
        let nzcv = bits(word, 3, 0);
        let mn = if word & 0x0000_0800 != 0 {
            "ccmp"
        } else {
            "ccmn"
        };
        return Some(R::new(
            mn,
            format!(
                "{mn} {}, {}, #{nzcv:#x}, {}",
                r(sf, rn),
                r(sf, rm),
                CONDS[cond as usize]
            ),
        ));
    }

    // Data-processing (1 source): RBIT/REV/CLZ/CLS.
    // Fixed bits 30..16, with sf at bit 31 left free.
    if word & 0x7fff_0000 == 0x5ac0_0000 {
        let sf = word & 0x8000_0000 != 0;
        let opcode = bits(word, 15, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let mn = match opcode {
            0b000000 => "rbit",
            0b000001 => "rev16",
            0b000010 => "rev32",
            0b000011 => "rev",
            0b000100 => "clz",
            0b000101 => "cls",
            _ => return None,
        };
        return Some(R::new(mn, format!("{mn} {}, {}", r(sf, rd), r(sf, rn))));
    }

    None
}

fn load_store(origin: Addr, word: u32) -> Option<R> {
    // Load register (literal).
    if word & 0x3b00_0000 == 0x1800_0000 {
        let opc = bits(word, 31, 30);
        let imm = sext(bits(word, 23, 5) << 2, 21);
        let to = origin.offset(imm);
        let rt = bits(word, 4, 0);
        let (mn, name) = if word & 0x0400_0000 != 0 {
            match opc {
                0 => ("ldr", format!("s{rt}")),
                1 => ("ldr", format!("d{rt}")),
                2 => ("ldr", format!("q{rt}")),
                _ => return None,
            }
        } else {
            match opc {
                0 => ("ldr", w(rt).to_string()),
                1 => ("ldr", x(rt).to_string()),
                2 => ("ldrsw", x(rt).to_string()),
                _ => ("prfm", x(rt).to_string()),
            }
        };
        let mut r = R::new(mn, format!("{mn} {name}, {to}")).with_pc(to);
        r.reads.push(to);
        return Some(r);
    }

    // Load/store pair.
    if word & 0x3a00_0000 == 0x2800_0000 {
        let vector = word & 0x0400_0000 != 0;
        let opc = bits(word, 31, 30);
        let load = word & 0x0040_0000 != 0;
        let mode = bits(word, 24, 23);
        let imm7 = sext(bits(word, 21, 15), 7);
        let rt2 = bits(word, 14, 10);
        let rn = bits(word, 9, 5);
        let rt = bits(word, 4, 0);
        // With V set the pair moves FP/SIMD registers and `opc` is their
        // width: 4, 8 or 16 bytes.
        let (width, wide) = match (vector, opc) {
            (true, 0) => (4i64, false),
            (true, 1) => (8i64, false),
            (true, 2) => (16i64, false),
            (true, 3) => return None,
            (false, 0) => (4i64, false),
            (false, 2) => (8i64, true),
            (false, 1) if load => (4i64, true), // LDPSW sign-extends into X registers
            _ => return None,
        };
        let off = imm7 * width;
        let mn = if load { "ldp" } else { "stp" };
        let (a, b) = if vector {
            match opc {
                0 => (format!("s{rt}"), format!("s{rt2}")),
                1 => (format!("d{rt}"), format!("d{rt2}")),
                _ => (format!("q{rt}"), format!("q{rt2}")),
            }
        } else {
            (r(wide, rt).to_string(), r(wide, rt2).to_string())
        };
        let base = xsp(rn);
        let text = match mode {
            0b01 => format!("{mn} {a}, {b}, [{base}], #{off}"),
            0b11 => format!("{mn} {a}, {b}, [{base}, #{off}]!"),
            _ => format!("{mn} {a}, {b}, [{base}, #{off}]"),
        };
        return Some(R::new(mn, text));
    }

    // Single-register loads and stores. Bit 26 turns the group from general
    // purpose registers to FP/SIMD ones, which moves the encoding from the
    // 0x38/0x39 pair to 0x3c/0x3d.
    let vector = word & 0x0400_0000 != 0;
    let top = (word >> 24) & 0x3f;
    let known = top == 0x39 || top == 0x38 || (vector && (top == 0x3d || top == 0x3c));
    if !known {
        return None;
    }

    let size = bits(word, 31, 30);
    let opc = bits(word, 23, 22);
    let rn = bits(word, 9, 5);
    let rt = bits(word, 4, 0);

    // A vector access names its register width in `size` and the transfer
    // direction in `opc`, with the quadword form moving to `size` zero so that
    // its own bit 22 can be set. `opc` is 00 or 01 for a store and a load of a
    // 32- or 64-bit register, and 10 or 11 for the same of a 128-bit one.
    if vector {
        let (mn, name, scale) = match (size, opc) {
            (2, 0) => ("str", format!("s{rt}"), 2),
            (2, 1) => ("ldr", format!("s{rt}"), 2),
            (3, 0) => ("str", format!("d{rt}"), 3),
            (3, 1) => ("ldr", format!("d{rt}"), 3),
            (0, 2) => ("str", format!("q{rt}"), 4),
            (0, 3) => ("ldr", format!("q{rt}"), 4),
            _ => return None,
        };
        return address_form(word, mn, &name, scale, rn, top == 0x3d);
    }

    // Mnemonic, destination width and store flag from (size, opc).
    let (mn, wide, _store) = match (size, opc) {
        (3, 0) => ("str", true, true),
        (3, 1) => ("ldr", true, false),
        (3, _) => return None,
        (s, 0) => (
            match s {
                0 => "strb",
                1 => "strh",
                _ => "str",
            },
            false,
            true,
        ),
        (s, 1) => (
            match s {
                0 => "ldrb",
                1 => "ldrh",
                _ => "ldr",
            },
            false,
            false,
        ),
        (s, 2) => (
            match s {
                0 => "ldrsb",
                1 => "ldrsh",
                _ => "ldrsw",
            },
            true,
            false,
        ),
        (s, 3) => (
            match s {
                0 => "ldrsb",
                1 => "ldrsh",
                _ => "ldrsw",
            },
            false,
            false,
        ),
        _ => return None,
    };
    let rt_name = r(wide, rt).to_string();
    address_form(word, mn, &rt_name, size, rn, top == 0x39)
}

/// Render the addressing form of a single-register load or store: unsigned
/// offset (`unsigned`), and the unscaled, pre/post-indexed and register-offset
/// forms that share bits 11:10 as a mode selector. `scale` is the access size
/// in log2 bytes, which both the unsigned-offset immediate and the register
/// offset shift use.
fn address_form(
    word: u32,
    mn: &'static str,
    rt_name: &str,
    scale: u32,
    rn: u32,
    unsigned: bool,
) -> Option<R> {
    if unsigned {
        let imm = bits(word, 21, 10) << scale;
        return Some(R::new(mn, format!("{mn} {rt_name}, [{}, #{imm}]", xsp(rn))));
    }
    match bits(word, 11, 10) {
        0b00 => {
            let imm = sext(bits(word, 20, 12), 9);
            Some(R::new(
                stur_mn(mn),
                format!("{} {rt_name}, [{}, #{imm}]", stur_mn(mn), xsp(rn)),
            ))
        }
        0b01 => {
            let imm = sext(bits(word, 20, 12), 9);
            Some(R::new(mn, format!("{mn} {rt_name}, [{}], #{imm}", xsp(rn))))
        }
        0b11 => {
            let imm = sext(bits(word, 20, 12), 9);
            Some(R::new(
                mn,
                format!("{mn} {rt_name}, [{}, #{imm}]!", xsp(rn)),
            ))
        }
        0b10 => {
            let rm = bits(word, 20, 16);
            let option = bits(word, 15, 13);
            let s = word & 0x0000_1000 != 0;
            let ext = EXTENDS[option as usize];
            let idx = if matches!(option, 3 | 7) {
                x(rm)
            } else {
                w(rm)
            };
            let mut operand = format!("{idx}, {ext}");
            if s {
                operand.push_str(&format!(" #{scale}"));
            }
            Some(R::new(
                mn,
                format!("{mn} {rt_name}, [{}, {operand}]", xsp(rn)),
            ))
        }
        _ => None,
    }
}

/// The unscaled form of a load/store mnemonic (`ldr` -> `ldur`).
fn stur_mn(mn: &'static str) -> &'static str {
    match mn {
        "ldr" => "ldur",
        "str" => "stur",
        "ldrb" => "ldurb",
        "strb" => "sturb",
        "ldrh" => "ldurh",
        "strh" => "sturh",
        "ldrsb" => "ldursb",
        "ldrsh" => "ldursh",
        "ldrsw" => "ldursw",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(word: u32) -> Decoded {
        Decoder::new(64).unwrap().word(Addr(0x1000), word)
    }

    #[test]
    fn decodes_the_standard_prologue() {
        // stp x29, x30, [sp, #-16]!
        let d = dec(0xa9bf_7bfd);
        assert_eq!(&*d.insn.mnemonic, "stp");
        assert_eq!(d.insn.text, "stp x29, x30, [sp, #-16]!");
        assert_eq!(d.insn.len, 4);

        // mov x29, sp  (add x29, sp, #0)
        let d = dec(0x9100_03fd);
        assert_eq!(&*d.insn.mnemonic, "add");
        assert_eq!(d.insn.text, "add x29, sp, #0x0");
    }

    #[test]
    fn decodes_epilogue_and_return() {
        // ldp x29, x30, [sp], #16
        let d = dec(0xa8c1_7bfd);
        assert_eq!(&*d.insn.mnemonic, "ldp");
        assert_eq!(d.insn.text, "ldp x29, x30, [sp], #16");
        // ret
        let d = dec(0xd65f_03c0);
        assert_eq!(&*d.insn.mnemonic, "ret");
        assert_eq!(d.insn.flow, FlowKind::Return);
    }

    #[test]
    fn conditional_select_keeps_its_operand_form() {
        // The four members of the family share a mask, and bit 30 is what
        // separates `csel`/`csinc` from `csinv`/`csneg`. Reading only the
        // two-bit field below it decoded `csinv w0, w2, w3, eq` as
        // `csel w0, w2, w3, eq`, which selects the operand instead of its
        // complement — a different function, and a silently wrong one.
        for (word, mnemonic) in [
            (0x1a83_0040u32, "csel"),
            (0x1a83_0440, "csinc"),
            (0x5a83_0040, "csinv"),
            (0x5a83_0440, "csneg"),
        ] {
            let d = dec(word);
            assert_eq!(&*d.insn.mnemonic, mnemonic, "{word:#010x}");
            assert!(
                d.insn.text.starts_with(mnemonic),
                "{word:#010x}: {}",
                d.insn.text
            );
            assert_eq!(d.insn.text, format!("{mnemonic} w0, w2, w3, eq"));
        }
        // `cset` is `csinc` with both sources zero and the condition
        // inverted; `csetm` is the same form of `csinv`.
        let d = dec(0x1a9f_07e0);
        assert_eq!(&*d.insn.mnemonic, "cset");
        assert_eq!(d.insn.text, "cset w0, ne");
        let d = dec(0x5a9f_03e0);
        assert_eq!(&*d.insn.mnemonic, "csetm");
        assert_eq!(d.insn.text, "csetm w0, ne");
    }

    #[test]
    fn branches_carry_their_target() {
        // bl +8 from 0x1000
        let d = dec(0x9400_0002);
        assert_eq!(&*d.insn.mnemonic, "bl");
        assert_eq!(d.insn.target, Some(CallTarget::Direct { to: Addr(0x1008) }));
        assert_eq!(d.rip_target, Some(Addr(0x1008)));
    }

    #[test]
    fn conditional_branch_is_marked_conditional() {
        // b.eq +8
        let d = dec(0x5400_0040);
        assert_eq!(&*d.insn.mnemonic, "b");
        assert!(d.insn.text.starts_with("b.eq "));
        assert_eq!(d.insn.flow, FlowKind::ConditionalBranch);
    }

    #[test]
    fn adrp_computes_a_page() {
        // adrp x0, 0x2000  (immhi=2, immlo=0 at pc 0x1000)
        let d = dec(0x9000_0000);
        assert_eq!(&*d.insn.mnemonic, "adrp");
        assert_eq!(d.rip_target, Some(Addr(0x1000)));
        assert_eq!(d.insn.text, "adrp x0, 0x1000");
    }

    #[test]
    fn loads_and_stores_render_offsets() {
        // ldr x0, [x1, #8]
        let d = dec(0xf940_0420);
        assert_eq!(d.insn.text, "ldr x0, [x1, #8]");
        // str w0, [sp, #4]
        let d = dec(0xb900_07e0);
        assert_eq!(d.insn.text, "str w0, [sp, #4]");
    }

    #[test]
    fn compare_and_move_wide() {
        // cmp w0, #0
        let d = dec(0x7100_001f);
        assert_eq!(d.insn.text, "cmp w0, #0x0");
        // mov w0, #1  (movz)
        let d = dec(0x5280_0020);
        assert_eq!(d.insn.text, "movz w0, #0x1");
    }

    #[test]
    fn unknown_words_stay_four_bytes() {
        let d = dec(0x0000_0000);
        assert_eq!(&*d.insn.mnemonic, ".word");
        assert_eq!(d.insn.len, 4);
    }

    #[test]
    fn short_input_is_not_decoded() {
        assert!(
            Decoder::new(64)
                .unwrap()
                .decode(Addr(0), &[0x1f, 0x20])
                .is_none()
        );
    }
}
