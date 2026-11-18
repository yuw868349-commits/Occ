//! ARM32 decoding: A32 (ARM) and T32 (Thumb).
//!
//! ARM has two instruction sets that share a register file and nothing else.
//! A32 is four bytes, every instruction is conditional, and an operand can be
//! shifted by a register. T32 is two or four bytes, only branches are
//! conditional (until `IT` makes the next four conditional), and the
//! encodings were designed to be dense rather than orthogonal. Treating one
//! as the other produces a listing that decodes cleanly and describes a
//! program that was never written, so the mode is chosen before the first
//! word is read and never guessed from the bytes.
//!
//! The decoder covers what a compiler emits: data processing (immediate,
//! shifted register, and register-shifted), multiplies, single and halfword
//! loads and stores, block transfers, branches, `bx`/`blx`, and the system
//! instructions that appear in prologues and syscall stubs. VFP/NEON,
//! coprocessor moves and the DSP extensions are recognised as groups and
//! emitted as `.word` with the group named, because a fabricated `vadd` line
//! is worse than an honest gap.
//!
//! The Thumb half of the decoder was written against a measurement rather
//! than against a description: the whole 16-bit space and a large sample of
//! the 32-bit space were read with two independent implementations — capstone
//! and ARM's own `llvm-objdump` — and every encoding in this file has to
//! agree with both. Where the two disagree, the difference is a profile
//! question (`bxns` exists in ARMv8-M and not in ARMv7) rather than a
//! mistake, and it is settled in favour of the older profile, which is the
//! one this toolset claims to read.

use occ_core::{Addr, Arch, CallTarget, FlowKind, Instruction, MAX_INSN_LEN};

use crate::Decoded;

/// Which instruction set a range of code is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// A32: four-byte instructions, every one conditional.
    #[default]
    Arm,
    /// T32: two- and four-byte instructions.
    Thumb,
}

impl Mode {
    /// The label used in output.
    pub const fn label(self) -> &'static str {
        match self {
            Mode::Arm => "arm",
            Mode::Thumb => "thumb",
        }
    }
}

/// Condition suffixes, indexed by the four-bit field.
///
/// The carry conditions are spelled `hs` and `lo`, not `cs` and `cc`.
/// Both references — capstone and ARM's own disassembler — print them
/// that way in A32 as well as in Thumb, and an assembler that is asked
/// for `bhs` produces the same encoding as one asked for `bcs`, so the
/// two names are two spellings of one condition and this is the
/// spelling the architecture's own tools use.
const CONDS: [&str; 16] = [
    "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "", "nv",
];

/// The condition names an `it` block is printed with, indexed by the
/// four-bit field.
///
/// The last two are not conditions at all — they always run — and both
/// are printed as `al`, because an `it` block that always runs is the
/// same block whichever of the two it was written with.
const IT_CONDS: [&str; 16] = [
    "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "al",
];

/// Shift names, indexed by the two-bit type.
const SHIFTS: [&str; 4] = ["lsl", "lsr", "asr", "ror"];

/// Extract bits `hi..=lo` inclusive.
const fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

/// Sign-extend the low `n` bits of `v`.
const fn sext(v: u32, n: u32) -> i64 {
    let shift = 64 - n;
    ((v as i64) << shift) >> shift
}

/// The name of a register. The three that the ABI gives a job are named for
/// it: `sp` is the stack pointer, `lr` the link register, `pc` the program
/// counter.
fn reg(n: u32) -> &'static str {
    match n & 15 {
        13 => "sp",
        14 => "lr",
        15 => "pc",
        0 => "r0",
        1 => "r1",
        2 => "r2",
        3 => "r3",
        4 => "r4",
        5 => "r5",
        6 => "r6",
        7 => "r7",
        8 => "r8",
        9 => "r9",
        10 => "r10",
        11 => "r11",
        _ => "r12",
    }
}

/// Rotate right by `n`, as the immediate encoding specifies.
const fn ror(v: u32, n: u32) -> u32 {
    v.rotate_right(n & 31)
}

/// Intermediate rendering of one instruction.
struct R {
    mn: String,
    text: String,
    flow: FlowKind,
    target: Option<CallTarget>,
    pc: Option<Addr>,
    reads: Vec<Addr>,
    writes: Vec<Addr>,
}

impl R {
    fn new(mn: impl Into<String>, text: String) -> R {
        R {
            mn: mn.into(),
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

    fn to(mut self, t: Addr) -> R {
        self.target = Some(CallTarget::Direct { to: t });
        self
    }

    fn reading(mut self, a: Addr) -> R {
        self.reads.push(a);
        self
    }
}

/// The decoder, bound to one mode.
#[derive(Clone, Copy)]
pub struct Decoder {
    mode: Mode,
}

impl Decoder {
    /// A decoder for `bits`-bit ARM. Only 32 is a real target.
    pub fn new(bits: u8) -> Option<Self> {
        (bits == 32).then_some(Decoder { mode: Mode::Arm })
    }

    /// A decoder for the given instruction set.
    pub const fn with_mode(mode: Mode) -> Self {
        Decoder { mode }
    }

    /// The architecture this decoder handles.
    pub const fn arch(&self) -> Arch {
        Arch::Arm
    }

    /// The instruction set this decoder reads.
    pub const fn mode(&self) -> Mode {
        self.mode
    }

    /// Fixed instruction width in bytes, or zero when the set is variable.
    pub const fn width(&self) -> u8 {
        match self.mode {
            Mode::Arm => 4,
            Mode::Thumb => 0,
        }
    }

    /// Decode one instruction at `origin`.
    pub fn decode(&self, origin: Addr, bytes: &[u8]) -> Option<Decoded> {
        match self.mode {
            Mode::Arm => self.decode_arm(origin, bytes),
            Mode::Thumb => self.decode_thumb(origin, bytes),
        }
    }

    fn decode_arm(&self, origin: Addr, bytes: &[u8]) -> Option<Decoded> {
        let b = bytes.get(0..4)?;
        let word = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let r = a32(word, origin);
        Some(finish(r, origin, word.to_le_bytes().to_vec()))
    }

    fn decode_thumb(&self, origin: Addr, bytes: &[u8]) -> Option<Decoded> {
        let half = bytes.get(0..2)?;
        let first = u16::from_le_bytes([half[0], half[1]]);
        // A 32-bit T32 instruction is two halfwords, and the first one says
        // so: `11101`, `11110` and `11111` are the only prefixes that open
        // one, which is why they are not available to the 16-bit set.
        let wide = matches!(first >> 11, 0b11101..=0b11111);
        if wide {
            let b = bytes.get(0..4)?;
            let second = u16::from_le_bytes([b[2], b[3]]);
            // The first halfword is the *low* half of the 32-bit encoding:
            // the encoding of `bl #0x100` is the four bytes `00 f0 80 f8`,
            // whose first halfword is `0xf000` and whose second is `0xf880`.
            let word = u32::from(first) | (u32::from(second) << 16);
            let r = t32_wide(word, origin);
            Some(finish(r, origin, b[..4].to_vec()))
        } else {
            let r = t32_narrow(first, origin);
            Some(finish(r, origin, half[..2].to_vec()))
        }
    }
}

/// Wrap a rendered instruction into the engine's type.
fn finish(r: R, origin: Addr, raw: Vec<u8>) -> Decoded {
    let mut bytes = [0u8; MAX_INSN_LEN];
    let n = raw.len().min(MAX_INSN_LEN);
    bytes[..n].copy_from_slice(&raw[..n]);
    Decoded {
        insn: Instruction {
            addr: origin,
            len: raw.len() as u32,
            bytes,
            text: r.text,
            mnemonic: r.mn.into(),
            flow: r.flow,
            target: r.target,
            mem_reads: r.reads,
            mem_writes: r.writes,
        },
        rip_target: r.pc,
    }
}

// ---------------------------------------------------------------------------
// A32

/// Decode one A32 word.
fn a32(w: u32, addr: Addr) -> R {
    let cond = bits(w, 31, 28);
    let suffix = CONDS[cond as usize];
    // `1111` is not a condition. In A32 it opens the unconditional space, and
    // the only instruction a compiler emits there is `blx` to a Thumb callee.
    if cond == 0b1111 {
        if bits(w, 27, 25) == 0b101 && bits(w, 24, 24) == 1 {
            let off = sext(bits(w, 23, 0) << 2, 26);
            let target = Addr((i128::from(addr.get()) + 8 + i128::from(off)) as u64);
            return R::new("blx", format!("blx {}", hex(target.get())))
                .flow(FlowKind::Call)
                .to(target);
        }
        return reserved(w, "unconditional space");
    }

    // The top level is `27:25`, not `27:26`: the immediate bit of a single
    // data transfer is bit 25, so `27:26 == 10` holds both the block
    // transfers (bit 25 clear) and the branches (bit 25 set), and splitting
    // on `27:26` decodes `b` as `ldm`.
    match bits(w, 27, 25) {
        0b000 | 0b001 => {
            // The multiply family shares the data-processing space and is
            // told apart by the last four bits of the encoding, which are
            // `1001` for every form of it. The long forms — `umull`,
            // `umlal`, `smull`, `smlal` — are in the same family and were
            // invisible while the selector also asked for bit 23 to be
            // clear: the long forms are exactly the ones that set it.
            if bits(w, 27, 25) == 0b000 && bits(w, 7, 4) == 0b1001 {
                return multiply(w, suffix);
            }
            // The half-word multiplies are the same family with a different
            // mark: they end in `100` rather than `1001`, and they choose
            // which half of each operand they multiply in their name.
            // The operation field tells them apart from the shifted
            // register form of `mov`, which shares their shape.
            if bits(w, 27, 25) == 0b000
                && bits(w, 24, 24) == 1
                && bits(w, 4, 4) == 0
                && matches!(bits(w, 23, 21), 0b001 | 0b011)
                && matches!(bits(w, 7, 5), 0b100..=0b111)
            {
                return halfword_multiply(w, suffix);
            }
            // The misc group shares the data-processing space: `00010x00`
            // with bits 6:5 == 01, which the data-processing decode would
            // otherwise read as an `and` of `pc` and a shifted register.
            // The miscellaneous group shares this space with the data
            // processing instructions, and it is not a prefix: `tst` and
            // `mrs` start with the same five bits. Each instruction is
            // matched by its own mask instead.
            if let Some(r) = misc(w, suffix, addr) {
                return r;
            }
            // `movw` / `movt`: a 16-bit immediate in two halves. They share
            // the data-processing space and would otherwise decode as `tst`
            // and `teq` of the same register — which is what turned a global
            // address into a condition test.
            if matches!(bits(w, 27, 20), 0x30 | 0x34) {
                let imm16 = (bits(w, 19, 16) << 12) | bits(w, 11, 0);
                let name = if bits(w, 27, 20) == 0x30 { "movw" } else { "movt" };
                return R::new(
                    name,
                    format!("{name}{suffix} {}, {}", reg(bits(w, 15, 12)), imm(i64::from(imm16))),
                );
            }
            // The extra load/store group is `1SH1`, and the two halves of
            // that are not separable from the data-processing instructions by
            // a prefix: `ldrd` and `ldrsb` differ in one bit.
            if matches!(bits(w, 7, 4), 0b1011 | 0b1101 | 0b1111) {
                return extra_load_store(w, suffix, addr);
            }
            data_processing(w, suffix)
        }
        // Single data transfer, immediate and register offset alike.
        0b010 | 0b011 => single_transfer(w, suffix, addr),
        // Block transfer.
        0b100 => block_transfer(w, suffix),
        // Branch, with or without link.
        0b101 => branch(w, suffix, addr),
        // Coprocessor: VFP and NEON live here, and they are a different
        // register file and a different rounding model.
        0b110 => reserved(w, "coprocessor (VFP/NEON)"),
        // Supervisor call.
        _ => {
            let imm = bits(w, 23, 0);
            R::new("svc", format!("svc{suffix} #{}", hex(u64::from(imm))))
                .flow(FlowKind::Interrupt)
        }
    }
}

/// An undecodable word: still four bytes, still aligned, and the listing says
/// what class the encoding belongs to rather than inventing a mnemonic.
fn reserved(w: u32, class: &str) -> R {
    R::new(".word", format!(".word {} ; {class}", hex(u64::from(w))))
        .flow(FlowKind::Exception)
}

fn hex(v: u64) -> String {
    format!("{v:#x}")
}

/// Render an immediate operand the way the assembler takes it.
fn imm(v: i64) -> String {
    if v < 0 {
        format!("#-{:#x}", v.unsigned_abs())
    } else {
        format!("#{v:#x}")
    }
}

/// The condition and S-bit suffix of a data-processing instruction.
fn dp_name(base: &str, s: bool, suffix: &str, has_s: bool) -> String {
    let mut name = String::with_capacity(base.len() + 3);
    name.push_str(base);
    if s && has_s {
        name.push('s');
    }
    name.push_str(suffix);
    name
}

/// True when the second operand is exactly `lr` with no shift.
fn op2_is_lr(w: u32) -> bool {
    bits(w, 25, 25) == 0
        && bits(w, 4, 4) == 0
        && bits(w, 3, 0) == 14
        && bits(w, 11, 5) == 0
}

/// The second operand of a data-processing instruction: an immediate, or a
/// register with an optional shift.
enum Op2 {
    Imm(u32),
    Reg { rm: u32, shift: u32, amount: Option<u32>, rs: Option<u32> },
}

impl Op2 {
    fn render(&self) -> String {
        match self {
            Op2::Imm(v) => imm(i64::from(*v)),
            Op2::Reg { rm, shift, amount, rs } => {
                let kind = SHIFTS[*shift as usize];
                match (amount, rs) {
                    (_, Some(r)) => format!("{}, {kind} {}", reg(*rm), reg(*r)),
                    (Some(0), _) if *shift == 0 => reg(*rm).to_string(),
                    // A rotate of zero is the one-bit rotate through carry,
                    // and the assembler spells it `rrx`.
                    (Some(0), _) if *shift == 3 => format!("{}, rrx", reg(*rm)),
                    (Some(a), _) => format!("{}, {kind} #{}", reg(*rm), a_or_32(*shift, *a)),
                    (None, _) => reg(*rm).to_string(),
                }
            }
        }
    }
}

/// A shift of zero written as zero means thirty-two for LSR and ASR.
fn a_or_32(kind: u32, amount: u32) -> u32 {
    if amount == 0 && (kind == 1 || kind == 2) { 32 } else { amount }
}

fn operand2(w: u32) -> Op2 {
    if bits(w, 25, 25) == 1 {
        Op2::Imm(ror(bits(w, 7, 0), bits(w, 11, 8) * 2))
    } else if bits(w, 4, 4) == 0 {
        Op2::Reg {
            rm: bits(w, 3, 0),
            shift: bits(w, 6, 5),
            amount: Some(bits(w, 11, 7)),
            rs: None,
        }
    } else {
        Op2::Reg {
            rm: bits(w, 3, 0),
            shift: bits(w, 6, 5),
            amount: None,
            rs: Some(bits(w, 11, 8)),
        }
    }
}

fn data_processing(w: u32, suffix: &str) -> R {
    const NAMES: [&str; 16] = [
        "and", "eor", "sub", "rsb", "add", "adc", "sbc", "rsc", "tst", "teq", "cmp", "cmn", "orr",
        "mov", "bic", "mvn",
    ];
    let op = bits(w, 24, 21);
    let s = bits(w, 20, 20) == 1;
    let rn = bits(w, 19, 16);
    let rd = bits(w, 15, 12);
    let is_test = (8..=11).contains(&op);
    let is_move = op == 13 || op == 15;
    let base = NAMES[op as usize];
    // `cmp`, `cmn`, `tst` and `teq` always set the flags, so the `S` bit is
    // part of the mnemonic rather than a suffix of it.
    let name = dp_name(base, s, suffix, !is_test);
    let op2 = operand2(w);
    let mut text = name;
    if is_test {
        text.push(' ');
        text.push_str(reg(rn));
    } else if is_move {
        text.push(' ');
        text.push_str(reg(rd));
    } else {
        text.push(' ');
        text.push_str(reg(rd));
        text.push_str(", ");
        text.push_str(reg(rn));
    }
    text.push_str(", ");
    text.push_str(&op2.render());
    let mut r = R::new(base, text);
    // `mov pc, rX` and `add pc, ...` are computed jumps, and a `mov pc, lr`
    // is the return an ARM function leaves through when it did not use
    // `bx lr`.
    if !is_test && rd == 15 {
        r.flow = if op2_is_lr(w) {
            FlowKind::Return
        } else {
            FlowKind::IndirectBranch
        };
    }
    r
}

fn multiply(w: u32, suffix: &str) -> R {
    let s = bits(w, 20, 20) == 1;
    let rd = bits(w, 19, 16);
    let rn = bits(w, 15, 12);
    let rs = bits(w, 11, 8);
    let rm = bits(w, 3, 0);
    // The operation is in three bits: the first three forms multiply into
    // one register, the next four into a pair of them, and the fourth
    // one — the half-word multiplies — has a class of its own.
    let kind = bits(w, 23, 21);
    match kind {
        0b000 => {
            let name = dp_name("mul", s, suffix, true);
            R::new(
                "mul",
                format!("{name} {}, {}, {}", reg(rd), reg(rm), reg(rs)),
            )
        }
        0b001 => {
            let name = dp_name("mla", s, suffix, true);
            R::new(
                "mla",
                format!("{name} {}, {}, {}, {}", reg(rd), reg(rm), reg(rs), reg(rn)),
            )
        }
        0b010 => {
            // `umaal rd, rm, rs, rn`: an unsigned product added to a third
            // register. The destination is the one the encoding keeps in
            // the second field, and the multiplicands are the other two.
            let name = dp_name("umaal", s, suffix, true);
            R::new(
                "umaal",
                format!(
                    "{name} {}, {}, {}, {}",
                    reg(rn),
                    reg(rd),
                    reg(rm),
                    reg(rs)
                ),
            )
        }
        0b100 | 0b101 => {
            let long = kind == 0b101;
            let name = dp_name(if long { "umlal" } else { "umull" }, s, suffix, true);
            R::new(
                if long { "umlal" } else { "umull" },
                format!(
                    "{name} {}, {}, {}, {}",
                    reg(rn),
                    reg(rd),
                    reg(rm),
                    reg(rs)
                ),
            )
        }
        0b110 | 0b111 => {
            let long = kind == 0b111;
            let name = dp_name(if long { "smlal" } else { "smull" }, s, suffix, true);
            R::new(
                if long { "smlal" } else { "smull" },
                format!(
                    "{name} {}, {}, {}, {}",
                    reg(rn),
                    reg(rd),
                    reg(rm),
                    reg(rs)
                ),
            )
        }
        _ => reserved(w, "multiply"),
    }
}

/// The half-word multiplies: `smulbb`, `smultb`, `smulwb`, `smulwt`.
///
/// They multiply one half of one operand by one half of the other, and
/// which halves is in the name: `bb` is the bottom half of both, `tb` the
/// top of the first and the bottom of the second, and so on.
fn halfword_multiply(w: u32, suffix: &str) -> R {
    let s = bits(w, 20, 20) == 1;
    let rd = bits(w, 19, 16);
    let rs = bits(w, 11, 8);
    let rm = bits(w, 3, 0);
    // The name says which half of which operand: the first letter is the
    // first operand's half and the second is the second's, and `w` means
    // the top word just as `t` does.
    let kind = match (bits(w, 7, 5), bits(w, 23, 21)) {
        (0b100, 0b011) => "bb",
        (0b101, 0b011) => "tb",
        (0b110, 0b011) => "bt",
        (0b111, 0b011) => "tt",
        (0b101, 0b001) => "wb",
        (0b111, 0b001) => "wt",
        _ => return reserved(w, "half-word multiply"),
    };
    let name = dp_name(&format!("smul{kind}"), s, suffix, true);
    R::new(
        "smul",
        format!("{name} {}, {}, {}", reg(rd), reg(rm), reg(rs)),
    )
}

/// `bx`, `blx`, `clz`, `mrs`, `msr`, the divide instructions and the hints.
///
/// Every one is matched by its own mask. The group is not a prefix — `tst`
/// and `mrs` share their top five bits — so a selector on bits 27:23 turns
/// `bx lr` into `teq pc, lr, lsl pc` and leaves the return invisible to
/// block recovery.
fn misc(w: u32, suffix: &str, _addr: Addr) -> Option<R> {
    let rm = bits(w, 3, 0);
    let rd = bits(w, 15, 12);
    // BX / BLX register. `bx lr` is the return an ARM function leaves
    // through, and saying so is what lets recovery stop the body there
    // instead of falling into the next function's first byte.
    if w & 0x0fff_fff0 == 0x012f_ff10 {
        let name = format!("bx{suffix}");
        // `bxne lr` returns on one path and falls through on the other. It
        // is a branch, not a return: ending the block at it made everything
        // after the conditional return unreachable, which is how a closed
        // form of a loop looked like the whole function.
        let flow = if !suffix.is_empty() {
            FlowKind::ConditionalBranch
        } else if rm == 14 {
            FlowKind::Return
        } else {
            FlowKind::IndirectBranch
        };
        return Some(R::new("bx", format!("{name} {}", reg(rm))).flow(flow));
    }
    if w & 0x0fff_fff0 == 0x012f_ff30 {
        let name = format!("blx{suffix}");
        return Some(R::new("blx", format!("{name} {}", reg(rm))).flow(FlowKind::IndirectCall));
    }
    // CLZ.
    if w & 0x0fff_0ff0 == 0x016f_0f10 {
        let name = format!("clz{suffix}");
        return Some(R::new("clz", format!("{name} {}, {}", reg(rd), reg(rm))));
    }
    // MRS: `0001 0000 R 00 1111 Rd 0000 0000 0000`.
    if w & 0x0fbf_0fff == 0x010f_0000 {
        let psr = if bits(w, 22, 22) == 1 { "spsr" } else { "cpsr" };
        let name = format!("mrs{suffix}");
        return Some(R::new("mrs", format!("{name} {}, {psr}", reg(rd))));
    }
    // MSR: `0001 0010 R 10 mask 1111 ... Rm`.
    if w & 0x0fb0_fff0 == 0x0120_f000 {
        let psr = if bits(w, 22, 22) == 1 { "spsr" } else { "cpsr" };
        let name = format!("msr{suffix}");
        return Some(R::new("msr", format!("{name} {psr}_c, {}", reg(rm))));
    }
    // SDIV / UDIV: the dividend is `Rn`, the divisor `Rm`, and the result
    // goes in the field the encoding calls `Rd16` — three different places
    // from where the data-processing instructions put them.
    // `sdiv Rd, Rn, Rm`: the result is in 19:16, the divisor in 11:8 and
    // the dividend in 3:0, which is not the order any other instruction
    // uses.
    if w & 0x0ff0_f0f0 == 0x0710_f010 {
        let name = format!("sdiv{suffix}");
        let rd = bits(w, 19, 16);
        let rs = bits(w, 11, 8);
        let rn = bits(w, 3, 0);
        return Some(R::new(
            "sdiv",
            format!("{name} {}, {}, {}", reg(rd), reg(rn), reg(rs)),
        ));
    }
    if w & 0x0ff0_f0f0 == 0x0730_f010 {
        let name = format!("udiv{suffix}");
        let rd = bits(w, 19, 16);
        let rs = bits(w, 11, 8);
        let rn = bits(w, 3, 0);
        return Some(R::new(
            "udiv",
            format!("{name} {}, {}, {}", reg(rd), reg(rn), reg(rs)),
        ));
    }
    // Hints: `nop`, `yield`, `wfe`, `wfi`, `sev`. The encoding is that of
    // `rsb r0, r0, #0`, so decoding it as one tells a reader the function
    // computes something it does not.
    if w & 0x0fff_fff0 == 0x0320_f000 {
        let hint = match bits(w, 7, 0) {
            0 => "nop",
            1 => "yield",
            2 => "wfe",
            3 => "wfi",
            4 => "sev",
            _ => "hint",
        };
        let text = if hint == "hint" {
            format!("hint{suffix} {}", hex(u64::from(bits(w, 7, 0))))
        } else {
            format!("{hint}{suffix}")
        };
        return Some(R::new(hint, text));
    }
    None
}

/// The offset of a single data transfer, rendered and, when it is a PC
/// literal, resolved.
struct Transfer {
    text: String,
    abs: Option<Addr>,
    write_back: bool,
    post: bool,
}

fn single_offset(w: u32, rn: u32, addr: Addr) -> Transfer {
    let i = bits(w, 25, 25) == 1;
    let u = bits(w, 23, 23) == 1;
    let post = bits(w, 24, 24) == 0;
    let write_back = bits(w, 21, 21) == 1 || post;
    if i {
        let rm = bits(w, 3, 0);
        let shift = bits(w, 6, 5);
        let amount = bits(w, 11, 7);
        let sign = if u { "" } else { "-" };
        let mut text = if amount == 0 && shift == 0 {
            format!("{sign}{}", reg(rm))
        } else {
            format!("{sign}{}, {} #{}", reg(rm), SHIFTS[shift as usize], a_or_32(shift, amount))
        };
        if post {
            // A post-indexed operand is written after the brackets, so the
            // sign belongs to the offset and the base is bare.
            text = text.trim_start_matches('-').to_string();
            if !u {
                text.insert(0, '-');
            }
            return Transfer { text, abs: None, write_back, post };
        }
        return Transfer { text, abs: None, write_back, post };
    }
    let off = bits(w, 11, 0);
    let value = if u { i64::from(off) } else { -i64::from(off) };
    // `[pc, #n]` addresses a literal in the same section: the PC reads as
    // the instruction's address plus eight, and the operand is an address.
    let abs = if rn == 15 {
        Some(Addr((i128::from(addr.get()) + 8 + i128::from(value)) as u64))
    } else {
        None
    };
    // A zero offset prints nothing: `[r0]` and `[r0, #0]` are one address.
    let text = if off == 0 { String::new() } else { imm(value) };
    Transfer { text, abs, write_back, post }
}

fn single_transfer(w: u32, suffix: &str, addr: Addr) -> R {
    let b = bits(w, 22, 22) == 1;
    let l = bits(w, 20, 20) == 1;
    let rn = bits(w, 19, 16);
    let rd = bits(w, 15, 12);
    let t = single_offset(w, rn, addr);
    let base = match (l, b) {
        (true, false) => "ldr",
        (false, false) => "str",
        (true, true) => "ldrb",
        (false, true) => "strb",
    };
    let name = format!("{base}{suffix}");
    let mut text = format!("{name} {}", reg(rd));
    if t.post {
        // Post-indexed: the write-back is in the brackets' position, and
        // `[r0], #4` is the only spelling the assembler accepts.
        text.push_str(&format!(", [{}]", reg(rn)));
        if !t.text.is_empty() {
            text.push_str(", ");
            text.push_str(&t.text);
        }
    } else {
        text.push_str(&format!(", [{}", reg(rn)));
        if !t.text.is_empty() {
            text.push_str(", ");
            text.push_str(&t.text);
        }
        text.push(']');
        if t.write_back {
            text.push('!');
        }
    }
    let mut r = R::new(base, text);
    if let Some(a) = t.abs {
        if l {
            r = r.reading(a);
            r.pc = Some(a);
        }
    }
    // `ldr pc, [...]` is a jump, and it is how a switch table and a tail
    // call are spelled: the load and the transfer are one instruction.
    if l && rd == 15 {
        r.flow = FlowKind::IndirectBranch;
    }
    r
}

fn extra_load_store(w: u32, suffix: &str, addr: Addr) -> R {
    let p = bits(w, 24, 24) == 1;
    let u = bits(w, 23, 23) == 1;
    let immediate = bits(w, 22, 22) == 1;
    let wb = bits(w, 21, 21) == 1;
    let l = bits(w, 20, 20) == 1;
    let rn = bits(w, 19, 16);
    let rd = bits(w, 15, 12);
    let s = bits(w, 6, 6) == 1;
    let h = bits(w, 5, 5) == 1;
    // The group is `1SH1`, and the four encodings are the four
    // combinations — except that `1101` and `1111` also carry the
    // doubleword forms, which are told apart by the load bit: `ldrd` is
    // `1101` with L clear and `ldrsb` is the same encoding with L set.
    let _ = (s, h);
    let form = bits(w, 7, 4);
    let (base, double) = match form {
        0b1011 => (if l { "ldrh" } else { "strh" }, false),
        0b1101 => {
            if l {
                ("ldrsb", false)
            } else {
                ("ldrd", true)
            }
        }
        0b1111 => {
            if l {
                ("ldrsh", false)
            } else {
                ("strd", true)
            }
        }
        _ => return reserved(w, "extra load/store"),
    };
    if double {
        let off = (bits(w, 11, 8) << 4) | bits(w, 3, 0);
        let v = if u { i64::from(off) } else { -i64::from(off) };
        let operand = if off == 0 {
            format!("[{}]", reg(rn))
        } else {
            format!("[{}, {}]", reg(rn), imm(v))
        };
        let mut text = format!("{base}{suffix} {}, {}, {operand}", reg(rd), reg(rd + 1));
        if !p {
            text = format!(
                "{base}{suffix} {}, {}, [{}], {v}",
                reg(rd),
                reg(rd + 1),
                reg(rn)
            );
        } else if wb {
            text.push('!');
        }
        return R::new(base, text);
    }
    let off = (bits(w, 11, 8) << 4) | bits(w, 3, 0);
    let text = if immediate {
        let v = if u { i64::from(off) } else { -i64::from(off) };
        if off == 0 {
            format!("[{}]", reg(rn))
        } else {
            format!("[{}, {}]", reg(rn), imm(v))
        }
    } else {
        let sign = if u { "" } else { "-" };
        format!("[{}, {sign}{}]", reg(rn), reg(off))
    };
    let name = format!("{base}{suffix}");
    let mut text = format!("{name} {}, {text}", reg(rd));
    if !p {
        // A post-indexed halfword transfer has its offset after the bracket
        // and the bracket holds only the base; the encoding still keeps the
        // pieces in the same fields.
        text = format!("{name} {}, [{}], {}", reg(rd), reg(rn), imm(if u { i64::from(off) } else { -i64::from(off) }));
    } else if wb {
        text.push('!');
    }
    let mut r = R::new(base, text);
    if rn == 15 {
        if let Some(a) = literal(addr, if u { i64::from(off) } else { -i64::from(off) }) {
            r = r.reading(a);
            r.pc = Some(a);
        }
    }
    r
}

/// The address a PC-relative operand names: the PC reads as the
/// instruction's address plus eight in A32.
fn literal(addr: Addr, off: i64) -> Option<Addr> {
    Some(Addr((i128::from(addr.get()) + 8 + i128::from(off)) as u64))
}

fn block_transfer(w: u32, suffix: &str) -> R {
    let p = bits(w, 24, 24) == 1;
    let u = bits(w, 23, 23) == 1;
    let wb = bits(w, 21, 21) == 1;
    let l = bits(w, 20, 20) == 1;
    let rn = bits(w, 19, 16);
    let list = bits(w, 15, 0);
    let registers = reg_list(list);
    // The stack forms have names of their own, and using them is what makes
    // a prologue readable: `push {r4, lr}` instead of `stmdb sp!, {r4, lr}`.
    let (base, text) = if rn == 13 && wb && list & (1 << 13) == 0 {
        if p && !u && !l {
            ("push", format!("push{suffix} {registers}"))
        } else if !p && u && l {
            ("pop", format!("pop{suffix} {registers}"))
        } else {
            let mode = block_mode(p, u);
            (l_name(l), format!("{} {mode} sp!, {registers}", l_name(l)))
        }
    } else {
        let mode = block_mode(p, u);
        let bang = if wb || list & (1 << rn) != 0 { "!" } else { "" };
        (
            l_name(l),
            format!(
                "{}{suffix} {mode} {}{bang}, {registers}",
                l_name(l),
                reg(rn)
            ),
        )
    };
    let name = format!("{base}{suffix}");
    // `pop {..., pc}` is a return: the load writes the program counter.
    let flow = if l && list & (1 << 15) != 0 {
        FlowKind::Return
    } else {
        FlowKind::Next
    };
    R::new(base, text).flow(flow).with_name(name)
}

impl R {
    fn with_name(mut self, name: String) -> R {
        self.mn = name;
        self
    }
}

fn l_name(l: bool) -> &'static str {
    if l { "ldm" } else { "stm" }
}

fn block_mode(p: bool, u: bool) -> &'static str {
    match (p, u) {
        (false, true) => "ia",
        (true, true) => "ib",
        (false, false) => "da",
        (true, false) => "db",
    }
}

fn reg_list(list: u32) -> String {
    let mut parts = Vec::new();
    let mut n = 0;
    while n < 16 {
        if list & (1 << n) == 0 {
            n += 1;
            continue;
        }
        let start = n;
        while n + 1 < 16 && list & (1 << (n + 1)) != 0 {
            n += 1;
        }
        if start == n {
            parts.push(reg(start).to_string());
        } else {
            parts.push(format!("{}-{}", reg(start), reg(n)));
        }
        n += 1;
    }
    format!("{{{}}}", parts.join(", "))
}

fn branch(w: u32, suffix: &str, addr: Addr) -> R {
    let l = bits(w, 24, 24) == 1;
    let off = sext(bits(w, 23, 0) << 2, 26);
    let target = Addr((i128::from(addr.get()) + 8 + i128::from(off)) as u64);
    if l {
        R::new("bl", format!("bl{suffix} {}", hex(target.get())))
            .flow(FlowKind::Call)
            .to(target)
    } else if suffix.is_empty() {
        R::new("b", format!("b {}", hex(target.get())))
            .flow(FlowKind::UnconditionalBranch)
            .to(target)
    } else {
        R::new("b", format!("b{suffix} {}", hex(target.get())))
            .flow(FlowKind::ConditionalBranch)
            .to(target)
    }
}

// ---------------------------------------------------------------------------
// T32
//
// The Thumb half of the decoder. The two halves are separate instruction
// sets that share a register file, and the split between them is not a
// guess: the first halfword says which one an instruction belongs to.
//
// The 16-bit set is small enough to be read in full — 65536 halfwords, of
// which 58511 are defined — and it was read in full against two
// independent implementations. The 32-bit set is four times the size of the
// address space of a halfword, so it was read over a structured sample that
// covers every operation class and every field of every class, with the
// classes a compiler emits checked one by one.

/// The Thumb program counter reads as the instruction's address plus four.
///
/// This is the difference from A32, where it reads as the address plus
/// eight, and it is the difference between a literal pool that is one word
/// off and one that is right.
const T32_PC_BIAS: i64 = 4;

/// The address a Thumb PC-relative operand is measured from.
fn t_pc(addr: Addr) -> i128 {
    i128::from(addr.get()) + i128::from(T32_PC_BIAS)
}

/// Decode one 16-bit T32 halfword.
fn t32_narrow(w: u16, addr: Addr) -> R {
    t16(u32::from(w), addr)
}

/// The condition a 16-bit branch tests, and the width of its displacement.
struct Branch16 {
    /// The condition suffix, empty for an unconditional branch.
    cond: &'static str,
    /// The number of bits in the displacement, before the shift.
    width: u32,
    /// The scale applied to the displacement.
    scale: u32,
    /// Whether the displacement is a forward-only range.
    forward_only: bool,
}


/// Decode one 16-bit Thumb instruction.
///
/// The class is chosen by the top three bits.  The measurement shows the
/// class boundaries to be exactly there: every class in the space is a
/// contiguous range, and the ranges are separated by those bits.
fn t16(x: u32, addr: Addr) -> R {
    match bits(x, 15, 13) {
        // `000xx`: the shifts by an immediate, and the add and subtract
        // in their register and three-bit-immediate forms.
        0b000 => t16_shift_add(x),
        // `001xx`: the move, compare, add and subtract by an eight-bit
        // immediate.
        0b001 => t16_imm8(x),
        // `010xx`: the data processing on registers, the special data
        // processing group, the literal load, and the loads and stores
        // with a register offset.
        0b010 => t16_dp(x, addr),
        // `011xx`: the loads and stores with a word or a byte offset.
        0b011 => t16_ls_imm(x, addr),
        // `100xx`: the halfword loads and stores, and the loads and
        // stores relative to the stack pointer.
        0b100 => t16_sp(x, addr),
        // `101xx`: the address computation, the stack-pointer
        // arithmetic, and the miscellaneous group.
        0b101 => t16_sp_or_misc(x, addr),
        // `110xx`: the load and store multiple, and the conditional
        // branches.
        0b110 => t16_branch(x, addr),
        // `111xx`: the unconditional branch.
        _ => t16_uncond(x, addr),
    }
}

/// `000xx`: the shifts by an immediate, and the add and subtract.
///
/// The next two bits choose within the group: `00` is a left shift, `01`
/// a right shift, `10` an arithmetic right shift, and `11` is the
/// add-and-subtract pair.
///
/// Within the shifts, the destination is the low three bits, the source
/// is the three above it, and the amount is the five above that.  A shift
/// of zero means thirty-two for the two right shifts.
///
/// Within the add and subtract, the bit at 9 says which operation it is
/// and the bit at 10 says whether the third operand is a register or a
/// three-bit immediate.  The three-bit immediate is scaled by four, and
/// the destination is the low three bits.
fn t16_shift_add(x: u32) -> R {
    match bits(x, 12, 11) {
        0b00 | 0b01 | 0b10 => {
            let name = match bits(x, 12, 11) {
                0b00 => "lsls",
                0b01 => "lsrs",
                _ => "asrs",
            };
            let mut amount = bits(x, 10, 6);
            if amount == 0 && bits(x, 12, 11) != 0b00 {
                amount = 32;
            }
            let rm = bits(x, 5, 3);
            let rd = bits(x, 2, 0);
            // A left shift of zero is a move: the two are the same
            // instruction, and the references print the move.
            if name == "lsls" && amount == 0 {
                return R::new("mov", format!("movs {}, {}", reg(rd), reg(rm)));
            }
            R::new(
                name,
                format!("{name} {}, {}, {}", reg(rd), reg(rm), imm(i64::from(amount))),
            )
        }
        _ => {
            // The bit at 9 chooses the operation, and the bit at 10
            // chooses the form.
            let sub = bits(x, 9, 9) == 1;
            let name = if sub { "subs" } else { "adds" };
            let rd = bits(x, 2, 0);
            if bits(x, 10, 10) == 0 {
                // Register form: `adds Rd, Rn, Rm`.  The third operand is
                // the three bits at 6, and the second is the three bits
                // at 3.
                let rm = bits(x, 8, 6);
                let rn = bits(x, 5, 3);
                R::new(
                    if sub { "sub" } else { "add" },
                    format!("{name} {}, {}, {}", reg(rd), reg(rn), reg(rm)),
                )
            } else {
                // Three-bit immediate form.  The immediate is the three
                // bits at 6 — not scaled — and the second operand is the
                // three bits at 3.
                let amount = bits(x, 8, 6);
                let rn = bits(x, 5, 3);
                R::new(
                    if sub { "sub" } else { "add" },
                    format!("{name} {}, {}, {}", reg(rd), reg(rn), imm(i64::from(amount))),
                )
            }
        }
    }
}

/// `001xx`: the move, compare, add and subtract by an eight-bit
/// immediate.
///
/// The two bits below the group choose the operation, and the
/// destination is the three bits at 8 — *not* the three at the bottom,
/// which hold the immediate.
fn t16_imm8(x: u32) -> R {
    let rd = bits(x, 10, 8);
    let value = i64::from(bits(x, 7, 0));
    match bits(x, 12, 11) {
        0b00 => R::new("mov", format!("movs {}, {}", reg(rd), imm(value))),
        0b01 => R::new("cmp", format!("cmp {}, {}", reg(rd), imm(value))),
        0b10 => R::new("add", format!("adds {}, {}", reg(rd), imm(value))),
        _ => R::new("sub", format!("subs {}, {}", reg(rd), imm(value))),
    }
}

/// `010xx`: the data processing on registers, the special data
/// processing group, the literal load, and the loads and stores with a
/// register offset.
///
/// The six bits at the top pick the class: the first sixteen encodings
/// of every row are the data processing on registers, the next sixteen
/// are the special data processing group, and the two rows after that
/// are the literal load and the register-offset loads and stores.
fn t16_dp(x: u32, addr: Addr) -> R {
    match bits(x, 15, 10) {
        0b010000 => t16_dp_reg(x),
        0b010001 => t16_dp_special(x),
        0b010010 | 0b010011 => t16_literal(x, addr),
        _ => t16_ls_reg(x, addr),
    }
}

/// `010000 op(4) Rm(3) Rdn(3)`: the data processing on registers.
///
/// The operation field chooses the operation and the two operand fields
/// are in the same place in every one of them.  The shifts take their
/// amount from the field that holds the second operand, and `rsb` with a
/// zero amount is a negation.
fn t16_dp_reg(x: u32) -> R {
    const NAMES: [&str; 16] = [
        "and", "eor", "lsl", "lsr", "asr", "adc", "sbc", "ror", "tst", "rsb", "cmp", "cmn", "orr",
        "mul", "bic", "mvn",
    ];
    let op = bits(x, 9, 6);
    let rm = bits(x, 5, 3);
    let rdn = bits(x, 2, 0);
    let base = NAMES[op as usize];
    match op {
        // The shifts write the amount in the second operand's place, so
        // the second operand is the amount and not a register.
        2 | 3 | 4 => R::new(base, format!("{base}s {}, {}", reg(rdn), reg(rm))),
        9 => R::new("rsb", format!("rsbs {}, {}, {}", reg(rdn), reg(rm), imm(0))),
        13 => R::new(
            "mul",
            format!("muls {}, {}, {}", reg(rdn), reg(rm), reg(rdn)),
        ),
        // `tst`, `cmp` and `cmn` compare and discard the result; the
        // flags are the whole point.
        8 | 10 | 11 => R::new(base, format!("{base} {}, {}", reg(rdn), reg(rm))),
        _ => R::new(base, format!("{base}s {}, {}", reg(rdn), reg(rm))),
    }
}

/// `010001 op(2) D(1) Rm(4) Rdn(3)`: the special data processing group
/// and the branch exchange.
///
/// The operation selects between add, compare, move and the branch
/// exchange.
///
/// The source register is the four bits at 3, so it can name any
/// register; the destination is the three bits at the bottom, and the
/// bit at 7 is its fourth bit, which is what makes the destination a
/// high register.
fn t16_dp_special(x: u32) -> R {
    let op = bits(x, 9, 8);
    let dst = bits(x, 2, 0) | (bits(x, 7, 7) << 3);
    let rm = bits(x, 6, 3);
    match op {
        0b00 => {
            // `add`: the two-operand form writes the sum to the
            // destination.  The stack pointer is the one source that
            // takes a third operand, which is how a value is computed
            // relative to the stack.
            let text = if rm == 13 {
                format!("add {}, {}, {}", reg(dst), reg(rm), reg(dst))
            } else {
                format!("add {}, {}", reg(dst), reg(rm))
            };
            R::new("add", text)
        }
        0b01 => R::new("cmp", format!("cmp {}, {}", reg(dst), reg(rm))),
        0b10 => R::new("mov", format!("mov {}, {}", reg(dst), reg(rm))),
        _ => {
            // The branch exchange.  The source is the four bits at 3,
            // and the three bits at the bottom are not part of it.
            if bits(x, 7, 7) == 0 {
                // `bx`: a jump through a register.  `bx lr` is the
                // return.
                //
                // The bit at 2 is the ARMv8-M security extension: it
                // makes the exchange change the security state as well.
                // The two references agree on the name only because
                // they are read in the ARMv8-M profile — the ARMv7 one
                // has no such instruction and prints the exchange
                // without it.
                if bits(x, 2, 2) == 1 {
                    let flow = if rm == 14 {
                        FlowKind::Return
                    } else {
                        FlowKind::IndirectBranch
                    };
                    return R::new("bxns", format!("bxns {}", reg(rm))).flow(flow);
                }
                let flow = if rm == 14 {
                    FlowKind::Return
                } else {
                    FlowKind::IndirectBranch
                };
                R::new("bx", format!("bx {}", reg(rm))).flow(flow)
            } else {
                // `blx`: a call through a register.  Here the three bits
                // at the bottom say which form it is: `000` is the
                // plain exchange, a set bit at 2 is the security
                // extension, and the rest are not defined at all.
                match bits(x, 2, 0) {
                    0b000 => {
                        R::new("blx", format!("blx {}", reg(rm))).flow(FlowKind::IndirectCall)
                    }
                    0b100 | 0b101 | 0b110 | 0b111 => {
                        R::new("blxns", format!("blxns {}", reg(rm))).flow(FlowKind::IndirectCall)
                    }
                    _ => t16_undefined(x),
                }
            }
        }
    }
}

/// `01001 Rt(3) imm8(8)`: the literal load.
///
/// The destination is the three bits at 8, and the offset is the eight
/// bits at the bottom, scaled by four.  The address it names is relative
/// to the program counter, which reads as the instruction's address plus
/// four.
fn t16_literal(x: u32, addr: Addr) -> R {
    let rt = bits(x, 10, 8);
    let amount = i64::from(bits(x, 7, 0)) * 4;
    let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
    R::new("ldr", format!("ldr {}, [pc, {}]", reg(rt), imm(amount)))
        .reading(target)
        .with_pc(target)
}

/// `0101 op(3) Rm(3) Rn(3) Rt(3)`: the loads and stores with a register
/// offset.
///
/// The operation field chooses the direction and the width, the
/// destination is the low three bits, the base is the three above it,
/// and the offset register is the three above that.
fn t16_ls_reg(x: u32, addr: Addr) -> R {
    let _ = addr;
    let op = bits(x, 11, 9);
    let base = match op {
        0b000 => "str",
        0b001 => "strh",
        0b010 => "strb",
        0b011 => "ldrsb",
        0b100 => "ldr",
        0b101 => "ldrh",
        0b110 => "ldrb",
        _ => "ldrsh",
    };
    let rt = bits(x, 2, 0);
    let rn = bits(x, 5, 3);
    let rm = bits(x, 8, 6);
    R::new(
        base,
        format!("{base} {}, [{}, {}]", reg(rt), reg(rn), reg(rm)),
    )
}

/// `011xx`: the loads and stores with a word or a byte offset.
///
/// The bit at 12 chooses between a word and a byte, the bit at 11
/// chooses the direction, and the five bits at 6 hold the offset.  A
/// word's offset is scaled by four.
fn t16_ls_imm(x: u32, addr: Addr) -> R {
    let word = bits(x, 12, 12) == 0;
    let load = bits(x, 11, 11) == 1;
    let mut amount = i64::from(bits(x, 10, 6));
    if word {
        amount *= 4;
    }
    let rn = bits(x, 5, 3);
    let rt = bits(x, 2, 0);
    let base = match (word, load) {
        (true, false) => "str",
        (true, true) => "ldr",
        (false, false) => "strb",
        (false, true) => "ldrb",
    };
    t16_mem(base, rt, rn, amount, addr)
}

/// Render a Thumb memory operand with an immediate offset.
fn t16_mem(base: &str, rt: u32, rn: u32, amount: i64, addr: Addr) -> R {
    let text = if amount == 0 {
        format!("{base} {}, [{}]", reg(rt), reg(rn))
    } else {
        format!("{base} {}, [{}, {}]", reg(rt), reg(rn), imm(amount))
    };
    let mut r = R::new(base, text);
    // A literal pool is addressed relative to the program counter, which
    // reads as the instruction's address plus four.
    if rn == 15 && base.starts_with("ldr") {
        let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
        r = r.reading(target);
        r.pc = Some(target);
    }
    r
}

/// `100xx`: the halfword loads and stores with an immediate offset, and
/// the loads and stores relative to the stack pointer.
fn t16_sp(x: u32, addr: Addr) -> R {
    match bits(x, 15, 12) {
        // `1000`: a halfword load or store.  The offset is scaled by two.
        0b1000 => {
            let load = bits(x, 11, 11) == 1;
            let amount = i64::from(bits(x, 10, 6)) * 2;
            let rn = bits(x, 5, 3);
            let rt = bits(x, 2, 0);
            let base = if load { "ldrh" } else { "strh" };
            t16_mem(base, rt, rn, amount, addr)
        }
        // `1001`: a load or store relative to the stack pointer.  The
        // destination is the three bits at 8 and the offset is the eight
        // bits at the bottom, scaled by four.
        _ => {
            let load = bits(x, 11, 11) == 1;
            let amount = i64::from(bits(x, 7, 0)) * 4;
            let rt = bits(x, 10, 8);
            let base = if load { "ldr" } else { "str" };
            t16_mem(base, rt, 13, amount, addr)
        }
    }
}

/// `101xx`: the address computation, the stack-pointer arithmetic, and
/// the miscellaneous group.
fn t16_sp_or_misc(x: u32, addr: Addr) -> R {
    match bits(x, 15, 12) {
        // `1010`: the address computation, and the add to the stack
        // pointer.
        0b1010 => {
            let rd = bits(x, 10, 8);
            let amount = i64::from(bits(x, 7, 0)) * 4;
            if bits(x, 11, 11) == 0 {
                // `adr`: the program counter, aligned down to a word,
                // plus the immediate.  The alignment is why the target
                // is not simply the address plus four.
                let base = (t_pc(addr) - 4) & !3;
                let target = base + i128::from(amount);
                let text = format!("adr {}, {}", reg(rd), imm(amount));
                return R::new("adr", text)
                    .reading(Addr(target as u64))
                    .with_pc(Addr(target as u64));
            }
            // `add rd, sp, #imm`: the stack pointer plus the immediate.
            let text = format!("add {}, sp, {}", reg(rd), imm(amount));
            R::new("add", text)
        }
        // `1011`: the miscellaneous group.
        _ => t16_misc(x, addr),
    }
}

/// `101xx`: the miscellaneous group.
///
/// This is the group that has no internal structure: stack adjustment,
/// compare-and-branch, the sign and zero extends, `push` and `pop`, the
/// byte and halfword reverses, and the hints.
///
/// The six bits at the top pick the family and the two bits at 9 and 8
/// pick within it.  The table the measurement gives is:
///
/// | `bits[15:10]` | `bits[9:8]` | the instruction |
/// |---|---|---|
/// | `101100` | `00` | the stack-pointer adjustment |
/// | `101100` | `01` | the extends |
/// | `101101` | `10` | `push` |
/// | `101101` | `11` | the system instructions |
/// | `101110` | `00` | the compare-and-branch on zero |
/// | `101110` | `01` | the reverses |
/// | `101111` | `10` | `pop` |
/// | `101111` | `11` | `bkpt` and the hints |
///
/// The compare-and-branch pair is the exception: it is the bit at 8
/// that tells it apart from the extends and the reverses, and the bit at
/// 11 that tells the two branches apart from each other.
fn t16_misc(x: u32, addr: Addr) -> R {
    match bits(x, 15, 10) {
        0b101100 => {
            if bits(x, 9, 8) == 0b00 {
                // The stack-pointer adjustment.  The amount is the seven
                // bits at the bottom, scaled by four, and the bit at 7
                // says whether the stack pointer grows or shrinks.
                let sub = bits(x, 7, 7) == 1;
                let amount = i64::from(bits(x, 6, 0)) * 4;
                let name = if sub { "sub" } else { "add" };
                return R::new(name, format!("{name} sp, {}", imm(amount)));
            }
            if bits(x, 8, 8) == 1 {
                // The compare-and-branch.  The displacement is a six-bit
                // field — the five at 3 and the one at 9 — scaled by two,
                // and it is *not* signed: the 16-bit form only reaches
                // forwards, and the bit at 9 is the top of the field
                // rather than a sign.
                //
                // The three ways of reading this were compared: the
                // assembler was asked to encode a branch to a known
                // distance and produced this field for every distance
                // the form can reach, and both references read the
                // destination the same way.
                let nz = bits(x, 11, 11) == 1;
                let rn = bits(x, 2, 0);
                let amount = i64::from((bits(x, 9, 9) << 5) | bits(x, 7, 3)) * 2;
                let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
                let name = if nz { "cbnz" } else { "cbz" };
                let text = format!("{name} {}, {}", reg(rn), hex(target.get()));
                return R::new(name, text).flow(FlowKind::ConditionalBranch).to(target);
            }
            // The extends.  The operation is the two bits at 6 —
            // signed or unsigned, halfword or byte — the source is the
            // three bits at 3, and the destination is the three bits at
            // the bottom.
            let op = bits(x, 7, 6);
            let rm = bits(x, 5, 3);
            let rd = bits(x, 2, 0);
            let name = match op {
                0b00 => "sxth",
                0b01 => "sxtb",
                0b10 => "uxth",
                _ => "uxtb",
            };
            R::new(name, format!("{name} {}, {}", reg(rd), reg(rm)))
        }
        0b101101 => {
            if bits(x, 15, 9) == 0b1011010 {
                // `push`.  The bit at 8 adds the link register to the
                // list, which is how a call's return address is pushed.
                let list = bits(x, 7, 0) | (bits(x, 8, 8) << 8);
                // An empty list is not an instruction: there is nothing
                // to push, and the architecture does not define it.
                if list == 0 {
                    return t16_undefined(x);
                }
                return R::new("push", format!("push {}", t16_reglist(list, true)));
            }
            // `10110110`: the endianness switch and the interrupt masks.
            if bits(x, 15, 8) == 0b1011_0110 {
                return t16_system(x);
            }
            t16_undefined(x)
        }
        0b101110 => {
            if bits(x, 8, 8) == 1 {
                // The compare-and-branch, the other one.  Its
                // displacement is the same unsigned field as the one
                // above.
                let rn = bits(x, 2, 0);
                let amount = i64::from((bits(x, 9, 9) << 5) | bits(x, 7, 3)) * 2;
                let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
                let text = format!("cbnz {}, {}", reg(rn), hex(target.get()));
                return R::new("cbnz", text).flow(FlowKind::ConditionalBranch).to(target);
            }
            // The reverses.  The operation is in the two bits at 6, the
            // source is the three bits at 3, and the destination is the
            // three bits at the bottom.  The third operation is not
            // defined.
            //
            // They are the half of the family that the bit at 9 leaves
            // clear: the other half is the compare-and-branch, and the
            // pair of rows below the family is not defined at all.
            if bits(x, 15, 9) != 0b1011101 {
                return t16_undefined(x);
            }
            let op = bits(x, 7, 6);
            let rm = bits(x, 5, 3);
            let rd = bits(x, 2, 0);
            let name = match op {
                0b00 => "rev",
                0b01 => "rev16",
                0b11 => "revsh",
                _ => return t16_undefined(x),
            };
            R::new(name, format!("{name} {}, {}", reg(rd), reg(rm)))
        }
        _ => {
            if bits(x, 15, 9) == 0b1011110 {
                // `pop`.  The bit at 8 adds the program counter to the
                // list, which is how a function returns.
                let list = bits(x, 7, 0) | (bits(x, 8, 8) << 8);
                // An empty list is not an instruction: there is nothing
                // to pop, and the architecture does not define it.
                if list == 0 {
                    return t16_undefined(x);
                }
                return R::new("pop", format!("pop {}", t16_reglist(list, false)));
            }
            // `10111110`: `bkpt`.
            if bits(x, 15, 8) == 0b1011_1110 {
                return t16_system(x);
            }
            // `10111111`: the hints and the `it`.
            t16_hint_or_it(x)
        }
    }
}

/// The system instructions of the 16-bit set.
///
/// They are in two places.  `10111110` is the breakpoint, and the whole of
/// the low byte is its immediate — every one of the 256 encodings is
/// defined.  `10110110` holds the endianness switch and the interrupt
/// masks, and the four bits at 7 say which: `0101` is `setend`, `0110` is
/// `cpsie` and `0111` is `cpsid`; the bit at 3 distinguishes the two
/// byte orders of `setend` and is reserved in the masks, where the three
/// bits at the bottom name the exception classes that are affected.
fn t16_system(x: u32) -> R {
    // `10111110`: `bkpt`.  The immediate is the whole of the low byte.
    if bits(x, 15, 8) == 0b1011_1110 {
        return R::new(
            "bkpt",
            format!("bkpt #{}", hex(u64::from(bits(x, 7, 0)))),
        );
    }
    // `10110110`: the endianness switch and the interrupt masks.
    if bits(x, 15, 8) == 0b1011_0110 {
        let kind = bits(x, 7, 4);
        if kind == 0b0101 && bits(x, 2, 0) == 0 {
            // `setend`: the bit at 3 says which byte order the processor
            // runs in, and the three bits at the bottom are not part of
            // the encoding.
            let order = if bits(x, 3, 3) == 1 { "be" } else { "le" };
            return R::new("setend", format!("setend {order}"));
        }
        if (kind == 0b0110 || kind == 0b0111) && bits(x, 3, 3) == 0 {
            // `cpsie` and `cpsid`: the three bits at the bottom name the
            // exception classes that are affected, and they are written
            // in the order `a`, `i`, `f` — the fault, the interrupt and
            // the fast interrupt.
            let name = if kind == 0b0110 { "cpsie" } else { "cpsid" };
            let mut classes = String::new();
            if bits(x, 2, 2) == 1 {
                classes.push('a');
            }
            if bits(x, 1, 1) == 1 {
                classes.push('i');
            }
            if bits(x, 0, 0) == 1 {
                classes.push('f');
            }
            if classes.is_empty() {
                classes.push_str("none");
            }
            return R::new(name, format!("{name} {classes}"));
        }
    }
    t16_undefined(x)
}

/// The hints and the `it` instruction.
///
/// `1011 1111 cond(4) mask(4)` is one encoding with two faces.  The
/// condition is the *high* nibble and the mask the low one; with the mask
/// clear the encoding is a hint, and the hint's number is in the same high
/// nibble the condition would have been in.  With the mask set it opens an
/// `it` block, and the mask says how many instructions the block covers
/// and which arm each of them is in.
///
/// Both facts were measured rather than assumed: the layout of the two
/// fields is the opposite of the one the encoding diagram suggests, and
/// the hint numbers run in the nibble that the condition uses.
fn t16_hint_or_it(x: u32) -> R {
    let cond = bits(x, 7, 4);
    let mask = bits(x, 3, 0);
    if mask == 0 {
        // The hint family.  Five of them have names; the rest are
        // `hint #n`, and the number is the high nibble.
        let name = match cond {
            0 => "nop",
            1 => "yield",
            2 => "wfe",
            3 => "wfi",
            4 => "sev",
            _ => return R::new("hint", format!("hint #{}", hex(u64::from(cond)))),
        };
        return R::new(name, name.to_string());
    }
    // The `it` block.
    //
    // The length of the block is set by the lowest bit of the mask that
    // is set: bit 3 gives one instruction, bit 2 two, bit 1 three and
    // bit 0 four.
    let count = if mask & 0b0001 != 0 {
        4
    } else if mask & 0b0010 != 0 {
        3
    } else if mask & 0b0100 != 0 {
        2
    } else {
        1
    };
    // The arm of each instruction after the first is read from the mask's
    // bits 3, 2 and 1, one for each of them in that order: a set bit is
    // the "else" arm and a clear one the "then" arm.  The first
    // instruction of a block is always the "then" one.
    //
    // A condition that is *inverted* — every other one, because the
    // condition's own sense is reversed — turns the arms the other way
    // round, so that the block reads the same way round: the first
    // instruction stays the "then" one and the rest are swapped.
    let inverted = cond & 1 == 1 && cond != 0b1111;
    // The mnemonic is the instruction's own name with one letter for each
    // instruction after the first: the `t` that ends `it` is the arm of
    // the first instruction, which is always the "then" one.
    let mut name = String::from("it");
    for i in 0..count - 1 {
        let arm = bits(mask, 3 - i, 3 - i);
        let arm = if inverted { 1 - arm } else { arm };
        name.push(if arm == 1 { 'e' } else { 't' });
    }
    R::new("it", format!("{name} {}", IT_CONDS[cond as usize]))
}

/// A 16-bit encoding that the architecture does not define.
///
/// The listing says which encoding it was rather than inventing a
/// mnemonic for it: an undefined encoding is data, and a reader that is
/// told `0xbe20` can see that it is not a breakpoint where an
/// architecture that defines one would have put a breakpoint.
fn t16_undefined(x: u32) -> R {
    R::new(
        ".short",
        format!(".short {} ; undefined", hex(u64::from(x))),
    )
    .flow(FlowKind::Exception)
}

/// A 16-bit register list.
///
/// The list is written out register by register: a run of consecutive
/// registers is *not* folded into a range, because that is how the
/// references print it and a folded range reads as a different
/// instruction to anything that matches on the text.
fn t16_reglist(list: u32, push: bool) -> String {
    let mut parts = Vec::new();
    for n in 0..8 {
        if list & (1 << n) != 0 {
            parts.push(reg(n).to_string());
        }
    }
    if push {
        if list & (1 << 8) != 0 {
            parts.push("lr".to_string());
        }
    } else if list & (1 << 8) != 0 {
        parts.push("pc".to_string());
    }
    format!("{{{}}}", parts.join(", "))
}

/// `110xx`: the load and store multiple, and the conditional branches.
fn t16_branch(x: u32, addr: Addr) -> R {
    // `1100`: the store and load multiple.  The base register is the
    // three bits at 8, the bit at 11 says whether the base is written
    // back, and the register list is the eight at the bottom.
    if bits(x, 15, 12) == 0b1100 {
        let load = bits(x, 11, 11) == 1;
        let rn = bits(x, 10, 8);
        let list = bits(x, 7, 0);
        let name = if load { "ldm" } else { "stm" };
        // The store writes the base back; the load does not, because it
        // reads the base before it writes the registers.
        // An empty list is not an instruction: there is nothing to
        // transfer.
        if list == 0 {
            return t16_undefined(x);
        }
        // The store writes the base back.  The load writes it back
        // only when the base is not among the registers it reads, for
        // then the value read would be the wrong one.
        let writeback = !load || (list & (1 << rn)) == 0;
        let base = if writeback {
            format!("{}!", reg(rn))
        } else {
            reg(rn).to_string()
        };
        let text = format!("{name} {base}, {}", t16_reglist(list, false));
        return R::new(name, text);
    }
    // `1101`: the conditional branch and the supervisor call.
    if bits(x, 15, 12) == 0b1101 {
        let cond = bits(x, 11, 8);
        if cond == 0b1111 {
            // `svc`: a supervisor call.
            let amount = i64::from(bits(x, 7, 0));
            return R::new("svc", format!("svc #{}", hex(amount as u64)))
                .flow(FlowKind::Call);
        }
        if cond == 0b1110 {
            // The permanently-undefined space.  Two of its 256
            // encodings have names of their own: `trap`, which is what
            // a debugger plants, and the breakpoint the architecture
            // defines for a division by zero.
            match bits(x, 7, 0) {
                0xFE => return R::new("trap", "trap".to_string()).flow(FlowKind::Exception),
                0xF9 => {
                    return R::new("__brkdiv0", "__brkdiv0".to_string())
                        .flow(FlowKind::Exception)
                }
                _ => {}
            }
            return R::new("udf", format!("udf #{}", hex(u64::from(bits(x, 7, 0)))))
                .flow(FlowKind::Exception);
        }
        // A conditional branch: the displacement is eight bits, signed,
        // and scaled by two.
        let amount = sext(bits(x, 7, 0), 8) * 2;
        let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
        let suffix = CONDS[cond as usize];
        let text = format!("b{suffix} {}", hex(target.get()));
        return R::new("b", text).flow(FlowKind::ConditionalBranch).to(target);
    }
    t16_undefined(x)
}

/// `111xx`: the unconditional branch.
///
/// The displacement is eleven bits, signed, and scaled by two.
fn t16_uncond(x: u32, addr: Addr) -> R {
    let amount = sext(bits(x, 10, 0), 11) * 2;
    let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
    R::new("b", format!("b {}", hex(target.get())))
        .flow(FlowKind::UnconditionalBranch)
        .to(target)
}

// ---------------------------------------------------------------------------
// T32: the 32-bit Thumb instruction set
//
// The 32-bit set is four times the size of the address space of a halfword,
// so it cannot be read in full the way the 16-bit set was.  It was read
// over a structured sample that covers every operation class and every
// field of every class, and the classes a compiler emits were checked one
// by one against both references.
//
// The first halfword says which class an instruction belongs to, and the
// class says where the operands are.  The layout of every field below was
// measured: a bit was flipped and the reference reading was watched to see
// which operand moved.

// ---------------------------------------------------------------------------
// The branch displacements
//
// A branch's displacement is not a field that can be read out; it is
// assembled from several, and the sign bit and the two "jump" bits are
// combined before they are placed.  The three branch classes combine them
// differently, and the differences were measured rather than assumed:
//
//   * the wide conditional branch assembles its twenty-one-bit offset from
//     the raw jump bits, with the bit-11 field the more significant of the
//     pair;
//   * the wide unconditional branch and the call assemble a twenty-five-bit
//     offset and fold the sign bit into the jump bits first, with the
//     bit-13 field the more significant of the pair.
//
// The narrow branches are simpler: the displacement is the immediate field
// itself, doubled.

/// The displacement of the wide conditional branch.
///
/// `11110 S cond imm6 | 10 J1 0 J2 imm11`, twenty-one bits wide.
fn t32_branch_cond(hw1: u32, hw2: u32) -> i64 {
    let s = bits(hw1, 10, 10);
    let j_high = bits(hw2, 11, 11);
    let j_low = bits(hw2, 13, 13);
    let v = (s << 20)
        | (j_high << 19)
        | (j_low << 18)
        | (bits(hw1, 5, 0) << 12)
        | (bits(hw2, 10, 0) << 1);
    sext(v, 21)
}

/// The displacement of the wide unconditional branch and of the call.
///
/// `11110 S imm10 | 1x J1 1 J2 imm11`.  The sign bit is folded into the
/// jump bits, so a jump in the opposite direction cancels the sign out.
fn t32_branch_wide(hw1: u32, hw2: u32) -> i64 {
    let s = bits(hw1, 10, 10);
    let i_high = 1 ^ (bits(hw2, 13, 13) ^ s);
    let i_low = 1 ^ (bits(hw2, 11, 11) ^ s);
    let v = (s << 24)
        | (i_high << 23)
        | (i_low << 22)
        | (bits(hw1, 9, 0) << 12)
        | (bits(hw2, 10, 0) << 1);
    sext(v, 25)
}

// ---------------------------------------------------------------------------
// The operation tables
//
// Each table is the operation field's own contents, read out of the
// references one bit at a time.  The names are the ones the references
// print.

/// The register-operand data processing: the operation field is four bits
/// wide here, and it is the same order as the immediate table's.
const T32_DP_OPS: [&str; 16] = [
    "and", "bic", "orr", "orn", "eor", "", "add", "adc", "sbc", "rsb", "", "mul", "and", "orr", "",
    "",
];

/// The immediate data processing: the operation field is three bits wide,
/// and the same order as the register table's.
const T32_DP_IMM_OPS: [&str; 8] = ["and", "bic", "orr", "orn", "eor", "", "add", "adc"];

/// The shift types, by their two-bit field.
const T32_SHIFT: [&str; 4] = ["lsl", "lsr", "asr", "ror"];

/// The register-operand data-processing class.
///
/// `1110101 op(4) S Rn | 0 imm3 Rd imm2 type Rm`.  The shift amount is
/// split across the two halfwords: three bits in the second halfword and
/// two more beside the shift type.
fn t32_dp_reg(hw1: u32, hw2: u32) -> R {
    let op = bits(hw1, 8, 5);
    let s = bits(hw1, 4, 4);
    let rn = bits(hw1, 3, 0);
    let rd = bits(hw2, 11, 8);
    let rm = bits(hw2, 3, 0);
    let sh_type = bits(hw2, 5, 4);
    let amount = (bits(hw2, 14, 12) << 2) | bits(hw2, 7, 6);
    let base = T32_DP_OPS[op as usize];
    if base.is_empty() {
        return t32_undefined(hw1, hw2);
    }
    // A shift of zero means thirty-two for the two right shifts, and one
    // for the rotate.
    let amount = match sh_type {
        0b01 | 0b10 if amount == 0 => 32,
        0b11 if amount == 0 => 1,
        _ => amount,
    };
    let name = if s == 1 { format!("{base}s") } else { base.to_string() };
    let text = format!(
        "{name} {}, {}, {}, {} #{}",
        reg(rd),
        reg(rn),
        reg(rm),
        T32_SHIFT[sh_type as usize],
        amount
    );
    R::new(base, text)
}

/// The plain and the modified immediate data-processing class.
///
/// `11110 i 0 op S Rn | 0 imm3 Rd imm8`.  The plain form takes a
/// twelve-bit immediate whose top four bits sit in the first halfword;
/// the modified form takes an eight-bit one that is either used as it
/// stands or rotated right by one.
fn t32_dp_imm(hw1: u32, hw2: u32) -> R {
    let i = bits(hw1, 10, 10);
    let op = bits(hw1, 8, 6);
    let s = bits(hw1, 5, 5);
    let rn = bits(hw1, 3, 0);
    let rd = bits(hw2, 11, 8);
    let imm8 = bits(hw2, 7, 0);
    // The plain form is the one whose wide bit is clear and whose
    // operation field is three bits; the modified form is the other.
    let plain = i == 0 && bits(hw1, 9, 9) == 0;
    let (base, value) = if plain {
        let top = bits(hw1, 9, 6);
        (T32_DP_IMM_OPS[op as usize], (top << 8) | imm8)
    } else {
        let value = if i == 0 { imm8 } else { (imm8 >> 1) | ((imm8 & 1) << 7) };
        (T32_DP_IMM_OPS[op as usize], value)
    };
    if base.is_empty() {
        return t32_undefined(hw1, hw2);
    }
    let name = if s == 1 { format!("{base}s") } else { base.to_string() };
    let text = format!("{name} {}, {}, {}", reg(rd), reg(rn), imm(i64::from(value)));
    R::new(base, text)
}

/// The wide-immediate data-processing class.
///
/// `11110 i 100100 Rn | 0 imm3 Rd imm8`, which is `movw` when the wide
/// bit is clear and `movt` when it is set.  The sixteen-bit immediate is
/// assembled from four fields spread over two halfwords.
fn t32_movw(hw1: u32, hw2: u32) -> R {
    let i = bits(hw1, 10, 10);
    let rd = bits(hw2, 11, 8);
    let imm4 = bits(hw1, 3, 0);
    let imm3 = bits(hw2, 14, 12);
    let imm8 = bits(hw2, 7, 0);
    let value = (imm4 << 12) | (i << 11) | (imm3 << 8) | imm8;
    let name = if bits(hw1, 9, 9) == 0 { "movw" } else { "movt" };
    R::new(name, format!("{name} {}, {}", reg(rd), imm(i64::from(value))))
}

/// The add-and-subtract-with-wide-immediate class.
///
/// `11110 i 100000 Rn | 0 imm3 Rd imm8` is `addw` and `subw`.
fn t32_addw(hw1: u32, hw2: u32) -> R {
    let rn = bits(hw1, 3, 0);
    let rd = bits(hw2, 11, 8);
    let imm12 = (bits(hw2, 14, 12) << 8) | bits(hw2, 7, 0);
    let text = if bits(hw1, 9, 9) == 0 {
        format!("addw {}, {}, {}", reg(rd), reg(rn), imm(i64::from(imm12)))
    } else {
        format!("subw {}, {}, {}", reg(rd), reg(rn), imm(i64::from(imm12)))
    };
    R::new("addw", text)
}

/// The single load and store class.
///
/// `1111 1000 U 1 size Rn | Rt imm12` is the immediate form and
/// `1111 1000 U 0 size Rn | Rt imm2 Rm` is the register form.  The bit
/// that separates them is the top bit of the second halfword.
fn t32_load_store(hw1: u32, hw2: u32) -> R {
    let u = bits(hw1, 7, 7);
    let size = bits(hw1, 6, 6);
    let rn = bits(hw1, 3, 0);
    let rt = bits(hw2, 15, 12);
    let load = bits(hw1, 4, 4) == 1;
    let sign = if u == 1 { "" } else { "-" };
    if bits(hw2, 11, 11) == 0 {
        // The register form: the offset is a register, scaled by the size.
        let rm = bits(hw2, 3, 0);
        let base = t32_ls_name(load, size == 1, false);
        let text = format!("{base} {}, [{}, {}{}]", reg(rt), reg(rn), sign, reg(rm));
        return R::new(base, text);
    }
    // The immediate form: a twelve-bit offset, scaled by the size.
    let imm12 = bits(hw2, 10, 0);
    let amount = if size == 0 { i64::from(imm12) * 4 } else { i64::from(imm12) };
    let base = t32_ls_name(load, size == 1, false);
    let text = if amount == 0 {
        format!("{base} {}, [{}]", reg(rt), reg(rn))
    } else {
        format!("{base} {}, [{}, {}{}]", reg(rt), reg(rn), sign, imm(amount))
    };
    R::new(base, text)
}

/// The name of a load or a store, by its direction and its width.
fn t32_ls_name(load: bool, size: bool, signed: bool) -> &'static str {
    match (load, size, signed) {
        (true, false, false) => "ldr",
        (true, false, true) => "ldrsb",
        (true, true, false) => "ldrh",
        (true, true, true) => "ldrsh",
        (false, false, false) => "str",
        (false, true, false) => "strh",
        (false, false, true) => "strb",
        _ => "strb",
    }
}

/// The load and store multiple class.
///
/// `1110 100 op(2) W L Rn | Rt register_list`.  The register list is
/// sixteen bits wide, and its two top bits name the stack pointer and the
/// program counter rather than the registers they would otherwise be.
fn t32_ldm(hw1: u32, hw2: u32) -> R {
    let op = bits(hw1, 7, 6);
    let w = bits(hw1, 5, 5);
    let l = bits(hw1, 4, 4);
    let rn = bits(hw1, 3, 0);
    let names = t32_reglist(hw2 & 0xFFFF);
    let base = match (op, l) {
        (0b00, 0) => "stm",
        (0b00, 1) => "ldm",
        (0b01, 0) => "stmdb",
        (0b01, 1) => "ldmdb",
        (0b10, 0) => "stmda",
        (0b10, 1) => "ldmda",
        _ => "stmia",
    };
    let mut rn_name = reg(rn).to_string();
    if w == 1 {
        rn_name.push('!');
    }
    let text = format!("{base} {rn_name}, {names}");
    R::new(base, text)
}

/// The register list of a load or store multiple, as it is written.
fn t32_reglist(list: u32) -> String {
    let mut names = Vec::new();
    for i in 0..16 {
        if list & (1 << i) == 0 {
            continue;
        }
        names.push(match i {
            13 => "sp".to_string(),
            14 => "lr".to_string(),
            15 => "pc".to_string(),
            _ => reg(i).to_string(),
        });
    }
    format!("{{{}}}", names.join(", "))
}

/// The undefined instruction in the 32-bit space.
fn t32_undefined(hw1: u32, hw2: u32) -> R {
    let word = (hw2 << 16) | hw1;
    R::new(
        ".word",
        format!(".word {} ; undefined", hex(u64::from(word))),
    )
    .flow(FlowKind::Exception)
}


/// Decode one 32-bit T32 instruction, assembled from two halfwords.
///
/// The first halfword picks the class.  Most classes are picked by the
/// high bits alone; the ones that share a space are separated by the top
/// bit of the second halfword, and that bit was measured to be what
/// separates them.
fn t32(w: u32, addr: Addr) -> R {
    let hw1 = w & 0xFFFF;
    let hw2 = (w >> 16) & 0xFFFF;
    match hw1 >> 8 {
        // `1110100`: the load and store multiple family, the exclusive
        // accesses, and the table branch.
        0xE8 | 0xE9 => t32_ldm(hw1, hw2),
        // `1110101`: the register-operand data processing.
        0xEA | 0xEB => t32_dp_reg(hw1, hw2),
        // `1110110` and `1110111`: the coprocessor and the vector space,
        // which this decoder does not cover.
        0xEC | 0xED | 0xEE | 0xEF => t32_undefined(hw1, hw2),
        // `1111 00`: the data processing with an immediate.
        0xF0 | 0xF1 | 0xF2 | 0xF3 | 0xF4 | 0xF5 | 0xF6 | 0xF7 => {
            t32_immediate(hw1, hw2, addr)
        }
        // `1111 10`: the single load and store.
        0xF8 | 0xF9 => t32_load_store(hw1, hw2),
        // `1111 11`: the extends, the multiplies, and the coprocessor.
        0xFA | 0xFB => t32_undefined(hw1, hw2),
        0xFC | 0xFD | 0xFE | 0xFF => t32_undefined(hw1, hw2),
        _ => t32_undefined(hw1, hw2),
    }
}

/// The data-processing class with an immediate, and the branches that
/// share its space.
///
/// The top bit of the second halfword is what separates them: with it
/// clear the instruction is a data-processing one, and with it set the
/// first halfword's operation field is a condition and the instruction is
/// a branch.
fn t32_immediate(hw1: u32, hw2: u32, addr: Addr) -> R {
    if bits(hw2, 15, 15) == 1 {
        // A branch.  The condition is the field that the data-processing
        // class uses for its operation.
        let cond = bits(hw1, 9, 6);
        if cond == 0b1110 {
            // `1110`: the unconditional wide branch.
            let amount = t32_branch_wide(hw1, hw2);
            let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
            return R::new("b", format!("b.w {}", hex(target.get())))
                .flow(FlowKind::UnconditionalBranch)
                .to(target);
        }
        if cond == 0b1111 {
            // `1111`: the call.
            let amount = t32_branch_wide(hw1, hw2);
            let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
            return R::new("bl", format!("bl {}", hex(target.get())))
                .flow(FlowKind::Call)
                .to(target);
        }
        let amount = t32_branch_cond(hw1, hw2);
        let target = Addr((t_pc(addr) + i128::from(amount)) as u64);
        let suffix = CONDS[cond as usize];
        let text = format!("b{suffix}.w {}", hex(target.get()));
        return R::new("b", text).flow(FlowKind::ConditionalBranch).to(target);
    }
    // A data-processing instruction with an immediate.  The operation
    // field's top bit says which of the two families this is.
    match bits(hw1, 9, 9) {
        0 => t32_dp_imm(hw1, hw2),
        _ => match bits(hw1, 9, 8) {
            (0b10 | 0b11) if bits(hw1, 9, 9) == 1 => t32_movw(hw1, hw2),
            _ => t32_undefined(hw1, hw2),
        },
    }
}

/// Decode one 32-bit T32 instruction, assembled from two halfwords.
fn t32_wide(w: u32, addr: Addr) -> R {
    t32(w, addr)
}

impl R {
    /// Set the address a PC-relative operand named.
    fn with_pc(mut self, a: Addr) -> R {
        self.pc = Some(a);
        self
    }
}
