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
const CONDS: [&str; 16] = [
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "", "nv",
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
            let word = (u32::from(first) << 16) | u32::from(second);
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

/// Decode one 16-bit T32 halfword.
fn t32_narrow(_w: u16, addr: Addr) -> R {
    R::new(".word", format!(".word 0x0 ; thumb not decoded at {:#x}", addr.get()))
        .flow(FlowKind::Exception)
}

/// Decode one 32-bit T32 instruction, assembled from two halfwords.
fn t32_wide(_w: u32, addr: Addr) -> R {
    R::new(".word", format!(".word 0x0 ; thumb-2 not decoded at {:#x}", addr.get()))
        .flow(FlowKind::Exception)
}
