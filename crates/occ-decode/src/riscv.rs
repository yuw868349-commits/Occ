//! RISC-V (RV32/RV64) decoding.
//!
//! The base integer ISA plus the compressed extension and the M (multiply /
//! divide) extension. Variable-length encoding is handled the only safe way:
//! the low two bits decide whether the unit is 16 or 32 bits wide, and a unit
//! wider than 32 bits is emitted as a `.word` with its length taken from the
//! spec, so the following instruction still lands where the hardware would
//! start it.
//!
//! Text uses the assembly spelling a compiler emits (`addi sp,sp,-16`,
//! `jal ra,0x1234`, `sd ra,8(sp)`), including the pseudo-instruction aliases
//! `ret`, `nop`, `mv`, `li`, `j`, `jr`, `call`. Aliases are not cosmetic: they
//! are what a reader and a model both recognise, and they collapse three
//! operand slots into one fact.

use occ_core::{Addr, Arch, CallTarget, FlowKind, Instruction, MAX_INSN_LEN};

use crate::Decoded;

/// Integer register names, ABI spelling.
const REGS: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

const BRANCHES: [&str; 8] = ["beq", "bne", "?", "?", "blt", "bge", "bltu", "bgeu"];
const LOADS: [&str; 8] = ["lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "?"];
const STORES: [&str; 8] = ["sb", "sh", "sw", "sd", "?", "?", "?", "?"];
const OPIMM: [&str; 8] = ["addi", "slli", "slti", "sltiu", "xori", "srli", "ori", "andi"];
const OP: [&str; 8] = ["add", "sll", "slt", "sltu", "xor", "srl", "or", "and"];
const MULDIV: [&str; 8] = ["mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"];
const CSR: [&str; 8] = ["?", "csrrw", "csrrs", "csrrc", "?", "csrrwi", "csrrsi", "csrrci"];

const fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

const fn sext(v: u32, n: u32) -> i64 {
    let shift = 64 - n;
    ((v as i64) << shift) >> shift
}

fn reg(n: u32) -> &'static str {
    REGS[n as usize & 31]
}

/// Decoder bound to one width.
#[derive(Clone, Copy, Debug)]
pub struct Decoder {
    bits: u8,
}

/// Intermediate rendering, shared with the AArch64 module in spirit.
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

    fn br(mut self, to: Addr, flow: FlowKind) -> R {
        self.flow = flow;
        self.pc = Some(to);
        // Calls name their callee, branches name their destination; both are
        // direct targets as far as a caller is concerned.
        if matches!(
            flow,
            FlowKind::Call | FlowKind::UnconditionalBranch | FlowKind::ConditionalBranch
        ) {
            self.target = Some(CallTarget::Direct { to });
        }
        self
    }
}

impl Decoder {
    /// Build a decoder for 32- or 64-bit RISC-V.
    pub const fn new(bits: u8) -> Option<Self> {
        match bits {
            32 | 64 => Some(Decoder { bits }),
            _ => None,
        }
    }

    /// Architecture label.
    pub const fn arch(self) -> Arch {
        if self.bits == 64 {
            Arch::RiscV64
        } else {
            Arch::RiscV32
        }
    }

    /// Decode one instruction unit at `origin`.
    pub fn decode(&self, origin: Addr, bytes: &[u8]) -> Option<Decoded> {
        if bytes.len() < 2 {
            return None;
        }
        let half = u16::from_le_bytes([bytes[0], bytes[1]]);
        if half & 0b11 != 0b11 {
            return Some(self.compressed(origin, half));
        }
        if bytes.len() < 4 {
            return None;
        }
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Some(self.full(origin, word))
    }

    /// Decode a 32-bit unit.
    pub fn full(&self, origin: Addr, word: u32) -> Decoded {
        let r = self.decode32(origin, word);
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
                mem_writes: r.writes,
            },
            rip_target: r.pc,
        }
    }

    /// Decode a 16-bit compressed unit.
    pub fn compressed(&self, origin: Addr, half: u16) -> Decoded {
        let r = self.decode16(origin, half);
        let mut raw = [0u8; MAX_INSN_LEN];
        raw[..2].copy_from_slice(&half.to_le_bytes());
        Decoded {
            insn: Instruction {
                addr: origin,
                len: 2,
                bytes: raw,
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

    fn decode32(&self, origin: Addr, w: u32) -> R {
        let opcode = w & 0x7f;
        let rd = bits(w, 11, 7);
        let funct3 = bits(w, 14, 12);
        let rs1 = bits(w, 19, 15);
        let rs2 = bits(w, 24, 20);
        let funct7 = bits(w, 31, 25);

        match opcode {
            0x37 => {
                // LUI
                let v = sext(w & 0xffff_f000, 32);
                R::new("lui", format!("lui {}, {:#x}", reg(rd), v >> 12))
            }
            0x17 => {
                let v = sext(w & 0xffff_f000, 32);
                R::new("auipc", format!("auipc {}, {:#x}", reg(rd), v >> 12))
            }
            0x6f => {
                let imm = bits(w, 31, 31) << 20
                    | bits(w, 19, 12) << 12
                    | bits(w, 20, 20) << 11
                    | bits(w, 30, 21) << 1;
                let off = sext(imm, 21);
                let to = origin.offset(off);
                if rd == 0 {
                    R::new("j", format!("j {to}")).br(to, FlowKind::UnconditionalBranch)
                } else if rd == 1 {
                    R::new("jal", format!("jal {to}")).br(to, FlowKind::Call)
                } else {
                    R::new("jal", format!("jal {}, {to}", reg(rd))).br(to, FlowKind::Call)
                }
            }
            0x67 => {
                let imm = sext(bits(w, 31, 20), 12);
                if rd == 0 && rs1 == 1 && imm == 0 {
                    return R::new("ret", "ret".into()).with_flow(FlowKind::Return);
                }
                if rd == 0 {
                    return R::new("jr", format!("jr {}", reg(rs1)))
                        .with_flow(FlowKind::IndirectBranch);
                }
                if rd == 1 && imm == 0 {
                    return R::new("jalr", format!("jalr {}", reg(rs1)))
                        .with_flow(FlowKind::Call)
                        .with_target(CallTarget::Indirect { via: None });
                }
                let mut text = format!("jalr {}, {}({})", reg(rd), imm, reg(rs1));
                if imm == 0 {
                    text = format!("jalr {}, ({})", reg(rd), reg(rs1));
                }
                R::new("jalr", text)
                    .with_flow(FlowKind::IndirectCall)
                    .with_target(CallTarget::Indirect { via: None })
            }
            0x63 => {
                let imm = bits(w, 31, 31) << 12
                    | bits(w, 7, 7) << 11
                    | bits(w, 30, 25) << 5
                    | bits(w, 11, 8) << 1;
                let to = origin.offset(sext(imm, 13));
                let mn = BRANCHES[funct3 as usize];
                if mn == "?" {
                    return undefined(origin, w, "reserved branch");
                }
                // Comparison against zero has its own spelling, and compilers
                // emit it constantly; the two-register form is the encoding,
                // the alias is what a reader recognises.
                //
                // The alias keeps the comparison's direction: `blt a0, zero`
                // is "a0 is less than zero", which is `bltz`, not `bgtz`.
                let alias = match (mn, rs1, rs2) {
                    ("beq", 0, _) => Some(("beqz", reg(rs2))),
                    ("bne", 0, _) => Some(("bnez", reg(rs2))),
                    ("blt", _, 0) => Some(("bltz", reg(rs1))),
                    ("bge", _, 0) => Some(("bgez", reg(rs1))),
                    ("bltu", _, 0) => Some(("bltuz", reg(rs1))),
                    ("bgeu", _, 0) => Some(("bgeuz", reg(rs1))),
                    _ => None,
                };
                if let Some((a, one)) = alias {
                    return R::new(a, format!("{a} {one}, {to}"))
                        .br(to, FlowKind::ConditionalBranch);
                }
                R::new(mn, format!("{mn} {}, {}, {to}", reg(rs1), reg(rs2)))
                    .br(to, FlowKind::ConditionalBranch)
            }
            0x03 => {
                let mn = LOADS[funct3 as usize];
                if mn == "?" {
                    return undefined(origin, w, "reserved load");
                }
                let imm = sext(bits(w, 31, 20), 12);
                R::new(mn, format!("{mn} {}, {}({})", reg(rd), imm, reg(rs1)))
            }
            0x23 => {
                let mn = STORES[funct3 as usize];
                if mn == "?" {
                    return undefined(origin, w, "reserved store");
                }
                let imm = sext(bits(w, 31, 25) << 5 | bits(w, 11, 7), 12);
                R::new(mn, format!("{mn} {}, {}({})", reg(rs2), imm, reg(rs1)))
            }
            0x13 => {
                let imm = sext(bits(w, 31, 20), 12);
                match funct3 {
                    0 => {
                        if rd == 0 && rs1 == 0 && imm == 0 {
                            return R::new("nop", "nop".into());
                        }
                        if rs1 == 0 {
                            return R::new("li", format!("li {}, {}", reg(rd), imm));
                        }
                        if imm == 0 {
                            return R::new("mv", format!("mv {}, {}", reg(rd), reg(rs1)));
                        }
                        R::new("addi", format!("addi {}, {}, {}", reg(rd), reg(rs1), imm))
                    }
                    1 => {
                        let sh = bits(w, 25, 20);
                        R::new("slli", format!("slli {}, {}, {}", reg(rd), reg(rs1), sh))
                    }
                    5 => {
                        let sh = bits(w, 25, 20);
                        if funct7 & 0b0100000 != 0 {
                            R::new("srai", format!("srai {}, {}, {}", reg(rd), reg(rs1), sh))
                        } else {
                            R::new("srli", format!("srli {}, {}, {}", reg(rd), reg(rs1), sh))
                        }
                    }
                    _ => R::new(
                        OPIMM[funct3 as usize],
                        format!("{} {}, {}, {}", OPIMM[funct3 as usize], reg(rd), reg(rs1), imm),
                    ),
                }
            }
            0x33 => {
                let alt = funct7 & 0b0100000 != 0;
                let mn = if funct7 & 0b0000001 != 0 {
                    MULDIV[funct3 as usize]
                } else if alt {
                    match funct3 {
                        0 => "sub",
                        5 => "sra",
                        _ => OP[funct3 as usize],
                    }
                } else {
                    OP[funct3 as usize]
                };
                R::new(
                    mn,
                    format!("{mn} {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
                )
            }
            0x1b => {
                let imm = sext(bits(w, 31, 20), 12);
                if funct3 == 0 {
                    if rs1 == 0 {
                        return R::new("li", format!("li {}, {}", reg(rd), imm));
                    }
                    R::new("addiw", format!("addiw {}, {}, {}", reg(rd), reg(rs1), imm))
                } else {
                    let sh = bits(w, 24, 20);
                    let mn = match funct3 {
                        1 => "slliw",
                        5 => {
                            if funct7 & 0b0100000 != 0 {
                                "sraiw"
                            } else {
                                "srliw"
                            }
                        }
                        _ => "?",
                    };
                    if mn == "?" {
                        return undefined(origin, w, "reserved op-imm-32");
                    }
                    R::new(mn, format!("{mn} {}, {}, {}", reg(rd), reg(rs1), sh))
                }
            }
            0x3b => {
                let alt = funct7 & 0b0100000 != 0;
                let mn = if funct7 & 0b0000001 != 0 {
                    match funct3 {
                        0 => "mulw",
                        4 => "divw",
                        5 => "divuw",
                        6 => "remw",
                        7 => "remuw",
                        _ => "?",
                    }
                } else if alt {
                    match funct3 {
                        0 => "subw",
                        5 => "sraw",
                        _ => "?",
                    }
                } else {
                    match funct3 {
                        0 => "addw",
                        1 => "sllw",
                        5 => "srlw",
                        _ => "?",
                    }
                };
                if mn == "?" {
                    return undefined(origin, w, "reserved op-32");
                }
                R::new(mn, format!("{mn} {}, {}, {}", reg(rd), reg(rs1), reg(rs2)))
            }
            0x0f => {
                let pred = bits(w, 27, 24);
                let succ = bits(w, 23, 20);
                R::new(
                    "fence",
                    format!("fence {:#x}, {:#x}", pred, succ),
                )
            }
            0x73 => match funct3 {
                0 => {
                    let imm = bits(w, 31, 20);
                    let mn = match imm {
                        0 => "ecall",
                        1 => "ebreak",
                        0x102 => "sret",
                        0x302 => "mret",
                        0x105 => "wfi",
                        _ => "system",
                    };
                    if mn == "system" {
                        undefined(origin, w, "reserved system")
                    } else {
                        R::new(mn, mn.to_string()).with_flow(if mn == "ebreak" {
                            FlowKind::Exception
                        } else {
                            FlowKind::Interrupt
                        })
                    }
                }
                f => {
                    let mn = CSR[f as usize];
                    if mn == "?" {
                        return undefined(origin, w, "reserved csr");
                    }
                    let csr = bits(w, 31, 20);
                    if f >= 5 {
                        R::new(
                            mn,
                            format!("{mn} {}, {:#x}, {}", reg(rd), csr, rs1),
                        )
                    } else if rd == 0 && f == 2 {
                        R::new("csrr", format!("csrr {}, {:#x}", reg(rd), csr))
                    } else {
                        R::new(
                            mn,
                            format!("{mn} {}, {:#x}, {}", reg(rd), csr, reg(rs1)),
                        )
                    }
                }
            },
            0x2f => {
                // Atomic memory operations, RV64/RV32 word and doubleword.
                let funct5 = bits(w, 31, 27);
                let aqrl = bits(w, 26, 25);
                let suffix = if funct3 == 3 { "d" } else { "w" };
                let mn: &'static str = match funct5 {
                    0b00010 => "lr",
                    0b00011 => "sc",
                    0b00001 => "amoswap",
                    0b00000 => "amoadd",
                    0b00100 => "amoxor",
                    0b01100 => "amoand",
                    0b01000 => "amoor",
                    0b10000 => "amomin",
                    0b10100 => "amomax",
                    0b11000 => "amominu",
                    0b11100 => "amomaxu",
                    _ => return undefined(origin, w, "reserved amo"),
                };
                let name = format!("{mn}.{suffix}{}", if aqrl == 3 { ".aqrl" } else { "" });
                let text = if funct5 == 0b00010 {
                    format!("{name} {}, ({})", reg(rd), reg(rs1))
                } else {
                    format!("{name} {}, {}, ({})", reg(rd), reg(rs2), reg(rs1))
                };
                // The mnemonic field is a fixed &'static str, so the suffix is
                // dropped there and kept in the text.
                R::new(mn, text)
            }
            _ => undefined(origin, w, "unknown opcode"),
        }
    }

    fn decode16(&self, origin: Addr, h: u16) -> R {
        let w = u32::from(h);
        let op = w & 0b11;
        let funct3 = bits(w, 15, 13);
        match op {
            0 => {
                let rd_p = 8 + bits(w, 4, 2);
                let rs1_p = 8 + bits(w, 9, 7);
                match funct3 {
                    0 => {
                        let imm = ((w >> 11) & 0x3) << 4
                            | ((w >> 7) & 0xf) << 6
                            | ((w >> 6) & 1) << 2
                            | ((w >> 5) & 1) << 3;
                        R::new(
                            "addi",
                            format!("addi {}, {}, {}", reg(rd_p), reg(rs1_p & 31), imm),
                        )
                    }
                    2 => {
                        let imm = ((w >> 10) & 0x7) << 3 | ((w >> 6) & 1) << 2 | ((w >> 5) & 1) << 6;
                        R::new(
                            "lw",
                            format!("lw {}, {}({})", reg(rd_p), imm, reg(rs1_p)),
                        )
                    }
                    3 => {
                        if self.bits == 64 {
                            let imm = ((w >> 10) & 0x7) << 3 | ((w >> 5) & 0x3) << 6;
                            R::new(
                                "ld",
                                format!("ld {}, {}({})", reg(rd_p), imm, reg(rs1_p)),
                            )
                        } else {
                            undefined(origin, w, "c.flw")
                        }
                    }
                    6 => {
                        let imm = ((w >> 10) & 0x7) << 3 | ((w >> 6) & 1) << 2 | ((w >> 5) & 1) << 6;
                        R::new(
                            "sw",
                            format!("sw {}, {}({})", reg(rd_p), imm, reg(rs1_p)),
                        )
                    }
                    7 => {
                        let imm = ((w >> 10) & 0x7) << 3 | ((w >> 5) & 0x3) << 6;
                        R::new(
                            "sd",
                            format!("sd {}, {}({})", reg(rd_p), imm, reg(rs1_p)),
                        )
                    }
                    _ => undefined(origin, w, "c.reserved"),
                }
            }
            1 => {
                let rd = bits(w, 11, 7);
                let imm6 = sext(bits(w, 12, 12) << 5 | bits(w, 6, 2), 6);
                match funct3 {
                    0 => {
                        if rd == 0 {
                            return R::new("nop", "nop".into());
                        }
                        R::new("addi", format!("addi {}, {}, {}", reg(rd), reg(rd), imm6))
                    }
                    1 => {
                        if self.bits == 32 {
                            let imm = sext(bits(w, 12, 12) << 11
                                | bits(w, 8, 8) << 10
                                | bits(w, 10, 9) << 8
                                | bits(w, 6, 6) << 7
                                | bits(w, 7, 7) << 6
                                | bits(w, 2, 2) << 5
                                | bits(w, 11, 11) << 4
                                | bits(w, 5, 3) << 1, 12);
                            let to = origin.offset(imm);
                            return R::new("jal", format!("jal {to}")).br(to, FlowKind::Call);
                        }
                        R::new("addiw", format!("addiw {}, {}, {}", reg(rd), reg(rd), imm6))
                    }
                    2 => {
                        if rd == 0 {
                            return R::new("nop", "nop".into());
                        }
                        R::new("li", format!("li {}, {}", reg(rd), imm6))
                    }
                    3 => {
                        if rd == 2 {
                            let imm = sext(
                                bits(w, 12, 12) << 9
                                    | bits(w, 6, 6) << 4
                                    | bits(w, 5, 5) << 6
                                    | bits(w, 4, 3) << 7
                                    | bits(w, 2, 2) << 5,
                                10,
                            );
                            return R::new(
                                "addi",
                                format!("addi sp, sp, {}", imm),
                            );
                        }
                        let imm = sext(bits(w, 12, 12) << 5 | bits(w, 6, 2), 6) << 12;
                        R::new("lui", format!("lui {}, {:#x}", reg(rd), imm >> 12))
                    }
                    4 => {
                        let rd_p = 8 + bits(w, 9, 7);
                        let rs2_p = 8 + bits(w, 4, 2);
                        let sub = bits(w, 11, 10);
                        let sh = bits(w, 12, 12) << 5 | bits(w, 6, 2);
                        match sub {
                            0 => R::new("srli", format!("srli {}, {}, {}", reg(rd_p), reg(rd_p), sh)),
                            1 => R::new("srai", format!("srai {}, {}, {}", reg(rd_p), reg(rd_p), sh)),
                            2 => {
                                let imm = sext(bits(w, 12, 12) << 5 | bits(w, 6, 2), 6);
                                R::new(
                                    "andi",
                                    format!("andi {}, {}, {}", reg(rd_p), reg(rd_p), imm),
                                )
                            }
                            _ => {
                                // Two distinct shapes share these bits: with
                                // bit 12 clear, `rs2'` selects the operation;
                                // with it set, bits 6..5 select the word form
                                // and `rs2'` is the source register.
                                let bit12 = w & 0x1000 != 0;
                                let mn = if bit12 {
                                    match bits(w, 6, 5) {
                                        0 => "subw",
                                        1 => "addw",
                                        _ => return undefined(origin, w, "c.reserved-alu"),
                                    }
                                } else {
                                    match rs2_p {
                                        0 => "sub",
                                        1 => "xor",
                                        2 => "or",
                                        _ => "and",
                                    }
                                };
                                R::new(mn, format!("{mn} {}, {}", reg(rd_p), reg(rs2_p)))
                            }
                        }
                    }
                    5 => {
                        let imm = sext(
                            bits(w, 12, 12) << 11
                                | bits(w, 8, 8) << 10
                                | bits(w, 10, 9) << 8
                                | bits(w, 6, 6) << 7
                                | bits(w, 7, 7) << 6
                                | bits(w, 2, 2) << 5
                                | bits(w, 11, 11) << 4
                                | bits(w, 5, 3) << 1,
                            12,
                        );
                        let to = origin.offset(imm);
                        R::new("j", format!("j {to}")).br(to, FlowKind::UnconditionalBranch)
                    }
                    6 | 7 => {
                        let rs1_p = 8 + bits(w, 9, 7);
                        let imm = sext(
                            bits(w, 12, 12) << 8
                                | bits(w, 11, 10) << 3
                                | bits(w, 6, 5) << 6
                                | bits(w, 4, 3) << 1
                                | bits(w, 2, 2) << 5,
                            9,
                        );
                        let to = origin.offset(imm);
                        let mn = if funct3 == 6 { "beqz" } else { "bnez" };
                        R::new(mn, format!("{mn} {}, {to}", reg(rs1_p)))
                            .br(to, FlowKind::ConditionalBranch)
                    }
                    _ => undefined(origin, w, "c.reserved"),
                }
            }
            2 => {
                let rd = bits(w, 11, 7);
                let rs2 = bits(w, 6, 2);
                match funct3 {
                    0 => {
                        let sh = bits(w, 12, 12) << 5 | bits(w, 6, 2);
                        R::new("slli", format!("slli {}, {}, {}", reg(rd), reg(rd), sh))
                    }
                    2 => {
                        let imm =
                            bits(w, 12, 12) << 5 | bits(w, 6, 4) << 2 | bits(w, 3, 2) << 6;
                        R::new("lw", format!("lw {}, {}(sp)", reg(rd), imm))
                    }
                    3 => {
                        if self.bits == 64 {
                            let imm =
                                bits(w, 12, 12) << 5 | bits(w, 6, 5) << 3 | bits(w, 4, 2) << 6;
                            R::new("ld", format!("ld {}, {}(sp)", reg(rd), imm))
                        } else {
                            undefined(origin, w, "c.flwsp")
                        }
                    }
                    4 => {
                        let bit12 = w & 0x1000 != 0;
                        if !bit12 {
                            if rs2 == 0 {
                                if rd == 0 {
                                    return R::new("ebreak", "ebreak".into())
                                        .with_flow(FlowKind::Exception);
                                }
                                // `jr ra` is the return, spelled as a jump to
                                // the return address. The encoding is a jump
                                // to `x1`, and both spellings mean the same
                                // thing, so the decoder has to recognise it
                                // rather than report a jump whose target it
                                // cannot know.
                                if rd == 1 {
                                    return R::new("ret", "ret".into())
                                        .with_flow(FlowKind::Return);
                                }
                                return R::new("jr", format!("jr {}", reg(rd)))
                                    .with_flow(FlowKind::IndirectBranch);
                            }
                            if rd == 0 {
                                return R::new("mv", format!("mv {}, {}", reg(rd), reg(rs2)));
                            }
                            R::new("mv", format!("mv {}, {}", reg(rd), reg(rs2)))
                        } else if rs2 == 0 {
                            if rd == 0 {
                                return R::new("ebreak", "ebreak".into())
                                    .with_flow(FlowKind::Exception);
                            }
                            R::new("jalr", format!("jalr {}", reg(rd)))
                                .with_flow(FlowKind::IndirectCall)
                                .with_target(CallTarget::Indirect { via: None })
                        } else {
                            R::new("add", format!("add {}, {}", reg(rd), reg(rs2)))
                        }
                    }
                    6 => {
                        let imm = bits(w, 12, 9) << 2 | bits(w, 8, 7) << 6;
                        R::new("sw", format!("sw {}, {}(sp)", reg(rs2), imm))
                    }
                    7 => {
                        let imm = bits(w, 12, 10) << 3 | bits(w, 9, 7) << 6;
                        R::new("sd", format!("sd {}, {}(sp)", reg(rs2), imm))
                    }
                    _ => undefined(origin, w, "c.reserved"),
                }
            }
            _ => undefined(origin, w, "48/64-bit encoding"),
        }
    }
}

impl R {
    fn with_flow(mut self, f: FlowKind) -> R {
        self.flow = f;
        self
    }

    fn with_target(mut self, t: CallTarget) -> R {
        self.target = Some(t);
        self
    }
}

fn undefined(origin: Addr, w: u32, why: &str) -> R {
    let _ = origin;
    let text = if w <= 0xffff {
        format!(".half {w:#06x}  // {why}")
    } else {
        format!(".word {w:#010x}  // {why}")
    };
    R::new(".word", text).with_flow(FlowKind::Exception)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d32(w: u32) -> Decoded {
        Decoder::new(64).unwrap().full(Addr(0x1000), w)
    }

    #[test]
    fn decodes_the_standard_prologue() {
        // addi sp,sp,-16
        let d = d32(0xff01_0113);
        assert_eq!(d.insn.text, "addi sp, sp, -16");
        // sd ra,8(sp)
        let d = d32(0x0011_3423);
        assert_eq!(d.insn.text, "sd ra, 8(sp)");
    }

    #[test]
    fn ret_is_a_return() {
        let d = d32(0x0000_8067);
        assert_eq!(d.insn.text, "ret");
        assert_eq!(d.insn.flow, FlowKind::Return);
    }

    #[test]
    fn call_and_jump_carry_targets() {
        // jal ra, +8
        let j = 0x0080_00ef;
        let d = d32(j);
        assert_eq!(d.insn.mnemonic.as_ref(), "jal");
        assert_eq!(d.insn.target, Some(CallTarget::Direct { to: Addr(0x1008) }));
    }

    #[test]
    fn compressed_instructions_are_two_bytes() {
        let dec = Decoder::new(64).unwrap();
        // c.nop = 0x0001
        let d = dec.decode(Addr(0x1000), &[0x01, 0x00]).unwrap();
        assert_eq!(d.insn.len, 2);
        assert_eq!(d.insn.text, "nop");
    }

    #[test]
    fn compressed_addi_prologue() {
        let dec = Decoder::new(64).unwrap();
        // c.addi sp, -16 => 0x1141
        let d = dec.decode(Addr(0x1000), &[0x41, 0x11]).unwrap();
        assert_eq!(d.insn.len, 2);
        assert_eq!(d.insn.text, "addi sp, sp, -16");
    }

    #[test]
    fn short_input_is_refused() {
        let dec = Decoder::new(64).unwrap();
        assert!(dec.decode(Addr(0), &[0x13]).is_none());
        assert!(dec.decode(Addr(0), &[0x13, 0x01]).is_none());
    }

    #[test]
    fn rv32_and_rv64_report_different_arch() {
        assert_eq!(Decoder::new(32).unwrap().arch(), Arch::RiscV32);
        assert_eq!(Decoder::new(64).unwrap().arch(), Arch::RiscV64);
    }
}
