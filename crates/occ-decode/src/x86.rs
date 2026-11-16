//! x86 and x86-64 instruction decoding.
//!
//! Written as a table over the two opcode maps rather than as a tree of
//! special cases, because the thing that must be right is *length*: one byte
//! of over- or under-read desynchronises every following instruction, and the
//! resulting listing looks plausible while being entirely wrong.
//!
//! Coverage is the instructions compilers actually emit. Anything outside it
//! decodes as a one-byte `.byte` entry, which keeps the stream aligned and lets
//! the caller see exactly which byte was not understood. That is the only
//! honest failure mode here: guessing a length is worse than admitting the gap.
//!
//! 16-bit real mode is out of scope. 32-bit protected mode is covered for the
//! cases that appear inside 64-bit code (operand-size prefixes) but not as a
//! standalone target.

use occ_core::{Addr, Arch, CallTarget, FlowKind, Instruction, MAX_INSN_LEN};

mod cmp_tables;
mod evex_tables;
mod pclmul;
mod sse_map;
mod vector;
mod vector_tables;
mod vsib_tables;

/// The segment registers, as the `reg` field of a `8c` or `8e` names them.
///
/// The last two entries are not segments at all, and a listing says so with
/// a question mark rather than by refusing to decode the encoding: the
/// bytes are still an instruction's worth of bytes, and a decoder that
/// stopped here would put everything after it in the wrong place.
const SEG_NAMES: [&str; 8] = ["es", "cs", "ss", "ds", "fs", "gs", "?", "?"];

/// Registers indexed by encoding, for each operand size.
const REG64: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];
const REG32: [&str; 16] = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d",
    "r13d", "r14d", "r15d",
];
const REG16: [&str; 16] = [
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w",
    "r14w", "r15w",
];
const REG8: [&str; 16] = [
    "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b",
    "r13b", "r14b", "r15b",
];
/// 8-bit registers without a REX prefix: the high-byte forms.
const REG8_NOREX: [&str; 16] = [
    "al", "cl", "dl", "bl", "ah", "ch", "dh", "bh", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b",
    "r14b", "r15b",
];

/// Conditions, indexed by the low nibble of a `jcc`/`setcc`/`cmovcc` opcode.
const CONDITIONS: [&str; 16] = [
    "o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g",
];

/// Mnemonics for the 8-entry arithmetic group, indexed by `opcode >> 3`.
const ARITH: [&str; 8] = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"];

/// Mnemonics for group 1 with an immediate operand.
const GROUP1: [&str; 8] = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"];

/// Mnemonics for group 2 (shifts), indexed by `modrm.reg`.
const GROUP2: [&str; 8] = ["rol", "ror", "rcl", "rcr", "shl", "shr", "sal", "sar"];

/// Mnemonics for group 3, indexed by `modrm.reg`.
const GROUP3: [&str; 8] = ["test", "test", "not", "neg", "mul", "imul", "div", "idiv"];

/// Mnemonics for group 5 (`0xff`), indexed by `modrm.reg`.
const GROUP5: [&str; 8] = ["inc", "dec", "call", "call", "jmp", "jmp", "push", "?"];

/// Operand size of the current instruction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Size {
    B = 1,
    W = 2,
    D = 4,
    Q = 8,
    /// 128-bit vector, used only for `xmmword ptr` in a memory operand.
    X = 16,
    /// 256-bit vector, the width of a `ymm` register.
    Y = 32,
    /// 512-bit vector, the width of a `zmm` register.
    Z = 64,
    /// 80-bit extended float, the x87 stack's own format.
    ///
    /// It is ten bytes, which is why it has a name of its own: the `xmmword`
    /// tag would claim sixteen.
    T = 10,
}

impl Size {
    fn regs(self, rex_present: bool) -> &'static [&'static str; 16] {
        match self {
            Size::B => {
                if rex_present {
                    &REG8
                } else {
                    &REG8_NOREX
                }
            }
            Size::W => &REG16,
            Size::D => &REG32,
            Size::Q => &REG64,
            // Vector registers are named by the SSE decoder, never through the
            // general-purpose table.
            Size::X | Size::Y | Size::Z | Size::T => &REG64,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Size::B => "byte ptr ",
            Size::W => "word ptr ",
            Size::D => "dword ptr ",
            Size::Q => "qword ptr ",
            Size::X => "xmmword ptr ",
            Size::Y => "ymmword ptr ",
            Size::Z => "zmmword ptr ",
            Size::T => "tbyte ptr ",
        }
    }
}

/// A decoded operand, before rendering.
enum Operand {
    Reg(&'static str),
    /// SSE/AVX register, `xmm0..xmm15`.
    Xmm(u8),
    /// MMX register, `mm0..mm7`.
    Mm(u8),
    /// A vector register: `xmm`, `ymm`, `zmm` or a mask register `k`.
    Vec {
        /// The register itself.
        reg: vector::VecReg,
        /// The EVEX write-mask register, or zero when the instruction is
        /// not masked.
        mask: u8,
        /// EVEX zeroing: the lanes the mask does not select are written as
        /// zero rather than left alone.
        zeroing: bool,
    },
    /// x87 stack slot, `st(0)..st(7)`.
    St(u8),
    /// An immediate operand and the width of the operand it belongs to.
    ///
    /// The value is sign-extended when it is read, because that is what the
    /// encoding means. The width is what decides how the listing prints it:
    /// an immediate is a field as wide as its operand, and a listing shows
    /// it as that many hex digits with no sign, so `add eax, -1` is written
    /// `add eax, 0xffffffff` and `push -1` is written
    /// `push 0xffffffffffffffff`.
    Imm(i64, Size),
    /// Memory reference. `base`/`index` are register names; `disp` is absolute
    /// for RIP-relative addressing and a signed offset otherwise.
    Mem {
        /// Base register number, `0..=15`.
        base: Option<u8>,
        /// Index register number.
        index: Option<u8>,
        scale: u8,
        disp: i64,
        rip_relative: bool,
        size: Option<Size>,
        /// EVEX broadcast: the memory operand holds one element that is
        /// copied to every lane, and this is how many lanes there are.
        ///
        /// `None` is the ordinary case. The listing says `{1to16}` for a
        /// dword read into a 512-bit register, and the count is the only
        /// thing that distinguishes that from reading a whole vector.
        broadcast: Option<u8>,
        /// The vector length of the index register, given in bits, when the
        /// addressing mode takes its index from one.
        ///
        /// The gather and scatter instructions add the lanes of a vector
        /// register to a base address, so `[rax+ymm1*1]` names a different
        /// index register than `[rax+rcx*1]` and the listing has to say
        /// which. `None` is the ordinary addressing mode, where the index
        /// is a general-purpose register.
        vsib: Option<u16>,
        /// The mask register the operand is written under, `1..=7`, or zero
        /// when it is written unconditionally.
        ///
        /// The scatter instructions put the mask on the memory operand
        /// rather than on a register: `vpscatterdd [rax+zmm1*1]{k1}, zmm0`
        /// writes the lanes `k1` selects and leaves the rest of memory
        /// alone.
        mask: u8,
    },
    /// Direct branch displacement, kept as a signed delta.
    Rel(i64),
}

impl Operand {
    /// A vector register with no write mask.
    ///
    /// Every encoding but EVEX has no mask to print, and an EVEX
    /// instruction whose `aaa` field is zero is unmasked, which is the same
    /// thing to a reader.
    fn vec(reg: vector::VecReg) -> Operand {
        Operand::Vec {
            reg,
            mask: 0,
            zeroing: false,
        }
    }
}

/// Decoder state for one architecture width.
#[derive(Clone, Copy)]
pub struct Decoder {
    bits: u8,
    /// Address the instruction starts at, needed to resolve RIP-relative
    /// operands into absolute addresses.
    origin: Addr,
}

pub use crate::Decoded;

impl Decoder {
    /// Create a decoder for `bits` (32 or 64). Returns `None` for other widths.
    pub fn new(bits: u8) -> Option<Self> {
        match bits {
            32 | 64 => Some(Decoder {
                bits,
                origin: Addr::NULL,
            }),
            _ => None,
        }
    }

    /// Architecture this decoder reports.
    pub const fn arch(self) -> Arch {
        if self.bits == 64 {
            Arch::X86_64
        } else {
            Arch::X86
        }
    }

    /// Decode at `origin` from `bytes`.
    ///
    /// Returns `None` only when `bytes` is empty. A byte that cannot be
    /// decoded comes back as a one-byte `.byte` instruction.
    pub fn decode(&self, origin: Addr, bytes: &[u8]) -> Option<Decoded> {
        if bytes.is_empty() {
            return None;
        }
        let mut d = Decoder {
            bits: self.bits,
            origin,
        };
        Some(d.decode_inner(bytes))
    }

    fn decode_inner(&mut self, bytes: &[u8]) -> Decoded {
        let mut p = Prefix::default();
        let mut i = 0usize;

        // Legacy prefixes. Bounded by the maximum instruction length rather
        // than by a loop count, so a run of 0x66 cannot spin.
        while i < bytes.len() && i < MAX_INSN_LEN - 1 {
            match bytes[i] {
                0x66 => p.op66 = true,
                0x67 => p.op67 = true,
                0xf0 => p.lock = true,
                0xf2 | 0xf3 => p.rep = Some(bytes[i]),
                0x2e | 0x36 | 0x3e | 0x26 | 0x64 | 0x65 => p.seg = Some(bytes[i]),
                _ => break,
            }
            i += 1;
            p.count += 1;
        }

        // REX must come immediately before the opcode.
        let mut rex = 0u8;
        if self.bits == 64 && i < bytes.len() && (0x40..=0x4f).contains(&bytes[i]) {
            rex = bytes[i];
            i += 1;
            p.count += 1;
        }
        p.rex = rex;
        p.rex_present = rex != 0;

        let op_size = if rex & 0x08 != 0 {
            Size::Q
        } else if p.op66 {
            Size::W
        } else {
            Size::D
        };
        // Address size is 64 in long mode unless 0x67 is present, and 32 in
        // protected mode unless 0x67 is present.
        let addr_size = if self.bits == 64 {
            if p.op67 { Size::D } else { Size::Q }
        } else if p.op67 {
            Size::W
        } else {
            Size::D
        };

        if i >= bytes.len() {
            return self.raw_byte(bytes);
        }

        let mut rip_target = None;
        let mut reads: Vec<Addr> = Vec::new();
        let mut writes: Vec<Addr> = Vec::new();

        // The vector encodings replace the legacy prefixes: they carry the
        // mandatory prefix in their own bits and come last, so there is no
        // REX byte before them and the opcode byte follows the prefix.
        //
        // The legacy maps share one decode path: only the opcode byte
        // differs, and folding them here keeps the "what happens after the
        // opcode" logic in a single place.
        let decoded = match bytes[i] {
            0xc4 | 0xc5 | 0x62 if self.bits == 64 => {
                let first = bytes[i];
                self.decode_vector(first, &p, op_size, addr_size, bytes, &mut i)
            }
            opcode => {
                i += 1;
                if opcode == 0x0f {
                    if i >= bytes.len() {
                        return self.raw_byte(bytes);
                    }
                    let op2 = bytes[i];
                    i += 1;
                    self.decode_two_byte(op2, &p, op_size, addr_size, bytes, &mut i)
                } else {
                    self.decode_one_byte(opcode, &p, op_size, addr_size, bytes, &mut i)
                }
            }
        };
        let Some((mnemonic, flow, mut target, ops)) = decoded else {
            return self.raw_bytes(bytes, i);
        };

        // The length is settled: no further bytes will be read. A RIP-relative
        // displacement is measured from the end of the instruction, so the
        // address it refers to only becomes computable here.
        let len = i.min(MAX_INSN_LEN);
        let rip_disp = ops.iter().find_map(|o| match o {
            Operand::Mem {
                rip_relative: true,
                disp,
                ..
            } => Some(*disp),
            _ => None,
        });
        if let Some(d) = rip_disp {
            rip_target = Some(self.origin().add(len as u64).offset(d));
        }

        // A direct branch's target is measured from the end of the instruction,
        // which is only known now.
        if let Some(Operand::Rel(d)) = ops.first() {
            target = Some(CallTarget::Direct {
                to: self.origin().add(len as u64).offset(*d),
            });
        }

        // A direct `call`/`jmp` through a RIP-relative slot was recorded with
        // its raw displacement; replace it with the resolved address.
        if let Some(rt) = rip_target
            && matches!(target, Some(CallTarget::Indirect { via: Some(_) }))
        {
            target = Some(CallTarget::Indirect { via: Some(rt) });
        }

        // Whether a memory operand is read or written is a property of the
        // instruction, not of its shape. `mov` writes its first operand,
        // `push` only reads, `pop` only writes, and `lea` touches no memory at
        // all — it computes an address.
        let writes_first = matches!(
            mnemonic,
            "mov"
                | "movsxd"
                | "movzx"
                | "movsx"
                | "stos"
                | "stosb"
                | "movaps"
                | "movups"
                | "movdqa"
                | "movdqu"
                | "movss"
                | "movsd"
                | "movq"
                | "movd"
                | "movapd"
                | "movlps"
                | "movlpd"
                | "movhps"
                | "movhpd"
                | "movntps"
                | "movntpd"
                | "movntdq"
                | "movntq"
                // The vector moves and extracts store into their first
                // operand.
                | "vmovups"
                | "vmovupd"
                | "vmovss"
                | "vmovsd"
                | "vmovlps"
                | "vmovhps"
                | "vmovlpd"
                | "vmovhpd"
                | "vmovaps"
                | "vmovapd"
                | "vmovdqa"
                | "vmovdqu"
                | "vmovq"
                | "vmovq2dq"
                | "vmovdq2q"
                | "vmovntps"
                | "vmovntpd"
                | "vmovntss"
                | "vmovntsd"
                | "vmovntq"
                | "vmovntdq"
                | "vmovntdqa"
                | "vextractps"
                | "vextractf128"
                | "vpextrb"
                | "vpextrw"
                | "vpextrd"
                | "vpextrq"
                // The legacy maps have their own storing instructions.
                | "movbe"
                | "movdiri"
                | "aadd"
                | "aand"
                | "axor"
                | "aor"
                | "wrussd"
                | "wrssd"
        );
        let touches_memory = !matches!(mnemonic, "lea" | "nop");

        if touches_memory {
            for (idx, op) in ops.iter().enumerate() {
                match op {
                    Operand::Mem {
                        rip_relative: true, ..
                    } => {
                        let Some(a) = rip_target else { continue };
                        let writing = (idx == 0 && writes_first) || mnemonic == "pop";
                        if writing {
                            writes.push(a);
                        } else {
                            reads.push(a);
                        }
                    }
                    Operand::Mem {
                        base: None,
                        index: None,
                        rip_relative: false,
                        disp,
                        ..
                    } => {
                        // A 32-bit displacement with no base is used as an
                        // absolute pointer, which is how 32-bit PIC code
                        // addresses the GOT.
                        let a = Addr(*disp as u64);
                        let writing = (idx == 0 && writes_first) || mnemonic == "pop";
                        if writing {
                            writes.push(a);
                        } else {
                            reads.push(a);
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut text = self.render(&p, mnemonic, &ops, len);
        // objdump's convention: keep the RIP-relative spelling, which shows how
        // the instruction encodes, and append the address it resolves to. Both
        // facts matter and dropping either one costs a reader a lookup.
        if let Some(rt) = rip_target {
            text.push_str(&format!("  # {rt}"));
        }
        let mut raw = [0u8; MAX_INSN_LEN];
        raw[..len].copy_from_slice(&bytes[..len]);

        Decoded {
            insn: Instruction {
                addr: self.origin(),
                len: len as u32,
                bytes: raw,
                text,
                mnemonic: mnemonic.into(),
                flow,
                target,
                mem_reads: reads,
                mem_writes: writes,
            },
            rip_target,
        }
    }

    fn origin(&self) -> Addr {
        self.origin
    }

    fn raw_byte(&mut self, bytes: &[u8]) -> Decoded {
        self.raw_bytes(bytes, 1)
    }

    /// Everything read so far, reported as uninterpreted bytes.
    ///
    /// `upto` is how far the decoder got before it gave up, and it is the
    /// length the listing prints: an unknown opcode after a `rep` prefix is
    /// two bytes there, not one, and a decoder that answers one byte moves
    /// every following instruction by one. The prefixes are part of the
    /// instruction that could not be decoded, so they are part of its length.
    fn raw_bytes(&mut self, bytes: &[u8], upto: usize) -> Decoded {
        let len = upto.clamp(1, bytes.len()).min(MAX_INSN_LEN);
        let mut raw = [0u8; MAX_INSN_LEN];
        raw[..len].copy_from_slice(&bytes[..len]);
        let text = if len == 1 {
            format!(".byte {:#04x}", bytes[0])
        } else {
            let parts: Vec<String> = bytes[..len].iter().map(|b| format!("{b:#04x}")).collect();
            format!(".byte {}", parts.join(", "))
        };
        Decoded {
            insn: Instruction {
                addr: self.origin(),
                len: len as u32,
                bytes: raw,
                text,
                mnemonic: ".byte".into(),
                flow: FlowKind::Exception,
                target: None,
                mem_reads: Vec::new(),
                mem_writes: Vec::new(),
            },
            rip_target: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_one_byte(
        &mut self,
        op: u8,
        p: &Prefix,
        os: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        // Arithmetic group: 00..=3f, eight families times eight forms.
        if op < 0x40 && op & 0x07 < 6 {
            let family = (op >> 3) as usize;
            if family < 8 {
                let name = ARITH[family];
                return self.arith_form(name, op & 7, p, os, asz, b, i);
            }
        }
        match op {
            0x40..=0x4f => {
                // In 64-bit mode these bytes are REX prefixes and were consumed
                // above, so reaching here means 32-bit mode, where they are the
                // single-byte inc/dec forms.
                let r = usize::from(op & 7);
                let name = if op & 8 == 0 { "inc" } else { "dec" };
                Some((name, FlowKind::Next, None, vec![Operand::Reg(REG32[r])]))
            }
            // The loop family: rel8 conditional branches that also decrement
            // rcx. `jrcxz` is the only one gcc emits, for `do { } while (n--)`.
            0xe0..=0xe3 => {
                let v = self.imm(b, i, 1)?;
                let name = match op {
                    0xe0 => "loopne",
                    0xe1 => "loope",
                    0xe2 => "loop",
                    _ => {
                        if self.bits == 64 {
                            "jrcxz"
                        } else {
                            "jecxz"
                        }
                    }
                };
                Some((
                    name,
                    FlowKind::ConditionalBranch,
                    None,
                    vec![Operand::Rel(v)],
                ))
            }
            0xcf => Some(("iret", FlowKind::Return, None, vec![])),
            // Port I/O. Only drivers use these, but they are two bytes long and
            // guessing one byte desynchronises everything after.
            0xe4..=0xe7 => {
                // A port number is an unsigned byte; `imm` sign-extends.
                let port = self.imm(b, i, 1)? & 0xff;
                let (name, reg) = match (op, os) {
                    (0xe4, _) => ("in", "al"),
                    (0xe5, Size::W) => ("in", "ax"),
                    (0xe5, Size::Q) | (0xe5, Size::D) => ("in", "eax"),
                    (0xe6, _) => ("out", "al"),
                    (0xe7, Size::W) => ("out", "ax"),
                    _ => ("out", "eax"),
                };
                let reg = Operand::Reg(reg);
                let ops = if name == "in" {
                    vec![reg, Operand::Imm(port, Size::B)]
                } else {
                    vec![Operand::Imm(port, Size::B), reg]
                };
                Some((name, FlowKind::Next, None, ops))
            }
            0xec..=0xef => {
                // The same four instructions as `e4` through `e7`, with
                // the port number in `dx` instead of in a byte of the
                // instruction. The operands are the accumulator and the
                // register, and which one comes first says which way the
                // data goes.
                let (name, reg) = match (op, os) {
                    (0xec, _) => ("in", "al"),
                    (0xed, Size::W) => ("in", "ax"),
                    (0xed, Size::Q) | (0xed, Size::D) => ("in", "eax"),
                    (0xee, _) => ("out", "al"),
                    (0xef, Size::W) => ("out", "ax"),
                    _ => ("out", "eax"),
                };
                let acc = Operand::Reg(reg);
                let port = Operand::Reg("dx");
                let ops = if name == "in" {
                    vec![acc, port]
                } else {
                    vec![port, acc]
                };
                Some((name, FlowKind::Next, None, ops))
            }
            0x50..=0x57 => {
                let r = ((op & 7) | p.rex_b()) as usize;
                Some((
                    "push",
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg(if self.bits == 64 {
                        REG64[r]
                    } else {
                        REG32[r]
                    })],
                ))
            }
            0x58..=0x5f => {
                let r = ((op & 7) | p.rex_b()) as usize;
                Some((
                    "pop",
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg(if self.bits == 64 {
                        REG64[r]
                    } else {
                        REG32[r]
                    })],
                ))
            }
            0x63 => {
                // movsxd r64, r/m32
                // `m.reg` already carries the REX.R extension; adding it again
                // pushed the index past the end of the register table.
                let m = self.modrm(p, Size::D, asz, b, i)?;
                Some((
                    "movsxd",
                    FlowKind::Next,
                    None,
                    vec![
                        Operand::Reg(REG64[usize::from(m.reg & 15)]),
                        // The source is r/m: using the register form here
                        // turned `movsxd rcx, dword ptr [rax]` into
                        // `movsxd rcx, ecx`.
                        mem_operand(&m, Size::D, p),
                    ],
                ))
            }
            0x68 => {
                let v = self.imm(b, i, if os == Size::W { 2 } else { 4 })?;
                let w = self.push_size(os);
                Some(("push", FlowKind::Next, None, vec![Operand::Imm(v, w)]))
            }
            0x69 => {
                let m = self.modrm(p, os, asz, b, i)?;
                let v = self.imm(b, i, imm_width(os))?;
                Some((
                    "imul",
                    FlowKind::Next,
                    None,
                    vec![
                        reg_operand(&m, os, p),
                        mem_operand(&m, os, p),
                        Operand::Imm(v, os),
                    ],
                ))
            }
            0x6a => {
                let v = self.imm(b, i, 1)?;
                let w = self.push_size(os);
                Some(("push", FlowKind::Next, None, vec![Operand::Imm(v, w)]))
            }
            0x6b => {
                let m = self.modrm(p, os, asz, b, i)?;
                let v = self.imm(b, i, 1)?;
                Some((
                    "imul",
                    FlowKind::Next,
                    None,
                    vec![
                        reg_operand(&m, os, p),
                        mem_operand(&m, os, p),
                        Operand::Imm(v, os),
                    ],
                ))
            }
            0x70..=0x7f => {
                let v = self.imm(b, i, 1)?;
                let name = jcc_name(op & 0x0f);
                Some((
                    name,
                    FlowKind::ConditionalBranch,
                    None,
                    vec![Operand::Rel(v)],
                ))
            }
            0x80..=0x83 => {
                // 0x80 is the byte form; 0x81 and 0x83 both use the operand
                // size, differing only in immediate width. 0x82 is a second
                // spelling of the byte form that 64-bit mode dropped, and
                // 32-bit code written before that still contains it.
                if op == 0x82 && self.bits != 32 {
                    return None;
                }
                let byte_form = op == 0x80 || op == 0x82;
                let size = if byte_form { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                let name = GROUP1[(m.reg & 7) as usize];
                let w = if byte_form || op == 0x83 {
                    1
                } else {
                    imm_width(size)
                };
                let v = self.imm(b, i, w)?;
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, size, p), Operand::Imm(v, size)],
                ))
            }
            0x84 => {
                let m = self.modrm(p, Size::B, asz, b, i)?;
                Some((
                    "test",
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, Size::B, p), reg_operand(&m, Size::B, p)],
                ))
            }
            0x85 => {
                let m = self.modrm(p, os, asz, b, i)?;
                Some((
                    "test",
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, os, p), reg_operand(&m, os, p)],
                ))
            }
            0x86 | 0x87 => {
                let size = if op == 0x86 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                Some((
                    "xchg",
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, size, p), reg_operand(&m, size, p)],
                ))
            }
            0x88..=0x8b => {
                let size = if op & 1 == 0 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                let r = reg_operand(&m, size, p);
                let rm = mem_operand(&m, size, p);
                let ops = if op & 2 == 0 {
                    vec![rm, r]
                } else {
                    vec![r, rm]
                };
                Some(("mov", FlowKind::Next, None, ops))
            }
            0x8c | 0x8e => {
                // The moves to and from a segment register. The `reg`
                // field of the ModRM byte names the segment, and the
                // other operand is always a word in memory: a segment
                // selector is sixteen bits wide, whatever the operand
                // size of the register form says.
                //
                // The register form prints at the operand size, which is
                // why `8c c0` is `mov eax, es` and `48 8c c0` is
                // `mov rax, es`.
                let after_opcode = *i;
                let m = self.modrm(p, Size::W, asz, b, i)?;
                // A `reg` field of six or seven names no segment: there are
                // only six, and the encoding spends the remaining two values
                // on nothing. The listing gives the opcode byte back as an
                // undecodable one, so the ModRM is not part of the length.
                if m.reg & 7 >= 6 {
                    *i = after_opcode;
                    return None;
                }
                let seg = SEG_NAMES[usize::from(m.reg & 7)];
                let rm = if m.mem.is_some() {
                    mem_operand(&m, Size::W, p)
                } else {
                    mem_operand(&m, os, p)
                };
                let seg = Operand::Reg(seg);
                let ops = if op == 0x8c {
                    vec![rm, seg]
                } else {
                    vec![seg, rm]
                };
                Some(("mov", FlowKind::Next, None, ops))
            }
            0x8d => {
                let m = self.modrm(p, os, asz, b, i)?;
                Some((
                    "lea",
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, os, p), mem_operand(&m, os, p)],
                ))
            }
            0x8f => {
                let after_opcode = *i;
                let m = self.modrm(p, os, asz, b, i)?;
                // The group has one defined entry: `/0` is `pop r/m`. The
                // other seven are not instructions, and the listing gives the
                // opcode byte back instead of naming one.
                if m.reg & 7 != 0 {
                    *i = after_opcode;
                    return None;
                }
                Some(("pop", FlowKind::Next, None, vec![mem_operand(&m, os, p)]))
            }
            // `f3 90` is the pause, which tells the processor that the
            // thread is in a spin loop.
            0x90 => Some((
                if p.rep == Some(0xf3) { "pause" } else { "nop" },
                FlowKind::Next,
                None,
                vec![],
            )),
            0x91..=0x97 => {
                let r = ((op & 7) | p.rex_b()) as usize;
                Some((
                    "xchg",
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg(REG64[r]), Operand::Reg(REG64[0])],
                ))
            }
            // The sign-extension instructions are named after the width
            // they extend to: `cbw` extends a byte into a word, `cwde`
            // extends a word into a double word, and `cdqe` extends a
            // double word into a quad word. The `66` prefix picks the
            // first and the `rex.w` bit picks the last.
            0x98 => Some((
                match os {
                    Size::Q => "cdqe",
                    Size::W => "cbw",
                    _ => "cwde",
                },
                FlowKind::Next,
                None,
                vec![],
            )),
            0x99 => Some((
                match os {
                    Size::Q => "cqo",
                    Size::W => "cwd",
                    _ => "cdq",
                },
                FlowKind::Next,
                None,
                vec![],
            )),
            // The byte `9b` is the wait instruction, but it is also the
            // prefix that an x87 instruction may carry: `9b d9 3d` is one
            // instruction, `fstcw`, and not two. Which of the two it is
            // depends on the byte that follows, so the byte is looked at
            // before the name is settled.
            0x9b => {
                match b.get(*i) {
                    Some(&next) if (0xd8..=0xdf).contains(&next) => {
                        // The wait is part of the x87 instruction that
                        // follows. The instruction is decoded as if the
                        // byte were not there, and the byte is consumed
                        // with it.
                        //
                        // The wait changes the name, because the machine
                        // has a spelling of its own for the instructions
                        // that wait: `fstcw` is `fnstcw` with a wait in
                        // front of it.
                        *i += 1;
                        self.decode_x87(next, p, asz, b, i, true)
                    }
                    _ => Some(("fwait", FlowKind::Next, None, vec![])),
                }
            }
            // The instructions 64-bit mode dropped. They are not decoration:
            // a 32-bit library full of hand-written assembly still contains
            // them, and a decoder that answers `.byte` for `0x07` emits one
            // byte where the listing has two, which shifts every instruction
            // after it. Each one is guarded on the mode rather than on a
            // table entry, because the encoding means nothing outside 32-bit
            // code.
            0x06 | 0x07 | 0x0e | 0x16 | 0x17 | 0x1e | 0x1f if self.bits == 32 => {
                // Segment-register push and pop. The four legacy segments
                // are the only ones that have an encoding: `fs` and `gs`
                // were added after this group was full and need `0f a0`
                // and `0f a8`.
                let (name, seg) = match op {
                    0x06 => ("push", "es"),
                    0x07 => ("pop", "es"),
                    0x0e => ("push", "cs"),
                    0x16 => ("push", "ss"),
                    0x17 => ("pop", "ss"),
                    0x1e => ("push", "ds"),
                    _ => ("pop", "ds"),
                };
                Some((name, FlowKind::Next, None, vec![Operand::Reg(seg)]))
            }
            0x27 | 0x2f | 0x37 | 0x3f if self.bits == 32 => {
                // The BCD adjust instructions. Decimal arithmetic stopped
                // being worth a circuit decades ago, and 32-bit code that
                // predates that still calls them.
                let name = match op {
                    0x27 => "daa",
                    0x2f => "das",
                    0x37 => "aaa",
                    _ => "aas",
                };
                Some((name, FlowKind::Next, None, vec![]))
            }
            0x60 | 0x61 if self.bits == 32 => {
                // Push and pop of every general-purpose register at once.
                // The `66` form moves 16-bit registers and the listing calls
                // it `pushaw`; the plain form is the 32-bit one.
                let (name, wide) = if op == 0x60 {
                    ("pusha", "pushaw")
                } else {
                    ("popa", "popaw")
                };
                Some((
                    if p.op66 { wide } else { name },
                    FlowKind::Next,
                    None,
                    vec![],
                ))
            }
            0x62 if self.bits == 32 => {
                // BOUND: an array-bounds check against a pair of limits in
                // memory. 64-bit mode reuses this byte as the EVEX prefix,
                // so the encoding only exists in 32-bit code.
                let m = self.modrm(p, Size::D, asz, b, i)?;
                m.mem.as_ref()?;
                Some((
                    "bound",
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, Size::D, p), mem_operand(&m, Size::D, p)],
                ))
            }
            0xce if self.bits == 32 => Some(("into", FlowKind::Next, None, vec![])),
            0x9a | 0xea if self.bits == 32 => {
                // A far call or jump: an offset at the operand size, then a
                // sixteen-bit selector. Both halves are part of the
                // instruction, so both are read even though the listing
                // prints them as a single target.
                let offset = self.imm(b, i, if p.op66 { 2 } else { 4 })?;
                let selector = self.imm(b, i, 2)?;
                let name = if op == 0x9a { "lcall" } else { "ljmp" };
                Some((
                    name,
                    if op == 0x9a {
                        FlowKind::Call
                    } else {
                        FlowKind::UnconditionalBranch
                    },
                    None,
                    vec![
                        Operand::Imm(offset, if p.op66 { Size::W } else { Size::D }),
                        Operand::Imm(selector, Size::W),
                    ],
                ))
            }
            0xd4 | 0xd5 if self.bits == 32 => {
                // ASCII adjust after multiply and after divide: both read one
                // immediate byte, the base.
                let base = self.imm(b, i, 1)?;
                let name = if op == 0xd4 { "aam" } else { "aad" };
                Some((name, FlowKind::Next, None, vec![Operand::Imm(base, Size::B)]))
            }
            0x6c..=0x6f => {
                // The string forms of `in` and `out`. The port is always `dx`
                // and the buffer is always `esi` or `edi`, so the encoding
                // carries no operand and only the width changes with `66`.
                let name = match (op, p.op66) {
                    (0x6c, true) => "insw",
                    (0x6c, false) => "insb",
                    (0x6d, true) => "insw",
                    (0x6d, false) => "insd",
                    (0x6e, true) => "outsw",
                    (0x6e, false) => "outsb",
                    (0x6f, true) => "outsw",
                    _ => "outsd",
                };
                Some((name, FlowKind::Next, None, vec![]))
            }
            0xc4 | 0xc5 if self.bits == 32 => {
                // Load far pointer: a general-purpose register and a memory
                // operand holding an offset with a selector after it. In
                // 64-bit mode these two bytes are the VEX prefix instead,
                // which is why the arm is guarded on the mode rather than on
                // the opcode.
                let m = self.modrm(p, Size::D, asz, b, i)?;
                let name = if op == 0xc4 { "les" } else { "lds" };
                m.mem.as_ref()?;
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, Size::D, p), mem_operand(&m, Size::D, p)],
                ))
            }
            0x9c => Some(("pushfq", FlowKind::Next, None, vec![])),
            0x9d => Some(("popfq", FlowKind::Next, None, vec![])),
            0x9e => Some(("sahf", FlowKind::Next, None, vec![])),
            0x9f => Some(("lahf", FlowKind::Next, None, vec![])),
            0xa0..=0xa3 => {
                // The moves to and from an absolute address. The address
                // is the whole of the rest of the instruction, and its
                // width is the address size, not the operand size: `67`
                // switches it to the other width, and which pair that is
                // depends on the mode. 64-bit code reads eight bytes or four,
                // 32-bit code four or two.
                //
                // The accumulator is the other operand, which is what
                // makes these four opcodes enough for every width.
                let addr_width = match (self.bits, p.op67) {
                    (64, false) => 8,
                    (64, true) => 4,
                    (_, true) => 2,
                    (_, false) => 4,
                };
                let raw = self.imm(b, i, addr_width)?;
                let size = if op & 1 == 0 {
                    Size::B
                } else if p.rex & 0x08 != 0 {
                    Size::Q
                } else if p.op66 {
                    Size::W
                } else {
                    Size::D
                };
                let acc = Operand::Reg(match size {
                    Size::B => "al",
                    Size::W => "ax",
                    Size::Q => "rax",
                    _ => "eax",
                });
                let mem = Operand::Mem {
                    base: None,
                    index: None,
                    scale: 1,
                    disp: raw,
                    rip_relative: false,
                    // The listing leaves the width off an absolute
                    // operand: the accumulator beside it already says
                    // how wide the move is.
                    size: None,
                    broadcast: None,
                    vsib: None,
                    mask: 0,
                };
                // An eight-byte address is what makes the instruction
                // long enough for the listing to call it `movabs`;
                // the four-byte one is an ordinary `mov`.
                let name = if addr_width == 8 { "movabs" } else { "mov" };
                let ops = if op & 2 == 0 {
                    vec![acc, mem]
                } else {
                    vec![mem, acc]
                };
                Some((name, FlowKind::Next, None, ops))
            }
            0xa8 => {
                let v = self.imm(b, i, 1)?;
                Some((
                    "test",
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg("al"), Operand::Imm(v, Size::B)],
                ))
            }
            0xa9 => {
                let v = self.imm(b, i, imm_width(os))?;
                let r = match os {
                    Size::Q => "rax",
                    Size::W => "ax",
                    _ => "eax",
                };
                Some((
                    "test",
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg(r), Operand::Imm(v, os)],
                ))
            }
            // The string instructions name their operands implicitly, but
            // printing them makes two facts visible that a reader otherwise
            // has to look up: which register holds the data, and how wide an
            // element is. The width is the operand-size prefix's business,
            // and `movs` alone does not say which of the three it is.
            0xa4..=0xaf => {
                let (name, size) = match op {
                    0xa4 => ("movsb", Size::B),
                    0xa5 => ("movs", os),
                    0xa6 => ("cmpsb", Size::B),
                    0xa7 => ("cmps", os),
                    0xaa => ("stosb", Size::B),
                    0xab => ("stos", os),
                    0xac => ("lodsb", Size::B),
                    0xad => ("lods", os),
                    0xae => ("scasb", Size::B),
                    _ => ("scas", os),
                };
                // `rsi` is the source, `rdi` the destination, and `rcx` the
                // count the `rep` prefix loops over.
                let mem = |n: u8| Operand::Mem {
                    base: Some(n),
                    index: None,
                    scale: 1,
                    disp: 0,
                    rip_relative: false,
                    size: Some(size),
                    broadcast: None,
                    vsib: None,
                    mask: 0,
                };
                // The accumulator holds the value that `stos` writes, the one
                // `lods` reads and the one `scas` compares against.
                let accumulator = Operand::Reg(match size {
                    Size::B => "al",
                    Size::W => "ax",
                    Size::Q => "rax",
                    _ => "eax",
                });
                let ops = match op {
                    0xa4..=0xa7 => vec![mem(6), mem(7)],
                    0xaa | 0xab => vec![mem(7), accumulator],
                    0xac | 0xad => vec![accumulator, mem(6)],
                    _ => vec![accumulator, mem(7)],
                };
                Some((name, FlowKind::Next, None, ops))
            }
            0xb0..=0xb7 => {
                let r = ((op & 7) | p.rex_b()) as usize;
                let v = self.imm(b, i, 1)?;
                Some((
                    "mov",
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg(REG8[r]), Operand::Imm(v, Size::B)],
                ))
            }
            0xb8..=0xbf => {
                let r = ((op & 7) | p.rex_b()) as usize;
                // The immediate is as wide as the destination: a 16-bit
                // operand takes two bytes, a 32-bit one takes four, and
                // only the 64-bit form takes eight.
                let w = match os {
                    Size::W => 2,
                    Size::Q => 8,
                    _ => 4,
                };
                let v = self.imm(b, i, w)?;
                let regs = os.regs(p.rex_present);
                // A 64-bit register loaded with a 64-bit immediate is a
                // different instruction from one loaded with a sign-extended
                // 32-bit one, and the name says so.
                let name = if os == Size::Q { "movabs" } else { "mov" };
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg(regs[r]), Operand::Imm(v, os)],
                ))
            }
            0xc0 | 0xc1 => {
                let size = if op == 0xc0 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                let v = self.imm(b, i, 1)?;
                Some((
                    GROUP2[(m.reg & 7) as usize],
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, size, p), Operand::Imm(v, size)],
                ))
            }
            0xc2 => {
                let v = self.imm(b, i, 2)?;
                Some(("ret", FlowKind::Return, None, vec![Operand::Imm(v, Size::W)]))
            }
            0xc3 => Some(("ret", FlowKind::Return, None, vec![])),
            // The far returns differ from the near ones in one bit, and
            // the bit is the whole difference: same stack effect for the
            // return address, plus a segment selector the near return
            // does not read.
            0xca => {
                let v = self.imm(b, i, 2)?;
                Some(("retf", FlowKind::Return, None, vec![Operand::Imm(v, Size::W)]))
            }
            0xcb => Some(("retf", FlowKind::Return, None, vec![])),
            0xc6 | 0xc7 => {
                let size = if op == 0xc6 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                // Group 9, extension 7, in its register form: the
                // transactional-memory instructions. `c6 f8 ff` aborts the
                // transaction and `c7 f8 rel32` begins one at the given
                // address.
                if m.mode == 3 && (m.reg & 7) == 7 {
                    if op == 0xc6 {
                        let v = self.imm(b, i, 1)?;
                        return Some(("xabort", FlowKind::Exception, None, vec![Operand::Imm(v, Size::B)]));
                    }
                    let v = self.imm(b, i, 4)?;
                    return Some((
                        "xbegin",
                        FlowKind::ConditionalBranch,
                        None,
                        vec![Operand::Rel(v)],
                    ));
                }
                let w = if op == 0xc6 { 1 } else { imm_width(size) };
                let v = self.imm(b, i, w)?;
                Some((
                    "mov",
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, size, p), Operand::Imm(v, size)],
                ))
            }
            0xc8 => {
                // `enter frame, level`: a sixteen-bit frame size and an
                // eight-bit nesting level, both part of the instruction.
                let frame = self.imm(b, i, 2)?;
                let level = self.imm(b, i, 1)?;
                Some((
                    "enter",
                    FlowKind::Next,
                    None,
                    vec![
                        Operand::Imm(frame, Size::W),
                        Operand::Imm(level, Size::B),
                    ],
                ))
            }
            0xc9 => Some(("leave", FlowKind::Next, None, vec![])),
            0xcc => Some(("int3", FlowKind::Interrupt, None, vec![])),
            0xcd => {
                let v = self.imm(b, i, 1)?;
                Some(("int", FlowKind::Interrupt, None, vec![Operand::Imm(v, Size::B)]))
            }
            0xd0..=0xd3 => {
                let size = if op & 1 == 0 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                let count = if op & 2 == 0 {
                    Operand::Imm(1, Size::B)
                } else {
                    Operand::Reg("cl")
                };
                Some((
                    GROUP2[(m.reg & 7) as usize],
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, size, p), count],
                ))
            }
            0xe8 => {
                let v = self.imm(b, i, 4)?;
                Some(("call", FlowKind::Call, None, vec![Operand::Rel(v)]))
            }
            0xe9 => {
                let v = self.imm(b, i, 4)?;
                Some((
                    "jmp",
                    FlowKind::UnconditionalBranch,
                    None,
                    vec![Operand::Rel(v)],
                ))
            }
            0xeb => {
                let v = self.imm(b, i, 1)?;
                Some((
                    "jmp",
                    FlowKind::UnconditionalBranch,
                    None,
                    vec![Operand::Rel(v)],
                ))
            }
            0xf4 => Some(("hlt", FlowKind::Interrupt, None, vec![])),
            0xf6 | 0xf7 => {
                let size = if op == 0xf6 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                let name = GROUP3[(m.reg & 7) as usize];
                // `test` has an immediate; the others do not.
                if m.reg & 7 < 2 {
                    let w = if op == 0xf6 { 1 } else { imm_width(size) };
                    let v = self.imm(b, i, w)?;
                    Some((
                        name,
                        FlowKind::Next,
                        None,
                        vec![mem_operand(&m, size, p), Operand::Imm(v, size)],
                    ))
                } else {
                    Some((name, FlowKind::Next, None, vec![mem_operand(&m, size, p)]))
                }
            }
            0xf8 => Some(("clc", FlowKind::Next, None, vec![])),
            0xf9 => Some(("stc", FlowKind::Next, None, vec![])),
            0xfa => Some(("cli", FlowKind::Next, None, vec![])),
            0xfb => Some(("sti", FlowKind::Next, None, vec![])),
            0xfc => Some(("cld", FlowKind::Next, None, vec![])),
            0xfd => Some(("std", FlowKind::Next, None, vec![])),
            0xfe => {
                let m = self.modrm(p, Size::B, asz, b, i)?;
                let name = if m.reg & 7 == 0 { "inc" } else { "dec" };
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, Size::B, p)],
                ))
            }
            0xff => {
                let after_opcode = *i;
                let m = self.modrm(p, os, asz, b, i)?;
                let idx = (m.reg & 7) as usize;
                // Seven of the eight entries in this group are defined: inc,
                // dec, call near, call far, jmp near, jmp far, push. The
                // eighth is not, and the listing gives the opcode byte back
                // rather than naming an instruction that does not exist.
                if idx == 7 {
                    *i = after_opcode;
                    return None;
                }
                let name = GROUP5[idx];
                // `mode == 3` means the operand is a register, so the target is
                // not in the instruction at all — that is an indirect call even
                // though the opcode is the direct form.
                let (flow, target) = match idx {
                    2 | 4 if m.mode != 3 => (
                        if idx == 2 {
                            FlowKind::Call
                        } else {
                            FlowKind::UnconditionalBranch
                        },
                        self.effective_target(&m),
                    ),
                    2 | 3 => (FlowKind::IndirectCall, None),
                    4 | 5 => (FlowKind::IndirectBranch, None),
                    _ => (FlowKind::Next, None),
                };
                // `call [mem]` and `jmp [mem]` read a pointer whose width is
                // the code model's, not the operand-size prefix's: 8 bytes in
                // long mode regardless of REX.W.
                let slot_size = if self.bits == 64 && (2..=5).contains(&idx) {
                    Size::Q
                } else {
                    os
                };
                Some((name, flow, target, vec![mem_operand(&m, slot_size, p)]))
            }
            // x87: every opcode in D8..DF takes a ModRM byte and no immediate,
            // so decoding it is what keeps `long double` code aligned. The
            // memory forms are named; an unnamed register form is reported as
            // such rather than guessed at.
            0xd8..=0xdf => self.decode_x87(op, p, asz, b, i, false),
            _ => None,
        }
    }

    /// Decode an x87 instruction.
    ///
    /// The opcodes `d8` through `df` all take a ModRM byte and no immediate
    /// byte, so decoding them is what keeps the `long double` code aligned.
    /// The memory forms are named; an unnamed register form is reported as
    /// such rather than guessed at.
    fn decode_x87(
        &mut self,
        op: u8,
        p: &Prefix,
        asz: Size,
        b: &[u8],
        i: &mut usize,
        wait: bool,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        let modrm = *b.get(*i)?;
        let m = self.modrm(p, Size::Q, asz, b, i)?;
        if m.mode == 3 {
            // The register forms come in two shapes: the ones that
            // name a stack slot, and the ones that stand alone.
            // The standalone ones are matched on the whole ModRM
            // byte, because that byte is the whole instruction.
            match x87_register_form(op, modrm) {
                Some((name, ops)) => {
                    let name = if wait && name == "fnstsw" { "fstsw" } else { name };
                    Some((name, FlowKind::Next, None, ops))
                }
                None => Some((
                    "fpu",
                    FlowKind::Next,
                    None,
                    vec![
                        Operand::Imm(i64::from(op), Size::B),
                        Operand::Imm(i64::from(modrm), Size::B),
                    ],
                )),
            }
        } else {
            let size = x87_mem_size(op, m.reg & 7);
            let name = X87_MEM[usize::from(op & 7)][usize::from(m.reg & 7)];
            if name.is_empty() {
                return None;
            }
            // The instructions that store the x87 state have a second
            // name for the form that waits first: `fstcw` is `fnstcw`
            // with a wait in front of it.
            let name = if wait {
                match name {
                    "fnstenv" => "fstenv",
                    "fnstcw" => "fstcw",
                    "fnsave" => "fsave",
                    "fnstsw" => "fstsw",
                    other => other,
                }
            } else {
                name
            };
            // The environment and state forms have no single width, so
            // they are printed without a size tag rather than with a
            // wrong one.
            let operand = if matches!(name, "fldenv" | "fnstenv" | "frstor" | "fnsave")
            {
                let mem = m.mem.as_ref()?;
                Operand::Mem {
                    base: mem.base,
                    index: mem.index,
                    scale: mem.scale,
                    disp: mem.disp,
                    rip_relative: mem.rip_relative,
                    size: None,
                    broadcast: None,
                    vsib: None,
                    mask: 0,
                }
            } else {
                mem_operand(&m, size, p)
            };
            Some((name, FlowKind::Next, None, vec![operand]))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn arith_form(
        &mut self,
        name: &'static str,
        form: u8,
        p: &Prefix,
        os: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        match form {
            0..=3 => {
                // Forms 0 and 2 are the 8-bit encodings; 1 and 3 use the
                // operand size. Checking `form < 2` here reversed it and made
                // `xor ebp, ebp` decode as `xor ch, ch`.
                let size = if form % 2 == 0 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                let r = reg_operand(&m, size, p);
                let rm = mem_operand(&m, size, p);
                // Forms 0/1 are `op r/m, r`; forms 2/3 are `op r, r/m`. The
                // direction bit is the low bit of the form, not its parity of
                // the byte size.
                let ops = if form < 2 {
                    vec![rm, r]
                } else {
                    vec![r, rm]
                };
                Some((name, FlowKind::Next, None, ops))
            }
            4 => {
                let v = self.imm(b, i, 1)?;
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg("al"), Operand::Imm(v, Size::B)],
                ))
            }
            5 => {
                let w = imm_width(os);
                let v = self.imm(b, i, w)?;
                let r = match os {
                    Size::Q => "rax",
                    Size::W => "ax",
                    _ => "eax",
                };
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg(r), Operand::Imm(v, os)],
                ))
            }
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_two_byte(
        &mut self,
        op: u8,
        p: &Prefix,
        os: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        match op {
            0x00 => {
                // Group 6: descriptor-table and task-register access.
                let m = self.modrm(p, os, asz, b, i)?;
                let name = match m.reg & 7 {
                    0 => "sldt",
                    1 => "str",
                    2 => "lldt",
                    3 => "ltr",
                    4 => "verr",
                    5 => "verw",
                    _ => return None,
                };
                Some((name, FlowKind::Next, None, vec![mem_operand(&m, Size::W, p)]))
            }
            0x01 => {
                // Group 7. The register forms are the privileged and
                // control-flow-integrity instructions, which have no operands
                // beyond their own encoding.
                let m = self.modrm(p, os, asz, b, i)?;
                if m.mode == 3 {
                    let name = group7_mod3(m.reg & 7, m.rm & 7, p)?;
                    // `smsw` and `lmsw` name a register when the operand is
                    // one, and read or write memory when it is not.
                    let ops = match name {
                        "smsw" => vec![gp_reg_operand(m.rm & 15, os, p)],
                        "lmsw" => vec![gp_reg_operand(m.rm & 15, os, p)],
                        _ => vec![],
                    };
                    return Some((name, FlowKind::Next, None, ops));
                }
                let name = match (m.reg & 7, p.rep) {
                    (0, _) => "sgdt",
                    (1, _) => "sidt",
                    (2, _) => "lgdt",
                    (3, _) => "lidt",
                    (4, _) => "smsw",
                    (5, Some(0xf3)) => "rstorssp",
                    (6, _) => "lmsw",
                    (7, _) => "invlpg",
                    _ => return None,
                };
                Some((name, FlowKind::Next, None, vec![mem_operand(&m, os, p)]))
            }
            0xae => {
                // Group 15: the SSE state save/restore pair and the MXCSR
                // control word, both emitted by every C runtime startup.
                let m = self.modrm(p, os, asz, b, i)?;
                // An `f3` prefix turns the whole group into the instructions
                // that read and write the base registers and the shadow
                // stack. They are a group of their own, and the `f3` is what
                // says which one is meant.
                if p.rep == Some(0xf3) {
                    let name = match m.reg & 7 {
                        0 => "rdfsbase",
                        1 => "rdgsbase",
                        2 => "wrfsbase",
                        3 => "wrgsbase",
                        4 => "ptwrite",
                        5 => {
                            if os == Size::Q {
                                "incsspq"
                            } else {
                                "incsspd"
                            }
                        }
                        6 => "umonitor",
                        _ => return None,
                    };
                    return Some((name, FlowKind::Next, None, vec![]));
                }
                // The register forms of this group are the fences: they
                // order the memory operations and name no operand.
                if m.mode == 3 {
                    let name = match m.reg & 7 {
                        5 => "lfence",
                        6 => "mfence",
                        7 => "sfence",
                        _ => return None,
                    };
                    return Some((name, FlowKind::Next, None, vec![]));
                }
                // Group 15: the state-save instructions and the cache
                // controls. The operand-width bit picks a 64-bit variant of
                // the four that save a state area, and those four name no
                // width at all: the area is a fixed layout of several
                // hundred bytes, not a value of any type the listing could
                // name.
                let wide = p.rex & 0x08 != 0;
                let (name, width) = match (m.reg & 7, p.rep, p.op66) {
                    (0, _, _) => (if wide { "fxsave64" } else { "fxsave" }, None),
                    (1, _, _) => (if wide { "fxrstor64" } else { "fxrstor" }, None),
                    (2, _, _) => ("ldmxcsr", Some(Size::D)),
                    (3, _, _) => ("stmxcsr", Some(Size::D)),
                    (4, _, _) => (if wide { "xsave64" } else { "xsave" }, None),
                    (5, _, _) => (if wide { "xrstor64" } else { "xrstor" }, None),
                    (6, _, _) => (if wide { "xsaveopt64" } else { "xsaveopt" }, None),
                    (7, Some(0xf3), _) => ("clrssbsy", None),
                    (7, _, true) => ("clflushopt", Some(Size::B)),
                    (7, _, _) => ("clflush", Some(Size::B)),
                    _ => return None,
                };
                let mem = m.mem.as_ref()?;
                let operand = match width {
                    Some(w) => mem_ref_operand(mem, w),
                    None => Operand::Mem {
                        base: mem.base,
                        index: mem.index,
                        scale: mem.scale,
                        disp: mem.disp,
                        rip_relative: mem.rip_relative,
                        size: None,
                        broadcast: None,
                        vsib: None,
                        mask: 0,
                    },
                };
                Some((name, FlowKind::Next, None, vec![operand]))
            }
            0x05 => Some(("syscall", FlowKind::Interrupt, None, vec![])),
            0x0b => Some(("ud2", FlowKind::Exception, None, vec![])),
            // `0f 0e` is the AMD 3DNow! state reset. It takes no operand and
            // survives in both modes, unlike the rest of the 3DNow! set.
            0x0e => Some(("femms", FlowKind::Next, None, vec![])),
            // `0f 1e` takes a ModRM byte. `fa` makes it endbr64 and `fb`
            // endbr32; anything else is the reserved `nop r/m` form. Not
            // consuming the byte left endbr64 three bytes long and shifted
            // every following instruction by one.
            0x18 => {
                // Prefetch hints. Compilers emit these in copy loops, and the
                // three bytes they occupy have to be consumed as one
                // instruction or the listing drifts by two.
                let m = self.modrm(p, Size::B, asz, b, i)?;
                let name = match m.reg & 7 {
                    0 => "prefetchnta",
                    1 => "prefetcht0",
                    2 => "prefetcht1",
                    3 => "prefetcht2",
                    _ => "nop",
                };
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, Size::B, p)],
                ))
            }
            0x0d => {
                let m = self.modrm(p, Size::B, asz, b, i)?;
                let name = if p.op66 { "prefetchwt1" } else { "prefetchw" };
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, Size::B, p)],
                ))
            }
            0x1e => {
                let byte = *b.get(*i)?;
                match byte {
                    // `f3 0f 1e fa` is `endbr64` and `f3 0f 1e fb` is
                    // `endbr32`. The `f3` prefix is part of the instruction:
                    // without it the same bytes are the reserved `nop r/m`
                    // form, which names the register and not the branch
                    // target.
                    0xfa if p.rep == Some(0xf3) => {
                        *i += 1;
                        Some(("endbr64", FlowKind::Next, None, vec![]))
                    }
                    0xfb if p.rep == Some(0xf3) => {
                        *i += 1;
                        Some(("endbr32", FlowKind::Next, None, vec![]))
                    }
                    _ => {
                        let m = self.modrm(p, os, asz, b, i)?;
                        // `f3 0f 1e /1` reads the shadow stack pointer. The
                        // `nop` forms of this opcode are the reserved
                        // encodings; this one is not reserved.
                        if p.rep == Some(0xf3) && m.mode == 3 && (m.reg & 7) == 1 {
                            let name = if os == Size::Q {
                                "rdsspq"
                            } else {
                                "rdsspd"
                            };
                            return Some((name, FlowKind::Next, None, vec![]));
                        }
                        Some(("nop", FlowKind::Next, None, vec![mem_operand(&m, os, p)]))
                    }
                }
            }
            0x1f => {
                let m = self.modrm(p, os, asz, b, i)?;
                Some(("nop", FlowKind::Next, None, vec![mem_operand(&m, os, p)]))
            }
            0x31 => Some(("rdtsc", FlowKind::Next, None, vec![])),
            0x40..=0x4f => {
                let m = self.modrm(p, os, asz, b, i)?;
                Some((
                    cmov_name(op & 0x0f),
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, os, p), mem_operand(&m, os, p)],
                ))
            }
            0x80..=0x8f => {
                let v = self.imm(b, i, 4)?;
                Some((
                    jcc_name(op & 0x0f),
                    FlowKind::ConditionalBranch,
                    None,
                    vec![Operand::Rel(v)],
                ))
            }
            0x90..=0x9f => {
                let m = self.modrm(p, Size::B, asz, b, i)?;
                Some((
                    setcc_name(op & 0x0f),
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, Size::B, p)],
                ))
            }
            0xa2 => Some(("cpuid", FlowKind::Next, None, vec![])),
            // Bit test and modify, and the double-precision shifts. Both appear
            // in bitset and switch lowering.
            0xa3 | 0xab | 0xb3 | 0xbb => {
                let m = self.modrm(p, os, asz, b, i)?;
                let name = match op {
                    0xa3 => "bt",
                    0xab => "bts",
                    0xb3 => "btr",
                    _ => "btc",
                };
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, os, p), reg_operand(&m, os, p)],
                ))
            }
            0xba => {
                let m = self.modrm(p, os, asz, b, i)?;
                let name = match m.reg & 7 {
                    4 => "bt",
                    5 => "bts",
                    6 => "btr",
                    7 => "btc",
                    _ => return None,
                };
                let v = self.imm(b, i, 1)?;
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, os, p), Operand::Imm(v, Size::B)],
                ))
            }
            0xa4 | 0xa5 | 0xac | 0xad => {
                let m = self.modrm(p, os, asz, b, i)?;
                let name = if op & 0x08 == 0 { "shld" } else { "shrd" };
                let mut ops = vec![mem_operand(&m, os, p), reg_operand(&m, os, p)];
                if op & 1 == 0 {
                    ops.push(Operand::Imm(self.imm(b, i, 1)?, Size::B));
                } else {
                    ops.push(Operand::Reg("cl"));
                }
                Some((name, FlowKind::Next, None, ops))
            }
            0xb0 | 0xb1 => {
                let size = if op == 0xb0 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                Some((
                    "cmpxchg",
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, size, p), reg_operand(&m, size, p)],
                ))
            }
            // `f3` turns these into the trailing-zero/leading-zero counts, which
            // compilers emit for `x & -x` and `31 - clz(x)`.
            0xbc | 0xbd => {
                let m = self.modrm(p, os, asz, b, i)?;
                let name = match (op, p.rep) {
                    (0xbc, Some(0xf3)) => "tzcnt",
                    (0xbd, Some(0xf3)) => "lzcnt",
                    (0xbc, _) => "bsf",
                    _ => "bsr",
                };
                Some((
                    name,
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, os, p), mem_operand(&m, os, p)],
                ))
            }
            0xb8 => {
                if p.rep != Some(0xf3) {
                    return None;
                }
                let m = self.modrm(p, os, asz, b, i)?;
                Some((
                    "popcnt",
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, os, p), mem_operand(&m, os, p)],
                ))
            }
            0xc7 => {
                // Group 9: the reg field selects, and only the memory forms are
                // defined.
                let m = self.modrm(p, os, asz, b, i)?;
                let name = match (m.reg & 7, p.rep) {
                    (1, _) => {
                        if os == Size::Q {
                            "cmpxchg16b"
                        } else {
                            "cmpxchg8b"
                        }
                    }
                    // XRSTORS, XSAVEC and XSAVES share this group. None of
                    // them takes a mandatory prefix: `0F C7 /3`, `/4` and
                    // `/5` are the whole encoding, and `66` or `F3` in front
                    // of them is noise the listing prints as `data16` or
                    // `repz` rather than as part of the name.
                    (3, _) => {
                        if p.rex & 0x08 != 0 {
                            "xrstors64"
                        } else {
                            "xrstors"
                        }
                    }
                    (4, _) => {
                        if p.rex & 0x08 != 0 {
                            "xsavec64"
                        } else {
                            "xsavec"
                        }
                    }
                    (5, _) => {
                        if p.rex & 0x08 != 0 {
                            "xsaves64"
                        } else {
                            "xsaves"
                        }
                    }
                    (6, Some(0xf3)) => "rdpid",
                    (6, _) => "rdrand",
                    (7, _) => "rdseed",
                    _ => return None,
                };
                // The state-save pair and the compare-exchange pair take an
                // operand whose size the encoding does not state: the
                // CMPXCHG names already carry the width, and the XSAVE family
                // works on a state area whose extent is not a memory operand
                // size, so the listing prints the address alone.
                let mut mem = mem_operand(&m, os, p);
                if matches!(
                    name,
                    "xsavec"
                        | "xsavec64"
                        | "xsaves"
                        | "xsaves64"
                        | "xrstors"
                        | "xrstors64"
                        | "cmpxchg8b"
                        | "cmpxchg16b"
                ) {
                    if let Operand::Mem { size, .. } = &mut mem {
                        *size = None;
                    }
                }
                Some((name, FlowKind::Next, None, vec![mem]))
            }
            0xaf => {
                let m = self.modrm(p, os, asz, b, i)?;
                Some((
                    "imul",
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, os, p), mem_operand(&m, os, p)],
                ))
            }
            0xb6 | 0xb7 => {
                let src = if op == 0xb6 { Size::B } else { Size::W };
                let m = self.modrm(p, src, asz, b, i)?;
                Some((
                    "movzx",
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, os, p), mem_operand(&m, src, p)],
                ))
            }
            0xbe | 0xbf => {
                let src = if op == 0xbe { Size::B } else { Size::W };
                let m = self.modrm(p, src, asz, b, i)?;
                Some((
                    "movsx",
                    FlowKind::Next,
                    None,
                    vec![reg_operand(&m, os, p), mem_operand(&m, src, p)],
                ))
            }
            0xc0 | 0xc1 => {
                let size = if op == 0xc0 { Size::B } else { os };
                let m = self.modrm(p, size, asz, b, i)?;
                Some((
                    "xadd",
                    FlowKind::Next,
                    None,
                    vec![mem_operand(&m, size, p), reg_operand(&m, size, p)],
                ))
            }
            0xc8..=0xcf => {
                let r = ((op & 7) | p.rex_b()) as usize;
                Some((
                    "bswap",
                    FlowKind::Next,
                    None,
                    vec![Operand::Reg(if os == Size::Q {
                        REG64[r]
                    } else {
                        REG32[r]
                    })],
                ))
            }
            0x38 | 0x3a => self.decode_legacy_sse(op, p, os, asz, b, i),
            _ => self.decode_sse(op, p, os, asz, b, i),
        }
    }

    /// The legacy `0f38` and `0f3a` maps.
    ///
    /// These two maps have no prefix of their own: the mandatory prefix in
    /// front of `0f` is what says which instruction an opcode byte means, and
    /// the same byte means four different instructions depending on it. The
    /// tables were built from the encodings themselves, so the names and the
    /// operand shapes are facts rather than recollections.
    #[allow(clippy::too_many_arguments)]
    fn decode_legacy_sse(
        &mut self,
        map: u8,
        p: &Prefix,
        os: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        let op = *b.get(*i)?;
        *i += 1;

        // The mandatory prefix. Two of them at once contradict each other,
        // with one exception: an instruction whose own prefix is `f2` and
        // whose operand size is set by `66` carries both bytes, and the
        // table is the only thing that knows which instructions those are.
        // `66 f2 0f 38 f1` is `crc32 eax, cx`; the `66` says how wide the
        // source is and says nothing about which instruction this is.
        let pp = match (p.op66, p.rep) {
            (true, None) => 1,
            (true, Some(0xf3 | 0xf2)) => {
                let rep_pp = if p.rep == Some(0xf3) { 2 } else { 3 };
                match sse_map::lookup(map, op, rep_pp) {
                    Some(e) if e.size66 => rep_pp,
                    _ => return None,
                }
            }
            (false, Some(0xf3)) => 2,
            (false, Some(0xf2)) => 3,
            (false, None) => 0,
            (false, Some(_)) | (true, Some(_)) => return None,
        };
        let entry = sse_map::lookup(map, op, pp)?;

        // The width of the r/m operand. The general-purpose forms follow the
        // operand-size prefix, except the ones the table says work on a byte.
        let rm_size = match entry.form {
            sse_map::SseForm::GpReg
            | sse_map::SseForm::GpMemStore
            | sse_map::SseForm::GpRegMem => {
                if entry.rm == Size::D {
                    os
                } else {
                    entry.rm
                }
            }
            _ => entry.rm,
        };
        let m = self.modrm(p, rm_size, asz, b, i)?;

        // A general-purpose operand of an SSE instruction is named at the
        // width the instruction works with, not at the width of the
        // element it moves: `pextrb eax, xmm0, 0x1` takes one byte out of
        // the vector and leaves it in a dword register. The exception is
        // the pair the operand-width bit splits, where the register is as
        // wide as the bit says.
        let gp_rm_size = if !entry.name_w.is_empty() && p.rex & 0x08 != 0 {
            Size::Q
        } else {
            Size::D
        };
        let gp_rm = |rm: &ModRm| {
            if rm.mem.is_some() {
                mem_operand(rm, rm_size, p)
            } else {
                // The `crc32` family reads its source at the width the
                // instruction works at, which the operand-size prefix and
                // the `W` bit set: `crc32 eax, cl`, `crc32 eax, cx`,
                // `crc32 rax, rcx`. The extraction family reads a vector
                // element and leaves it in a full-width register, so its
                // r/m register is a container and not the operand.
                let size = if matches!(entry.form, sse_map::SseForm::GpReg) {
                    rm_size
                } else {
                    gp_rm_size
                };
                Operand::Reg(size.regs(p.rex_present)[usize::from(rm.rm) & 15])
            }
        };

        let vec_dst = Operand::Xmm(m.reg & 15);
        // The `66` that named the operand size of an instruction does not
        // reach its destination: `crc32 eax, cx` still writes a dword, and
        // only `rex.W` widens that.
        let dst_size = if entry.size66 {
            if p.rex & 0x08 != 0 { Size::Q } else { Size::D }
        } else {
            os
        };
        let gp_dst = gp_reg_operand(m.reg & 15, dst_size, p);
        let mut ops = match entry.form {
            sse_map::SseForm::VecReg => vec![vec_dst, xmm_operand(&m, rm_size)],
            sse_map::SseForm::VecMem => {
                // `movntdqa` has no register form: the source is always
                // memory.
                let mem = m.mem.as_ref()?;
                vec![vec_dst, mem_ref_operand(mem, rm_size)]
            }
            sse_map::SseForm::GpStore => vec![gp_rm(&m), vec_dst],
            sse_map::SseForm::GpReg => vec![gp_dst, gp_rm(&m)],
            sse_map::SseForm::GpMemStore => {
                // `movbe` and the atomic-memory instructions write to memory;
                // the r/m field must be a memory reference.
                let mem = m.mem.as_ref()?;
                vec![
                    mem_ref_operand(mem, rm_size),
                    gp_reg_operand(m.reg & 15, os, p),
                ]
            }
            sse_map::SseForm::GpRegMem => {
                let mem = m.mem.as_ref()?;
                vec![gp_dst, mem_ref_operand(mem, rm_size)]
            }
            sse_map::SseForm::VecGp => vec![vec_dst, gp_rm(&m)],
            sse_map::SseForm::VecRegXmm0 => {
                // The third operand is the implicit `xmm0` that the
                // instruction blends in.
                vec![vec_dst, xmm_operand(&m, rm_size), Operand::Xmm(0)]
            }
            sse_map::SseForm::MemOnly => {
                let mem = m.mem.as_ref()?;
                vec![mem_ref_operand(mem, rm_size)]
            }
        };
        // The operand-width bit names a different instruction for the
        // pairs the table records two names for: `pinsrd` moves a dword
        // and `pinsrq` a qword, and they are one opcode.
        let mut name = if !entry.name_w.is_empty() && p.rex & 0x08 != 0 {
            entry.name_w
        } else {
            entry.name
        };
        if entry.imm8 {
            let v = self.imm(b, i, 1)?;
            // The carry-less multiplication names the quadwords it works
            // on, and the immediate byte says which ones. A byte that
            // names no choice leaves the plain name in place.
            if map == 0x3a && op == 0x44 {
                if let Some(n) = pclmul::name(false, v as u8) {
                    name = n;
                }
            }
            ops.push(Operand::Imm(v, Size::B));
        }
        Some((name, FlowKind::Next, None, ops))
    }

    /// SSE/MMX opcodes.
    ///
    /// Every instruction here has the same shape: a ModRM byte, an optional
    /// immediate, and a mnemonic chosen by the mandatory prefix. The table says
    /// which of those apply; the special cases below are the handful of
    /// instructions whose operands are not `xmm, xmm/mem`.
    #[allow(clippy::too_many_arguments)]
    fn decode_sse(
        &mut self,
        op: u8,
        p: &Prefix,
        os: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        // `emms` is the only opcode in this range with no ModRM byte.
        if op == 0x77 {
            if p.op66 || p.rep.is_some() {
                return None;
            }
            return Some(("emms", FlowKind::Next, None, vec![]));
        }
        // The shift groups pick their mnemonic from the reg field, not from a
        // prefix, and always take an immediate count.
        if (0x71..=0x73).contains(&op) {
            let m = self.modrm(p, Size::X, asz, b, i)?;
            let name = match (op, m.reg & 7) {
                (0x71, 2) => "psrlw",
                (0x71, 4) => "psraw",
                (0x71, 6) => "psllw",
                (0x72, 2) => "psrld",
                (0x72, 4) => "psrad",
                (0x72, 6) => "pslld",
                (0x73, 2) => "psrlq",
                (0x73, 3) if p.op66 => "psrldq",
                (0x73, 6) => "psllq",
                (0x73, 7) if p.op66 => "pslldq",
                _ => return None,
            };
            let v = self.imm(b, i, 1)?;
            let dst = if p.op66 {
                xmm_operand(&m, Size::X)
            } else {
                mm_operand(&m, Size::X)
            };
            return Some((name, FlowKind::Next, None, vec![dst, Operand::Imm(v, Size::B)]));
        }
        // Two mandatory prefixes at once is not an encoding.
        let idx = match (p.op66, p.rep) {
            (true, Some(_)) => return None,
            (true, None) => 1,
            (false, Some(0xf3)) => 2,
            (false, Some(0xf2)) => 3,
            (false, Some(_)) => return None,
            (false, None) => 0,
        };
        // Conversions move between vector and general-purpose registers, so
        // which operand is which depends on the prefix.
        if op == 0x2a {
            let m = self.modrm(p, Size::D, asz, b, i)?;
            let dst = Operand::Xmm(m.reg & 15);
            let src = match idx {
                0 | 1 => mm_operand(&m, Size::Q),
                _ => mem_operand(&m, if os == Size::Q { Size::Q } else { Size::D }, p),
            };
            let name = ["cvtpi2ps", "cvtpi2pd", "cvtsi2ss", "cvtsi2sd"][idx];
            return Some((name, FlowKind::Next, None, vec![dst, src]));
        }
        if op == 0x2c || op == 0x2d {
            // The opcode says whether the conversion truncates or rounds:
            // `0f 2c` truncates and `0f 2d` rounds. The prefix says what
            // it converts.
            let m = self.modrm(p, Size::D, asz, b, i)?;
            let (name, dst, src) = match idx {
                0 | 1 => {
                    let n = match (op, idx) {
                        (0x2c, 0) => "cvttps2pi",
                        (0x2d, 0) => "cvtps2pi",
                        (0x2c, _) => "cvttpd2pi",
                        (0x2d, _) => "cvtpd2pi",
                        _ => unreachable!(),
                    };
                    (n, Operand::Mm(m.reg & 7), xmm_operand(&m, Size::Q))
                }
                _ => {
                    let n = match (op, idx) {
                        (0x2c, 2) => "cvttss2si",
                        (0x2d, 2) => "cvtss2si",
                        (0x2c, _) => "cvttsd2si",
                        (0x2d, _) => "cvtsd2si",
                        _ => unreachable!(),
                    };
                    let r = if os == Size::Q { REG64 } else { REG32 };
                    (
                        n,
                        Operand::Reg(r[usize::from(m.reg & 15)]),
                        xmm_operand(&m, if idx == 2 { Size::D } else { Size::Q }),
                    )
                }
            };
            return Some((name, FlowKind::Next, None, vec![dst, src]));
        }

        let entry = SSE_OPS.iter().find(|e| e.op == op)?;
        let mut name = entry.names[idx];
        if name.is_empty() {
            return None;
        }
        let size = sse_mem_size(name);
        let m = self.modrm(p, size, asz, b, i)?;
        // Two of these opcodes have a register form with another name: the
        // register form of `movlps` moves the high half of the destination
        // into the low half of the source, and the register form of `movhps`
        // does the opposite. The memory forms of the same opcodes are
        // loads and stores of a single 64-bit half.
        //
        // Only the prefix-free encodings are renamed. Under `F2` and `F3`
        // the same opcodes are `movddup`, `movsldup` and `movshdup`, which
        // keep their names in both forms: the register form of `movddup` is
        // still `movddup`.
        if m.mode == 3 {
            match (op, idx) {
                (0x12, 0) => name = "movhlps",
                (0x16, 0) => name = "movlhps",
                (0x13 | 0x17, _) => return None,
                _ => {}
            }
        }
        // `movd` with a 64-bit operand is `movq`.
        if entry.gp_rm && os == Size::Q && name == "movd" {
            name = "movq";
        }
        // `pextrw r32, xmm, imm8` puts the general-purpose register in the reg
        // field, the opposite of every other instruction in the table.
        let gp_reg = entry.gp_reg || op == 0xc5;
        let reg = if gp_reg {
            let r = if os == Size::Q { REG64 } else { REG32 };
            Operand::Reg(r[usize::from(m.reg & 15)])
        } else if entry.mmx && idx == 0 {
            Operand::Mm(m.reg & 7)
        } else {
            Operand::Xmm(m.reg & 15)
        };
        let rm = if entry.gp_rm {
            mem_operand(&m, if os == Size::Q { Size::Q } else { Size::D }, p)
        } else if entry.mmx && idx == 0 && !gp_reg {
            mm_operand(&m, size)
        } else if entry.mmx && gp_reg {
            // `pmovmskb r32, mm` — the source is still an MMX register.
            if m.mem.is_none() {
                Operand::Mm(m.rm & 7)
            } else {
                mem_ref_operand(m.mem.as_ref()?, size)
            }
        } else {
            // Both the register and the memory form of a packed operand read as
            // an XMM register or a sized memory reference.
            xmm_operand(&m, size)
        };
        let mut ops = if entry.store {
            vec![rm, reg]
        } else {
            vec![reg, rm]
        };
        if entry.imm8 {
            let v = self.imm(b, i, 1)?;
            if entry.cmp {
                // The comparison is named after its condition, and the
                // condition is the immediate byte.
                name = LEGACY_CMP_NAMES[usize::from(v as u8 & 0x0f)][idx];
            }
            ops.push(Operand::Imm(v, Size::B));
        }
        Some((name, FlowKind::Next, None, ops))
    }

    fn modrm(
        &mut self,
        p: &Prefix,
        size: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<ModRm> {
        if *i >= b.len() {
            return None;
        }
        let byte = b[*i];
        *i += 1;
        let mode = byte >> 6;
        let reg = ((byte >> 3) & 7) | p.rex_r();
        let rm_low = byte & 7;
        let rm = rm_low | p.rex_b();

        if mode == 3 {
            return Some(ModRm {
                mode,
                reg,
                rm,
                mem: None,
            });
        }

        let mut base: Option<u8> = None;
        let mut index: Option<u8> = None;
        let mut scale = 1u8;
        let mut disp = 0i64;
        let mut rip_relative = false;

        if rm_low == 4 {
            // SIB byte.
            if *i >= b.len() {
                return None;
            }
            let sib = b[*i];
            *i += 1;
            scale = 1 << (sib >> 6);
            let idx_low = (sib >> 3) & 7;
            // An index field of 4 with no REX.X means "no index".
            let idx = idx_low | ((p.rex >> 1 & 1) << 3);
            if idx_low != 4 || p.rex >> 1 & 1 != 0 {
                index = Some(idx);
            }
            let base_low = sib & 7;
            if base_low == 5 && mode == 0 {
                // No base, displacement follows.
                disp = self.imm(b, i, 4)?;
            } else {
                base = Some(base_low | p.rex_b());
            }
        } else if rm_low == 5 && mode == 0 {
            if asz == Size::Q && self.bits == 64 {
                // RIP-relative: displacement is from the end of the
                // instruction, which is not yet known. Stored relative here and
                // fixed up by the caller once the length is settled.
                let d = self.imm(b, i, 4)?;
                rip_relative = true;
                disp = d;
            } else {
                disp = self.imm(b, i, 4)?;
            }
        } else {
            base = Some(rm);
        }

        if mode == 1 {
            disp = self.imm(b, i, 1)?;
        } else if mode == 2 {
            disp = self.imm(b, i, 4)?;
        }

        let _ = size;
        Some(ModRm {
            mode,
            reg,
            rm,
            mem: Some(MemRef {
                base,
                index,
                scale,
                disp,
                rip_relative,
            }),
        })
    }

    fn imm(&mut self, b: &[u8], i: &mut usize, width: u8) -> Option<i64> {
        let w = width as usize;
        if *i + w > b.len() {
            return None;
        }
        let mut v: u64 = 0;
        for k in 0..w {
            v |= u64::from(b[*i + k]) << (8 * k);
        }
        *i += w;
        // Sign-extend to 64 bits, because that is what the encoding means:
        // an `imm8` in `add eax, imm8` is a signed field. How the listing
        // prints it is a separate question, and the width carried by
        // `Operand::Imm` answers that one.
        let shift = 64 - 8 * w as u32;
        Some(((v << shift) as i64) >> shift)
    }

    /// The width of a value pushed on the stack.
    ///
    /// A push moves a whole stack slot whatever the encoded immediate is:
    /// in 64-bit mode the slot is 64 bits and an `imm8` is sign-extended
    /// into it, which is why `6a ff` prints as `push 0xffffffffffffffff`.
    /// The `66` prefix is the one thing that narrows the slot.
    fn push_size(&self, os: Size) -> Size {
        if os == Size::W {
            Size::W
        } else if self.bits == 64 {
            Size::Q
        } else {
            Size::D
        }
    }

    /// Absolute target of a direct `call`/`jmp` through memory, when the
    /// operand is a plain absolute address.
    fn effective_target(&self, m: &ModRm) -> Option<CallTarget> {
        let mem = m.mem.as_ref()?;
        if mem.rip_relative {
            // The displacement is not yet resolvable to an address; a null
            // placeholder marks the slot so the caller can substitute the real
            // one once the instruction length is known.
            return Some(CallTarget::Indirect {
                via: Some(Addr::NULL),
            });
        }
        if mem.base.is_none() && mem.index.is_none() {
            return Some(CallTarget::Indirect {
                via: Some(Addr(mem.disp as u64)),
            });
        }
        None
    }

    fn render(&self, p: &Prefix, mnemonic: &str, ops: &[Operand], len: usize) -> String {
        let mut s = String::with_capacity(40);
        if p.lock {
            s.push_str("lock ");
        }
        if let Some(r) = p.rep
            && !mnemonic.starts_with("endbr")
            // A repeat prefix means a repeat for the instructions that
            // actually repeat, and for nothing else. Everywhere else `f3` and
            // `f2` are mandatory prefixes that select the instruction: they
            // are part of the opcode, not a count, and printing them as `rep`
            // would be wrong.
            && (is_string_op(mnemonic) || mnemonic == "ret")
        {
            s.push_str(if r == 0xf3 { "rep " } else { "repne " });
        }
        s.push_str(mnemonic);

        // A branch target is rendered as an absolute address, because that is
        // what a reader needs; the raw displacement is in `Operand::Rel`.
        let rendered: Vec<String> = ops
            .iter()
            .map(|o| self.render_operand(o, p.seg, len))
            .collect();
        if !rendered.is_empty() {
            s.push(' ');
            s.push_str(&rendered.join(", "));
        }
        s
    }

    fn render_operand(&self, op: &Operand, seg: Option<u8>, len: usize) -> String {
        match op {
            Operand::Reg(r) => (*r).to_string(),
            Operand::Xmm(n) => format!("xmm{n}"),
            Operand::Mm(n) => format!("mm{n}"),
            Operand::Vec { reg, mask, zeroing } => {
                let mut s = match reg {
                    vector::VecReg::Xmm(n) => format!("xmm{n}"),
                    vector::VecReg::Ymm(n) => format!("ymm{n}"),
                    vector::VecReg::Zmm(n) => format!("zmm{n}"),
                    vector::VecReg::Mask(n) => format!("k{n}"),
                };
                // `{k1}` is the write mask, and `{z}` beside it says the
                // lanes it does not select are zeroed instead of kept. A
                // mask of zero is not a mask: the encoding reads it as
                // "unmasked", which is why it is not printed.
                if *mask != 0 {
                    s.push_str(&format!("{{k{mask}}}"));
                    if *zeroing {
                        s.push_str("{z}");
                    }
                }
                s
            }
            Operand::St(n) => format!("st({n})"),
            Operand::Imm(v, size) => {
                // The immediate is a field as wide as the operand it
                // belongs to, printed as that many hex digits with no sign:
                // `add eax, 0xffffffff`, not `add eax, -1`.
                let bits = match size {
                    Size::B => 8,
                    Size::W => 16,
                    Size::D => 32,
                    // A wider operand than this has no immediate form; the
                    // 64-bit mask is what a listing would print anyway.
                    _ => 64,
                };
                let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
                format!("{:#x}", (*v as u64) & mask)
            }
            Operand::Rel(v) => {
                // A relative displacement counts from the *end* of the
                // instruction, not from its first byte. Using the origin here
                // made every `jmp` and `jcc` print a target one instruction
                // length too low, disagreeing with `Instruction::target`.
                let target = self.origin().add(len as u64).offset(*v);
                if *v < 0 {
                    format!("{target} (rel -{:#x})", v.unsigned_abs())
                } else {
                    format!("{target} (rel +{:#x})", v)
                }
            }
            Operand::Mem {
                base,
                index,
                scale,
                disp,
                rip_relative,
                size,
                broadcast,
                vsib,
                mask,
            } => {
                let mut s = String::new();
                if let Some(sz) = size {
                    s.push_str(sz.tag());
                }
                // A segment override changes which address space the operand
                // names. `fs:0x28` is the stack canary; the same displacement
                // without the prefix is an unrelated absolute address.
                if let Some(seg_byte) = seg {
                    s.push_str(match seg_byte {
                        0x64 => "fs:",
                        0x65 => "gs:",
                        0x2e => "cs:",
                        0x36 => "ss:",
                        0x3e => "ds:",
                        0x26 => "es:",
                        _ => "",
                    });
                } else if base.is_none() && index.is_none() && !*rip_relative {
                    // An address with no base and no index is absolute,
                    // and an absolute address is read through a segment:
                    // the listing says which one rather than leaving it
                    // to be assumed, and it does not bracket the number.
                    s.push_str("ds:");
                }
                if *rip_relative {
                    // Signed, because the target is usually behind the
                    // instruction and `0xffffffffffffe051` hides that.
                    if *disp < 0 {
                        s.push_str(&format!("[rip-{:#x}]", disp.unsigned_abs()));
                    } else {
                        s.push_str(&format!("[rip+{disp:#x}]"));
                    }
                } else if base.is_none() && index.is_none() {
                    s.push_str(&format!("{:#x}", *disp as u64));
                } else {
                    s.push('[');
                    // Address registers follow the address size, which is the
                    // machine width unless a 0x67 prefix narrowed it.
                    let regs: &[&str; 16] = if self.bits == 64 { &REG64 } else { &REG32 };
                    let mut first = true;
                    if let Some(b) = base {
                        s.push_str(regs[usize::from(*b & 15)]);
                        first = false;
                    }
                    if let Some(ix) = index {
                        if !first {
                            s.push_str(" + ");
                        }
                        // A vector index register is named at the width
                        // of the vector the instruction works on: the
                        // gather reads its lanes from `ymm1`, not from
                        // `rcx`.
                        match vsib {
                            Some(bits) => {
                                let prefix = match bits {
                                    256 => "ymm",
                                    512 => "zmm",
                                    _ => "xmm",
                                };
                                // A vector index has thirty-two names,
                                // not sixteen: EVEX spends a second bit
                                // on it.
                                s.push_str(&format!("{prefix}{}", ix & 0x1f));
                            }
                            None => s.push_str(regs[usize::from(*ix & 15)]),
                        }
                        if *scale > 1 {
                            s.push_str(&format!("*{scale}"));
                        }
                        first = false;
                    }
                    if *disp != 0 || first {
                        if !first {
                            if *disp < 0 {
                                s.push_str(&format!(" - {:#x}", disp.unsigned_abs()));
                            } else {
                                s.push_str(&format!(" + {disp:#x}"));
                            }
                        } else if *disp < 0 {
                            s.push_str(&format!("-{:#x}", disp.unsigned_abs()));
                        } else {
                            s.push_str(&format!("{disp:#x}"));
                        }
                    }
                    s.push(']');
                }
                // The broadcast is written after the operand it applies
                // to: `dword ptr [rax]{1to16}` is one dword read and
                // copied to sixteen lanes.
                if let Some(n) = broadcast {
                    s.push_str(&format!("{{1to{n}}}"));
                }
                // The scatter writes memory under a mask, and the mask is
                // written on the operand it guards.
                if *mask != 0 {
                    s.push_str(&format!("{{k{mask}}}"));
                }
                s
            }
        }
    }
}

#[derive(Default)]
struct Prefix {
    op66: bool,
    op67: bool,
    lock: bool,
    rep: Option<u8>,
    seg: Option<u8>,
    rex: u8,
    rex_present: bool,
    count: usize,
}



impl Prefix {
    /// REX.R as a register-number offset: 0 or 8.
    fn rex_r(&self) -> u8 {
        ((self.rex >> 2) & 1) << 3
    }
    /// REX.B as a register-number offset: 0 or 8.
    fn rex_b(&self) -> u8 {
        (self.rex & 1) << 3
    }
}

struct MemRef {
    base: Option<u8>,
    index: Option<u8>,
    scale: u8,
    disp: i64,
    rip_relative: bool,
}

struct ModRm {
    mode: u8,
    reg: u8,
    rm: u8,
    mem: Option<MemRef>,
}

fn imm_width(size: Size) -> u8 {
    match size {
        Size::B => 1,
        Size::W => 2,
        Size::D => 4,
        Size::Q => 4, // most 64-bit immediates are sign-extended 32-bit
        Size::X | Size::Y | Size::Z | Size::T => 4,
    }
}

fn reg_operand(m: &ModRm, size: Size, p: &Prefix) -> Operand {
    let regs = size.regs(p.rex_present);
    Operand::Reg(regs[m.reg as usize & 15])
}

/// A general-purpose register named by its number.
fn gp_reg_operand(n: u8, size: Size, p: &Prefix) -> Operand {
    Operand::Reg(size.regs(p.rex_present)[usize::from(n & 15)])
}

/// The register forms of group 7, the one that holds the privileged and the
/// control-flow-integrity instructions.
///
/// The group holds more than twenty instructions, and several of them differ
/// only in the r/m field, so the table was taken from the encodings rather
/// than from the manual. A mandatory prefix picks a different instruction
/// for the same encoding: `0f 01 ee` is `rdpkru` on its own and `clui` with
/// an `f3` prefix in front.
fn group7_mod3(reg: u8, rm: u8, p: &Prefix) -> Option<&'static str> {
    let key = (reg & 7, rm & 7);
    match p.rep {
        Some(0xf3) => match key {
            (0, 6) => return Some("wrmsrlist"),
            (1, 2) => return Some("eretu"),
            (3, 1) => return Some("vmgexit"),
            (5, 0) => return Some("setssbsy"),
            (5, 2) => return Some("saveprevssp"),
            (5, 4) => return Some("uiret"),
            (5, 5) => return Some("testui"),
            (5, 6) => return Some("clui"),
            (5, 7) => return Some("stui"),
            (7, 2) => return Some("mcommit"),
            (7, 5) => return Some("rmpquery"),
            (7, 6) => return Some("rmpadjust"),
            (7, 7) => return Some("psmash"),
            _ => {}
        },
        Some(0xf2) => match key {
            (0, 6) => return Some("rdmsrlist"),
            (1, 2) => return Some("erets"),
            (3, 1) => return Some("vmgexit"),
            (5, 0) => return Some("xsusldtrk"),
            (5, 1) => return Some("xresldtrk"),
            (7, 6) => return Some("rmpupdate"),
            (7, 7) => return Some("pvalidate"),
            _ => {}
        },
        _ => {}
    }
    if p.op66 {
        match key {
            (1, 4) => return Some("tdcall"),
            (1, 5) => return Some("seamret"),
            (1, 6) => return Some("seamops"),
            (1, 7) => return Some("seamcall"),
            _ => {}
        }
    }
    Some(match key {
        (0, 0) => "enclv",
        (0, 1) => "vmcall",
        (0, 2) => "vmlaunch",
        (0, 3) => "vmresume",
        (0, 4) => "vmxoff",
        (0, 5) => "pconfig",
        (0, 6) => "wrmsrns",
        (0, 7) => "pbndkb",
        (1, 0) => "monitor",
        (1, 1) => "mwait",
        (1, 2) => "clac",
        (1, 3) => "stac",
        (1, 7) => "encls",
        (2, 0) => "xgetbv",
        (2, 1) => "xsetbv",
        (2, 4) => "vmfunc",
        (2, 5) => "xend",
        (2, 6) => "xtest",
        (2, 7) => "enclu",
        (3, 0) => "vmrun",
        (3, 1) => "vmmcall",
        (3, 2) => "vmload",
        (3, 3) => "vmsave",
        (3, 4) => "stgi",
        (3, 5) => "clgi",
        (3, 6) => "skinit",
        (3, 7) => "invlpga",
        (4, _) => "smsw",
        (5, 0) => "serialize",
        (5, 6) => "rdpkru",
        (5, 7) => "wrpkru",
        (6, _) => "lmsw",
        (7, 0) => "swapgs",
        (7, 1) => "rdtscp",
        (7, 2) => "monitorx",
        (7, 3) => "mwaitx",
        (7, 4) => "clzero",
        (7, 5) => "rdpru",
        (7, 6) => "invlpgb",
        (7, 7) => "tlbsync",
        _ => return None,
    })
}

/// Whether the instruction is one the `rep` prefix makes repeat.
///
/// The string instructions are the only ones a repeat prefix applies to.
/// Everywhere else the prefix selects the instruction.
fn is_string_op(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "movsb" | "movs" | "cmpsb" | "cmps" | "stosb" | "stos" | "lodsb" | "lods" | "scasb"
            | "scas" | "insb" | "ins" | "outsb" | "outs"
    )
}

fn mem_operand(m: &ModRm, size: Size, p: &Prefix) -> Operand {
    match &m.mem {
        None => {
            let regs = size.regs(p.rex_present);
            Operand::Reg(regs[m.rm as usize & 15])
        }
        Some(mem) => Operand::Mem {
            base: mem.base,
            index: mem.index,
            scale: mem.scale,
            disp: mem.disp,
            rip_relative: mem.rip_relative,
            size: Some(size),
            broadcast: None,
            vsib: None,
            mask: 0,
        },
    }
}

/// One `0f xx` SSE/MMX opcode.
///
/// The mnemonic depends on the mandatory prefix, which is why the table has
/// four names per opcode rather than one: `0f 58` is `addps`, `addpd`, `addss`
/// or `addsd` depending on nothing else.
struct SseOp {
    op: u8,
    /// Mnemonic per prefix state: none, `66`, `f3`, `f2`. Empty means the
    /// combination is not an instruction.
    names: [&'static str; 4],
    /// The memory operand is the destination (a store form).
    store: bool,
    /// A trailing immediate byte.
    imm8: bool,
    /// The r/m operand is a general-purpose or MMX register, not an XMM one.
    gp_rm: bool,
    /// The `reg` field names a general-purpose register (`movmskps`, `pmovmskb`).
    gp_reg: bool,
    /// Without a prefix the register operands are MMX.
    mmx: bool,
    /// The instruction is named after the condition in its immediate byte.
    ///
    /// The packed comparisons are the case: `cmpps` with a condition of one
    /// is `cmpltps`, and a listing that calls them all `cmpps` has lost the
    /// only thing that tells them apart.
    cmp: bool,
}

/// `const` constructor so the table below stays readable.
const fn sse(op: u8, names: [&'static str; 4]) -> SseOp {
    SseOp {
        op,
        names,
        store: false,
        imm8: false,
        gp_rm: false,
        gp_reg: false,
        mmx: false,
        cmp: false,
    }
}

/// `const` constructor for the comparisons that are named after the
/// condition in their immediate byte.
const fn sse_cmp(op: u8, names: [&'static str; 4]) -> SseOp {
    SseOp {
        imm8: true,
        cmp: true,
        ..sse(op, names)
    }
}

const fn sse_store(op: u8, names: [&'static str; 4]) -> SseOp {
    SseOp {
        store: true,
        ..sse(op, names)
    }
}

const fn sse_imm(op: u8, names: [&'static str; 4]) -> SseOp {
    SseOp {
        imm8: true,
        ..sse(op, names)
    }
}

/// The names of the legacy packed comparison instructions.
///
/// The condition is part of the instruction's name: `cmpsd` and `cmpnlesd`
/// are different instructions, and a listing that calls them both `cmpsd`
/// has lost the only thing that tells them apart. The first eight
/// conditions have a name of their own; the other eight keep the plain
/// name, because the machine has no other spelling for them.
const LEGACY_CMP_NAMES: [[&str; 4]; 16] = [
    [
        "cmpeqps", "cmpeqpd", "cmpeqss", "cmpeqsd",
    ],
    [
        "cmpltps", "cmpltpd", "cmpltss", "cmpltsd",
    ],
    [
        "cmpleps", "cmplepd", "cmpless", "cmplesd",
    ],
    [
        "cmpunordps", "cmpunordpd", "cmpunordss", "cmpunordsd",
    ],
    [
        "cmpneqps", "cmpneqpd", "cmpneqss", "cmpneqsd",
    ],
    [
        "cmpnltps", "cmpnltpd", "cmpnltss", "cmpnltsd",
    ],
    [
        "cmpnleps", "cmpnlepd", "cmpnless", "cmpnlesd",
    ],
    [
        "cmpordps", "cmpordpd", "cmpordss", "cmpordsd",
    ],
    [
        "cmpps", "cmppd", "cmpss", "cmpsd",
    ],
    [
        "cmpps", "cmppd", "cmpss", "cmpsd",
    ],
    [
        "cmpps", "cmppd", "cmpss", "cmpsd",
    ],
    [
        "cmpps", "cmppd", "cmpss", "cmpsd",
    ],
    [
        "cmpps", "cmppd", "cmpss", "cmpsd",
    ],
    [
        "cmpps", "cmppd", "cmpss", "cmpsd",
    ],
    [
        "cmpps", "cmppd", "cmpss", "cmpsd",
    ],
    [
        "cmpps", "cmppd", "cmpss", "cmpsd",
    ],
];

/// The SSE, SSE2, SSE3 and SSSE3 opcode map. Ops that are not listed decode as
/// `.byte`, which keeps the stream aligned and shows exactly which opcode was
/// not understood.
const SSE_OPS: &[SseOp] = &[
    // Moves and conversions.
    sse(0x10, ["movups", "movupd", "movss", "movsd"]),
    sse_store(0x11, ["movups", "movupd", "movss", "movsd"]),
    sse(0x12, ["movlps", "movlpd", "movsldup", "movddup"]),
    sse_store(0x13, ["movlps", "movlpd", "", ""]),
    sse(0x14, ["unpcklps", "unpcklpd", "", ""]),
    sse(0x15, ["unpckhps", "unpckhpd", "", ""]),
    sse(0x16, ["movhps", "movhpd", "movshdup", ""]),
    sse_store(0x17, ["movhps", "movhpd", "", ""]),
    sse(0x28, ["movaps", "movapd", "", ""]),
    sse_store(0x29, ["movaps", "movapd", "", ""]),
    sse_store(0x2b, ["movntps", "movntpd", "movntss", "movntsd"]),
    sse(0x2e, ["ucomiss", "ucomisd", "", ""]),
    sse(0x2f, ["comiss", "comisd", "", ""]),
    // `movmskps`/`movmskpd` put a general-purpose register in the reg
    // field: the mask they build is an integer, not a vector.
    SseOp {
        gp_reg: true,
        ..sse(0x50, ["movmskps", "movmskpd", "", ""])
    },
    sse(0x51, ["sqrtps", "sqrtpd", "sqrtss", "sqrtsd"]),    sse(0x52, ["rsqrtps", "", "rsqrtss", ""]),
    sse(0x53, ["rcpps", "", "rcpss", ""]),
    sse(0x54, ["andps", "andpd", "", ""]),
    sse(0x55, ["andnps", "andnpd", "", ""]),
    sse(0x56, ["orps", "orpd", "", ""]),
    sse(0x57, ["xorps", "xorpd", "", ""]),
    sse(0x58, ["addps", "addpd", "addss", "addsd"]),
    sse(0x59, ["mulps", "mulpd", "mulss", "mulsd"]),
    sse(0x5a, ["cvtps2pd", "cvtpd2ps", "cvtss2sd", "cvtsd2ss"]),
    sse(0x5b, ["cvtdq2ps", "cvtps2dq", "cvttps2dq", ""]),
    sse(0x5c, ["subps", "subpd", "subss", "subsd"]),
    sse(0x5d, ["minps", "minpd", "minss", "minsd"]),
    sse(0x5e, ["divps", "divpd", "divss", "divsd"]),
    sse(0x5f, ["maxps", "maxpd", "maxss", "maxsd"]),
    // Integer SIMD. Without a prefix these are MMX instructions on mm0..mm7.
    SseOp {
        mmx: true,
        ..sse(0x60, ["punpcklbw", "punpcklbw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x61, ["punpcklwd", "punpcklwd", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x62, ["punpckldq", "punpckldq", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x63, ["packsswb", "packsswb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x64, ["pcmpgtb", "pcmpgtb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x65, ["pcmpgtw", "pcmpgtw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x66, ["pcmpgtd", "pcmpgtd", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x67, ["packuswb", "packuswb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x68, ["punpckhbw", "punpckhbw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x69, ["punpckhwd", "punpckhwd", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x6a, ["punpckhdq", "punpckhdq", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x6b, ["packssdw", "packssdw", "", ""])
    },
    sse(0x6c, ["", "punpcklqdq", "", ""]),
    sse(0x6d, ["", "punpckhqdq", "", ""]),
    SseOp {
        mmx: true,
        gp_rm: true,
        ..sse(0x6e, ["movd", "movd", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x6f, ["movq", "movdqa", "movdqu", ""])
    },
    SseOp {
        imm8: true,
        mmx: true,
        ..sse(0x70, ["pshufw", "pshufd", "pshufhw", "pshuflw"])
    },
    SseOp {
        mmx: true,
        ..sse(0x74, ["pcmpeqb", "pcmpeqb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x75, ["pcmpeqw", "pcmpeqw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0x76, ["pcmpeqd", "pcmpeqd", "", ""])
    },
    sse(0x7c, ["haddps", "haddpd", "", ""]),
    sse(0x7d, ["hsubps", "hsubpd", "", ""]),
    SseOp {
        mmx: true,
        gp_rm: true,
        ..sse(0x7e, ["movd", "movd", "movq", ""])
    },
    sse_store(0x7f, ["movq", "movdqa", "movdqu", ""]),
    // SSE3 additions and the integer set continued.
    sse(0xd0, ["addsubps", "addsubpd", "", ""]),
    SseOp {
        mmx: true,
        ..sse(0xd1, ["psrlw", "psrlw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xd2, ["psrld", "psrld", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xd3, ["psrlq", "psrlq", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xd4, ["paddq", "paddq", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xd5, ["pmullw", "pmullw", "", ""])
    },
    sse_store(0xd6, ["", "movq", "movq2dq", "movdq2q"]),
    // `pmovmskb` has no MMX form: the instruction was added with SSE2 and
    // reads an XMM register.
    SseOp {
        gp_reg: true,
        ..sse(0xd7, ["pmovmskb", "pmovmskb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xd8, ["psubusb", "psubusb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xd9, ["psubusw", "psubusw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xda, ["pminub", "pminub", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xdb, ["pand", "pand", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xdc, ["paddusb", "paddusb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xdd, ["paddusw", "paddusw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xde, ["pmaxub", "pmaxub", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xdf, ["pandn", "pandn", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xe0, ["pavgb", "pavgb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xe1, ["psraw", "psraw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xe2, ["psrad", "psrad", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xe3, ["pavgw", "pavgw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xe4, ["pmulhuw", "pmulhuw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xe5, ["pmulhw", "pmulhw", "", ""])
    },
    sse(0xe6, ["", "cvttpd2dq", "cvtdq2pd", "cvtpd2dq"]),
    sse_store(0xe7, ["movntq", "movntdq", "", ""]),
    SseOp {
        mmx: true,
        ..sse(0xe8, ["psubsb", "psubsb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xe9, ["psubsw", "psubsw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xea, ["pminsw", "pminsw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xeb, ["por", "por", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xec, ["paddsb", "paddsb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xed, ["paddsw", "paddsw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xee, ["pmaxsw", "pmaxsw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xef, ["pxor", "pxor", "", ""])
    },
    sse(0xf0, ["", "", "", "lddqu"]),
    SseOp {
        mmx: true,
        ..sse(0xf1, ["psllw", "psllw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xf2, ["pslld", "pslld", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xf3, ["psllq", "psllq", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xf4, ["pmuludq", "pmuludq", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xf5, ["pmaddwd", "pmaddwd", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xf6, ["psadbw", "psadbw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xf7, ["maskmovq", "maskmovdqu", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xf8, ["psubb", "psubb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xf9, ["psubw", "psubw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xfa, ["psubd", "psubd", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xfb, ["psubq", "psubq", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xfc, ["paddb", "paddb", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xfd, ["paddw", "paddw", "", ""])
    },
    SseOp {
        mmx: true,
        ..sse(0xfe, ["paddd", "paddd", "", ""])
    },
    // Comparisons and shuffles with an immediate selector.
    sse_cmp(0xc2, ["cmpps", "cmppd", "cmpss", "cmpsd"]),
    sse_imm(0xc4, ["pinsrw", "pinsrw", "", ""]),
    sse_imm(0xc5, ["pextrw", "pextrw", "", ""]),
    sse_imm(0xc6, ["shufps", "shufpd", "", ""]),
];

/// x87 opcodes, indexed by the low three bits of the opcode and the ModRM `reg`
/// field. The memory forms are what `long double` code emits; the register forms
/// are named for the few that appear in practice and otherwise reported with
/// their own encoding.
const X87_MEM: [[&str; 8]; 8] = [
    ["fadd", "fmul", "fcom", "fcomp", "fsub", "fsubr", "fdiv", "fdivr"],
    ["fld", "", "fst", "fstp", "fldenv", "fldcw", "fnstenv", "fnstcw"],
    ["fiadd", "fimul", "ficom", "ficomp", "fisub", "fisubr", "fidiv", "fidivr"],
    ["fild", "fisttp", "fist", "fistp", "", "fld", "", "fstp"],
    ["fadd", "fmul", "fcom", "fcomp", "fsub", "fsubr", "fdiv", "fdivr"],
    ["fld", "fisttp", "fst", "fstp", "frstor", "", "fnsave", "fnstsw"],
    ["fiadd", "fimul", "ficom", "ficomp", "fisub", "fisubr", "fidiv", "fidivr"],
    ["fild", "fisttp", "fist", "fistp", "fbld", "fild", "fbstp", "fistp"],
];

/// Width of an x87 memory operand. The encodings differ per opcode and, in a
/// few slots, per register field.
fn x87_mem_size(op: u8, reg: u8) -> Size {
    match (op & 7, reg) {
        (0, _) => Size::D,          // D8: m32fp
        (1, 0) | (1, 2) | (1, 3) => Size::D, // D9: fld/fst/fstp m32fp
        (1, _) => Size::W,          // D9: fldcw/fnstcw m16
        (2, _) => Size::D,          // DA: m32int
        (3, 5) | (3, 7) => Size::T, // DB: m80fp
        (3, _) => Size::D,          // DB: m32int
        (4, _) => Size::Q,          // DC: m64fp
        (5, 7) => Size::W,          // DD /7: fnstsw m16
        (5, _) => Size::Q,          // DD: m64fp
        (6, _) => Size::W,          // DE: m16int
        (7, 4) | (7, 6) => Size::X, // DF: packed BCD
        (7, 5) | (7, 7) => Size::Q, // DF: m64int
        _ => Size::W,
    }
}

/// The register forms of the x87 instructions.
///
/// The key is the whole ModRM byte: these instructions differ only in the
/// `C0`-to-`FF` range, and masking the byte down to its opcode extension
/// loses exactly the bit that says which one it is.
///
/// The names came from the encodings rather than from the manual, because
/// the two disagree about the subtraction and division families. The
/// encodings are what the hardware runs, and the disagreement is the
/// historical accident that made the two names for the same opcode mean
/// opposite things in the two syntaxes.
fn x87_register_form(op: u8, modrm: u8) -> Option<(&'static str, Vec<Operand>)> {
    let i = modrm & 7;
    let reg = (modrm >> 3) & 7;
    let two = |a: u8, b: u8| vec![Operand::St(a), Operand::St(b)];
    let one = |a: u8| vec![Operand::St(a)];

    // The instructions that name no stack slot at all. They are checked
    // first, because a few of them share an encoding range with the
    // instructions that do name one: `d9 d0` is `fnop`, even though the
    // reg field of that byte says `fst`.
    let standalone = match (op & 7, modrm) {
        (1, 0xd0) => "fnop",
        (1, 0xe0) => "fchs",
        (1, 0xe1) => "fabs",
        (1, 0xe4) => "ftst",
        (1, 0xe5) => "fxam",
        (1, 0xe8) => "fld1",
        (1, 0xe9) => "fldl2t",
        (1, 0xea) => "fldl2e",
        (1, 0xeb) => "fldpi",
        (1, 0xec) => "fldlg2",
        (1, 0xed) => "fldln2",
        (1, 0xee) => "fldz",
        (1, 0xf0) => "f2xm1",
        (1, 0xf1) => "fyl2x",
        (1, 0xf2) => "fptan",
        (1, 0xf3) => "fpatan",
        (1, 0xf4) => "fxtract",
        (1, 0xf5) => "fprem1",
        (1, 0xf6) => "fdecstp",
        (1, 0xf7) => "fincstp",
        (1, 0xf8) => "fprem",
        (1, 0xf9) => "fyl2xp1",
        (1, 0xfa) => "fsqrt",
        (1, 0xfb) => "fsincos",
        (1, 0xfc) => "frndint",
        (1, 0xfd) => "fscale",
        (1, 0xfe) => "fsin",
        (1, 0xff) => "fcos",
        (2, 0xe9) => "fucompp",
        (3, 0xe0) => "feni",
        (3, 0xe1) => "fdisi",
        (3, 0xe2) => "fnclex",
        (3, 0xe3) => "fninit",
        (3, 0xe4) => "fnsetpm",
        (3, 0xe5) => "frstpm",
        (3, 0xe8..=0xef) => "fucomi",
        (3, 0xf0..=0xf7) => "fcomi",
        (7, 0xe0) => "fnstsw",
        (7, 0xe8..=0xef) => "fucomip",
        (7, 0xf0..=0xf7) => "fcomip",
        _ => "",
    };
    if !standalone.is_empty() {
        // The comparisons against `st(0)` name both operands; the store of
        // the status word names a general-purpose register.
        let ops = match standalone {
            "fucomi" | "fcomi" | "fucomip" | "fcomip" => two(0, i),
            "fnstsw" => vec![Operand::Reg("ax")],
            _ => Vec::new(),
        };
        return Some((standalone, ops));
    }

    match op {
        // The three arithmetic groups differ in where the result lands, not
        // in what it computes: `D8` writes `st(0)` from `st(i)`, `DC` writes
        // `st(i)` from `st(0)`, and `DE` writes `st(i)` and pops. The
        // arithmetic is the same in all three, so the operand order follows
        // the destination, as it does everywhere else.
        0xd8 => Some(match reg {
            0 => ("fadd", two(0, i)),
            1 => ("fmul", two(0, i)),
            2 => ("fcom", one(i)),
            3 => ("fcomp", one(i)),
            4 => ("fsub", two(0, i)),
            5 => ("fsubr", two(0, i)),
            6 => ("fdiv", two(0, i)),
            _ => ("fdivr", two(0, i)),
        }),
        // `DC /2` and `DC /3` are not encodings.
        0xdc => Some(match reg {
            0 => ("fadd", two(i, 0)),
            1 => ("fmul", two(i, 0)),
            4 => ("fsubr", two(i, 0)),
            5 => ("fsub", two(i, 0)),
            6 => ("fdivr", two(i, 0)),
            _ => ("fdiv", two(i, 0)),
        }),
        0xde => Some(match (reg, i) {
            (0, _) => ("faddp", two(i, 0)),
            (1, _) => ("fmulp", two(i, 0)),
            // `fcompp` has one encoding; the second is the same instruction
            // with the bit the manual calls C1 set.
            (2, 1) | (3, 1) => ("fcompp", Vec::new()),
            (4, _) => ("fsubrp", two(i, 0)),
            (5, _) => ("fsubp", two(i, 0)),
            (6, _) => ("fdivrp", two(i, 0)),
            _ => ("fdivp", two(i, 0)),
        }),
        0xd9 => Some(match reg {
            0 => ("fld", one(i)),
            1 => ("fxch", one(i)),
            2 => ("fst", one(i)),
            3 => ("fstp", one(i)),
            _ => return None,
        }),
        0xdd => Some(match reg {
            // `DD /0` is `ffree` in its register form; the `fld` of that
            // group is the memory form only.
            0 => ("ffree", one(i)),
            2 => ("fst", one(i)),
            3 => ("fstp", one(i)),
            4 => ("fucom", one(i)),
            5 => ("fucomp", one(i)),
            _ => return None,
        }),
        0xdf => Some(match reg {
            0 => ("ffreep", one(i)),
            _ => return None,
        }),
        // The conditional moves: the destination is `st(0)` and the source
        // is `st(i)`.
        0xda => Some(match reg {
            0 => ("fcmovb", two(0, i)),
            1 => ("fcmove", two(0, i)),
            2 => ("fcmovbe", two(0, i)),
            3 => ("fcmovu", two(0, i)),
            _ => return None,
        }),
        0xdb => Some(match reg {
            0 => ("fcmovnb", two(0, i)),
            1 => ("fcmovne", two(0, i)),
            2 => ("fcmovnbe", two(0, i)),
            3 => ("fcmovnu", two(0, i)),
            _ => return None,
        }),
        _ => None,
    }
}



/// Width of the memory operand, from the mnemonic's suffix.
fn sse_mem_size(name: &str) -> Size {
    // MOVDDUP reads one 64-bit element and writes it twice. Its name ends in
    // neither `ss` nor `sd`, so the suffix rules below would call it a full
    // vector and overstate what it reads. MOVSLDUP and MOVSHDUP do read a full
    // vector, and the default is right for them.
    if name == "movddup" {
        return Size::Q;
    }
    if name.ends_with("ss") || name.ends_with("d") && name.len() > 3 {
        Size::D
    } else if name.ends_with("sd") || name.ends_with('q') {
        Size::Q
    } else {
        Size::X
    }
}

/// Memory form of an r/m operand with an explicit width.
fn mem_ref_operand(mem: &MemRef, size: Size) -> Operand {
    Operand::Mem {
        base: mem.base,
        index: mem.index,
        scale: mem.scale,
        disp: mem.disp,
        rip_relative: mem.rip_relative,
        size: Some(size),
        broadcast: None,
        vsib: None,
        mask: 0,
    }
}

fn xmm_operand(m: &ModRm, size: Size) -> Operand {
    match &m.mem {
        None => Operand::Xmm(m.rm & 15),
        Some(mem) => mem_ref_operand(mem, size),
    }
}

fn mm_operand(m: &ModRm, size: Size) -> Operand {
    match &m.mem {
        None => Operand::Mm(m.rm & 7),
        Some(mem) => mem_ref_operand(mem, size),
    }
}

fn jcc_name(cond: u8) -> &'static str {
    const NAMES: [&str; 16] = [
        "jo", "jno", "jb", "jae", "je", "jne", "jbe", "ja", "js", "jns", "jp", "jnp", "jl", "jge",
        "jle", "jg",
    ];
    NAMES[(cond & 0x0f) as usize]
}

fn setcc_name(cond: u8) -> &'static str {
    const NAMES: [&str; 16] = [
        "seto", "setno", "setb", "setae", "sete", "setne", "setbe", "seta", "sets", "setns",
        "setp", "setnp", "setl", "setge", "setle", "setg",
    ];
    NAMES[(cond & 0x0f) as usize]
}

fn cmov_name(cond: u8) -> &'static str {
    const NAMES: [&str; 16] = [
        "cmovo", "cmovno", "cmovb", "cmovae", "cmove", "cmovne", "cmovbe", "cmova", "cmovs",
        "cmovns", "cmovp", "cmovnp", "cmovl", "cmovge", "cmovle", "cmovg",
    ];
    NAMES[(cond & 0x0f) as usize]
}

/// Unused but kept as the single source of truth for condition names.
#[allow(dead_code)]
fn condition_name(cond: u8) -> &'static str {
    CONDITIONS[(cond & 0x0f) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d64(bytes: &[u8]) -> Decoded {
        Decoder::new(64)
            .unwrap()
            .decode(Addr(0x1000), bytes)
            .unwrap()
    }

    fn text(bytes: &[u8]) -> String {
        d64(bytes).insn.text
    }

    #[test]
    fn alu_operand_direction_matches_the_encoding() {
        // `29 c2` is SUB r/m32, r32: the destination is edx, not eax.
        assert_eq!(text(&[0x29, 0xc2]), "sub edx, eax");
        assert_eq!(text(&[0x01, 0xd8]), "add eax, ebx");
        assert_eq!(text(&[0x39, 0xd8]), "cmp eax, ebx");
        // `03 45 fc` is ADD r32, r/m32, so the memory operand is the source.
        assert_eq!(text(&[0x03, 0x45, 0xfc]), "add eax, dword ptr [rbp - 0x4]");
        // `2b 45 fc` is SUB r32, r/m32.
        assert_eq!(text(&[0x2b, 0x45, 0xfc]), "sub eax, dword ptr [rbp - 0x4]");
    }

    #[test]
    fn segment_overrides_are_rendered() {
        // mov rax, qword ptr fs:0x28 — the stack canary. Without the prefix the
        // same displacement is an unrelated absolute address, and the listing
        // names the segment rather than bracketing the number: `fs:0x28`, not
        // `fs:[0x28]`.
        let t = text(&[0x64, 0x48, 0x8b, 0x04, 0x25, 0x28, 0, 0, 0]);
        assert!(t.contains("fs:0x28"), "{t}");
    }

    #[test]
    fn sse_opcodes_decode_with_the_right_length() {
        // (bytes, mnemonic, expected length). The lengths are what matters:
        // one byte short and every following instruction is misread.
        let cases: &[(&[u8], &str, u8)] = &[
            (&[0x66, 0x0f, 0xef, 0xc0], "pxor", 4),
            (&[0x0f, 0x28, 0x07], "movaps", 3),
            (&[0x0f, 0x29, 0x07], "movaps", 3),
            (&[0xf3, 0x0f, 0x10, 0x45, 0xfc], "movss", 5),
            (&[0xf2, 0x0f, 0x10, 0x05, 0x10, 0, 0, 0], "movsd", 8),
            (&[0x66, 0x0f, 0x6e, 0xc0], "movd", 4),
            (&[0x0f, 0x6f, 0x07], "movq", 3),
            (&[0x66, 0x0f, 0x73, 0xfa, 0x04], "pslldq", 5),
            (&[0x66, 0x0f, 0x70, 0xc1, 0x1b], "pshufd", 5),
            (&[0x0f, 0x77], "emms", 2),
        ];
        for (bytes, mnemonic, len) in cases {
            let d = d64(bytes);
            assert_eq!(d.insn.len, u32::from(*len), "{bytes:02x?} decoded as {}", d.insn.len);
            assert_eq!(d.insn.mnemonic.as_ref(), *mnemonic, "{}", d.insn.text);
        }
    }

    #[test]
    fn sse_operands_use_vector_registers() {
        assert_eq!(text(&[0x66, 0x0f, 0xef, 0xc0]), "pxor xmm0, xmm0");
        assert_eq!(text(&[0x0f, 0x28, 0x07]), "movaps xmm0, xmmword ptr [rdi]");
        assert_eq!(text(&[0xf3, 0x0f, 0x58, 0xc1]), "addss xmm0, xmm1");
        assert_eq!(text(&[0x0f, 0x6f, 0x07]), "movq mm0, qword ptr [rdi]");
    }

    #[test]
    fn the_x87_groups_name_the_destination_first() {
        // The three arithmetic groups write to different places: `D8` writes
        // `st(0)`, `DC` writes `st(i)`, `DE` writes `st(i)` and pops. The
        // operand order follows the destination, because that is the only
        // thing that distinguishes the three.
        assert_eq!(text(&[0xd8, 0xe1]), "fsub st(0), st(1)");
        // The subtraction and division families are named the other way round
        // in `DC` and `DE` than in `D8`. The names below were checked
        // against an independent decoder rather than written from memory,
        // because the two spellings of the same opcode have meant opposite
        // things in the two syntaxes for as long as there have been two
        // syntaxes.
        assert_eq!(text(&[0xdc, 0xe1]), "fsubr st(1), st(0)");
        assert_eq!(text(&[0xde, 0xe1]), "fsubrp st(1), st(0)");
        assert_eq!(text(&[0xde, 0xe9]), "fsubp st(1), st(0)");
        assert_eq!(text(&[0xd8, 0xf1]), "fdiv st(0), st(1)");
        assert_eq!(text(&[0xdc, 0xf9]), "fdiv st(1), st(0)");
        assert_eq!(text(&[0xde, 0xf1]), "fdivrp st(1), st(0)");
        // A comparison names only the other operand; the top is implied.
        assert_eq!(text(&[0xd8, 0xd1]), "fcom st(1)");
        assert_eq!(text(&[0xd8, 0xd9]), "fcomp st(1)");
        assert_eq!(text(&[0xde, 0xd9]), "fcompp");
        // `DC /2` and `DC /3` are not encodings.
        assert!(!text(&[0xdc, 0xd1]).starts_with("fcom"));
    }

    #[test]
    fn the_x87_instructions_that_were_missing_are_there_now() {
        // The conditional moves, the frees and the comparisons were absent,
        // and every one of them is three bytes of code that used to decode
        // as an unnamed `fpu` instruction.
        assert_eq!(text(&[0xda, 0xc1]), "fcmovb st(0), st(1)");
        assert_eq!(text(&[0xda, 0xd9]), "fcmovu st(0), st(1)");
        assert_eq!(text(&[0xdb, 0xc1]), "fcmovnb st(0), st(1)");
        assert_eq!(text(&[0xdb, 0xd9]), "fcmovnu st(0), st(1)");
        assert_eq!(text(&[0xdd, 0xc1]), "ffree st(1)");
        assert_eq!(text(&[0xdd, 0xe1]), "fucom st(1)");
        assert_eq!(text(&[0xdd, 0xe9]), "fucomp st(1)");
        assert_eq!(text(&[0xdf, 0xc1]), "ffreep st(1)");
        // The comparison against `st(0)`: the two families sit on either
        // side of the `db` opcode's extension field.
        assert_eq!(text(&[0xdb, 0xe9]), "fucomi st(0), st(1)");
        assert_eq!(text(&[0xdb, 0xf1]), "fcomi st(0), st(1)");
        assert_eq!(text(&[0xdf, 0xe9]), "fucomip st(0), st(1)");
        assert_eq!(text(&[0xdf, 0xf1]), "fcomip st(0), st(1)");
        // `d9 d0` is the no-operation, even though the reg field of that
        // byte says `fst`.
        assert_eq!(text(&[0xd9, 0xd0]), "fnop");
        assert_eq!(text(&[0xd9, 0xd1]), "fst st(1)");
        // The constants and the transcendentals.
        assert_eq!(text(&[0xd9, 0xe8]), "fld1");
        assert_eq!(text(&[0xd9, 0xee]), "fldz");
        assert_eq!(text(&[0xd9, 0xfa]), "fsqrt");
        assert_eq!(text(&[0xd9, 0xff]), "fcos");
    }

    #[test]
    fn the_legacy_vector_maps_decode() {
        // The `0f38` map: the instructions that arrived with SSSE3 and
        // SSE4.1.
        assert_eq!(text(&[0x0f, 0x38, 0x00, 0xc1]), "pshufb xmm0, xmm1");
        assert_eq!(text(&[0x66, 0x0f, 0x38, 0x17, 0xc1]), "ptest xmm0, xmm1");
        assert_eq!(text(&[0x66, 0x0f, 0x38, 0x40, 0xc1]), "pmulld xmm0, xmm1");
        // `movbe` moves a word the slow way, and it is the only instruction
        // in these maps that moves between a register and memory in the
        // direction its name suggests.
        assert_eq!(
            text(&[0x0f, 0x38, 0xf0, 0x00]),
            "movbe eax, dword ptr [rax]"
        );
        assert_eq!(
            text(&[0x0f, 0x38, 0xf1, 0x00]),
            "movbe dword ptr [rax], eax"
        );
        // `crc32` lives at the same opcode and is told apart by its prefix.
        assert_eq!(text(&[0xf2, 0x0f, 0x38, 0xf0, 0xc1]), "crc32 eax, cl");
        // The `0f3a` map: every instruction in it carries an immediate byte.
        assert_eq!(
            text(&[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x04]),
            "palignr xmm0, xmm1, 0x4"
        );
        // The extraction instructions put the general-purpose destination in
        // the r/m field and the vector source in the reg field. That is what
        // the manual says and what the hardware does; some other listings
        // print them the other way round.
        //
        // The destination register is a full-width container, not an
        // element: `pextrb` takes one byte out of the vector and leaves it
        // in a dword register, which both disassemblers print as `ecx`.
        assert_eq!(
            text(&[0x66, 0x0f, 0x3a, 0x14, 0xc1, 0x01]),
            "pextrb ecx, xmm0, 0x1"
        );
        // The carry-less multiplication names the quad it works on, and the
        // immediate byte says which one.
        assert_eq!(
            text(&[0x66, 0x0f, 0x3a, 0x44, 0xc1, 0x00]),
            "pclmullqlqdq xmm0, xmm1, 0x0"
        );
        assert_eq!(
            text(&[0x66, 0x0f, 0x3a, 0x44, 0xc1, 0x01]),
            "pclmulhqlqdq xmm0, xmm1, 0x1"
        );
    }

    #[test]
    fn the_group_seven_instructions_are_named_right() {
        // The privileged instructions of group 7, several of which differ
        // only in the r/m field.
        assert_eq!(text(&[0x0f, 0x01, 0xc1]), "vmcall");
        assert_eq!(text(&[0x0f, 0x01, 0xd0]), "xgetbv");
        assert_eq!(text(&[0x0f, 0x01, 0xd5]), "xend");
        assert_eq!(text(&[0x0f, 0x01, 0xee]), "rdpkru");
        assert_eq!(text(&[0x0f, 0x01, 0xef]), "wrpkru");
        assert_eq!(text(&[0x0f, 0x01, 0xf8]), "swapgs");
        // The same encoding is a different instruction with a prefix in
        // front of it.
        assert_eq!(text(&[0xf3, 0x0f, 0x01, 0xe8]), "setssbsy");
        assert_eq!(text(&[0xf2, 0x0f, 0x01, 0xe8]), "xsusldtrk");
    }

    #[test]
    fn the_fences_and_the_pause_are_named() {
        assert_eq!(text(&[0x0f, 0xae, 0xe8]), "lfence");
        assert_eq!(text(&[0x0f, 0xae, 0xf0]), "mfence");
        assert_eq!(text(&[0x0f, 0xae, 0xf8]), "sfence");
        assert_eq!(text(&[0xf3, 0x90]), "pause");
        assert_eq!(text(&[0x90]), "nop");
    }

    #[test]
    fn the_transactional_instructions_are_named() {
        // `c7 f8 rel32` begins a transaction and `c6 f8 ff` aborts one.
        assert!(text(&[0xc7, 0xf8, 0x00, 0x00, 0x00, 0x00]).starts_with("xbegin"));
        assert!(text(&[0xc6, 0xf8, 0xff]).starts_with("xabort"));
    }

    #[test]
    fn the_shadow_stack_instructions_are_named() {
        assert_eq!(text(&[0xf3, 0x0f, 0xae, 0xc0]), "rdfsbase");
        assert_eq!(text(&[0xf3, 0x0f, 0xae, 0xc8]), "rdgsbase");
        assert_eq!(text(&[0xf3, 0x0f, 0xae, 0xe8]), "incsspd");
        assert_eq!(text(&[0xf3, 0x48, 0x0f, 0xae, 0xe8]), "incsspq");
        assert_eq!(text(&[0xf3, 0x48, 0x0f, 0x1e, 0xc8]), "rdsspq");
    }

    #[test]
    fn the_half_register_moves_are_named() {
        // The register forms of `movlps` and `movhps` move halves around.
        assert_eq!(text(&[0x0f, 0x12, 0xc1]), "movhlps xmm0, xmm1");
        assert_eq!(text(&[0x0f, 0x16, 0xc1]), "movlhps xmm0, xmm1");
        assert_eq!(text(&[0x0f, 0x12, 0x00]), "movlps xmm0, xmmword ptr [rax]");
    }

    #[test]
    fn a_prefixed_instruction_keeps_its_prefix() {
        assert_eq!(text(&[0xf0, 0x0f, 0xb1, 0x07]), "lock cmpxchg dword ptr [rdi], eax");
        assert_eq!(text(&[0xf3, 0xa4]), "rep movsb byte ptr [rsi], byte ptr [rdi]");
    }

    #[test]
    fn movsxd_takes_a_memory_source() {
        // 48 63 08 is `movsxd rcx, dword ptr [rax]`, not `movsxd rcx, eax`.
        let t = text(&[0x48, 0x63, 0x08]);
        assert!(t.contains("[rax]"), "{t}");
    }

    #[test]
    fn decodes_a_prologue() {
        assert_eq!(text(&[0x55]), "push rbp");
        assert_eq!(text(&[0x48, 0x89, 0xe5]), "mov rbp, rsp");
        assert_eq!(text(&[0x48, 0x83, 0xec, 0x20]), "sub rsp, 0x20");
        assert_eq!(text(&[0xc3]), "ret");
    }

    #[test]
    fn lengths_are_exact() {
        // (bytes, expected length)
        let cases: &[(&[u8], u8)] = &[
            (&[0x90], 1),
            (&[0x48, 0x89, 0xe5], 3),
            (&[0x48, 0x8b, 0x05, 0x10, 0x00, 0x00, 0x00], 7),
            (&[0xe8, 0x00, 0x00, 0x00, 0x00], 5),
            (&[0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8], 10),
            (&[0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00], 6),
        ];
        for (bytes, want) in cases {
            let got = d64(bytes).insn.len;
            assert_eq!(got, u32::from(*want), "{bytes:02x?} decoded as {got} bytes");
        }
    }

    #[test]
    fn rip_relative_operand_resolves_to_absolute() {
        // mov rax, [rip+0x10] at 0x1000 -> the operand is at 0x1017.
        let bytes = [0x48, 0x8b, 0x05, 0x10, 0x00, 0x00, 0x00];
        let d = d64(&bytes);
        assert_eq!(d.insn.len, 7);
        assert_eq!(d.rip_target, Some(Addr(0x1000 + 7 + 0x10)));
        assert_eq!(d.insn.mem_reads, vec![Addr(0x1000 + 7 + 0x10)]);
        assert!(d.insn.text.contains("rip+0x10"), "{}", d.insn.text);
        assert!(d.insn.text.contains("0x1017"), "{}", d.insn.text);
    }

    #[test]
    fn branches_report_absolute_targets() {
        // jmp 0x1000+5+0x20
        let d = d64(&[0xe9, 0x20, 0x00, 0x00, 0x00]);
        assert_eq!(d.insn.flow, FlowKind::UnconditionalBranch);
        assert_eq!(d.insn.target, Some(CallTarget::Direct { to: Addr(0x1025) }));

        // call backwards
        let d = d64(&[0xe8, 0xf0, 0xff, 0xff, 0xff]);
        assert_eq!(d.insn.flow, FlowKind::Call);
        assert_eq!(d.insn.target, Some(CallTarget::Direct { to: Addr(0x0ff5) }));
    }

    #[test]
    fn indirect_call_through_got_yields_a_slot() {
        // call qword ptr [rip+0x2f5c]
        let d = d64(&[0xff, 0x15, 0x5c, 0x2f, 0x00, 0x00]);
        assert_eq!(d.insn.flow, FlowKind::Call);
        assert_eq!(
            d.insn.target,
            Some(CallTarget::Indirect {
                via: Some(Addr(0x1000 + 6 + 0x2f5c))
            })
        );
    }

    #[test]
    fn register_indirect_call_has_no_slot() {
        // call rax
        let d = d64(&[0xff, 0xd0]);
        assert_eq!(d.insn.flow, FlowKind::IndirectCall);
        assert_eq!(d.insn.target, None);
    }

    #[test]
    fn rex_extends_register_numbers() {
        // mov r8, r9
        let t = text(&[0x4d, 0x89, 0xc8]);
        assert!(t.contains("r8") && t.contains("r9"), "{t}");
        // mov r8d, eax
        let t = text(&[0x41, 0xb8, 0x00, 0x00, 0x00, 0x00]);
        assert!(t.contains("r8d"), "{t}");
    }

    #[test]
    fn operand_size_prefix_selects_16_bit_registers() {
        // mov ax, bx
        let t = text(&[0x66, 0x89, 0xd8]);
        assert!(t.contains("ax") && t.contains("bx"), "{t}");
    }

    #[test]
    fn high_byte_registers_require_no_rex() {
        // mov ah, bl  (no REX)
        let t = text(&[0x88, 0xdc]);
        assert!(t.contains("ah"), "{t}");
        // With REX the same encoding means spl, a different register.
        let t = text(&[0x40, 0x88, 0xdc]);
        assert!(t.contains("spl"), "{t}");
    }

    #[test]
    fn sib_with_index_and_scale() {
        // mov eax, [rbx+rcx*4+0x10]
        let t = text(&[0x8b, 0x44, 0x8b, 0x10]);
        assert!(t.contains("rbx") && t.contains("rcx*4"), "{t}");
    }

    #[test]
    fn sib_without_base_uses_disp32() {
        // mov eax, [rcx*4+0x1000]
        let t = text(&[0x8b, 0x04, 0x8d, 0x00, 0x10, 0x00, 0x00]);
        assert!(t.contains("rcx*4"), "{t}");
        assert!(t.contains("0x1000"), "{t}");
    }

    #[test]
    fn negative_displacements_render_with_a_minus() {
        // mov eax, [rbp-0x8]
        let t = text(&[0x8b, 0x45, 0xf8]);
        assert!(t.contains("- 0x8"), "{t}");
    }

    #[test]
    fn immediates_print_at_the_width_of_their_operand() {
        // The field is signed, but a listing prints it as the operand's
        // width with no sign: `push -1` is a 64-bit slot holding all ones,
        // and `add eax, -1` is a 32-bit one.
        assert!(text(&[0x6a, 0xff]).contains("0xffffffffffffffff"));
        assert!(text(&[0x83, 0xc0, 0xff]).contains("0xffffffff"));
        assert!(text(&[0xb0, 0xff]).contains("0xff"));
        assert!(text(&[0x66, 0xc7, 0xc0, 0xff, 0xff]).contains("0xffff"));
    }

    #[test]
    fn conditional_branches_map_to_the_right_mnemonic() {
        // je rel8 and jne rel8 differ in one bit.
        assert!(text(&[0x74, 0x00]).starts_with("je"));
        assert!(text(&[0x75, 0x00]).starts_with("jne"));
        assert!(text(&[0x0f, 0x84, 0, 0, 0, 0]).starts_with("je"));
        assert!(text(&[0x0f, 0x85, 0, 0, 0, 0]).starts_with("jne"));
    }

    #[test]
    fn setcc_and_cmovcc_are_named() {
        assert!(text(&[0x0f, 0x94, 0xc0]).starts_with("sete"));
        assert!(text(&[0x0f, 0x45, 0xc1]).starts_with("cmovne"));
    }

    #[test]
    fn common_compiler_output_decodes() {
        // endbr64, then a 64-bit compare and a conditional move.
        assert_eq!(text(&[0xf3, 0x0f, 0x1e, 0xfa]), "endbr64");
        assert!(text(&[0x48, 0x39, 0xf8]).starts_with("cmp"));
        assert!(text(&[0x0f, 0x1f, 0x44, 0x00, 0x00]).starts_with("nop"));
    }

    #[test]
    fn an_unknown_opcode_consumes_its_prefix() {
        // `0f ff` names nothing this decoder knows, and the listing prints
        // both bytes as the undecodable instruction. Answering one byte would
        // leave the `0f` behind and shift every instruction after it by one,
        // which is exactly the failure this test exists to prevent.
        let d = d64(&[0x0f, 0xff, 0x90]);
        assert_eq!(d.insn.len, 2);
        assert_eq!(d.insn.mnemonic.as_ref(), ".byte");
        // The byte after it must decode on its own, which is what keeps a
        // caller's scan aligned after a gap.
        let next = d64(&[0x90]);
        assert_eq!(next.insn.mnemonic.as_ref(), "nop");
    }

    #[test]
    fn truncated_input_returns_a_partial_decode_not_a_panic() {
        // A 7-byte instruction cut to 3 bytes cannot be decoded whole, and the
        // three bytes it does have are reported as one uninterpreted group
        // rather than as a prefix and a hole.
        let d = d64(&[0x48, 0x8b, 0x05]);
        assert_eq!(d.insn.len, 3);
    }

    /// EVEX: a write mask, a zeroing bit and a broadcast, each of which
    /// has to survive the trip from the prefix to the listing.
    #[test]
    fn evex_masks_zeroing_and_broadcast_are_rendered() {
        // vaddps zmm0, zmm1, [rax] — no mask, whole vector.
        assert_eq!(
            text(&[0x62, 0xf1, 0x74, 0x48, 0x58, 0x00]),
            "vaddps zmm0, zmm1, zmmword ptr [rax]"
        );
        // The same with the broadcast bit: one dword copied to all
        // sixteen lanes, and the operand says so instead of claiming a
        // whole vector was read.
        assert_eq!(
            text(&[0x62, 0xf1, 0x74, 0x58, 0x58, 0x00]),
            "vaddps zmm0, zmm1, dword ptr [rax]{1to16}"
        );
        // And with a mask: `{k1}` alone leaves the other lanes untouched.
        assert_eq!(
            text(&[0x62, 0xf1, 0x74, 0x59, 0x58, 0x00]),
            "vaddps zmm0{k1}, zmm1, dword ptr [rax]{1to16}"
        );
        // `{z}` is the difference between "leave the other lanes alone"
        // and "write zeros there", and it is one bit away.
        assert_eq!(
            text(&[0x62, 0xf1, 0x74, 0xf9, 0x58, 0x00]),
            "vaddps zmm0{k1}{z}, zmm1, dword ptr [rax]{1to16}"
        );
        // The lane count follows the element width, not the register
        // width: eight qwords is a different `{1toN}` than sixteen dwords.
        assert_eq!(
            text(&[0x62, 0xf1, 0xf5, 0x58, 0x58, 0x00]),
            "vaddpd zmm0, zmm1, qword ptr [rax]{1to8}"
        );
    }

    /// The store direction is only visible in the memory form: the
    /// register form of `vmovdqu64` names the same two registers either
    /// way round.
    #[test]
    fn evex_store_direction_comes_from_the_memory_operand() {
        // `6f` loads and `7f` stores. The register form of both is
        // `vmovdqu64 zmm1, zmm0`, so only the memory form says which way
        // the data goes.
        assert_eq!(
            text(&[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x08]),
            "vmovdqu64 zmm1, zmmword ptr [rax]"
        );
        assert_eq!(
            text(&[0x62, 0xf1, 0xfe, 0x48, 0x7f, 0x08]),
            "vmovdqu64 zmmword ptr [rax], zmm1"
        );
        // The AVX-512 stores put the write mask on the memory operand,
        // because it is memory that is partly written.
        assert_eq!(
            text(&[0x62, 0xf1, 0xfe, 0x49, 0x7f, 0x08]),
            "vmovdqu64 zmmword ptr [rax]{k1}, zmm1"
        );
    }

    /// A one-byte displacement in an EVEX encoding counts units, not
    /// bytes, and the unit is whatever the instruction reads.
    #[test]
    fn evex_displacement_is_scaled_by_the_operand_size() {
        // `01` is one 64-byte vector, so the address moves by 0x40.
        assert_eq!(
            text(&[0x62, 0xf1, 0x7e, 0x48, 0x6f, 0x47, 0x01]),
            "vmovdqu32 zmm0, zmmword ptr [rdi + 0x40]"
        );
        // A scalar move counts one element: four bytes, not sixty-four.
        assert_eq!(
            text(&[0x62, 0xf1, 0x7e, 0x08, 0x10, 0x47, 0x01]),
            "vmovss xmm0, dword ptr [rdi + 0x4]"
        );
    }

    /// The gather and scatter family: a vector index, two widths that
    /// differ, and a mask that sits on a different operand in each.
    #[test]
    fn gather_and_scatter_index_memory_with_a_vector() {
        // AVX2 gathers name their write mask as a third operand.
        assert_eq!(
            text(&[0xc4, 0xe2, 0x69, 0x90, 0x04, 0x08]),
            "vpgatherdd xmm0, dword ptr [rax + xmm1], xmm2"
        );
        assert_eq!(
            text(&[0xc4, 0xe2, 0x6d, 0x90, 0x04, 0x08]),
            "vpgatherdd ymm0, dword ptr [rax + ymm1], ymm2"
        );
        // A qword index halves the number of lanes, so the data register
        // is narrower than the index one.
        assert_eq!(
            text(&[0xc4, 0xe2, 0x6d, 0x91, 0x04, 0x08]),
            "vpgatherqd xmm0, dword ptr [rax + ymm1], xmm2"
        );
        // The AVX-512 form takes the mask from the prefix instead.
        assert_eq!(
            text(&[0x62, 0xf2, 0x7d, 0x49, 0x90, 0x04, 0x08]),
            "vpgatherdd zmm0{k1}, dword ptr [rax + zmm1]"
        );
        // A scatter writes memory, so its mask is written on the memory
        // operand and the data register is the source.
        assert_eq!(
            text(&[0x62, 0xf2, 0x7d, 0x49, 0xa0, 0x04, 0x08]),
            "vpscatterdd dword ptr [rax + zmm1]{k1}, zmm0"
        );
        assert_eq!(
            text(&[0x62, 0xf2, 0xfd, 0x29, 0xa1, 0x04, 0x08]),
            "vpscatterqq qword ptr [rax + ymm1]{k1}, ymm0"
        );
    }

    /// EVEX has two extension bits above bit 3 and they move different
    /// registers: `R'` the destination, `V'` the `vvvv` field and the
    /// `r/m` register. Confusing them puts the answer in the wrong
    /// register while still looking like a valid listing.
    #[test]
    fn evex_extension_bits_reach_the_register_they_extend() {
        // R' set: the destination is zmm16, the source stays xmm1.
        // The source is `xmm` because a broadcast reads one element and
        // there is no narrower vector register to name.
        assert_eq!(
            text(&[0x62, 0xe2, 0x7d, 0x49, 0x58, 0xc1]),
            "vpbroadcastd zmm16{k1}, xmm1"
        );
        // V' cleared: the same bit extends the `r/m` register instead.
        // GNU objdump prints `xmm1` here, which cannot be right — the
        // encoding says register 17 — so this follows the encoding.
        assert_eq!(
            text(&[0x62, 0xf2, 0x7d, 0x41, 0x58, 0xc1]),
            "vpbroadcastd zmm0{k1}, xmm17"
        );
        // The same bit extends a vector index, which is how a gather
        // reaches the upper sixteen vector registers.
        assert_eq!(
            text(&[0x62, 0xf2, 0x7d, 0x41, 0x90, 0x04, 0x08]),
            "vpgatherdd zmm0{k1}, dword ptr [rax + zmm17]"
        );
    }

    /// The moves to and from a segment register, and the absolute
    /// accumulator moves: both name an operand the ordinary forms never
    /// do, and both are still emitted by real compilers.
    #[test]
    fn segment_and_absolute_moves_are_named() {
        assert_eq!(text(&[0x8c, 0xc0]), "mov eax, es");
        assert_eq!(text(&[0x48, 0x8c, 0xc0]), "mov rax, es");
        assert_eq!(text(&[0x8c, 0x00]), "mov word ptr [rax], es");
        assert_eq!(text(&[0x8e, 0xe0]), "mov fs, eax");
        assert_eq!(text(&[0x8e, 0x00]), "mov es, word ptr [rax]");
        // An address with no base and no index is absolute, and the
        // listing names the segment rather than bracketing the number.
        assert_eq!(
            text(&[0x8b, 0x04, 0x25, 0x00, 0x10, 0x40, 0x00]),
            "mov eax, dword ptr ds:0x401000"
        );
        assert_eq!(
            text(&[0xa1, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]),
            "movabs eax, ds:0x8877665544332211"
        );
    }

    #[test]
    fn absolute_moves_take_the_address_width_of_the_mode() {
        // `A1` is a move from an absolute address, and the address is as wide
        // as the mode says, not as wide as the operand. Reading eight bytes in
        // 32-bit code over-reads by four and every following instruction is
        // decoded from the middle of this one.
        assert_eq!(
            text(&[0x48, 0xa1, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]),
            "movabs rax, ds:0x8877665544332211"
        );
        assert_eq!(
            text(&[0x67, 0xa1, 0x11, 0x22, 0x33, 0x44]),
            "mov eax, ds:0x44332211"
        );

        let d32 = Decoder::new(32).unwrap();
        let d = d32
            .decode(Addr(0x1000), &[0xa1, 0x11, 0x22, 0x33, 0x44])
            .unwrap();
        assert_eq!(d.insn.text, "mov eax, ds:0x44332211");
        assert_eq!(d.insn.len, 5);
        // The `67` prefix halves the address again, to two bytes.
        let d = d32.decode(Addr(0x1000), &[0x67, 0xa1, 0x11, 0x22]).unwrap();
        assert_eq!(d.insn.text, "mov eax, ds:0x2211");
        assert_eq!(d.insn.len, 4);
        // The byte moves take the accumulator, not a register number.
        let d = d32
            .decode(Addr(0x1000), &[0xa0, 0x11, 0x22, 0x33, 0x44])
            .unwrap();
        assert_eq!(d.insn.text, "mov al, ds:0x44332211");
    }

    #[test]
    fn the_state_save_group_is_selected_by_the_reg_field() {
        // Group 9 is selected by the ModRM reg field, and its upper half is
        // the state-area family: `/3`, `/4` and `/5` are XRSTORS, XSAVEC and
        // XSAVES. A decoder that leaves them out emits one byte where the
        // listing has five and loses the instruction stream for the rest of
        // the function, which is how a missing entry in one group turns into
        // hundreds of `not decoded` lines.
        assert_eq!(text(&[0x0f, 0xc7, 0x5c, 0x24, 0x40]), "xrstors [rsp + 0x40]");
        assert_eq!(
            text(&[0x48, 0x0f, 0xc7, 0x5c, 0x24, 0x40]),
            "xrstors64 [rsp + 0x40]"
        );
        assert_eq!(text(&[0x0f, 0xc7, 0x64, 0x24, 0x40]), "xsavec [rsp + 0x40]");
        assert_eq!(
            text(&[0x48, 0x0f, 0xc7, 0x64, 0x24, 0x40]),
            "xsavec64 [rsp + 0x40]"
        );
        assert_eq!(text(&[0x0f, 0xc7, 0x6c, 0x24, 0x40]), "xsaves [rsp + 0x40]");
        assert_eq!(
            text(&[0x48, 0x0f, 0xc7, 0x6c, 0x24, 0x40]),
            "xsaves64 [rsp + 0x40]"
        );
        // The prefix in front of one of them is not part of the encoding: the
        // same three bytes are the instruction with or without it.
        assert_eq!(text(&[0x66, 0x0f, 0xc7, 0x27]), "xsavec [rdi]");
        assert_eq!(text(&[0xf3, 0x0f, 0xc7, 0x2f]), "xsaves [rdi]");
        // The compare-and-exchange pair share the group, and their names
        // already carry the width, so the operand prints without one.
        assert_eq!(text(&[0x0f, 0xc7, 0x08]), "cmpxchg8b [rax]");
        assert_eq!(text(&[0x48, 0x0f, 0xc7, 0x08]), "cmpxchg16b [rax]");
        // The register forms in the same group are rdrand and rdseed, not
        // loads, and they name a general-purpose register.
        assert_eq!(text(&[0x0f, 0xc7, 0xf0]), "rdrand eax");
    }

    #[test]
    fn the_prefix_decides_the_name_of_a_shared_opcode() {
        // `0F 12` is `movhlps` in its register form, but under `F2` and `F3`
        // the same opcode is `movddup` and `movsldup`, which keep their names
        // in both forms. Renaming every register form gives the wrong
        // instruction a right-looking length.
        assert_eq!(text(&[0x0f, 0x12, 0xc1]), "movhlps xmm0, xmm1");
        assert_eq!(text(&[0xf2, 0x0f, 0x12, 0xc1]), "movddup xmm0, xmm1");
        assert_eq!(text(&[0xf3, 0x0f, 0x12, 0xc1]), "movsldup xmm0, xmm1");
        assert_eq!(text(&[0x0f, 0x16, 0xc1]), "movlhps xmm0, xmm1");
        assert_eq!(text(&[0xf3, 0x0f, 0x16, 0xc1]), "movshdup xmm0, xmm1");
        assert_eq!(text(&[0xf2, 0x0f, 0x12, 0x01]), "movddup xmm0, qword ptr [rcx]");
    }

    #[test]
    fn thirty_two_bit_mode_has_no_rex() {
        let dec = Decoder::new(32).unwrap();
        // In 32-bit mode 0x40 is `inc eax`, not a REX prefix.
        let d = dec.decode(Addr(0x1000), &[0x40]).unwrap();
        assert!(d.insn.text.contains("eax"), "{}", d.insn.text);
    }
}
