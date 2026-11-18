//! Architecture-neutral lifting front end.
//!
//! The decompiler's back end — SSA construction, structuring, rendering, and
//! the symbolic executor on top of it — is architecture-independent: it works
//! on [`crate::ir`], which knows nothing about registers.
//!
//! What is architecture-dependent is the front end: turning decoded
//! instructions into statements. That is what this module separates.
//!
//! # The split
//!
//! * [`Reg`] — one architectural register, with the width and the bit offset
//!   of the view being accessed.
//! * [`Operand`] — a register, an immediate, or a memory reference.
//! * [`Target`] — the facts that differ per architecture: how a name maps to a
//!   register, which registers carry arguments, which one carries the return
//!   value, and what a conditional branch's condition is called.
//! * [`Lifter`] — the trait the front end implements.
//!
//! # Why a trait and not a match on architecture
//!
//! A `match image.arch` inside the lifter would put AArch64's condition codes
//! and x86's flags in the same function, and the two do not compose: x86
//! compares through a flags register that survives an arbitrary number of
//! instructions, while AArch64's condition flags are set by the instruction
//! that computes them and consumed by the branch. Modelling that difference
//! is the whole reason the front end is per-architecture.
//!
//! # What stays shared
//!
//! Memory operands, arithmetic, calls, and the stack frame are modelled the
//! same way on every architecture, so the shared helpers in this module do
//! that work once. An architecture that differs — RISC-V, where the frame
//! pointer is optional and arguments are named rather than numbered — says so
//! through [`Target`].

pub mod aarch64;
pub mod riscv;
pub mod x86;

use std::collections::HashSet;

use occ_core::{Addr, FlowKind, Instruction};

use crate::ir::{
    BinOp, CastKind, CondOp, Expr, FlagCond, FlagState, Intrinsic, Stmt, UnOp,
};

/// One architectural register.
///
/// `idx` is the architecture's own register number, so `Reg { idx: 0 }` is
/// `rax` on x86-64 and `x0` on AArch64. The name comes from [`Target`].
///
/// `size` and `offset` describe the view being accessed. The architectural
/// register is `width` bytes wide; a narrower view reads and writes a slice
/// of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reg {
    /// Register number, in the architecture's own numbering.
    pub idx: u8,
    /// Access width in bytes.
    pub size: u8,
    /// Bit offset of the view within the architectural register.
    ///
    /// Zero for the whole register and for its low half; 8 for x86's `ah`.
    pub offset: u8,
}

impl Reg {
    /// A register accessed at its full width.
    pub const fn full(idx: u8, size: u8) -> Self {
        Reg {
            idx,
            size,
            offset: 0,
        }
    }

    /// A narrow view of a register, starting at `offset` bits.
    pub const fn view(idx: u8, size: u8, offset: u8) -> Self {
        Reg { idx, size, offset }
    }
}

/// One operand.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    /// Register access.
    Reg(Reg),
    /// Immediate.
    Imm(u64),
    /// A condition name, as AArch64's `csel` and `cset` print it.
    Cond(String),
    /// A register with a shift or an extension, as AArch64 writes the third
    /// operand of `add` and the index of a load: `w1, sxtw #2`.
    ///
    /// It is a separate operand because the modifier belongs to the operand
    /// it follows, not to the instruction: `add x3, x2, w1, sxtw #2` has four
    /// comma-separated fields and three operands. Lifting it as a fourth
    /// operand lost it, and the instruction became "unparsed" — which is how
    /// a whole array index disappeared from the output.
    Modified {
        /// The register being shifted or extended.
        reg: Reg,
        /// How it is modified.
        kind: ModKind,
        /// Shift amount in bits, zero when the text omitted it.
        amount: u8,
    },
    /// A vector register.
    Vec(VecClass, u8),
    /// A slot of the x87 register stack.
    Fpu(u8),
    /// Memory reference.
    Mem {
        /// Access width in bytes; zero when the instruction did not say.
        size: u8,
        /// Base register.
        base: Option<Reg>,
        /// Index register.
        index: Option<Reg>,
        /// Index scale, 1/2/4/8.
        scale: u64,
        /// Displacement.
        disp: i64,
        /// Absolute address, when the decoder resolved one.
        abs: Option<u64>,
        /// Segment override, which selects a different address space for the
        /// same arithmetic. Only x86 has these.
        seg: Option<&'static str>,
        /// Change the base register by this much after the access, for the
        /// pre-indexed and post-indexed forms AArch64 uses.
        base_update: Option<i64>,
    },
}

/// How a register operand is modified before it is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModKind {
    /// Logical shift left.
    Lsl,
    /// Logical shift right.
    Lsr,
    /// Arithmetic shift right.
    Asr,
    /// Rotate right.
    Ror,
    /// Sign-extend a byte.
    Sxtb,
    /// Sign-extend a halfword.
    Sxth,
    /// Sign-extend a word.
    Sxtw,
    /// Sign-extend the register, which is a no-op on a 64-bit one.
    Sxtx,
    /// Zero-extend a byte.
    Uxtb,
    /// Zero-extend a halfword.
    Uxth,
    /// Zero-extend a word.
    Uxtw,
    /// Zero-extend the register, which is a no-op on a 64-bit one.
    Uxtx,
}

impl ModKind {
    /// The width the extension starts from, in bytes.
    const fn from(self) -> Option<u8> {
        Some(match self {
            ModKind::Sxtb | ModKind::Uxtb => 1,
            ModKind::Sxth | ModKind::Uxth => 2,
            ModKind::Sxtw | ModKind::Uxtw => 4,
            ModKind::Sxtx | ModKind::Uxtx => return None,
            _ => return None,
        })
    }

    /// True when the extension is a signed one.
    const fn signed(self) -> bool {
        matches!(
            self,
            ModKind::Sxtb | ModKind::Sxth | ModKind::Sxtw | ModKind::Sxtx
        )
    }

    /// True when the modifier is a shift rather than an extension.
    const fn is_shift(self) -> bool {
        matches!(
            self,
            ModKind::Lsl | ModKind::Lsr | ModKind::Asr | ModKind::Ror
        )
    }

    /// The modifier a mnemonic names, if it names one.
    pub fn of_name(name: &str) -> Option<ModKind> {
        Some(match name {
            "lsl" => ModKind::Lsl,
            "lsr" => ModKind::Lsr,
            "asr" => ModKind::Asr,
            "ror" => ModKind::Ror,
            "sxtb" => ModKind::Sxtb,
            "sxth" => ModKind::Sxth,
            "sxtw" => ModKind::Sxtw,
            "sxtx" => ModKind::Sxtx,
            "uxtb" => ModKind::Uxtb,
            "uxth" => ModKind::Uxth,
            "uxtw" => ModKind::Uxtw,
            "uxtx" => ModKind::Uxtx,
            _ => return None,
        })
    }
}

/// Which vector register file a register belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VecClass {
    /// An SSE register, `xmm0..xmm15`.
    Xmm,
    /// An AVX register, `ymm0..ymm15`.
    Ymm,
    /// An MMX register, `mm0..mm7`.
    Mm,
}

impl VecClass {
    /// The register file's width in bytes.
    pub const fn size(self) -> u8 {
        match self {
            VecClass::Xmm => 16,
            VecClass::Ymm => 32,
            VecClass::Mm => 8,
        }
    }
}

/// The access a load or store instruction performs.
///
/// The width is not always in the instruction text. x86 prints it —
/// `qword ptr [rbx]` — but AArch64 and RISC-V spell it in the mnemonic
/// alone: `strb` is one byte, `str` is eight, `sw` is four, `ld` is eight.
/// Getting this wrong is not a cosmetic problem: a four-byte store lifted as
/// an eight-byte one makes the executor read and write twice the memory it
/// should, and the analysis that follows inherits the mistake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    /// Width in bytes.
    pub width: u8,
    /// True when a load sign-extends its result into the destination.
    ///
    /// `ldrsb` does, `ldrb` does not. On RISC-V `lw` sign-extends into a
    /// 64-bit register and `lwu` does not, which is the same distinction
    /// spelled differently.
    pub signed: bool,
}

impl Default for Access {
    fn default() -> Self {
        Access {
            width: 8,
            signed: false,
        }
    }
}

impl Operand {
    /// Access width in bytes, defaulting to 8 when unspecified.
    pub fn size(&self) -> u8 {
        match self {
            Operand::Reg(r) => r.size,
            Operand::Modified { reg, .. } => reg.size,
            Operand::Imm(_) => 8,
            Operand::Cond(_) => 1,
            Operand::Vec(class, _) => class.size(),
            Operand::Fpu(_) => 8,
            Operand::Mem { size, .. } => {
                if *size == 0 {
                    8
                } else {
                    *size
                }
            }
        }
    }

    /// The condition suffix of an operand that names one, for the
    /// architectures that spell a condition as a bare word.
    pub fn cond_suffix(&self) -> Option<&str> {
        match self {
            Operand::Cond(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// A parsed instruction.
#[derive(Clone, Debug)]
pub struct Parsed {
    /// Mnemonic, lower case.
    pub mnemonic: String,
    /// Operands, in the order the architecture prints them.
    pub ops: Vec<Operand>,
    /// Absolute address a PC-relative operand refers to, when there was one.
    pub pc_target: Option<u64>,
    /// Prefixes the instruction carried, lower case.
    ///
    /// `lock` says the instruction is atomic, `rep` says it is a loop. Both
    /// change what the instruction does, so the lifter has to see them; they
    /// are not decoration.
    pub prefixes: Vec<String>,
}

/// What one block lifted to.
#[derive(Default)]
pub struct Lifted {
    /// Straight-line statements.
    pub stmts: Vec<Stmt>,
    /// Address of the instruction each statement came from, parallel to
    /// [`stmts`](Self::stmts).
    ///
    /// One instruction can lift to several statements and to none, so the
    /// addresses repeat and skip. The line table is keyed by address, which
    /// is why the mapping is kept instead of recomputed from instruction
    /// offsets.
    pub stmt_at: Vec<Addr>,
    /// Condition of a trailing conditional branch.
    pub branch: Option<Expr>,
    /// Target expression of a trailing indirect jump.
    pub indirect: Option<Expr>,
}

/// The facts that differ between architectures.
///
/// Every method is a constant fact about the architecture, not a decision
/// made per instruction, so an implementation is a static table.
pub trait Target {
    /// The register a name refers to.
    ///
    /// `None` for a name that is not a register, which is how the operand
    /// parser tells a register from an immediate.
    fn lookup_reg(&self, name: &str) -> Option<Reg>;

    /// The canonical name of a register, at its full width.
    fn reg_name(&self, idx: u8) -> String;

    /// The full width of an architectural register, in bytes.
    fn reg_width(&self, idx: u8) -> u8;

    /// Registers that carry integer arguments, in ABI order.
    fn arg_regs(&self) -> &[u8];

    /// The register that carries the return value.
    fn ret_reg(&self) -> u8;

    /// The stack pointer.
    fn sp_reg(&self) -> u8;

    /// The frame pointer, when the architecture has one.
    fn fp_reg(&self) -> Option<u8>;

    /// True when the register holds the program counter, so a memory operand
    /// based on it is an address rather than arithmetic.
    fn is_pc(&self, idx: u8) -> bool;

    /// The name the stack pointer is printed under.
    ///
    /// Calling it `rsp` invites the reader to think the value is still the
    /// live stack pointer, when at function entry it is the frame base.
    fn sp_alias(&self) -> &'static str;

    /// The condition a conditional branch tests.
    ///
    /// The condition is a question about the flags, because that is what
    /// the machine's condition codes are. Some of them are relations
    /// between two values and the rest ask about one flag or about two
    /// of them together.
    ///
    /// `None` for a mnemonic that is not a conditional branch.
    fn branch_cond(&self, mnemonic: &str) -> Option<FlagCond>;

    /// The condition a `setcc`-style instruction produces.
    ///
    /// x86 spells these `sete`, `setl` and so on; AArch64 spells the same
    /// conditions `cset` with a condition suffix. The suffix is passed in
    /// without the architecture's prefix.
    fn setcc_cond(&self, suffix: &str) -> Option<FlagCond>;

    /// The condition a conditional-select instruction tests.
    fn cmov_cond(&self, suffix: &str) -> Option<FlagCond>;

    /// The condition a branch tests, when the flags come from a
    /// floating-point comparison.
    ///
    /// A floating-point compare sets three flags — below, equal, unordered —
    /// and a branch asks about them in combinations that the integer
    /// conditions cannot express: "below or unordered" is neither less-than
    /// nor not-less-than. The mapping is per architecture, so it belongs
    /// here; `None` means the architecture has no floating-point compare.
    fn float_branch_cond(
        &self,
        _mnemonic: &str,
        _lhs: &Expr,
        _rhs: &Expr,
        _bits: u8,
    ) -> Option<Expr> {
        None
    }

    /// The access a load or store with this mnemonic performs.
    ///
    /// `data_size` is the width of the register that carries the data: the
    /// destination of a load, the source of a store. An architecture whose
    /// decoder prints the width in the instruction text never needs this.
    ///
    /// The default is a full-width unsigned access.
    fn mem_access(&self, mnemonic: &str, data_size: u8) -> Access {
        let _ = (mnemonic, data_size);
        Access::default()
    }

    /// A register whose value the linker fixes, by symbol name.
    ///
    /// RISC-V's `gp` points at `__global_pointer$`, which the startup code
    /// establishes before any function runs. A static analysis cannot follow
    /// that setup, but the symbol table states the value directly. Returning
    /// `None` means the architecture has no such register.
    fn fixed_register(&self, _idx: u8) -> Option<&'static str> {
        None
    }

    /// The register that always reads as zero, when the architecture has one.
    ///
    /// RISC-V's `x0` is hardwired to zero: reading it returns zero and
    /// writing it discards the value. Lifting it as an ordinary register
    /// turned every comparison against it into a comparison against an
    /// unknown, and left the solver with no constant to work from.
    fn zero_reg(&self) -> Option<u8> {
        None
    }

    /// Parse one instruction's text into a mnemonic and operands.
    ///
    /// `text` is the decoder's own rendering. Each architecture's syntax
    /// differs, so this cannot be shared.
    fn parse(&self, text: &str) -> Option<Parsed>;
}

/// A lookup of a symbol's address, by name.
///
/// Only the registers the linker fixes need this, and only one architecture
/// has them; the lookup is absent when the caller has no symbol table to
/// offer.
pub type SymbolLookup<'a> = &'a dyn Fn(&str) -> Option<u64>;

/// The comparison a flag-setting operation left behind.
///
/// The carry flag.
pub const FLAG_CF: u8 = 1 << 0;
/// The zero flag.
pub const FLAG_ZF: u8 = 1 << 1;
/// The sign flag.
pub const FLAG_SF: u8 = 1 << 2;
/// The overflow flag.
pub const FLAG_OF: u8 = 1 << 3;
/// The parity flag.
pub const FLAG_PF: u8 = 1 << 4;

/// Every flag a comparison defines.
pub const ALL_FLAGS: u8 = FLAG_CF | FLAG_ZF | FLAG_SF | FLAG_OF | FLAG_PF;

/// The width matters. `cmp edi, 0` compares two 32-bit values, and the
/// branch that consumes the flags decides whether they are read as signed or
/// unsigned. Comparing the 64-bit register instead would read the upper half
/// of `rdi`, which the comparison never looked at.
#[derive(Clone, Debug)]
pub struct PendingCmp {
    /// Left operand.
    pub lhs: Expr,
    /// Right operand.
    pub rhs: Expr,
    /// Width of the compared values in bytes; eight for a full register.
    pub width: u8,
    /// True when the comparison is a floating-point one.
    ///
    /// A floating-point compare sets three flags where an integer compare
    /// sets one condition, and the branch that reads them asks about
    /// combinations the integer conditions cannot express. The operands are
    /// bit patterns, not numbers.
    pub float: bool,
    /// The flags the operation that set these actually defined.
    ///
    /// A comparison defines all of them. A bit test defines the carry flag
    /// and nothing else, and a branch that reads the zero flag after one
    /// cannot be answered from it: the machine leaves that flag alone, and
    /// what it holds is whatever came before.
    pub flags: u8,
}

impl PendingCmp {
    /// An integer comparison of two values of the given width.
    pub fn int(lhs: Expr, rhs: Expr, width: u8) -> Self {
        PendingCmp::with_flags(lhs, rhs, width, ALL_FLAGS)
    }

    /// A floating-point comparison of two bit patterns of the given width.
    pub fn float(lhs: Expr, rhs: Expr, width: u8) -> Self {
        // A floating-point compare sets the carry, zero and parity flags;
        // the sign and overflow flags are cleared.
        PendingCmp::with_flags(lhs, rhs, width, FLAG_CF | FLAG_ZF | FLAG_PF)
    }

    /// A comparison that defines only the flags named.
    pub fn with_flags(lhs: Expr, rhs: Expr, width: u8, flags: u8) -> Self {
        PendingCmp {
            lhs,
            rhs,
            width,
            float: false,
            flags,
        }
    }

    /// The flags a comparison leaves behind.
    ///
    /// A comparison is a subtraction that threw the difference away, and
    /// the flags are the ones that subtraction leaves: the carry flag is
    /// the borrow, the zero flag says the values were equal, the sign flag
    /// is the sign of the *difference* and not of either operand, and the
    /// overflow flag says whether the difference was representable.
    ///
    /// The difference matters. `cmp 0, 0x80000000` leaves the sign flag
    /// set, because the difference is negative, even though the first
    /// operand is not.
    pub fn derived_flags(&self) -> FlagState {
        let bits = self.width;
        let difference = Expr::Bin {
            op: BinOp::Sub,
            lhs: Box::new(self.lhs.clone()),
            rhs: Box::new(self.rhs.clone()),
            bits,
        };
        FlagState {
            cf: Some(Expr::Cond {
                op: CondOp::Ult,
                lhs: Box::new(self.lhs.clone()),
                rhs: Box::new(self.rhs.clone()),
                bits,
            }),
            zf: Some(Expr::Cond {
                op: CondOp::Eq,
                lhs: Box::new(self.lhs.clone()),
                rhs: Box::new(self.rhs.clone()),
                bits,
            }),
            sf: Some(Expr::Cond {
                op: CondOp::Slt,
                lhs: Box::new(difference.clone()),
                rhs: Box::new(Expr::Const(0)),
                bits,
            }),
            of: Some(Expr::Cond {
                op: CondOp::Slt,
                lhs: Box::new(crate::ir::sub_overflow(
                    &self.lhs,
                    &self.rhs,
                    &difference,
                    bits,
                )),
                rhs: Box::new(Expr::Const(0)),
                bits,
            }),
            pf: Some(Expr::Intrinsic {
                op: Intrinsic::ParityEven,
                args: vec![difference],
                bits,
            }),
        }
    }
}

/// State shared by every architecture's lifter.
///
/// The frame bookkeeping, the flag handling, and the register read/write rules
/// are the same on every architecture this toolset handles, so they live here
/// and the per-architecture code calls into them.
pub struct Core<'a> {
    /// The architecture.
    pub target: &'static dyn Target,
    /// Name lookup for direct call targets.
    pub name_of: &'a dyn Fn(Addr) -> String,
    /// Lookup of a symbol by name.
    ///
    /// Only the registers the linker fixes need this, and only one
    /// architecture has them; the lookup is absent when the caller has no
    /// symbol table to offer.
    pub symbols: Option<SymbolLookup<'a>>,
    /// Stack offset of the stack pointer relative to the frame base.
    pub sp_off: i64,
    /// Frame offset of the frame pointer once anchored.
    pub fp_off: Option<i64>,
    /// Where the frame pointer was anchored, kept after `pop rbp` restores
    /// the register.
    ///
    /// Debug info locates locals relative to the frame pointer, and the
    /// question "where was `rbp`?" has to be answered after the whole
    /// function is lifted — by which time the epilogue has cleared `fp_off`.
    /// The anchor is therefore recorded when the prologue sets it and never
    /// cleared.
    pub fp_anchor: Option<i64>,
    /// Set once the stack pointer moves by a non-constant amount; frame slots
    /// stop being trustworthy from that point.
    pub dynamic_sp: bool,
    /// The operands of the last flag-setting operation, when it was a
    /// comparison.
    ///
    /// A comparison is the one operation whose flags are a relation
    /// between two values, and the branch that follows it can be written
    /// as that relation. An arithmetic operation's flags are not a
    /// relation between two values, and they live in `flags`.
    pub pending_cmp: Option<PendingCmp>,
    /// The value of each condition-code flag, when it is known.
    ///
    /// A flag holds the expression that holds when the flag is set. A
    /// flag the last operation did not define has no value here, because
    /// the machine left that flag alone and what it holds is whatever
    /// came before.
    pub flags: FlagState,
    /// Registers written anywhere in this block.
    pub written: HashSet<u8>,
    /// Registers written anywhere in this function.
    ///
    /// The block-level set is cleared for every block, because a block that
    /// prepares a call's arguments is not the block that calls. This one
    /// is not: it says what the function as a whole touched, which is how
    /// the return value's register is worked out.
    pub ever_written: HashSet<u8>,
    /// Registers a call has overwritten since their last definition.
    ///
    /// A call is where the lifted text stops modelling: the callee may write
    /// any caller-saved register, and the lifted statements say nothing about
    /// it. Reading such a register afterwards used to print the value it held
    /// *before* the call, which is a number the program no longer has. The
    /// register is marked here and reads become explicitly unknown instead.
    pub clobbered: HashSet<u8>,
    /// Notes about approximations, one per address.
    pub notes: Vec<(Addr, String)>,
    /// Absolute address the current instruction's PC-relative operand refers
    /// to.
    pub pc_target: Option<u64>,
}

impl<'a> Core<'a> {
    /// Build the shared state.
    pub fn new(target: &'static dyn Target, name_of: &'a dyn Fn(Addr) -> String) -> Self {
        Core {
            target,
            name_of,
            symbols: None,
            sp_off: 0,
            fp_off: None,
            fp_anchor: None,
            dynamic_sp: false,
            pending_cmp: None,
            flags: FlagState::default(),
            written: HashSet::new(),
            ever_written: HashSet::new(),
            clobbered: HashSet::new(),
            notes: Vec::new(),
            pc_target: None,
        }
    }

    /// Attach a symbol lookup, for the registers the linker fixes.
    pub fn with_symbols(mut self, symbols: SymbolLookup<'a>) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// The value of a register the linker fixes, when it is known.
    fn fixed_value(&self, idx: u8) -> Option<u64> {
        let name = self.target.fixed_register(idx)?;
        (self.symbols?)(name)
    }

    /// Name of a register, at the architecture's canonical width.
    pub fn name(&self, idx: u8) -> String {
        self.target.reg_name(idx)
    }

    /// Read an operand as an expression.
    pub fn read(&mut self, op: &Operand, stmts: &mut Vec<Stmt>) -> Expr {
        match op {
            Operand::Reg(r) => self.read_reg(*r),
            Operand::Modified { reg, kind, amount } => self.read_modified(*reg, *kind, *amount),
            Operand::Imm(v) => Expr::Const(*v),
            Operand::Mem { size, .. } => {
                if let Some(slot) = self.stack_slot(op) {
                    return Expr::var(slot_name(slot, op.size()));
                }
                let addr = self.addr_of(op, stmts);
                Expr::Load {
                    addr: Box::new(addr),
                    size: if *size == 0 { 8 } else { *size },
                    signed: false,
                }
            }
            Operand::Cond(_) => Expr::Unknown("condition".into()),
            // A vector register is not one value: it is a set of elements,
            // and an x87 slot is a slot of a stack. Reading one of them as a
            // single word would be a lie, so the architectures that have
            // them read them element by element instead.
            Operand::Vec(..) => Expr::Unknown("vector register".into()),
            Operand::Fpu(_) => Expr::Unknown("x87 slot".into()),
        }
    }

    /// Read a register, narrowing to the access width.
    ///
    /// A view that starts above the bottom of the register is shifted down
    /// first, so `ah` reads bits 8 to 15 of `rax`.
    pub fn read_reg(&self, r: Reg) -> Expr {
        // A register that is hardwired to zero reads as zero, whatever the
        // program last tried to write into it and whatever width the view
        // asks for: the low half of zero is zero too.
        if Some(r.idx) == self.target.zero_reg() {
            return Expr::Const(0);
        }
        // A caller-saved register a call has overwritten holds whatever the
        // callee left there, which this tool does not model.
        if self.clobbered.contains(&r.idx) && self.target.fixed_register(r.idx).is_none() {
            return Expr::Unknown(format!("{} after a call", self.name(r.idx)));
        }
        // A register the linker fixes reads as the value the linker gave it,
        // not as an unknown name.
        if r.offset == 0
            && let Some(value) = self.fixed_value(r.idx)
        {
            return Expr::Const(value);
        }
        let full = Expr::var(self.name(r.idx));
        let width = self.target.reg_width(r.idx);
        if r.offset == 0 && r.size >= width {
            return full;
        }
        let shifted = if r.offset == 0 {
            full
        } else {
            Expr::Bin {
                op: BinOp::Shr,
                lhs: Box::new(full),
                rhs: Box::new(Expr::Const(u64::from(r.offset))),
                bits: 8,
            }
        };
        Expr::Cast {
            kind: CastKind::Trunc,
            arg: Box::new(shifted),
            from: 8,
            to: r.size,
        }
    }

    /// Read a register with a shift or an extension applied to it.
    ///
    /// The order matters: on AArch64 the extension happens first and the
    /// shift second, so `w1, sxtw #2` is `(i64)(i32)w1 << 2` and not the
    /// other way round. The extended value is always 64 bits wide, because
    /// that is what the register it lands in holds.
    pub fn read_modified(&self, reg: Reg, kind: ModKind, amount: u8) -> Expr {
        let mut value = self.read_reg(reg);
        if !kind.is_shift()
            && let Some(from) = kind.from()
        {
            value = if kind.signed() {
                // A signed extension of a narrow view is one cast, not a
                // truncation followed by a sign-extending cast of the
                // truncated value: `(i64)(u32)w1` reads unsigned and is not
                // what `sxtw` does.
                widen_signed(value, from)
            } else {
                match value {
                    Expr::Cast {
                        kind: CastKind::Trunc,
                        arg,
                        from: 8,
                        to,
                    } if to == from => Expr::Cast {
                        kind: CastKind::Trunc,
                        arg,
                        from: 8,
                        to,
                    },
                    other => Expr::Cast {
                        kind: CastKind::Zext,
                        arg: Box::new(other),
                        from,
                        to: 8,
                    },
                }
            };
        }
        if amount == 0 {
            return value;
        }
        let op = match kind {
            ModKind::Lsr => BinOp::Shr,
            ModKind::Asr => BinOp::Sar,
            ModKind::Ror => {
                // The IR has no rotate, so a rotate is spelled out the way
                // the instruction does it: the high half shifted up and the
                // low half shifted down, or'd together.
                let bits = u32::from(reg.size) * 8;
                let complement = Expr::Bin {
                    op: BinOp::Sub,
                    lhs: Box::new(Expr::Const(u64::from(bits))),
                    rhs: Box::new(Expr::Const(u64::from(amount))),
                    bits: reg.size,
                };
                let high = Expr::Bin {
                    op: BinOp::Shl,
                    lhs: Box::new(value.clone()),
                    rhs: Box::new(complement),
                    bits: reg.size,
                };
                let low = Expr::Bin {
                    op: BinOp::Shr,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(u64::from(amount))),
                    bits: reg.size,
                };
                return Expr::Bin {
                    op: BinOp::Or,
                    lhs: Box::new(high),
                    rhs: Box::new(low),
                    bits: reg.size,
                };
            }
            _ => BinOp::Shl,
        };
        Expr::Bin {
            op,
            lhs: Box::new(value),
            rhs: Box::new(Expr::Const(u64::from(amount))),
            bits: 8,
        }
    }

    /// The access an instruction performs.
    ///
    /// The instruction text may already carry the width — x86 prints it —
    /// in which case that wins. Otherwise the architecture says what the
    /// mnemonic means, with the data register's width for the architectures
    /// that put the width there.
    pub fn access_of(&self, mnemonic: &str, operand: &Operand, data_size: u8) -> Access {
        if let Operand::Mem { size, .. } = operand
            && *size != 0
        {
            return Access {
                width: *size,
                signed: false,
            };
        }
        self.target.mem_access(mnemonic, data_size)
    }

    /// Read a memory operand as a load of a known width.
    ///
    /// The width comes from the instruction, not from the operand, on the
    /// architectures whose decoder does not print it. A frame slot is still
    /// a named local rather than a memory read, because the frame is the
    /// one piece of memory the tool can name.
    pub fn read_load(
        &mut self,
        op: &Operand,
        access: Access,
        stmts: &mut Vec<Stmt>,
    ) -> Expr {
        let Operand::Mem { .. } = op else {
            return self.read(op, stmts);
        };
        if let Some(slot) = self.stack_slot(op) {
            return Expr::var(slot_name(slot, access.width));
        }
        let addr = self.addr_of(op, stmts);
        Expr::Load {
            addr: Box::new(addr),
            size: access.width,
            signed: access.signed,
        }
    }

    /// Store a value through a memory operand.
    ///
    /// A frame slot becomes an assignment to the named local, so a store to
    /// the frame is visible to the analysis as a variable rather than as an
    /// opaque write to an address nobody can name.
    pub fn store(
        &mut self,
        op: &Operand,
        value: Expr,
        access: Access,
        stmts: &mut Vec<Stmt>,
    ) {
        let Operand::Mem { .. } = op else {
            return;
        };
        if let Some(slot) = self.stack_slot(op) {
            self.assign(&slot_name(slot, access.width), value, stmts);
            return;
        }
        let addr = self.addr_of(op, stmts);
        stmts.push(Stmt::Store {
            addr,
            value,
            size: access.width,
        });
    }

    /// Write an expression into a destination operand.
    pub fn write(&mut self, op: &Operand, value: Expr, stmts: &mut Vec<Stmt>) {
        match op {
            Operand::Reg(r) => {
                // A write to the hardwired zero register is discarded: the
                // register does not hold what was written.
                if Some(r.idx) == self.target.zero_reg() {
                    return;
                }
                let dest = self.name(r.idx);
                self.written.insert(r.idx);
                self.ever_written.insert(r.idx);
                self.clobbered.remove(&r.idx);
                let width = self.target.reg_width(r.idx);
                let value = if r.size == width && r.offset == 0 {
                    value
                } else if r.size == 4 && width == 8 && r.offset == 0 {
                    // Writing a 32-bit view of a 64-bit register clears the
                    // upper half.
                    Expr::Cast {
                        kind: CastKind::Zext,
                        arg: Box::new(value),
                        from: 4,
                        to: 8,
                    }
                } else {
                    // A narrower view leaves the rest of the register alone.
                    merge_write(&dest, value, r.size, r.offset, width)
                };
                self.assign(&dest, value, stmts);
            }
            Operand::Mem { size, .. } => {
                if let Some(slot) = self.stack_slot(op) {
                    self.assign(&slot_name(slot, op.size()), value, stmts);
                } else {
                    let addr = self.addr_of(op, stmts);
                    stmts.push(Stmt::Store {
                        addr,
                        value,
                        size: if *size == 0 { 8 } else { *size },
                    });
                }
            }
            Operand::Imm(_) | Operand::Cond(_) => {}
            // A modified register is not a destination: the modifier only
            // says how its value is read.
            Operand::Modified { .. } => {}
            // Writing a whole vector register or an x87 slot as one word
            // would lose the elements the register holds, so the
            // architectures that have them write them element by element.
            Operand::Vec(..) | Operand::Fpu(_) => {}
        }
    }

    /// Assign a value to a name.
    pub fn assign(&mut self, dest: &str, value: Expr, stmts: &mut Vec<Stmt>) {
        stmts.push(Stmt::Assign {
            dest: dest.to_string(),
            value: value.simplify(),
        });
    }

    /// Address expression for a memory operand.
    pub fn addr_of(&mut self, op: &Operand, _stmts: &mut Vec<Stmt>) -> Expr {
        let Operand::Mem {
            base,
            index,
            scale,
            disp,
            abs,
            seg,
            ..
        } = op
        else {
            return Expr::Unknown("not memory".into());
        };
        if let Some(seg) = seg {
            if base.is_none() && index.is_none() {
                // A segmented absolute address belongs to another address
                // space, so it gets a name rather than an integer that would
                // read as a plain pointer.
                return Expr::var(format!("{seg}:{:#x}", *disp as u64));
            }
            self.notes.push((
                Addr::NULL,
                format!("segment override {seg} on a computed address ignored"),
            ));
        }
        if let Some(abs) = abs {
            return Expr::Const(*abs);
        }
        // A base register that holds a program counter is an address, not
        // arithmetic on a register of that name.
        if let Some(b) = base
            && self.target.is_pc(b.idx)
            && let Some(target) = self.pc_target
        {
            return Expr::Const(target);
        }
        let mut parts: Vec<Expr> = Vec::new();
        if let Some(b) = base {
            parts.push(self.read_reg(*b));
        }
        if let Some(i) = index {
            let mut e = self.read_reg(*i);
            if *scale > 1 {
                e = Expr::Bin {
                    op: BinOp::Mul,
                    lhs: Box::new(e),
                    rhs: Box::new(Expr::Const(*scale)),
                    bits: 8,
                };
            }
            parts.push(e);
        }
        let mut acc = parts
            .into_iter()
            .reduce(|a, b| Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(a),
                rhs: Box::new(b),
                bits: 8,
            })
            .unwrap_or(Expr::Const(0));
        if *disp != 0 {
            acc = if *disp < 0 {
                Expr::Bin {
                    op: BinOp::Sub,
                    lhs: Box::new(acc),
                    rhs: Box::new(Expr::Const((*disp).unsigned_abs())),
                    bits: 8,
                }
            } else {
                Expr::Bin {
                    op: BinOp::Add,
                    lhs: Box::new(acc),
                    rhs: Box::new(Expr::Const(*disp as u64)),
                    bits: 8,
                }
            };
        }
        // A base the linker fixed folds with the displacement into a plain
        // address, which is what the reader wants to see.
        acc.simplify()
    }

    /// Frame slot for a memory operand, when it is a fixed stack slot.
    pub fn stack_slot(&self, op: &Operand) -> Option<i64> {
        if self.dynamic_sp {
            // Once the stack pointer moves by a variable amount, offsets from
            // it are not frame slots any more.
            if let Operand::Mem { base, .. } = op
                && base.is_some_and(|b| b.idx == self.target.sp_reg())
            {
                return None;
            }
        }
        let Operand::Mem {
            base, index, disp, ..
        } = op
        else {
            return None;
        };
        if index.is_some() {
            return None;
        }
        let base = (*base)?;
        if base.idx == self.target.sp_reg() {
            return Some(self.sp_off.wrapping_add(*disp));
        }
        if Some(base.idx) == self.target.fp_reg() {
            return self.fp_off.map(|off| off.wrapping_add(*disp));
        }
        None
    }

    /// The operands of the last flag-setting operation.
    pub fn pending_pair(&self) -> (Expr, Expr) {
        match &self.pending_cmp {
            Some(p) => (p.lhs.clone(), p.rhs.clone()),
            None => (
                Expr::Unknown("flags".into()),
                Expr::Unknown("flags".into()),
            ),
        }
    }

    /// The condition a branch or a `setcc` tests.
    ///
    /// A floating-point comparison is not an integer comparison of two
    /// numbers, so the integer conditions cannot answer it: the machine's
    /// compare reports "less than", "equal" and "unordered" separately, and
    /// "below" means "less than *or* unordered". The architecture knows how
    /// its branches read those flags, so it answers this one.
    ///
    /// The conditions that ask about a single flag, or about two flags
    /// together, are answered from the flags themselves. A comparison
    /// answers most of the others with a relation between the two values
    /// it compared, which is the form a reader wants to see.
    pub fn condition_of(&self, mnemonic: &str, cond: FlagCond) -> Expr {
        if let Some(p) = &self.pending_cmp {
            if p.float
                && let Some(e) =
                    self.target
                        .float_branch_cond(mnemonic, &p.lhs, &p.rhs, p.width)
            {
                return e;
            }
            // A comparison answers most of the conditions with a relation
            // between the two values it compared, which is the form a
            // reader wants to see. The rest — the sign flag on its own,
            // the overflow flag on its own — come from the flags.
            if p.flags == ALL_FLAGS
                && let Some(e) = compare_condition(p, cond)
            {
                return e;
            }
        }
        match self.flags.condition(cond) {
            Some(e) => e,
            // The condition reads a flag the operation did not define. The
            // machine leaves it alone, so what it holds is whatever came
            // before, and nothing here says what that was.
            None => Expr::Unknown("flags".into()),
        }
    }

    /// Record the flags an operation left behind.
    ///
    /// A comparison gives both the comparison and the flags it derives: the
    /// branch that follows can be written as a relation between the two
    /// values, and the flags themselves say the same thing in a longer
    /// form. An arithmetic operation gives the flags and no comparison,
    /// because there is no pair of values the branch could be written
    /// against.
    pub fn set_flags(&mut self, flags: FlagState, cmp: Option<PendingCmp>) {
        self.flags = flags;
        self.pending_cmp = cmp;
    }

    /// Record the flags an operation changed, leaving the others alone.
    ///
    /// An operation that does not touch a flag leaves it holding what it
    /// held before, and the machine's rules say so: a bit test changes the
    /// carry and nothing else, a multiplication changes the carry and the
    /// overflow, and an increment leaves the carry alone on purpose.
    pub fn update_flags(&mut self, flags: FlagState) {
        if flags.cf.is_some() {
            self.flags.cf = flags.cf.clone();
        }
        if flags.zf.is_some() {
            self.flags.zf = flags.zf.clone();
        }
        if flags.sf.is_some() {
            self.flags.sf = flags.sf.clone();
        }
        if flags.of.is_some() {
            self.flags.of = flags.of.clone();
        }
        if flags.pf.is_some() {
            self.flags.pf = flags.pf.clone();
        }
    }

    /// Name for the value an operand holds *after* being written.
    pub fn dest_value(&self, op: &Operand) -> Expr {
        match op {
            Operand::Reg(r) => {
                let mut full = Expr::var(self.name(r.idx));
                let width = self.target.reg_width(r.idx);
                // A high-byte register is a view at an offset, not at zero:
                // `ah` is bits 8..16 of `rax`, and the offset is already a bit
                // count.
                if r.offset > 0 {
                    full = Expr::Bin {
                        op: BinOp::Shr,
                        lhs: Box::new(full),
                        rhs: Box::new(Expr::Const(u64::from(r.offset))),
                        bits: 8,
                    };
                }
                // A narrower view holds only part of the register, and a
                // flag that reads the result reads that part.
                if r.size >= width && r.offset == 0 {
                    full
                } else {
                    Expr::Cast {
                        kind: CastKind::Trunc,
                        arg: Box::new(full),
                        from: width,
                        to: r.size,
                    }
                }
            }
            _ => Expr::Unknown("memory flags".into()),
        }
    }

    /// Record that a call overwrote `regs`.
    ///
    /// The return register is not among them: the call defines it, so it
    /// holds the callee's result rather than an unknown.
    pub fn clobber(&mut self, regs: &[u8], ret: u8) {
        self.clobbered.clear();
        for r in regs {
            if *r != ret {
                self.clobbered.insert(*r);
            }
        }
    }

    /// Adjust the stack pointer by a constant.
    pub fn adjust_sp(&mut self, delta: i64) {
        self.sp_off = self.sp_off.wrapping_add(delta);
    }

    /// The condition for a conditional branch, from the last flag-setting
    /// operation.
    pub fn take_condition(&mut self, parsed: &Parsed) -> Expr {
        let Some(cond) = self.target.branch_cond(&parsed.mnemonic) else {
            return Expr::Unknown("condition".into());
        };
        // Flags survive until something writes them, so two conditional
        // branches can follow one comparison. Consuming here turned the
        // second branch into `<flags> <= <flags>`.
        self.condition_of(&parsed.mnemonic, cond)
    }

    /// The arguments a call at this point passes: only the registers this
    /// function has already set.
    pub fn call_args(&self) -> Vec<Expr> {
        let regs = self.target.arg_regs();
        let mut argc = 0usize;
        for (i, reg) in regs.iter().enumerate() {
            if self.written.contains(reg) {
                argc = i + 1;
            }
        }
        regs[..argc]
            .iter()
            .map(|r| Expr::var(self.name(*r)))
            .collect()
    }
}

/// The condition a comparison answers, when the condition is a relation
/// between the two compared values.
///
/// The relation has to be the one the machine asks about. "Less" after a
/// comparison is the signed comparison of the two values, and it holds
/// even when the difference overflowed: the overflow flag is what tells
/// the two apart, and the branch that asks about the overflow flag is the
/// one that needs it.
///
/// The sign flag on its own and the overflow flag on its own are not
/// relations between the two values. The sign flag is the sign of the
/// difference, and the overflow flag says whether the difference was
/// representable, so those two come from the flags instead.
fn compare_condition(p: &PendingCmp, cond: FlagCond) -> Option<Expr> {
    let bits = p.width;
    let (op, signed) = match cond {
        FlagCond::Z => (CondOp::Eq, false),
        FlagCond::Nz => (CondOp::Ne, false),
        FlagCond::C => (CondOp::Ult, false),
        FlagCond::Nc => (CondOp::Uge, false),
        FlagCond::A => (CondOp::Ugt, false),
        FlagCond::Be => (CondOp::Ule, false),
        FlagCond::L => (CondOp::Slt, true),
        FlagCond::Nl => (CondOp::Sge, true),
        FlagCond::G => (CondOp::Sgt, true),
        FlagCond::Le => (CondOp::Sle, true),
        _ => return None,
    };
    // The width travels with the comparison, and it says how the operands
    // are read. The operands keep the truncation that says which part of
    // a register the comparison looked at.
    let _ = signed;
    Some(Expr::Cond {
        op,
        lhs: Box::new(p.lhs.clone()),
        rhs: Box::new(p.rhs.clone()),
        bits,
    })
}

/// Widen a comparison operand so a signed comparison reads it the way the
/// branch does.
///
/// A 32-bit comparison reads the low half of a register, and a signed branch
/// reads that half as a signed number. The operand arrives truncated, which
/// zero-extends it, so the sign has to be restored: `(u32)rdi` is not the
/// value `jl` looks at, `(i64)rdi` is.
///
/// A truncated register view is extended at the source, so the result reads
/// as one cast rather than two.
pub fn widen_signed(value: Expr, width: u8) -> Expr {
    if width == 0 || width >= 8 {
        return value;
    }
    let arg: Box<Expr> = match value {
        Expr::Cast {
            kind: CastKind::Trunc,
            arg,
            from: 8,
            to,
        } if to == width => arg,
        other => Box::new(other),
    };
    Expr::Cast {
        kind: CastKind::Sext,
        arg,
        from: width,
        to: 8,
    }
}
/// Merge a narrow write into the full register.
pub fn merge_write(dest: &str, value: Expr, size: u8, offset: u8, width: u8) -> Expr {    let bits = u32::from(size) * 8;
    let field_mask = if bits >= u32::from(width) * 8 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let mask = field_mask << offset;
    let cleared = Expr::Bin {
        op: BinOp::And,
        lhs: Box::new(Expr::var(dest)),
        rhs: Box::new(Expr::Const(!mask)),
        bits: 8,
    };
    let mut field = Expr::Bin {
        op: BinOp::And,
        lhs: Box::new(value),
        rhs: Box::new(Expr::Const(field_mask)),
        bits: size,
    };
    if offset > 0 {
        field = Expr::Bin {
            op: BinOp::Shl,
            lhs: Box::new(field),
            rhs: Box::new(Expr::Const(u64::from(offset))),
            bits: 8,
        };
    }
    Expr::Bin {
        op: BinOp::Or,
        lhs: Box::new(cleared),
        rhs: Box::new(field),
        bits: 8,
    }
}

/// Name for a frame slot: locals below the frame base, arguments above it.
pub fn slot_name(slot: i64, size: u8) -> String {
    let suffix = match size {
        1 => "u8",
        2 => "u16",
        4 => "u32",
        // Ten bytes is the x87 extended-precision format, and on a 64-bit
        // target nothing else is ten bytes wide. The slot holds a float,
        // and saying so is what tells a reader why the value ends up in an
        // `f80_*` call.
        10 => "f80",
        _ => "u64",
    };
    let base = if slot < 0 {
        format!("local_{:x}", -slot)
    } else {
        format!("arg_{:x}", slot)
    };
    format!("{base}_{suffix}")
}

/// A lift of one function's blocks.
pub trait Lifter {
    /// Lift one block. The caller attaches the terminator from the last
    /// instruction's flow.
    fn lift_block(&mut self, insns: &[Instruction]) -> Lifted;

    /// Argument expressions for a call at this point: only the registers this
    /// function has already set.
    fn call_args(&self) -> Vec<Expr>;

    /// The register a function returns its value in.
    fn ret_reg(&self) -> String;

    /// The type of the value the function returns, when the body says so.
    ///
    /// A function's return type is a fact about its declaration, not about
    /// its code, so this is `None` unless the architecture can tell from
    /// the body alone. The renderer prints the type in the signature when
    /// there is one.
    fn ret_type(&self) -> Option<&'static str> {
        None
    }

    /// Whether the function's return value is one it chose.
    ///
    /// A function that returns a value has to write the register the ABI
    /// names for it. One that never writes that register returns whatever
    /// was in it, which is not a value the function produced, and printing
    /// it would claim otherwise.
    fn ret_defined(&self) -> bool {
        true
    }

    /// The argument registers, in ABI order, by name.
    fn arg_reg_names(&self) -> Vec<String>;

    /// The name the stack pointer is printed under.
    fn sp_alias(&self) -> &'static str;

    /// The name of the stack pointer register itself.
    fn sp_reg_name(&self) -> String;

    /// Notes about approximations made while lifting.
    fn notes(&mut self) -> Vec<(Addr, String)>;

    /// The frame slot of a register that a debug frame base names.
    ///
    /// A `DW_AT_frame_base` of `DW_OP_reg6` says every `fbreg` offset is
    /// relative to `rbp`, not to the canonical frame address. Converting one
    /// to a lifter slot needs the slot `rbp` itself was anchored at, which is
    /// what this returns. `None` means the register does not carry a frame
    /// base in this function, and the debug names are left unapplied.
    fn frame_base_slot(&self, _reg: &str) -> Option<i64> {
        None
    }
}

/// Split on commas that are not inside brackets.
pub fn split_operands(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Parse an integer in the forms the decoders emit.
pub fn parse_int(s: &str) -> Option<u64> {
    let s = s.trim();
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let rest = rest.trim_start_matches('+');
    let v = if let Some(hex) = rest.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()?
    } else if let Some(hex) = rest.strip_suffix('h') {
        u64::from_str_radix(hex, 16).ok()?
    } else {
        rest.parse::<u64>().ok()?
    };
    Some(if neg { v.wrapping_neg() } else { v })
}

/// True when the instruction leaves the block.
pub fn is_terminator(flow: FlowKind) -> bool {
    matches!(
        flow,
        FlowKind::Return
            | FlowKind::Call
            | FlowKind::ConditionalBranch
            | FlowKind::UnconditionalBranch
            | FlowKind::IndirectBranch
            | FlowKind::IndirectCall
    )
}

/// The arithmetic operator for a mnemonic that names one.
pub fn alu_op(mnemonic: &str) -> Option<BinOp> {
    Some(match mnemonic {
        "add" => BinOp::Add,
        "sub" => BinOp::Sub,
        "and" => BinOp::And,
        "or" => BinOp::Or,
        "xor" => BinOp::Xor,
        "mul" => BinOp::Mul,
        "shl" | "sal" => BinOp::Shl,
        "shr" | "lsr" => BinOp::Shr,
        "sar" | "asr" => BinOp::Sar,
        _ => return None,
    })
}

/// The unary operator for a mnemonic that names one.
pub fn unary_op(mnemonic: &str) -> Option<UnOp> {
    Some(match mnemonic {
        "neg" => UnOp::Neg,
        "not" => UnOp::Not,
        _ => return None,
    })
}
