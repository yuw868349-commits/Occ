//! Instruction decoding for the architectures this toolset can analyse.
//!
//! Recognition is per-architecture and per-width: an `Arch` value plus a
//! bitness, and nothing else. A decoder that guesses the architecture from the
//! bytes produces confident nonsense on data that happens to look like code.
//!
//! Three decoders share one result type, [`Decoded`], and one entry point,
//! [`AnyDecoder::new`]. The x86 decoder keeps its own module because its
//! encoding is genuinely irregular; AArch64 and RISC-V are table-driven and get
//! a module each. The shared type means the analysis layer never branches on
//! architecture except where an architecture-specific fact matters.
//!
//! Failure policy is uniform: an undecodable unit becomes a `.byte` (x86) or
//! `.word` (AArch64, RISC-V) directive of the correct width, so the stream
//! stays aligned and the caller sees exactly which unit was not understood.
//! Guessing a length is worse than admitting the gap: one byte of over-read
//! desynchronises every following instruction and the listing still looks
//! plausible.


#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod aarch64;
pub mod riscv;
pub mod wasm;
pub mod x86;

use occ_core::{Addr, Arch, Instruction};

pub use x86::Decoder;
pub use x86::Decoder as X86Decoder;

/// Result of decoding one instruction.
pub struct Decoded {
    /// The instruction, with text already rendered.
    pub insn: Instruction,
    /// Absolute address a PC-relative operand referred to, when there was one.
    pub rip_target: Option<Addr>,
}

/// A decoder bound to one architecture and width.
#[derive(Clone, Copy)]
pub enum AnyDecoder {
    /// x86 / x86-64.
    X86(x86::Decoder),
    /// AArch64 (A64).
    Aarch64(aarch64::Decoder),
    /// RISC-V, 32- or 64-bit.
    Riscv(riscv::Decoder),
}

impl AnyDecoder {
    /// Build a decoder for `arch` at `bits`.
    ///
    /// Returns `None` when the pair is not a real target: the caller asked for
    /// x86-16 or AArch64-32, which this toolset does not claim to decode.
    pub fn new(arch: Arch, bits: u8) -> Option<Self> {
        match arch {
            Arch::X86 | Arch::X86_64 => x86::Decoder::new(bits).map(AnyDecoder::X86),
            Arch::Aarch64 => aarch64::Decoder::new(bits).map(AnyDecoder::Aarch64),
            Arch::RiscV32 | Arch::RiscV64 => riscv::Decoder::new(bits).map(AnyDecoder::Riscv),
            _ => None,
        }
    }

    /// The architecture this decoder handles.
    pub const fn arch(self) -> Arch {
        match self {
            AnyDecoder::X86(d) => d.arch(),
            AnyDecoder::Aarch64(_) => Arch::Aarch64,
            AnyDecoder::Riscv(d) => d.arch(),
        }
    }

    /// Fixed instruction width in bytes, or zero when the encoding is variable.
    pub const fn width(self) -> u8 {
        match self {
            AnyDecoder::X86(_) => 0,
            AnyDecoder::Aarch64(_) => 4,
            AnyDecoder::Riscv(_) => 0,
        }
    }

    /// Decode one instruction at `origin`.
    ///
    /// Returns `None` only when `bytes` is empty or too short to hold even a
    /// single unit. Undecodable content inside a valid unit comes back as a
    /// data directive of the correct width.
    pub fn decode(&self, origin: Addr, bytes: &[u8]) -> Option<Decoded> {
        match self {
            AnyDecoder::X86(d) => d.decode(origin, bytes),
            AnyDecoder::Aarch64(d) => d.decode(origin, bytes),
            AnyDecoder::Riscv(d) => d.decode(origin, bytes),
        }
    }
}

impl core::fmt::Debug for AnyDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("AnyDecoder").field(&self.arch()).finish()
    }
}

/// True when this architecture uses fixed-width instructions.
pub const fn is_fixed_width(arch: Arch) -> bool {
    matches!(arch, Arch::Aarch64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_impossible_widths() {
        assert!(AnyDecoder::new(Arch::X86, 16).is_none());
        assert!(AnyDecoder::new(Arch::Aarch64, 32).is_none());
        assert!(AnyDecoder::new(Arch::Other, 32).is_none());
    }

    #[test]
    fn builds_every_supported_pair() {
        assert!(AnyDecoder::new(Arch::X86_64, 64).is_some());
        assert!(AnyDecoder::new(Arch::X86, 32).is_some());
        assert!(AnyDecoder::new(Arch::Aarch64, 64).is_some());
        assert!(AnyDecoder::new(Arch::RiscV64, 64).is_some());
        assert!(AnyDecoder::new(Arch::RiscV32, 32).is_some());
    }

    #[test]
    fn dispatch_agrees_with_width() {
        let a = AnyDecoder::new(Arch::Aarch64, 64).unwrap();
        assert_eq!(a.width(), 4);
        assert!(is_fixed_width(a.arch()));
        let x = AnyDecoder::new(Arch::X86_64, 64).unwrap();
        assert_eq!(x.width(), 0);
        assert!(!is_fixed_width(x.arch()));
    }
}
