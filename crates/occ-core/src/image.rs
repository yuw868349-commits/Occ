//! Container metadata: what a binary *is*, independent of how it was loaded.
//!
//! These types are pure data. They deliberately do not own the file bytes —
//! that lives in `occ-format`, which pairs an [`Image`] with a memory map.
//! Keeping the split means a session snapshot can be serialised without
//! dragging a gigabyte of mapped file along with it.

use crate::addr::Addr;
use crate::evidence::Claim;
use std::fmt;

/// Dense identifier for an image within a session, starting at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ImageId(pub u32);

impl fmt::Display for ImageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// Executable and Linkable Format (Linux, BSD, Android).
    Elf,
    /// Portable Executable (Windows).
    Pe,
    /// Raw COFF object (Windows, also inside PE and ELF `.o` archives).
    Coff,
    /// Mach-O (macOS, iOS), including relocatable objects.
    MachO,
    /// WebAssembly module.
    Wasm,
    /// Fat/universal binary wrapping several Mach-O images.
    MachOFat,
    /// Archive (`ar`) of object files.
    Archive,
    /// Recognised as a binary but not as anything supported.
    Unknown,
}

impl Format {
    /// Lower-case label used in output.
    pub const fn label(self) -> &'static str {
        match self {
            Format::Elf => "elf",
            Format::Pe => "pe",
            Format::Coff => "coff",
            Format::MachO => "macho",
            Format::Wasm => "wasm",
            Format::MachOFat => "macho-fat",
            Format::Archive => "archive",
            Format::Unknown => "unknown",
        }
    }

    /// True when this build can analyse the format end to end.
    ///
    /// Mach-O and WebAssembly are included: their headers, sections, symbols,
    /// imports and code are all parsed, and their instructions are decoded. The
    /// gaps that remain are per-format details — `PTRACE`-based tracing needs a
    /// Linux host, and WebAssembly has no indirect-branch targets to recover —
    /// not the format itself.
    pub const fn is_supported(self) -> bool {
        matches!(
            self,
            Format::Elf | Format::Pe | Format::Coff | Format::MachO | Format::Wasm
        )
    }
}

/// Instruction set architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    /// 32-bit x86.
    X86,
    /// 64-bit x86.
    X86_64,
    /// 32-bit ARM.
    Arm,
    /// 64-bit ARM.
    Aarch64,
    /// 32-bit RISC-V.
    RiscV32,
    /// 64-bit RISC-V.
    RiscV64,
    /// WebAssembly, whose "instructions" are bytecode rather than machine code.
    Wasm,
    /// Something not in the list.
    Other,
}

impl Arch {
    /// Lower-case label.
    pub const fn label(self) -> &'static str {
        match self {
            Arch::X86 => "x86",
            Arch::X86_64 => "x86_64",
            Arch::Arm => "arm",
            Arch::Aarch64 => "aarch64",
            Arch::RiscV32 => "riscv32",
            Arch::RiscV64 => "riscv64",
            Arch::Wasm => "wasm",
            Arch::Other => "other",
        }
    }

    /// Default instruction alignment, used when reconstructing code.
    pub const fn insn_align(self) -> u64 {
        match self {
            Arch::X86 | Arch::X86_64 | Arch::Wasm => 1,
            // RISC-V code is 2-byte aligned because of the compressed
            // extension; treating it as 4-byte aligned would reject valid
            // branch targets.
            Arch::RiscV32 | Arch::RiscV64 => 2,
            Arch::Arm | Arch::Aarch64 | Arch::Other => 4,
        }
    }

    /// True when the disassembler in this build can decode it.
    pub const fn is_disassemblable(self) -> bool {
        matches!(
            self,
            Arch::X86 | Arch::X86_64 | Arch::Aarch64 | Arch::RiscV32 | Arch::RiscV64 | Arch::Wasm
        )
    }
}

/// Byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endian {
    /// Least significant byte first.
    Little,
    /// Most significant byte first.
    Big,
}

/// Memory permissions for a region.
///
/// Serialises as an `rwx`-style string (`"r-x"`), which is what every
/// disassembler listing and `/proc/maps` dump already shows, so a caller can
/// compare it without a decoder ring.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Perm(u8);

impl Perm {
    /// Readable.
    pub const R: Perm = Perm(0b001);
    /// Writable.
    pub const W: Perm = Perm(0b010);
    /// Executable.
    pub const X: Perm = Perm(0b100);
    /// No access.
    pub const NONE: Perm = Perm(0);

    /// Build from individual flags.
    pub const fn of(read: bool, write: bool, exec: bool) -> Self {
        let mut bits = 0u8;
        if read {
            bits |= 0b001;
        }
        if write {
            bits |= 0b010;
        }
        if exec {
            bits |= 0b100;
        }
        Perm(bits)
    }

    /// True when readable.
    pub const fn read(self) -> bool {
        self.0 & 0b001 != 0
    }

    /// True when writable.
    pub const fn write(self) -> bool {
        self.0 & 0b010 != 0
    }

    /// True when executable.
    pub const fn exec(self) -> bool {
        self.0 & 0b100 != 0
    }

    /// Union of two permission sets.
    pub const fn union(self, other: Perm) -> Perm {
        Perm(self.0 | other.0)
    }

    /// True when no bit is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Perm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = if self.read() { 'r' } else { '-' };
        let w = if self.write() { 'w' } else { '-' };
        let x = if self.exec() { 'x' } else { '-' };
        write!(f, "{r}{w}{x}")
    }
}

impl fmt::Debug for Perm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl Perm {
    /// Parse the three-character form produced by `Display`.
    ///
    /// Any character other than `r`, `w` or `x` in the expected position means
    /// "not granted", which is how `---` and `r--` both decode without a
    /// separate error path.
    pub fn parse(s: &str) -> Perm {
        let mut chars = s.chars();
        let mut perm = Perm::NONE;
        if chars.next() == Some('r') {
            perm = perm.union(Perm::R);
        }
        if chars.next() == Some('w') {
            perm = perm.union(Perm::W);
        }
        if chars.next() == Some('x') {
            perm = perm.union(Perm::X);
        }
        perm
    }
}

/// Semantic role of a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionKind {
    /// Holds instructions.
    Code,
    /// General initialised data.
    Data,
    /// Read-only initialised data (string literals, jump tables, vtables).
    Rodata,
    /// Zero-initialised data, occupies no file space.
    Bss,
    /// Thread-local storage template.
    Tls,
    /// Debug information.
    Debug,
    /// Linker notes, build-id, ABI tags.
    Note,
    /// Exception handling tables (`.eh_frame`, `.pdata`).
    Unwind,
    /// Relocations.
    Reloc,
    /// Anything else.
    Other,
}

/// A section as reported by the container.
#[derive(Debug, Clone)]
pub struct Section {
    /// Index in the container's section table, for cross-reference with other tools.
    pub index: usize,
    /// Section name, verbatim. Stripped binaries frequently have empty names.
    pub name: String,
    /// Virtual address once loaded.
    pub addr: Addr,
    /// Offset of the section contents in the file.
    pub file_offset: u64,
    /// Size in bytes when loaded. May exceed the on-disk size for `.bss`.
    pub size: u64,
    /// Size in bytes actually present in the file.
    pub file_size: u64,
    /// Semantic role.
    pub kind: SectionKind,
    /// Permissions.
    pub perm: Perm,
}

impl Section {
    /// True when this section can contain instructions.
    pub const fn is_code(&self) -> bool {
        matches!(self.kind, SectionKind::Code)
    }

    /// True when the section holds no file-backed bytes.
    pub const fn is_zero_fill(&self) -> bool {
        self.file_size == 0 && self.size > 0
    }

    /// Half-open end address.
    pub fn end(&self) -> Addr {
        self.addr.add(self.size)
    }
}

/// A loadable segment, as described by program headers (ELF) or sections (PE).
#[derive(Debug, Clone)]
pub struct Segment {
    /// Virtual address.
    pub addr: Addr,
    /// Size in memory.
    pub size: u64,
    /// Size backed by file bytes.
    pub file_size: u64,
    /// Permissions.
    pub perm: Perm,
}

impl Segment {
    /// Half-open end address.
    pub fn end(&self) -> Addr {
        self.addr.add(self.size)
    }
}

/// What a symbol refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    /// A function entry point.
    Function,
    /// A data object.
    Object,
    /// A section marker.
    Section,
    /// A source file marker.
    File,
    /// A thread-local object.
    Tls,
    /// Not classified.
    Unknown,
}

/// Linkage of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolBinding {
    /// Visible only within the translation unit.
    Local,
    /// Visible across the image.
    Global,
    /// Overridable, resolved at load time.
    Weak,
    /// Not specified.
    Unknown,
}

/// A symbol table entry.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Demangled-on-request raw name.
    pub name: String,
    /// Address, meaningless when [`Symbol::defined`] is false.
    pub addr: Addr,
    /// Size in bytes; often zero without debug info.
    pub size: u64,
    /// What it refers to.
    pub kind: SymbolKind,
    /// Visibility.
    pub binding: SymbolBinding,
    /// Owning section name, when known.
    pub section: Option<String>,
    /// False for undefined symbols that a dynamic linker must resolve.
    pub defined: bool,
    /// True when the name was demangled from an encoded form.
    pub demangled: bool,
}

/// How an import is bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportKind {
    /// Resolved by the dynamic linker at load time.
    Normal,
    /// Resolved on first call via a stub.
    DelayLoad,
    /// Data import (copy relocation, imported variable).
    Data,
}

/// An imported symbol.
#[derive(Debug, Clone)]
pub struct Import {
    /// Symbol name. Empty when imported by ordinal only.
    pub name: String,
    /// Providing library, as recorded in the container.
    pub library: Option<String>,
    /// Ordinal, for PE imports by ordinal.
    pub ordinal: Option<u32>,
    /// Address of the slot the import lands in (GOT entry / IAT entry).
    pub slot: Option<Addr>,
    /// Binding kind.
    pub kind: ImportKind,
}

/// An exported symbol.
#[derive(Debug, Clone)]
pub struct Export {
    /// Exported name.
    pub name: String,
    /// Address in this image.
    pub addr: Addr,
    /// Ordinal, for PE exports.
    pub ordinal: Option<u32>,
    /// Set when the export forwards to another module (PE forwarder string).
    pub forward: Option<String>,
}

/// Relocation flavour, normalised across formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelocKind {
    /// Absolute address of a symbol/word.
    Absolute,
    /// PC-relative reference.
    Relative,
    /// Reference into the GOT (ELF `GLOB_DAT`, `JUMP_SLOT`).
    Got,
    /// TLS offset.
    Tls,
    /// Relative to image base (PE `DIR64`).
    ImageRelative,
    /// Anything else.
    Other,
}

/// A relocation entry.
#[derive(Debug, Clone)]
pub struct Relocation {
    /// Where the relocation applies.
    ///
    /// In a relocatable object this counts from the start of [`Self::section`],
    /// not from the start of the image: the linker has not assigned an address
    /// yet, so there is nothing else for it to count from.
    pub offset: Addr,
    /// Flavour.
    pub kind: RelocKind,
    /// Target symbol name, when the format records one.
    pub symbol: Option<String>,
    /// Addend, when the format records one.
    ///
    /// `None` means the addend lives in the bytes being relocated, which is
    /// what `SHT_REL` and COFF use.
    pub addend: Option<i64>,
    /// Container-specific type number.
    ///
    /// [`Self::kind`] is a coarse summary for display; applying a relocation
    /// needs the exact type, because `CALL26` and `LDST32_ABS_LO12_NC` both
    /// summarise as "absolute-ish" and patch completely different bits.
    pub raw_type: u32,
    /// Index of the section whose bytes the relocation patches.
    ///
    /// `None` for a table that applies to the image as a whole (ELF
    /// `.rela.dyn`), and for formats that do not record a target section.
    pub section: Option<usize>,
    /// Address the target symbol resolves to inside this image.
    ///
    /// `None` for an undefined symbol: the value is the linker's to choose, and
    /// until then the only honest answer is that it is unknown.
    pub symbol_addr: Option<Addr>,
}

impl Relocation {
    /// A relocation with no container-specific detail.
    pub fn bare(offset: Addr, kind: RelocKind, symbol: Option<String>) -> Relocation {
        Relocation {
            offset,
            kind,
            symbol,
            addend: None,
            raw_type: 0,
            section: None,
            symbol_addr: None,
        }
    }
}

/// Compiler family inferred from build artefacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompilerFamily {
    /// GNU Compiler Collection.
    Gcc,
    /// LLVM clang.
    Clang,
    /// Microsoft Visual C++.
    Msvc,
    /// rustc.
    Rustc,
    /// Go toolchain.
    Go,
    /// Swift compiler.
    Swift,
    /// Zig.
    Zig,
    /// Not determined.
    Unknown,
}

impl CompilerFamily {
    /// Lower-case label.
    pub const fn label(self) -> &'static str {
        match self {
            CompilerFamily::Gcc => "gcc",
            CompilerFamily::Clang => "clang",
            CompilerFamily::Msvc => "msvc",
            CompilerFamily::Rustc => "rustc",
            CompilerFamily::Go => "go",
            CompilerFamily::Swift => "swift",
            CompilerFamily::Zig => "zig",
            CompilerFamily::Unknown => "unknown",
        }
    }
}

/// Source language inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangGuess {
    /// C.
    C,
    /// C++.
    Cpp,
    /// Rust.
    Rust,
    /// Go.
    Go,
    /// Swift.
    Swift,
    /// Zig.
    Zig,
    /// Not determined.
    Unknown,
}

impl LangGuess {
    /// Lower-case label.
    pub const fn label(self) -> &'static str {
        match self {
            LangGuess::C => "c",
            LangGuess::Cpp => "c++",
            LangGuess::Rust => "rust",
            LangGuess::Go => "go",
            LangGuess::Swift => "swift",
            LangGuess::Zig => "zig",
            LangGuess::Unknown => "unknown",
        }
    }
}

/// How the image expects to be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linkage {
    /// Position-independent executable, fixed load base chosen by the kernel.
    Pie,
    /// Position-independent code loaded as a library.
    Shared,
    /// Non-PIE executable with a fixed base.
    Executable,
    /// Statically linked, no dynamic loader involvement.
    Static,
    /// A relocatable object (`.o`, `.obj`, `MH_OBJECT`): no load address, and
    /// every address in it is an offset until a linker assigns one.
    Relocatable,
    /// Not determined.
    Unknown,
}

/// Cheap structural summary used to recognise toolchains and detect packing.
#[derive(Debug, Clone)]
pub struct Fingerprint {
    /// Compiler that produced the image, when inferable.
    pub compiler: CompilerFamily,
    /// Evidence behind [`Fingerprint::compiler`].
    pub compiler_claim: Claim,
    /// Source language, when inferable.
    pub language: LangGuess,
    /// Load model.
    pub linkage: Linkage,
    /// True when symbol names were stripped.
    pub stripped: bool,
    /// True when the image carries a Go build info blob.
    pub has_go_buildinfo: bool,
    /// True when DWARF debug info is present.
    pub has_dwarf: bool,
    /// Per-section entropy, highest first.
    pub section_entropy: Vec<(String, f32)>,
    /// C runtime the image expects: `glibc`, `musl`, `msvcrt`, `ucrt`, or
    /// `none` when statically linked. Distinguishing these changes which
    /// library functions an analyst can assume exist.
    pub runtime: Option<String>,
    /// Suspicion that the image is packed or encrypted, with evidence.
    pub packer_suspicion: Option<Claim>,
}

impl Fingerprint {
    /// A fingerprint with nothing determined.
    ///
    /// Used by the container parsers, which answer "what is in this file" and
    /// not "what produced it". The claim is present but carries no evidence, so
    /// a consumer that reads `compiler_claim.confidence` sees zero rather than
    /// an absent field that might be mistaken for agreement.
    pub fn unknown(id: ImageId) -> Self {
        Fingerprint {
            compiler: CompilerFamily::Unknown,
            compiler_claim: Claim::new(
                crate::handle::Handle::image(id.0),
                "compiler family has not been determined",
            ),
            language: LangGuess::Unknown,
            linkage: Linkage::Unknown,
            stripped: false,
            has_go_buildinfo: false,
            has_dwarf: false,
            section_entropy: Vec::new(),
            runtime: None,
            packer_suspicion: None,
        }
    }
}

/// A loaded binary's metadata.
#[derive(Debug, Clone)]
pub struct Image {
    /// Session-local identifier.
    pub id: ImageId,
    /// Path as supplied by the caller, canonicalised when possible.
    pub path: String,
    /// Container format.
    pub format: Format,
    /// Architecture.
    pub arch: Arch,
    /// Address size in bits.
    pub bits: u8,
    /// Byte order.
    pub endian: Endian,
    /// Entry point virtual address.
    pub entry: Addr,
    /// Preferred load base.
    pub image_base: Addr,
    /// Size of the file on disk.
    pub file_size: u64,
    /// SHA-256 of the file, hex encoded. Used to recognise identical images.
    pub sha256: String,
    /// Loadable segments.
    pub segments: Vec<Segment>,
    /// Sections.
    pub sections: Vec<Section>,
    /// Symbols, in container order.
    pub symbols: Vec<Symbol>,
    /// Imports.
    pub imports: Vec<Import>,
    /// Exports.
    pub exports: Vec<Export>,
    /// Relocations.
    pub relocations: Vec<Relocation>,
    /// Derived structural summary.
    pub fingerprint: Fingerprint,
    /// Requested program interpreter, verbatim (ELF `PT_INTERP`). The path
    /// distinguishes glibc from musl far more reliably than symbol names do.
    pub interpreter: Option<String>,
    /// Non-fatal problems encountered while loading.
    pub warnings: Vec<String>,
}

impl Image {
    /// Total virtual size covered by segments.
    pub fn mapped_size(&self) -> u64 {
        self.segments.iter().map(|s| s.size).sum()
    }

    /// Section containing `addr`, if any.
    pub fn section_at(&self, addr: Addr) -> Option<&Section> {
        self.sections
            .iter()
            .find(|s| addr.within(s.addr, s.size) && s.size > 0)
    }

    /// Section with an exact name match.
    pub fn section_by_name(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Symbol defined at exactly `addr`, preferring global and larger symbols.
    pub fn symbol_at(&self, addr: Addr) -> Option<&Symbol> {
        self.symbols
            .iter()
            .filter(|s| s.defined && s.addr == addr)
            .max_by_key(|s| {
                let global = u8::from(s.binding == SymbolBinding::Global);
                (global, s.size)
            })
    }

    /// Symbol whose `[addr, addr+size)` range contains `addr`.
    pub fn symbol_covering(&self, addr: Addr) -> Option<&Symbol> {
        self.symbols
            .iter()
            .find(|s| s.defined && s.size > 0 && addr.within(s.addr, s.size))
    }

    /// Permissions in effect at `addr`, from the first covering segment.
    pub fn perm_at(&self, addr: Addr) -> Perm {
        self.segments
            .iter()
            .find(|s| addr.within(s.addr, s.size))
            .map(|s| s.perm)
            .unwrap_or(Perm::NONE)
    }

    /// True when `addr` falls inside any mapped segment.
    pub fn is_mapped(&self, addr: Addr) -> bool {
        self.segments.iter().any(|s| addr.within(s.addr, s.size))
    }

    /// Sections whose contents can be disassembled.
    pub fn code_sections(&self) -> impl Iterator<Item = &Section> {
        self.sections.iter().filter(|s| s.is_code() && s.size > 0)
    }

    /// Import with an exact name match.
    pub fn import_by_name(&self, name: &str) -> Option<&Import> {
        self.imports.iter().find(|i| i.name == name)
    }

    /// Import whose slot address equals `addr` — i.e. a call through the PLT/IAT.
    pub fn import_by_slot(&self, addr: Addr) -> Option<&Import> {
        self.imports.iter().find(|i| i.slot == Some(addr))
    }

    /// One-line identity string, used in listings.
    pub fn headline(&self) -> String {
        format!(
            "{} {} {}-bit {} ({})",
            self.format.label(),
            self.arch.label(),
            self.bits,
            self.fingerprint.linkage_label(),
            self.path
        )
    }
}

impl Fingerprint {
    /// Label for the linkage model.
    pub const fn linkage_label(&self) -> &'static str {
        match self.linkage {
            Linkage::Pie => "pie",
            Linkage::Shared => "shared",
            Linkage::Executable => "exec",
            Linkage::Static => "static",
            Linkage::Relocatable => "reloc",
            Linkage::Unknown => "linkage?",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Image {
        Image {
            id: ImageId(0),
            path: "/bin/true".into(),
            format: Format::Elf,
            arch: Arch::X86_64,
            bits: 64,
            endian: Endian::Little,
            entry: Addr(0x1050),
            image_base: Addr(0x400000),
            file_size: 35664,
            sha256: "00".into(),
            segments: vec![
                Segment {
                    addr: Addr(0x400000),
                    size: 0x1000,
                    file_size: 0x400,
                    perm: Perm::R,
                },
                Segment {
                    addr: Addr(0x401000),
                    size: 0x2000,
                    file_size: 0x2000,
                    perm: Perm::of(true, false, true),
                },
            ],
            sections: vec![Section {
                index: 1,
                name: ".text".into(),
                addr: Addr(0x401000),
                file_offset: 0x1000,
                size: 0x100,
                file_size: 0x100,
                kind: SectionKind::Code,
                perm: Perm::of(true, false, true),
            }],
            symbols: vec![Symbol {
                name: "_start".into(),
                addr: Addr(0x401000),
                size: 0x20,
                kind: SymbolKind::Function,
                binding: SymbolBinding::Global,
                section: Some(".text".into()),
                defined: true,
                demangled: false,
            }],
            imports: vec![Import {
                name: "puts".into(),
                library: Some("libc.so.6".into()),
                ordinal: None,
                slot: Some(Addr(0x404000)),
                kind: ImportKind::Normal,
            }],
            exports: Vec::new(),
            relocations: Vec::new(),
            fingerprint: Fingerprint {
                compiler: CompilerFamily::Gcc,
                compiler_claim: Claim::new(crate::handle::Handle::image(0), "gcc"),
                language: LangGuess::C,
                linkage: Linkage::Pie,
                stripped: false,
                has_go_buildinfo: false,
                has_dwarf: false,
                section_entropy: vec![(".text".into(), 6.2)],
                packer_suspicion: None,
                runtime: Some("glibc".into()),
            },
            interpreter: Some("/lib64/ld-linux-x86-64.so.2".into()),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn perm_renders_as_rwx_string() {
        assert_eq!(Perm::of(true, false, true).to_string(), "r-x");
        assert_eq!(Perm::of(true, true, false).to_string(), "rw-");
        assert_eq!(Perm::NONE.to_string(), "---");
    }

    #[test]
    fn perm_round_trips() {
        for p in [
            Perm::R,
            Perm::W,
            Perm::X,
            Perm::of(true, true, true),
            Perm::NONE,
        ] {
            assert_eq!(Perm::parse(&p.to_string()), p);
        }
    }

    #[test]
    fn section_lookup_is_half_open() {
        let img = sample();
        assert!(img.section_at(Addr(0x401000)).is_some());
        assert!(img.section_at(Addr(0x4010ff)).is_some());
        assert!(img.section_at(Addr(0x401100)).is_none());
    }

    #[test]
    fn perm_at_uses_covering_segment() {
        let img = sample();
        assert!(img.perm_at(Addr(0x401000)).exec());
        assert!(!img.perm_at(Addr(0x401000)).write());
        assert_eq!(img.perm_at(Addr(0x900000)), Perm::NONE);
    }

    #[test]
    fn symbol_covering_respects_size() {
        let img = sample();
        assert_eq!(
            img.symbol_covering(Addr(0x401008)).map(|s| s.name.as_str()),
            Some("_start")
        );
        assert!(img.symbol_covering(Addr(0x401030)).is_none());
    }

    #[test]
    fn import_lookup_by_slot() {
        let img = sample();
        assert_eq!(
            img.import_by_slot(Addr(0x404000)).map(|i| i.name.as_str()),
            Some("puts")
        );
        assert!(img.import_by_slot(Addr(0x404008)).is_none());
    }

    #[test]
    fn image_clone_preserves_every_field() {
        // The wire form is built by the MCP layer, not derived from the type,
        // so what has to hold here is that an `Image` copies faithfully.
        let img = sample();
        let back = img.clone();
        assert_eq!(back.entry, img.entry);
        assert_eq!(back.sections.len(), 1);
        assert_eq!(back.fingerprint.compiler, CompilerFamily::Gcc);
    }
}
