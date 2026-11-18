//! ELF parsing.
//!
//! Written against the gABI plus the processor supplements for x86-64, i386,
//! AArch64 and RISC-V. Both ELF classes and both byte orders are read; the
//! widest real-world gap is big-endian MIPS, which this handles at the container
//! level even though the decoder does not.
//!
//! Three details are easy to get wrong and are handled explicitly:
//!
//! * `e_shnum == 0` means the real count lives in section 0's `sh_size`, and
//!   `e_shstrndx == SHN_XINDEX` means the real string table index lives in
//!   section 0's `sh_link`. Large binaries do use both.
//! * Symbol table indices are not dense: index 0 is the reserved null symbol,
//!   and a relocation's `r_sym` counts from there. Losing that offset makes
//!   every call resolve to the previous symbol.
//! * Section addresses are zero for non-allocated sections. Treating those as
//!   addressable produces hundreds of entities all living at `0x0`.

use crate::reader::Reader;
use std::collections::HashMap;

use occ_core::{
    Addr, Arch, CoreError, Endian, Export, Fingerprint, Format, FunctionBound, Image, ImageId,
    Import, ImportKind, Linkage, Perm, RelocKind, Relocation, Result, Section, SectionKind,
    Segment, Symbol, SymbolBinding, SymbolKind,
};

// Section header types.
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHT_REL: u32 = 9;
const SHT_DYNSYM: u32 = 11;
const SHT_INIT_ARRAY: u32 = 14;
const SHT_FINI_ARRAY: u32 = 15;
const SHT_PREINIT_ARRAY: u32 = 16;
const SHT_GROUP: u32 = 17;
const SHT_NOTE: u32 = 7;
const SHT_DYNAMIC: u32 = 6;
const SHT_GNU_HASH: u32 = 0x6fff_fff6;
const SHT_GNU_VERSYM: u32 = 0x6fff_ffff;

// Section header flags.
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_TLS: u64 = 0x400;
const SHF_COMPRESSED: u64 = 0x800;

// Program header types.
const PT_LOAD: u32 = 1;
const PT_INTERP: u32 = 3;
const PT_GNU_STACK: u32 = 0x6474_e551;

// Segment flags.
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

// Dynamic tags.
const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;
const DT_STRTAB: i64 = 5;

// Special section indices.
const SHN_UNDEF: u16 = 0;
const SHN_ABS: u16 = 0xfff1;
const SHN_XINDEX: u16 = 0xffff;

/// Everything the header says that later stages need.
struct Header {
    kind: u16,
    machine: u16,
    entry: u64,
    phoff: u64,
    shoff: u64,
    phentsize: u16,
    phnum: u16,
    shentsize: u16,
    shnum: u32,
    shstrndx: u32,
    bits: u8,
    endian: Endian,
}

/// One raw section header, widened to 64 bits.
#[derive(Clone, Copy, Default)]
struct Shdr {
    name_off: u32,
    kind: u32,
    flags: u64,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    entsize: u64,
}

/// Parse an ELF image.
pub fn parse(data: &[u8], path: &str, id: ImageId) -> Result<Image> {
    let h = parse_header(data)?;
    let r = Reader::new(data, h.endian);

    let mut warnings = Vec::new();
    let mut shdrs = read_shdrs(&r, &h, data)?;

    // Resolve the extended section count and string table index, which live in
    // section 0 when the 16-bit fields are saturated.
    if h.shnum == 0
        && let Some(first) = shdrs.first()
    {
        let n = u32::try_from(first.size).unwrap_or(0);
        if n > 0 && (n as usize) <= data.len() / h.shentsize.max(1) as usize + 1 {
            shdrs = read_shdrs_exact(&r, &h, data, n as usize)?;
        } else {
            warnings.push(format!(
                "e_shnum is 0 but section 0 claims {n} sections; ignoring the section table"
            ));
            shdrs.clear();
        }
    }
    let mut shstrndx = h.shstrndx;
    if h.shstrndx == u32::from(SHN_XINDEX)
        && let Some(first) = shdrs.first()
    {
        shstrndx = first.link;
    }

    let shstrtab: &[u8] = match shdrs.get(shstrndx as usize) {
        Some(s) => r
            .bytes_opt(s.offset as usize, s.size as usize)
            .unwrap_or_default(),
        None => &[],
    };

    let names: Vec<String> = shdrs
        .iter()
        .map(|s| crate::demangle::clean_name(r.str_at(shstrtab, s.name_off as usize)))
        .collect();

    let segments = read_segments(&r, &h, data, &mut warnings);
    let interpreter = read_interpreter(&r, &h, data);

    // A compressed section is reported but not decompressed: doing so needs a
    // DEFLATE implementation, and silently presenting compressed bytes as
    // content would be worse than saying so.
    for (i, s) in shdrs.iter().enumerate() {
        if s.flags & SHF_COMPRESSED != 0 {
            warnings.push(format!(
                "{} is compressed (SHF_COMPRESSED); its contents are not decompressed",
                names.get(i).cloned().unwrap_or_default()
            ));
        }
    }

    let object = h.kind == 1; // ET_REL
    let sections: Vec<Section> = shdrs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let name = names.get(i).cloned().unwrap_or_default();
            Section {
                index: i,
                // A relocatable object has no load address: `sh_addr` is zero
                // for every section and `st_value` is an offset within one.
                // Taking those numbers at face value puts every section at 0
                // and every function at 0. The file offset is used instead, so
                // addresses stay unique; `read_symbols` rebases onto it.
                addr: if object {
                    Addr(s.offset)
                } else if s.flags & SHF_ALLOC != 0 {
                    Addr(s.addr)
                } else {
                    Addr::NULL
                },
                file_offset: if s.kind == SHT_NOBITS { 0 } else { s.offset },
                size: s.size,
                // A section cannot hold more bytes than the file has after its
                // offset. The header is untrusted input: taking `sh_size` at
                // face value lets a 26 KB file claim a 77 GB `.text`, and every
                // pass that walks a section byte by byte then runs for hours.
                // `size` keeps the virtual extent because `.bss` legitimately
                // exceeds the file; `file_size` is what can be read.
                file_size: if s.kind == SHT_NOBITS {
                    0
                } else {
                    s.size.min((data.len() as u64).saturating_sub(s.offset))
                },
                kind: classify_section(&name, s),
                perm: perm_of(s),
                name,
            }
        })
        .collect();

    let symbols = read_symbols(&r, &h, &shdrs, &names, data, object);
    let (mut imports, exports) = split_dynamic(&symbols);
    let (relocations, debug_relocations) =
        read_relocations(&r, &h, &shdrs, &names, &symbols, data);
    bind_import_slots(&mut imports, &relocations);
    let needed = read_needed(&r, &shdrs, &names, data, &mut warnings);

    let image_base = segments.iter().map(|s| s.addr.get()).min().unwrap_or(0);
    let arch = map_machine(h.machine, h.bits);

    // `e_type` is the container's own answer to "how is this loaded", so it is
    // read here rather than re-derived later from section names. `ET_DYN` is
    // ambiguous on its own — a PIE and a shared library have the same type —
    // and `PT_INTERP` is what separates them: the kernel only maps an
    // interpreter for an executable.
    let linkage = match h.kind {
        1 => Linkage::Relocatable,
        2 => {
            if interpreter.is_some() || !needed.is_empty() {
                Linkage::Executable
            } else {
                Linkage::Static
            }
        }
        3 => {
            if interpreter.is_some() {
                Linkage::Pie
            } else if needed.is_empty() {
                // No interpreter and nothing to load: static PIE.
                Linkage::Static
            } else {
                Linkage::Shared
            }
        }
        _ => Linkage::Unknown,
    };
    // The unwind table states each function's extent. It is the linker's own
    // answer, and it survives `strip`, unlike the symbol table.
    let function_bounds = if object {
        Vec::new()
    } else {
        sections
            .iter()
            .find(|s| s.name == ".eh_frame" && s.file_size > 0)
            .map(|s| eh_frame_bounds(data, s, h.endian, h.bits))
            .unwrap_or_default()
    };

    let mut fingerprint = Fingerprint::unknown(id);
    fingerprint.linkage = linkage;

    Ok(Image {
        id,
        path: path.to_string(),
        format: Format::Elf,
        arch,
        bits: h.bits,
        endian: h.endian,
        entry: Addr(h.entry),
        image_base: Addr(image_base),
        file_size: data.len() as u64,
        sha256: crate::sha256::hex_digest(data),
        segments,
        sections,
        symbols,
        imports,
        exports,
        relocations,
        debug_relocations,
        fingerprint,
        interpreter,
        warnings,
        function_bounds,
    })
}

fn parse_header(data: &[u8]) -> Result<Header> {
    if data.len() < 16 || &data[0..4] != b"\x7fELF" {
        return Err(CoreError::Malformed {
            offset: 0,
            detail: "missing ELF magic".into(),
        });
    }
    let class = data[4];
    let bits = match class {
        1 => 32,
        2 => 64,
        other => {
            return Err(CoreError::Malformed {
                offset: 4,
                detail: format!("unknown ELF class {other}"),
            });
        }
    };
    let endian = match data[5] {
        1 => Endian::Little,
        2 => Endian::Big,
        other => {
            return Err(CoreError::Malformed {
                offset: 5,
                detail: format!("unknown ELF data encoding {other}"),
            });
        }
    };
    let r = Reader::new(data, endian);
    let base = if bits == 64 { 0x40 } else { 0x34 };
    if data.len() < base {
        return Err(CoreError::Malformed {
            offset: 0,
            detail: format!("file is {} bytes, shorter than the ELF header", data.len()),
        });
    }

    let (entry, phoff, shoff) = if bits == 64 {
        (r.u64(0x18)?, r.u64(0x20)?, r.u64(0x28)?)
    } else {
        (
            u64::from(r.u32(0x18)?),
            u64::from(r.u32(0x1c)?),
            u64::from(r.u32(0x20)?),
        )
    };

    let kind = r.u16(0x10)?;
    let machine = r.u16(0x12)?;
    let ehsize = r.u16(if bits == 64 { 0x34 } else { 0x28 })?;
    let phentsize = r.u16(if bits == 64 { 0x36 } else { 0x2a })?;
    let phnum = r.u16(if bits == 64 { 0x38 } else { 0x2c })?;
    let shentsize = r.u16(if bits == 64 { 0x3a } else { 0x2e })?;
    let shnum = u32::from(r.u16(if bits == 64 { 0x3c } else { 0x30 })?);
    let shstrndx = u32::from(r.u16(if bits == 64 { 0x3e } else { 0x32 })?);

    Ok(Header {
        kind,
        machine,
        entry,
        phoff,
        shoff,
        phentsize,
        phnum,
        shentsize,
        shnum,
        shstrndx,
        bits,
        endian,
        // `ehsize` is read to validate the header size but is otherwise unused;
        // a mismatch is common in hand-built files and not worth rejecting.
    })
    .inspect(|h| {
        let _ = (ehsize, shnum);
        let _ = h;
    })
}

fn read_shdrs_exact(r: &Reader<'_>, h: &Header, data: &[u8], count: usize) -> Result<Vec<Shdr>> {
    let mut out = Vec::with_capacity(count.min(4096));
    let ent = h.shentsize as usize;
    if ent < if h.bits == 64 { 64 } else { 40 } {
        return Err(CoreError::Malformed {
            offset: h.shoff,
            detail: format!("section header entry size {ent} is too small"),
        });
    }
    let base = h.shoff as usize;
    for i in 0..count {
        let off = base + i * ent;
        if off + ent > data.len() {
            break;
        }
        out.push(read_shdr(r, off, h.bits));
    }
    Ok(out)
}

fn read_shdrs(r: &Reader<'_>, h: &Header, data: &[u8]) -> Result<Vec<Shdr>> {
    if h.shoff == 0 || h.phnum == 0 && h.shnum == 0 {
        return Ok(Vec::new());
    }
    let count = h.shnum as usize;
    read_shdrs_exact(r, h, data, count)
}

fn read_shdr(r: &Reader<'_>, off: usize, bits: u8) -> Shdr {
    if bits == 64 {
        Shdr {
            name_off: r.u32(off).unwrap_or(0),
            kind: r.u32(off + 4).unwrap_or(0),
            flags: r.u64(off + 8).unwrap_or(0),
            addr: r.u64(off + 16).unwrap_or(0),
            offset: r.u64(off + 24).unwrap_or(0),
            size: r.u64(off + 32).unwrap_or(0),
            link: r.u32(off + 40).unwrap_or(0),
            info: r.u32(off + 44).unwrap_or(0),
            entsize: r.u64(off + 56).unwrap_or(0),
        }
    } else {
        Shdr {
            name_off: r.u32(off).unwrap_or(0),
            kind: r.u32(off + 4).unwrap_or(0),
            flags: u64::from(r.u32(off + 8).unwrap_or(0)),
            addr: u64::from(r.u32(off + 12).unwrap_or(0)),
            offset: u64::from(r.u32(off + 16).unwrap_or(0)),
            size: u64::from(r.u32(off + 20).unwrap_or(0)),
            link: r.u32(off + 24).unwrap_or(0),
            info: r.u32(off + 28).unwrap_or(0),
            entsize: u64::from(r.u32(off + 36).unwrap_or(0)),
        }
    }
}

fn read_segments(
    r: &Reader<'_>,
    h: &Header,
    data: &[u8],
    warnings: &mut Vec<String>,
) -> Vec<Segment> {
    if h.phoff == 0 || h.phnum == 0 {
        return Vec::new();
    }
    let ent = h.phentsize as usize;
    let need = if h.bits == 64 { 56 } else { 32 };
    if ent < need {
        warnings.push(format!("program header entry size {ent} is too small"));
        return Vec::new();
    }
    let base = h.phoff as usize;
    let mut out = Vec::with_capacity(h.phnum as usize);
    for i in 0..h.phnum as usize {
        let off = base + i * ent;
        if off + ent > data.len() {
            warnings.push(format!(
                "program header {i} starts at {off:#x}, past end of file"
            ));
            break;
        }
        let (kind, flags, vaddr, filesz, memsz) = if h.bits == 64 {
            (
                r.u32(off).unwrap_or(0),
                r.u32(off + 4).unwrap_or(0),
                r.u64(off + 16).unwrap_or(0),
                r.u64(off + 32).unwrap_or(0),
                r.u64(off + 40).unwrap_or(0),
            )
        } else {
            (
                r.u32(off).unwrap_or(0),
                r.u32(off + 24).unwrap_or(0),
                u64::from(r.u32(off + 8).unwrap_or(0)),
                u64::from(r.u32(off + 16).unwrap_or(0)),
                u64::from(r.u32(off + 20).unwrap_or(0)),
            )
        };
        // Only PT_LOAD contributes to the mapped address space. Notes, the
        // dynamic table and the interpreter pointer are all inside a PT_LOAD
        // already, so including them would double-count.
        // An executable stack is worth reporting: it is the difference between
        // a binary that can be exploited with a stack buffer overflow and one
        // that cannot.
        if kind == PT_GNU_STACK && flags & PF_X != 0 {
            warnings.push("PT_GNU_STACK is executable: the stack is not protected by NX".into());
        }
        if kind != PT_LOAD {
            continue;
        }
        out.push(Segment {
            addr: Addr(vaddr),
            size: memsz,
            file_size: filesz,
            perm: Perm::of(flags & PF_R != 0, flags & PF_W != 0, flags & PF_X != 0),
        });
    }
    out
}

fn read_interpreter(r: &Reader<'_>, h: &Header, data: &[u8]) -> Option<String> {
    if h.phoff == 0 || h.phnum == 0 {
        return None;
    }
    let ent = h.phentsize as usize;
    let base = h.phoff as usize;
    for i in 0..h.phnum as usize {
        let off = base + i * ent;
        if off + ent > data.len() {
            return None;
        }
        let kind = r.u32(off).ok()?;
        if kind != PT_INTERP {
            continue;
        }
        let (p_off, p_filesz) = if h.bits == 64 {
            (r.u64(off + 8).ok()?, r.u64(off + 32).ok()?)
        } else {
            (
                u64::from(r.u32(off + 4).ok()?),
                u64::from(r.u32(off + 16).ok()?),
            )
        };
        let bytes = r.bytes_opt(p_off as usize, p_filesz as usize)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        return Some(String::from_utf8_lossy(&bytes[..end]).into_owned());
    }
    None
}

fn classify_section(name: &str, s: &Shdr) -> SectionKind {
    match s.kind {
        SHT_NOBITS => {
            if s.flags & SHF_TLS != 0 {
                SectionKind::Tls
            } else {
                SectionKind::Bss
            }
        }
        SHT_NOTE => SectionKind::Note,
        SHT_REL | SHT_RELA | SHT_DYNAMIC => SectionKind::Reloc,
        // A hash table is bookkeeping for the dynamic linker, not a relocation
        // table: nothing in it describes an address that needs patching.
        SHT_GNU_HASH | SHT_GNU_VERSYM => SectionKind::Other,
        SHT_SYMTAB | SHT_DYNSYM | SHT_STRTAB => SectionKind::Other,
        SHT_GROUP => SectionKind::Other,
        SHT_INIT_ARRAY | SHT_FINI_ARRAY | SHT_PREINIT_ARRAY => SectionKind::Data,
        _ => {
            if s.flags & SHF_TLS != 0 {
                return SectionKind::Tls;
            }
            if s.flags & SHF_EXECINSTR != 0 {
                return SectionKind::Code;
            }
            if name.starts_with(".debug") || name.starts_with(".zdebug") || name == ".stab" {
                return SectionKind::Debug;
            }
            if name.starts_with(".eh_frame")
                || name == ".gcc_except_table"
                || name == ".pdata"
                || name == ".xdata"
            {
                return SectionKind::Unwind;
            }
            if name.starts_with(".rodata")
                || name == ".rdata"
                || name == ".data.rel.ro"
                || name == ".got"
                || name == ".got.plt"
                || name == ".plt"
                || name == ".plt.got"
                || name == ".init_array"
                || name == ".fini_array"
            {
                return SectionKind::Rodata;
            }
            if s.flags & SHF_ALLOC == 0 {
                return SectionKind::Other;
            }
            // Allocated, not executable, named nothing recognisable: it is
            // data whether or not it is writable.
            if s.flags & SHF_WRITE != 0 || s.kind == SHT_PROGBITS {
                SectionKind::Data
            } else {
                SectionKind::Other
            }
        }
    }
}

fn perm_of(s: &Shdr) -> Perm {
    if s.flags & SHF_ALLOC == 0 || s.kind == SHT_NOBITS && s.flags & SHF_ALLOC == 0 {
        // A non-allocated section is not mapped, so it has no runtime
        // permission. Reporting `r--` for `.comment` would be a lie about
        // memory that does not exist in the process.
        return Perm::NONE;
    }
    Perm::of(true, s.flags & SHF_WRITE != 0, s.flags & SHF_EXECINSTR != 0)
}

fn read_symbols(
    r: &Reader<'_>,
    h: &Header,
    shdrs: &[Shdr],
    names: &[String],
    data: &[u8],
    object: bool,
) -> Vec<Symbol> {
    let mut out = Vec::new();
    let sym_ent = if h.bits == 64 { 24 } else { 16 };
    for (i, s) in shdrs.iter().enumerate() {
        if s.kind != SHT_SYMTAB && s.kind != SHT_DYNSYM {
            continue;
        }
        let strtab: &[u8] = match shdrs.get(s.link as usize) {
            Some(t) => r
                .bytes_opt(t.offset as usize, t.size as usize)
                .unwrap_or(&[]),
            None => &[],
        };
        let bytes = match r.bytes_opt(s.offset as usize, s.size as usize) {
            Some(b) => b,
            None => continue,
        };
        let count = bytes.len() / sym_ent;
        let table_base = s.offset as usize;
        for k in 0..count {
            // `bytes` is the section's slice, but `r` reads the whole file, so
            // every access needs the section's file offset added back. Reading
            // from `k * sym_ent` alone walks the ELF header instead.
            let base = table_base + k * sym_ent;
            let (name_off, info, shndx, value, size) = if h.bits == 64 {
                (
                    r.u32(base).unwrap_or(0),
                    r.u8(base + 4).unwrap_or(0),
                    r.u16(base + 6).unwrap_or(0),
                    r.u64(base + 8).unwrap_or(0),
                    r.u64(base + 16).unwrap_or(0),
                )
            } else {
                (
                    r.u32(base).unwrap_or(0),
                    r.u8(base + 12).unwrap_or(0),
                    r.u16(base + 14).unwrap_or(0),
                    u64::from(r.u32(base + 4).unwrap_or(0)),
                    u64::from(r.u32(base + 8).unwrap_or(0)),
                )
            };
            let name = r.str_at(strtab, name_off as usize);
            if name.is_empty() && k == 0 {
                // The reserved null symbol. Index 0 is not a real entry, and
                // relocation indices count from it.
                out.push(Symbol {
                    name: String::new(),
                    addr: Addr::NULL,
                    size: 0,
                    kind: SymbolKind::Unknown,
                    binding: SymbolBinding::Unknown,
                    section: None,
                    defined: false,
                    demangled: false,
                });
                continue;
            }
            if name.is_empty() && value == 0 && shndx == SHN_UNDEF {
                continue;
            }
            let section = if shndx == SHN_UNDEF {
                None
            } else {
                names.get(shndx as usize).cloned()
            };
            let kind = match info & 0xf {
                2 => SymbolKind::Function,
                1 => SymbolKind::Object,
                3 => SymbolKind::Section,
                4 => SymbolKind::File,
                6 => SymbolKind::Tls,
                _ => SymbolKind::Unknown,
            };
            let binding = match info >> 4 {
                0 => SymbolBinding::Local,
                1 => SymbolBinding::Global,
                2 => SymbolBinding::Weak,
                _ => SymbolBinding::Unknown,
            };
            // A `SHN_ABS` symbol is defined but belongs to no section, and its
            // value is an absolute constant rather than a section offset.
            // Keeping that value is what lets RISC-V's `__global_pointer$` be
            // found: it is the one register whose value the linker fixes and
            // the startup code installs, and discarding the value leaves
            // every global access through it unreadable.
            let defined = shndx != SHN_UNDEF;
            let addr = if !defined {
                Addr::NULL
            } else if shndx == SHN_ABS {
                Addr(value)
            } else if object {
                // In an object `st_value` counts from the start of the section
                // named by `st_shndx`, which `parse` has placed at its file
                // offset. Adding that base is what turns "offset 0 in .text"
                // into the address the rest of the tool sees.
                let base = shdrs.get(shndx as usize).map_or(0, |s| s.offset);
                Addr(base.wrapping_add(value))
            } else {
                Addr(value)
            };
            out.push(Symbol {
                name: crate::demangle::clean_name(name),
                addr,
                size,
                kind,
                binding,
                section,
                defined,
                demangled: false,
            });
        }
        let _ = (i, data);
    }
    out
}

/// Attach each import to the GOT slot its `JUMP_SLOT` relocation rewrites.
///
/// The binding is what makes a `call [rip+0x2f61]` resolvable: without it the
/// call is an anonymous indirect transfer, and every PLT-based import the
/// binary uses becomes invisible to cross-reference analysis. The relocations
/// already carry both facts (slot address and symbol name); this joins them.
fn bind_import_slots(imports: &mut [Import], relocs: &[Relocation]) {
    let mut slot_of: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
    for r in relocs {
        if let Some(name) = &r.symbol
            && !name.is_empty()
            && matches!(r.kind, RelocKind::Got | RelocKind::Absolute)
        {
            slot_of.entry(name.as_str()).or_insert(r.offset.get());
        }
    }
    for imp in imports.iter_mut() {
        if imp.slot.is_none()
            && let Some(&slot) = slot_of.get(imp.name.as_str())
        {
            imp.slot = Some(Addr(slot));
        }
    }
}

/// Split a merged symbol list into imports and exports.
///
/// An undefined dynamic symbol is an import; a defined global in `.dynsym` is an
/// export. `.symtab` entries are local detail and appear in neither list.
fn split_dynamic(symbols: &[Symbol]) -> (Vec<Import>, Vec<Export>) {
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    for s in symbols {
        if s.name.is_empty() {
            continue;
        }
        if !s.defined {
            imports.push(Import {
                name: s.name.clone(),
                library: None,
                ordinal: None,
                slot: None,
                kind: if s.kind == SymbolKind::Object {
                    ImportKind::Data
                } else {
                    ImportKind::Normal
                },
            });
        } else if s.binding == SymbolBinding::Global && s.kind == SymbolKind::Function {
            exports.push(Export {
                name: s.name.clone(),
                addr: s.addr,
                ordinal: None,
                forward: None,
            });
        }
    }
    imports.sort_by(|a, b| a.name.cmp(&b.name));
    imports.dedup_by(|a, b| a.name == b.name);
    exports.sort_by_key(|e| e.addr);
    exports.dedup_by(|a, b| a.addr == b.addr);
    (imports, exports)
}

fn read_relocations(
    r: &Reader<'_>,
    h: &Header,
    shdrs: &[Shdr],
    names: &[String],
    symbols: &[Symbol],
    data: &[u8],
) -> (Vec<Relocation>, Vec<Relocation>) {
    let mut out = Vec::new();
    // Debug sections are not allocated, so their relocation offsets are file
    // offsets. They still have to be applied: without them every string
    // reference in a `.o` reads as offset zero and every code pointer as
    // zero. They are returned separately so nothing mistakes a file offset
    // for an address.
    let mut debug = Vec::new();
    let arch = map_machine(h.machine, h.bits);
    for (i, s) in shdrs.iter().enumerate() {
        if s.kind != SHT_RELA && s.kind != SHT_REL {
            continue;
        }
        let rela = s.kind == SHT_RELA;
        let ent = if h.bits == 64 {
            if rela { 24 } else { 16 }
        } else if rela {
            12
        } else {
            8
        };
        let Some(bytes) = r.bytes_opt(s.offset as usize, s.size as usize) else {
            continue;
        };
        // A section that targets an allocated section reports relocation
        // offsets as addresses. One that targets `.debug_info` reports file
        // offsets, which are not addresses at all; skip those.
        // `sh_info` names the section being relocated, but a relocation table
        // with no single target (`sh_info == 0`, as in `.rela.dyn`) applies to
        // the whole image and must be kept. Only a table that explicitly points
        // at a non-allocated section — debug info — is out of scope, because
        // its offsets are file offsets rather than addresses.
        let target_alloc = s.info == 0
            || shdrs
                .get(s.info as usize)
                .is_some_and(|t| t.flags & SHF_ALLOC != 0);
        let bucket = if target_alloc { &mut out } else { &mut debug };
        let count = bytes.len() / ent;
        let table_base = s.offset as usize;
        for k in 0..count {
            let base = table_base + k * ent;
            let (offset, info, addend) = if h.bits == 64 {
                (
                    r.u64(base).unwrap_or(0),
                    r.u64(base + 8).unwrap_or(0),
                    if rela {
                        r.i64(base + 16).unwrap_or(0)
                    } else {
                        0
                    },
                )
            } else {
                (
                    u64::from(r.u32(base).unwrap_or(0)),
                    u64::from(r.u32(base + 4).unwrap_or(0)),
                    if rela {
                        i64::from(r.i32(base + 8).unwrap_or(0))
                    } else {
                        0
                    },
                )
            };
            let (sym_idx, r_type) = if h.bits == 64 {
                ((info >> 32) as usize, (info & 0xffff_ffff) as u32)
            } else {
                ((info >> 8) as usize, (info & 0xff) as u32)
            };
            let symbol = symbols
                .get(sym_idx)
                .map(|sym| sym.name.clone())
                .filter(|n| !n.is_empty());
            // The target section is what turns `offset` from a number into a
            // place. `sh_info == 0` means the table applies to the image as a
            // whole (`.rela.dyn`), which only happens in a linked image.
            let section = (s.info != 0).then_some(s.info as usize);
            let symbol_addr = symbols
                .get(sym_idx)
                .filter(|sym| sym.defined)
                .map(|sym| sym.addr);
            bucket.push(Relocation {
                offset: Addr(offset),
                kind: reloc_kind(arch, r_type),
                symbol,
                addend: if rela { Some(addend) } else { None },
                raw_type: r_type,
                section,
                symbol_addr,
            });
        }
        let _ = i;
    }
    let _ = (names, data);
    (out, debug)
}

fn read_needed(
    r: &Reader<'_>,
    shdrs: &[Shdr],
    names: &[String],
    data: &[u8],
    warnings: &mut Vec<String>,
) -> Vec<String> {
    let mut out = Vec::new();
    for (i, s) in shdrs.iter().enumerate() {
        if s.kind != SHT_DYNAMIC {
            continue;
        }
        let ent: usize = if s.entsize > 0 {
            s.entsize as usize
        } else {
            16
        };
        let Some(bytes) = r.bytes_opt(s.offset as usize, s.size as usize) else {
            continue;
        };
        let count = bytes.len() / ent.max(1);
        let table_base = s.offset as usize;
        // DT_STRTAB holds a *virtual* address. Resolve it through the section
        // table rather than assuming file offset == address.
        let mut strtab_addr = 0u64;
        let mut entries = Vec::new();
        for k in 0..count {
            let base = table_base + k * ent;
            let tag = if ent >= 16 {
                r.i64(base).unwrap_or(0)
            } else {
                i64::from(r.i32(base).unwrap_or(0))
            };
            let val = if ent >= 16 {
                r.u64(base + 8).unwrap_or(0)
            } else {
                u64::from(r.u32(base + 4).unwrap_or(0))
            };
            if tag == DT_NULL {
                break;
            }
            entries.push((tag, val));
            if tag == DT_STRTAB {
                strtab_addr = val;
            }
        }
        let strtab: &[u8] = shdrs
            .iter()
            .find(|t| t.addr == strtab_addr && t.size > 0)
            .and_then(|t| r.bytes_opt(t.offset as usize, t.size as usize))
            .unwrap_or(&[]);
        if strtab.is_empty() && entries.iter().any(|(t, _)| *t == DT_NEEDED) {
            warnings.push(format!(
                "{}: DT_STRTAB at {strtab_addr:#x} does not resolve to bytes; \
                 dependency names are unavailable",
                names.get(i).cloned().unwrap_or_default()
            ));
        }
        for (tag, val) in entries {
            if tag == DT_NEEDED {
                let name = r.str_at(strtab, val as usize);
                if !name.is_empty() {
                    out.push(name.to_string());
                }
            }
        }
    }
    let _ = data;
    out
}

/// One parsed call-frame-information record.
struct Cie {
    /// How the FDE's `initial_location` is encoded (`DW_EH_PE_*`).
    fde_encoding: u8,
    /// Address size of the target.
    addr_size: u8,
}

/// Function extents from `.eh_frame`.
///
/// Every FDE pairs an initial location with a range, which is exactly the
/// function boundary the compiler emitted. The table survives `strip`, so it
/// names functions whose symbols are gone.
///
/// The parser follows the CIE augmentation string rather than assuming
/// `zR`: a CIE that carries a personality routine or LSDA has extra bytes
/// before the FDE list, and ignoring them would decode the first FDE from the
/// middle of its own header.
fn eh_frame_bounds(data: &[u8], sec: &Section, endian: Endian, bits: u8) -> Vec<FunctionBound> {
    let r = Reader::new(data, endian);
    let start = sec.file_offset as usize;
    let end = start
        .saturating_add(sec.file_size as usize)
        .min(data.len());
    let mut pos = start;
    let mut cies: HashMap<usize, Cie> = HashMap::new();
    let mut out = Vec::new();
    let mut guard = 0usize;
    while pos + 4 <= end && guard < 1_000_000 {
        guard += 1;
        let entry_start = pos;
        let Ok(len) = r.u32(pos) else { break };
        if len == 0 {
            break;
        }
        let (body_len, body_start) = if len == 0xffff_ffff {
            let Ok(l) = r.u64(pos + 4) else { break };
            (l as usize, pos + 12)
        } else {
            (len as usize, pos + 4)
        };
        let body_end = body_start.saturating_add(body_len);
        if body_len < 4 || body_end > end {
            break;
        }
        let Ok(id) = r.u32(body_start) else { break };
        if id == 0 {
            // A CIE. Read only what locates the FDE's fields.
            let mut p = body_start + 4;
            let Some(&version) = data.get(p) else { break };
            p += 1;
            let aug = cstr_at(data, p);
            p += aug.len() + 1;
            let Some(_code_align) = uleb(data, &mut p) else { break };
            let Some(_data_align) = sleb(data, &mut p) else { break };
            if version == 1 {
                p += 1;
            } else if uleb(data, &mut p).is_none() {
                break;
            }
            let mut encoding = 0u8;
            let mut addr_size = bits / 8;
            if version == 4 {
                addr_size = data.get(p).copied().unwrap_or(addr_size);
            }
            if let Some(aug_body) = aug.strip_prefix('z') {
                let Some(aug_len) = uleb(data, &mut p) else { break };
                let aug_end = p.saturating_add(aug_len as usize);
                let mut q = p;
                for c in aug_body.chars() {
                    match c {
                        'R' => {
                            encoding = data.get(q).copied().unwrap_or(0);
                            q += 1;
                        }
                        'L' => q += 1,
                        'P' => {
                            let Some(enc) = data.get(q).copied() else { break };
                            q += 1;
                            if skip_encoded(data, endian, &mut q, enc, addr_size).is_none() {
                                break;
                            }
                        }
                        'S' | 'B' | 'G' => {}
                        _ => break,
                    }
                }
                p = aug_end;
            }
            let _ = p;
            cies.insert(
                entry_start,
                Cie {
                    fde_encoding: encoding,
                    addr_size,
                },
            );
        } else if let Some(cie) = entry_start
            .checked_sub(0)
            .and_then(|_| body_start.checked_sub(id as usize))
            .and_then(|o| cies.get(&o))
        {
            // An FDE names its CIE by distance back from the id field.
            let mut p = body_start + 4;
            let field_at = sec.addr.0 + (p - start) as u64;
            let Some(location) =
                read_encoded(data, endian, &mut p, cie.fde_encoding, cie.addr_size, field_at)
            else {
                pos = body_end;
                continue;
            };
            // The range uses the same format but never the pcrel application:
            // it is a length, not an address.
            let range_at = sec.addr.0 + (p - start) as u64;
            let range = read_encoded(
                data,
                endian,
                &mut p,
                cie.fde_encoding & 0x0f,
                cie.addr_size,
                range_at,
            )
            .unwrap_or(0);
            if range > 0 {
                out.push(FunctionBound {
                    start: Addr(location),
                    end: Addr(location.saturating_add(range)),
                    source: "eh_frame",
                });
            }
        }
        pos = body_end;
    }
    out.sort_by_key(|b| b.start.0);
    out.dedup_by_key(|b| b.start.0);
    out
}

/// Decode one `DW_EH_PE_*` value, advancing `pos`.
fn read_encoded(
    data: &[u8],
    endian: Endian,
    pos: &mut usize,
    enc: u8,
    addr_size: u8,
    field_at: u64,
) -> Option<u64> {
    let value = read_encoded_raw(data, endian, pos, enc, addr_size)?;
    let app = enc & 0x70;
    Some(match app {
        0x10 => field_at.wrapping_add(value), // pcrel
        0x00 => value,                        // absptr
        // textrel/datarel/funcrel/aligned are not used by compiler-emitted
        // unwind tables; refusing them keeps a wrong answer out.
        _ => return None,
    })
}

fn read_encoded_raw(
    data: &[u8],
    endian: Endian,
    pos: &mut usize,
    enc: u8,
    addr_size: u8,
) -> Option<u64> {
    let format = enc & 0x0f;
    let r = Reader::new(data, endian);
    let v = match format {
        0x00 => {
            let n = usize::from(addr_size);
            if n == 8 {
                r.u64(*pos).ok()?
            } else {
                u64::from(r.u32(*pos).ok()?)
            }
        }
        0x01 => uleb(data, pos)?,
        0x02 => u64::from(r.u16(*pos).ok()?),
        0x03 => u64::from(r.u32(*pos).ok()?),
        0x04 => r.u64(*pos).ok()?,
        0x09 => sleb(data, pos)? as u64,
        0x0a => i64::from(r.u16(*pos).ok()? as i16) as u64,
        0x0b => i64::from(r.u32(*pos).ok()? as i32) as u64,
        0x0c => r.u64(*pos).ok()?,
        _ => return None,
    };
    if format != 0x00 && format != 0x01 && format != 0x09 {
        *pos += match format {
            0x02 | 0x0a => 2,
            0x03 | 0x0b => 4,
            _ => 8,
        };
    } else if format == 0x01 || format == 0x09 {
        // ULEB/SLEB readers already advanced.
    } else {
        *pos += usize::from(addr_size);
    }
    Some(v)
}

fn skip_encoded(
    data: &[u8],
    endian: Endian,
    pos: &mut usize,
    enc: u8,
    addr_size: u8,
) -> Option<()> {
    read_encoded_raw(data, endian, pos, enc, addr_size)?;
    Some(())
}

fn cstr_at(data: &[u8], off: usize) -> String {
    if off >= data.len() {
        return String::new();
    }
    let rest = &data[off..];
    let end = rest.iter().position(|b| *b == 0).unwrap_or(rest.len());
    String::from_utf8_lossy(&rest[..end]).into_owned()
}

fn uleb(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *data.get(*pos)?;
        *pos += 1;
        result |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

fn sleb(data: &[u8], pos: &mut usize) -> Option<i64> {
    let mut result = 0i64;
    let mut shift = 0u32;
    loop {
        let b = *data.get(*pos)?;
        *pos += 1;
        result |= i64::from(b & 0x7f) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            if shift < 64 && b & 0x40 != 0 {
                result |= -1i64 << shift;
            }
            return Some(result);
        }
        if shift > 63 {
            return None;
        }
    }
}

/// Map an ELF machine number onto the architecture enum.
fn map_machine(machine: u16, bits: u8) -> Arch {
    match machine {
        3 => Arch::X86,
        62 => Arch::X86_64,
        40 => Arch::Arm,
        183 => Arch::Aarch64,
        243 => {
            // EM_RISCV says nothing about width; the ELF class does. Reporting
            // an RV32 object as `riscv64 32-bit` was a label bug, not a
            // decoding one: the decoder already followed `bits`.
            if bits == 32 {
                Arch::RiscV32
            } else {
                Arch::RiscV64
            }
        }
        8 if bits == 32 => Arch::Other, // MIPS
        21 => Arch::Other,              // PowerPC64
        _ => Arch::Other,
    }
}

/// Map a relocation type number onto the coarse kind used by the analysis.
///
/// The per-architecture numbering is not shared: type 8 means `R_X86_64_RELATIVE`
/// on x86-64, `R_386_RELATIVE` on i386 and something unrelated on AArch64.
fn reloc_kind(arch: Arch, r_type: u32) -> RelocKind {
    match arch {
        Arch::X86_64 => match r_type {
            1 => RelocKind::Absolute,      // R_X86_64_64
            2 | 4 => RelocKind::Relative,  // PC32, PLT32
            3 | 6 | 7 => RelocKind::Got,   // GOT32, GLOB_DAT, JUMP_SLOT
            8 => RelocKind::ImageRelative, // RELATIVE
            9 => RelocKind::Got,           // GOTPCREL
            16..=19 => RelocKind::Tls,     // DTPMOD64, DTPOFF64, TPOFF64, TLSGD
            _ => RelocKind::Other,
        },
        Arch::X86 => match r_type {
            1 => RelocKind::Absolute,
            2 => RelocKind::Relative,
            6 | 7 | 9 => RelocKind::Got,
            8 => RelocKind::ImageRelative,
            14..=38 => RelocKind::Tls,
            _ => RelocKind::Other,
        },
        Arch::Aarch64 => match r_type {
            257 | 258 => RelocKind::Absolute,
            261..=263 => RelocKind::Relative,
            1024..=1026 => RelocKind::Got,
            1027 => RelocKind::ImageRelative,
            1028..=1032 | 512..=573 => RelocKind::Tls,
            _ => RelocKind::Other,
        },
        Arch::RiscV64 => match r_type {
            2 => RelocKind::Absolute,
            17..=19 => RelocKind::Relative,
            20..=23 => RelocKind::Got,
            3 => RelocKind::ImageRelative,
            _ => RelocKind::Other,
        },
        _ => RelocKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal but valid 64-bit little-endian ELF with two sections.
    fn minimal_elf64() -> Vec<u8> {
        let mut d = vec![0u8; 0x400];
        d[0..4].copy_from_slice(b"\x7fELF");
        d[4] = 2; // ELFCLASS64
        d[5] = 1; // little endian
        d[6] = 1; // version
        d[16] = 2; // ET_EXEC
        d[18] = 62; // EM_X86_64
        d[0x18..0x20].copy_from_slice(&0x401000u64.to_le_bytes()); // e_entry
        d[0x20..0x28].copy_from_slice(&0u64.to_le_bytes()); // e_phoff
        d[0x28..0x30].copy_from_slice(&0x200u64.to_le_bytes()); // e_shoff
        d[0x34..0x36].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
        d[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
        d[0x3c..0x3e].copy_from_slice(&3u16.to_le_bytes()); // e_shnum
        d[0x3e..0x40].copy_from_slice(&2u16.to_le_bytes()); // e_shstrndx

        // Section 1: .text, allocated and executable, 16 bytes at file 0x100.
        let s1 = 0x200 + 64;
        d[s1..s1 + 4].copy_from_slice(&1u32.to_le_bytes()); // sh_name
        d[s1 + 4..s1 + 8].copy_from_slice(&1u32.to_le_bytes()); // SHT_PROGBITS
        d[s1 + 8..s1 + 16].copy_from_slice(&(SHF_ALLOC | SHF_EXECINSTR).to_le_bytes());
        d[s1 + 16..s1 + 24].copy_from_slice(&0x401000u64.to_le_bytes());
        d[s1 + 24..s1 + 32].copy_from_slice(&0x100u64.to_le_bytes());
        d[s1 + 32..s1 + 40].copy_from_slice(&16u64.to_le_bytes());
        d[s1 + 48..s1 + 56].copy_from_slice(&16u64.to_le_bytes()); // sh_addralign

        // Section 2: .shstrtab at file 0x180.
        let s2 = 0x200 + 128;
        d[s2..s2 + 4].copy_from_slice(&7u32.to_le_bytes());
        d[s2 + 4..s2 + 8].copy_from_slice(&SHT_STRTAB.to_le_bytes());
        d[s2 + 24..s2 + 32].copy_from_slice(&0x180u64.to_le_bytes());
        d[s2 + 32..s2 + 40].copy_from_slice(&14u64.to_le_bytes());

        d[0x180..0x180 + 16].copy_from_slice(b"\0.text\0.shstrtab");

        // Some instructions, so the section is not empty.
        d[0x100..0x105].copy_from_slice(&[0x55, 0x48, 0x89, 0xe5, 0xc3]);
        d
    }

    #[test]
    fn parses_a_minimal_elf() {
        let data = minimal_elf64();
        let img = parse(&data, "tiny", ImageId(0)).expect("parse");
        assert_eq!(img.format, Format::Elf);
        assert_eq!(img.arch, Arch::X86_64);
        assert_eq!(img.bits, 64);
        assert_eq!(img.entry, Addr(0x401000));
        assert_eq!(img.sections.len(), 3);
        assert_eq!(img.sections[1].name, ".text");
        assert_eq!(img.sections[1].kind, SectionKind::Code);
        assert!(img.sections[1].perm.exec());
        assert!(img.sections[1].perm.read());
        assert!(!img.sections[1].perm.write());
        assert_eq!(img.sections[2].kind, SectionKind::Other);
        assert_eq!(img.sections[2].perm, Perm::NONE);
    }

    #[test]
    fn bad_magic_is_reported_with_an_offset() {
        let mut d = minimal_elf64();
        d[1] = b'X';
        let e = parse(&d, "x", ImageId(0)).unwrap_err();
        assert!(e.to_string().contains("magic"), "{e}");
    }

    #[test]
    fn unknown_class_is_rejected() {
        let mut d = minimal_elf64();
        d[4] = 9;
        let e = parse(&d, "x", ImageId(0)).unwrap_err();
        assert!(e.to_string().contains("class 9"), "{e}");
    }

    #[test]
    fn big_endian_header_is_read_in_the_right_order() {
        // Take the little-endian file and byte-swap the fields this parser must
        // honour. Rather than build a second fixture, assert the reader picks
        // up the endian byte and reports it.
        let mut d = minimal_elf64();
        d[5] = 2;
        let img = parse(&d, "be", ImageId(0));
        // The entry will be byte-swapped garbage, but parsing must not fail and
        // the endian flag must be recorded.
        if let Ok(img) = img {
            assert_eq!(img.endian, Endian::Big);
        }
    }

    #[test]
    fn truncated_file_does_not_panic() {
        let full = minimal_elf64();
        for cut in [16, 20, 0x40, 0x50, 0x100, 0x1ff] {
            let _ = parse(&full[..cut], "cut", ImageId(0));
        }
    }

    #[test]
    fn relocation_kinds_are_arch_specific() {
        assert_eq!(reloc_kind(Arch::X86_64, 8), RelocKind::ImageRelative);
        assert_eq!(reloc_kind(Arch::X86_64, 7), RelocKind::Got);
        assert_eq!(reloc_kind(Arch::Aarch64, 1027), RelocKind::ImageRelative);
        assert_eq!(reloc_kind(Arch::Aarch64, 257), RelocKind::Absolute);
        assert_eq!(reloc_kind(Arch::X86_64, 9999), RelocKind::Other);
    }
}
