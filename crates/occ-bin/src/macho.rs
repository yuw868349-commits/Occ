//! Mach-O parsing.
//!
//! Mach-O differs from ELF and PE in a way that shapes this parser: there is no
//! section header table. Everything — segments, sections, symbol table, entry
//! point, bindings — is a load command, and the command list is the only index.
//! A loader that reads the header and stops has nothing usable.
//!
//! Fat binaries are handled by choosing one slice: the first whose CPU type
//! this build can represent, preferring 64-bit. The other slices are recorded
//! as a warning rather than silently dropped, because "the analysis is of a
//! different architecture than the file name suggests" is exactly the kind of
//! thing an analyst needs to know.
//!
//! Imports come from the dyld bind opcode stream rather than from undefined
//! symbols. The symbol table says *that* `_printf` is imported; only the bind
//! stream says *where* its pointer slot is, and without the slot a call through
//! it is an anonymous indirect branch.

use occ_core::{
    Addr, Arch, CoreError, Endian, Export, Fingerprint, Format, FunctionBound, Image, ImageId,
    Import, ImportKind, Linkage, Perm, RelocKind, Relocation, Result, Section, SectionKind,
    Segment, Symbol, SymbolBinding, SymbolKind,
};

use crate::reader::Reader;

const MH_MAGIC: u32 = 0xfeed_face;
const MH_MAGIC_64: u32 = 0xfeed_facf;
const FAT_MAGIC: u32 = 0xcafe_babe;
const FAT_MAGIC_64: u32 = 0xcafe_babf;

const LC_SEGMENT: u32 = 0x1;
const LC_SYMTAB: u32 = 0x2;
const LC_UNIXTHREAD: u32 = 0x5;
const LC_LOAD_DYLIB: u32 = 0xc;
const LC_ID_DYLIB: u32 = 0xd;
const LC_SEGMENT_64: u32 = 0x19;
const LC_LOAD_WEAK_DYLIB: u32 = 0x8000_0018;
const LC_DYLD_INFO: u32 = 0x22;
const LC_DYLD_INFO_ONLY: u32 = 0x8000_0022;
const LC_FUNCTION_STARTS: u32 = 0x26;
const LC_MAIN: u32 = 0x8000_0028;
const LC_REEXPORT_DYLIB: u32 = 0x8000_001f;
const LC_DYLD_CHAINED_FIXUPS: u32 = 0x8000_0034;

/// Parse a Mach-O or fat Mach-O file.
pub fn parse(data: &[u8], path: &str, id: ImageId) -> Result<Image> {
    if data.len() < 8 {
        return Err(malformed(0, "file is shorter than a Mach-O header"));
    }
    let magic_le = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let magic_be = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    match (magic_le, magic_be) {
        (FAT_MAGIC, _) | (FAT_MAGIC_64, _) => parse_fat(data, path, id),
        (MH_MAGIC_64, _) | (_, MH_MAGIC_64) => parse_slice(data, 0, path, id),
        (MH_MAGIC, _) | (_, MH_MAGIC) => parse_slice(data, 0, path, id),
        _ => Err(malformed(0, "not a Mach-O image")),
    }
}

fn malformed(offset: usize, detail: impl Into<String>) -> CoreError {
    CoreError::Malformed {
        offset: offset as u64,
        detail: detail.into(),
    }
}

fn parse_fat(data: &[u8], path: &str, id: ImageId) -> Result<Image> {
    // Fat headers are always big-endian.
    let r = Reader::new(data, Endian::Big);
    let is64 = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) == FAT_MAGIC_64;
    let n = r.u32(4)? as usize;
    if n == 0 || n > 32 {
        return Err(malformed(4, format!("fat header claims {n} slices")));
    }
    let entry = if is64 { 32 } else { 20 };
    let mut best: Option<(usize, usize, u32)> = None;
    for i in 0..n {
        let off = 8 + i * entry;
        let cputype = r.u32(off)?;
        let (slice_off, slice_size) = if is64 {
            (r.u64(off + 8)? as usize, r.u64(off + 16)? as usize)
        } else {
            (r.u32(off + 8)? as usize, r.u32(off + 12)? as usize)
        };
        if slice_off >= data.len() {
            continue;
        }
        let rank = match map_cpu(cputype) {
            Arch::X86_64 | Arch::Aarch64 => 0,
            Arch::X86 | Arch::Arm => 1,
            _ => 2,
        };
        if best.map(|(r, _, _)| rank < r).unwrap_or(true) {
            best = Some((rank, slice_off, slice_size as u32));
        }
    }
    let (_, off, size) = best.ok_or_else(|| malformed(8, "no usable slice in fat binary"))?;
    let end = off.saturating_add(size as usize).min(data.len());
    let mut image = parse_slice(&data[..end], off, path, id)?;
    image.warnings.push(format!(
        "fat binary: analysed the slice at offset {off:#x} ({n} slices present)"
    ));
    Ok(image)
}

fn map_cpu(cputype: u32) -> Arch {
    match cputype {
        7 => Arch::X86,
        0x0100_0007 => Arch::X86_64,
        12 => Arch::Arm,
        0x0100_000c | 0x0200_000c => Arch::Aarch64,
        _ => Arch::Other,
    }
}

/// Where the symbol table lives.
struct Symtab {
    /// File offset of the first `nlist`.
    symoff: u32,
    /// Number of entries.
    nsyms: u32,
    /// File offset of the string table.
    stroff: u32,
}

struct Seg {
    name: String,
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    prot: u32,
}

fn parse_slice(data: &[u8], base: usize, path: &str, id: ImageId) -> Result<Image> {
    let le = data
        .get(..4)
        .map(|b| {
            let m = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            m == MH_MAGIC || m == MH_MAGIC_64
        })
        .unwrap_or(false);
    let endian = if le { Endian::Little } else { Endian::Big };
    let r = Reader::new(data, endian);
    let magic = r.u32(0)?;
    let is64 = magic == MH_MAGIC_64;
    if !is64 && magic != MH_MAGIC {
        return Err(malformed(0, format!("unexpected Mach-O magic {magic:#x}")));
    }
    let cputype = r.u32(4)?;
    let filetype = r.u32(12)?;
    // A relocatable object has no load address: every section starts at zero and
    // symbol values are offsets into a section. Reporting those numbers verbatim
    // makes every section overlap and puts every function at address 0, so the
    // whole file looks empty. Sections are given their file offset instead, and
    // symbol values are rebased onto it — addresses stay unique and readable,
    // and `View::synthetic` says they are ours rather than the loader's.
    let object = filetype == 1;
    let ncmds = r.u32(16)? as usize;
    if ncmds > 4096 {
        return Err(malformed(
            16,
            format!("header claims {ncmds} load commands"),
        ));
    }
    let header_size = if is64 { 32 } else { 28 };

    let mut segs: Vec<Seg> = Vec::new();
    let mut sections: Vec<Section> = Vec::new();
    let mut symtab: Option<Symtab> = None;
    let mut entry_off: Option<u64> = None;
    let mut dylibs: Vec<String> = Vec::new();
    let mut bind_info: Option<(u32, u32, u32, u32)> = None;
    let mut export_info: Option<(u32, u32)> = None;
    let mut function_starts: Option<(u32, u32)> = None;
    let mut chained: Option<(u32, u32)> = None;
    let mut thread_entry: Option<u64> = None;
    let mut warnings: Vec<String> = Vec::new();
    let mut pie = false;

    let mut off = header_size;
    for _ in 0..ncmds {
        let cmd = r.u32(off)?;
        let cmdsize = r.u32(off + 4)? as usize;
        if cmdsize < 8 || off + cmdsize > data.len() {
            return Err(malformed(
                off,
                format!("load command size {cmdsize} is invalid"),
            ));
        }
        match cmd {
            LC_SEGMENT | LC_SEGMENT_64 => {
                let seg64 = cmd == LC_SEGMENT_64;
                let name = fixed(data, off + 8, 16);
                let (vmaddr, vmsize, fileoff, filesize, nsects, prot_off) = if seg64 {
                    (
                        r.u64(off + 24)?,
                        r.u64(off + 32)?,
                        r.u64(off + 40)?,
                        r.u64(off + 48)?,
                        r.u32(off + 64)? as usize,
                        off + 60,
                    )
                } else {
                    (
                        u64::from(r.u32(off + 24)?),
                        u64::from(r.u32(off + 28)?),
                        u64::from(r.u32(off + 32)?),
                        u64::from(r.u32(off + 36)?),
                        r.u32(off + 48)? as usize,
                        off + 44,
                    )
                };
                let prot = r.u32(prot_off)?;
                segs.push(Seg {
                    name: name.clone(),
                    vmaddr,
                    vmsize,
                    fileoff,
                    filesize,
                    prot,
                });
                // Section structures follow the segment command.
                let sec_size = if seg64 { 80 } else { 68 };
                let mut soff = off + if seg64 { 72 } else { 56 };
                for _ in 0..nsects {
                    if soff + sec_size > off + cmdsize {
                        break;
                    }
                    let sectname = fixed(data, soff, 16);
                    let segname = fixed(data, soff + 16, 16);
                    let addr = if seg64 {
                        r.u64(soff + 32)?
                    } else {
                        u64::from(r.u32(soff + 32)?)
                    };
                    let size = if seg64 {
                        r.u64(soff + 40)?
                    } else {
                        u64::from(r.u32(soff + 36)?)
                    };
                    let file_offset = if seg64 {
                        r.u32(soff + 48)?
                    } else {
                        r.u32(soff + 40)?
                    };
                    let flags = if seg64 {
                        r.u32(soff + 64)?
                    } else {
                        r.u32(soff + 56)?
                    };
                    if size > 0 {
                        let file_offset = file_offset as u64 + base as u64;
                        sections.push(Section {
                            index: sections.len() + 1,
                            name: format!("{segname},{sectname}"),
                            addr: if object {
                                Addr(file_offset)
                            } else {
                                Addr(addr)
                            },
                            file_offset,
                            size,
                            file_size: if flags & 0xff == 0x1 { 0 } else { size },
                            kind: classify(&segname, &sectname, flags),
                            perm: section_perm(&segname, &sectname, prot),
                        });
                    }
                    soff += sec_size;
                }
            }
            LC_SYMTAB => {
                symtab = Some(Symtab {
                    symoff: r.u32(off + 8)?,
                    nsyms: r.u32(off + 12)?,
                    stroff: r.u32(off + 16)?,
                });
            }
            LC_DYLD_INFO | LC_DYLD_INFO_ONLY => {
                bind_info = Some((
                    r.u32(off + 16)?,
                    r.u32(off + 20)?,
                    r.u32(off + 24)?,
                    r.u32(off + 28)?,
                ));
                export_info = Some((r.u32(off + 40)?, r.u32(off + 44)?));
            }
            LC_MAIN => {
                entry_off = Some(r.u64(off + 8)?);
            }
            LC_LOAD_DYLIB | LC_LOAD_WEAK_DYLIB | LC_REEXPORT_DYLIB | LC_ID_DYLIB => {
                let name_off = r.u32(off + 8)? as usize;
                if name_off < cmdsize {
                    dylibs.push(fixed(data, off + name_off, cmdsize - name_off));
                }
            }
            LC_FUNCTION_STARTS => {
                function_starts = Some((r.u32(off + 8)?, r.u32(off + 12)?));
            }
            LC_DYLD_CHAINED_FIXUPS => {
                chained = Some((r.u32(off + 8)?, r.u32(off + 12)?));
            }
            LC_UNIXTHREAD => {
                // The thread state is the only place an old-style executable
                // states where execution begins. Refusing to read it left the
                // entry at the image base, which is a header, not code.
                thread_entry = read_thread_entry(&r, data, off, cmdsize, cputype, is64);
                if thread_entry.is_none() {
                    warnings.push(
                        "LC_UNIXTHREAD has no program counter for this CPU; entry stays at the image base"
                            .into(),
                    );
                }
            }
            _ => {}
        }
        let flags = r.u32(24).unwrap_or(0);
        pie = flags & 0x0020_0000 != 0;
        off += cmdsize;
    }

    if segs.is_empty() {
        return Err(malformed(header_size, "no loadable segments"));
    }
    let image_base = segs.iter().map(|s| s.vmaddr).min().unwrap_or(0);

    // Entry: LC_MAIN records a file offset into __TEXT.
    let entry = match entry_off {
        Some(o) => segs
            .iter()
            .find(|s| o >= s.fileoff && o < s.fileoff + s.filesize)
            .map(|s| Addr(s.vmaddr + (o - s.fileoff)))
            .unwrap_or(Addr(image_base + o)),
        None => thread_entry.map_or(Addr(image_base), Addr),
    };

    // An object has no loadable segments: its single `__TEXT` is a section
    // container with `vmaddr` 0 and protection 7, describing nothing the kernel
    // would map. Emitting it would claim a rwx mapping at address 0.
    let segments: Vec<Segment> = if object {
        Vec::new()
    } else {
        segs.iter()
            .map(|s| Segment {
                addr: Addr(s.vmaddr),
                size: s.vmsize,
                file_size: s.filesize,
                perm: seg_perm(s.prot),
            })
            .collect()
    };

    let (symbols, mut imports, mut exports) = match &symtab {
        Some(table) => read_symbols(&r, data, base, table, &sections, object),
        None => (Vec::new(), Vec::new(), Vec::new()),
    };

    // Bind opcodes give each import its slot.
    if let Some((bind_off, bind_size, _, _)) = bind_info
        && bind_size > 0
    {
        let bound = read_binds(&r, data, base, bind_off, bind_size, &segs, &dylibs);
        for (name, slot, ordinal) in bound {
            if let Some(existing) = imports.iter_mut().find(|i| i.name == name) {
                existing.slot = Some(slot);
                existing.library = ordinal.and_then(|o| dylibs.get(o.saturating_sub(1)).cloned());
            } else {
                imports.push(Import {
                    name,
                    library: ordinal.and_then(|o| dylibs.get(o.saturating_sub(1)).cloned()),
                    ordinal: None,
                    slot: Some(slot),
                    kind: ImportKind::Normal,
                });
            }
        }
    }

    // Chained fixups are the only import record in a current arm64 binary.
    if let Some((fix_off, fix_size)) = chained
        && fix_size > 0
        && !object
    {
        let chained_imports = read_chained_fixups(
            data,
            base,
            fix_off as usize,
            fix_size as usize,
            &segs,
            &dylibs,
        );
        let before = imports.len();
        for (name, slot, library) in chained_imports {
            if let Some(existing) = imports.iter_mut().find(|i| i.name == name) {
                if existing.slot.is_none() {
                    existing.slot = Some(slot);
                }
                if existing.library.is_none() {
                    existing.library = library;
                }
            } else {
                imports.push(Import {
                    name,
                    library,
                    ordinal: None,
                    slot: Some(slot),
                    kind: ImportKind::Normal,
                });
            }
        }
        if imports.len() > before {
            warnings.push(format!(
                "chained fixups named {} import slot(s)",
                imports.len() - before
            ));
        }
    }

    // Export trie, when present, is more accurate than N_EXT symbols.
    if let Some((export_off, export_size)) = export_info
        && export_size > 0
    {
        let trie = read_exports(&r, data, base, export_off, export_size, image_base);
        if !trie.is_empty() {
            exports = trie;
        }
    }

    imports.sort_by(|a, b| a.name.cmp(&b.name));
    imports.dedup_by(|a, b| a.name == b.name);
    exports.sort_by_key(|e| e.addr);
    // The compiler's own function list, when the linker kept it.
    let mut function_bounds = Vec::new();
    if let Some((starts_off, starts_size)) = function_starts
        && starts_size > 0
        && !object
    {
        let text_base = segs
            .iter()
            .find(|s| s.name == "__TEXT")
            .map(|s| s.vmaddr)
            .unwrap_or(image_base);
        let text_end = segs
            .iter()
            .filter(|s| s.vmaddr == text_base)
            .map(|s| s.vmaddr + s.vmsize)
            .max()
            .unwrap_or(text_base);
        function_bounds = read_function_starts(
            data,
            base + starts_off as usize,
            starts_size as usize,
            text_base,
            text_end,
        );
    }

    let mut relocations = Vec::new();
    for s in &sections {
        // Mach-O section relocations live in the section structure; recording
        // one entry per section keeps the count honest without pretending to
        // know each target.
        if s.kind == SectionKind::Code {
            relocations.push(Relocation::bare(s.addr, RelocKind::Other, None));
        }
    }
    if !dylibs.is_empty() {
        warnings.push(format!(
            "links against {} dylib(s), first: {}",
            dylibs.len(),
            dylibs.first().cloned().unwrap_or_default()
        ));
    }

    let mut fp = Fingerprint::unknown(id);
    fp.linkage = match filetype {
        1 => Linkage::Relocatable, // MH_OBJECT
        2 => {
            if pie {
                Linkage::Pie
            } else {
                Linkage::Executable
            }
        }
        6 | 8 => Linkage::Shared, // MH_DYLIB / MH_BUNDLE
        _ => Linkage::Unknown,
    };
    fp.stripped = symbols.is_empty();

    Ok(Image {
        id,
        path: path.to_string(),
        format: Format::MachO,
        arch: map_cpu(cputype),
        bits: if is64 { 64 } else { 32 },
        endian,
        entry,
        image_base: Addr(image_base),
        file_size: data.len() as u64,
        sha256: crate::sha256::hex_digest(data),
        segments,
        sections,
        resources: Vec::new(),
        symbols,
        imports,
        exports,
        relocations,
        debug_relocations: Vec::new(),
        fingerprint: fp,
        interpreter: None,
        warnings,
        function_bounds,
        arm_modes: Default::default(),
        wasm_types: Default::default(),
        wasm_funcs: Default::default(),
    })
}

/// Read a fixed-width, NUL-padded name field.
fn fixed(data: &[u8], off: usize, len: usize) -> String {
    let end = off.saturating_add(len).min(data.len());
    if off >= end {
        return String::new();
    }
    let s = &data[off..end];
    let n = s.iter().position(|b| *b == 0).unwrap_or(s.len());
    String::from_utf8_lossy(&s[..n]).into_owned()
}

fn seg_perm(prot: u32) -> Perm {
    Perm::of(prot & 1 != 0, prot & 2 != 0, prot & 4 != 0)
}

/// Permissions for one section.
///
/// A relocatable object has no segments to speak of — Mach-O gives it a single
/// `__TEXT` with `initprot` 7 — so taking the segment's protection at face value
/// reports `rwx` for the debug info and for the string table. The section name is
/// the reliable signal: only `__text` and the stub sections execute, `__const`
/// and friends only read, and the debug sections are not mapped at all.
fn section_perm(seg: &str, sect: &str, prot: u32) -> Perm {
    match (seg, sect) {
        ("__TEXT", "__text") => Perm::R.union(Perm::X),
        (_, s) if s.contains("stub") || s.contains("__stub_helper") => Perm::R.union(Perm::X),
        ("__TEXT", _) => Perm::R,
        ("__DATA", _) | ("__DATA_CONST", _) | ("__DATA_DIRTY", _) | ("__OBJC", _) => {
            Perm::R.union(Perm::W)
        }
        ("__DWARF", _) | ("__LINKEDIT", _) => Perm::R,
        _ => seg_perm(prot),
    }
}

fn classify(seg: &str, sect: &str, flags: u32) -> SectionKind {
    match (seg, sect) {
        (_, "__text") => SectionKind::Code,
        (_, s) if s.contains("stub") => SectionKind::Code,
        ("__TEXT", "__const") | ("__TEXT", "__cstring") => SectionKind::Rodata,
        (_, "__bss") => SectionKind::Bss,
        (_, "__eh_frame") | (_, "__unwind_info") => SectionKind::Unwind,
        (_, "__debug_info") | (_, "__debug_line") => SectionKind::Debug,
        ("__DWARF", _) => SectionKind::Debug,
        ("__DATA", _) | ("__DATA_CONST", _) | ("__DATA_DIRTY", _) => SectionKind::Data,
        _ => {
            if flags & 0x8000_0000 != 0 {
                SectionKind::Code
            } else {
                SectionKind::Other
            }
        }
    }
}

fn read_symbols(
    r: &Reader<'_>,
    data: &[u8],
    base: usize,
    table: &Symtab,
    sections: &[Section],
    object: bool,
) -> (Vec<Symbol>, Vec<Import>, Vec<Export>) {
    let (symoff, nsyms, stroff) = (table.symoff, table.nsyms, table.stroff);
    let mut symbols = Vec::new();
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let str_base = base + stroff as usize;
    let count = (nsyms as usize).min(1_000_000);
    for i in 0..count {
        let off = base + symoff as usize + i * 16;
        let Ok(strx) = r.u32(off) else { break };
        let Ok(ntype) = r.u8(off + 4) else { break };
        let Ok(nsect) = r.u8(off + 5) else { break };
        let Ok(nvalue) = r.u64(off + 8) else { break };
        if ntype & 0xe0 != 0 {
            continue; // debug stab
        }
        let kind_bits = ntype & 0x0e;
        let ext = ntype & 0x01 != 0;
        let name = r
            .str_at(data.get(str_base..).unwrap_or(&[]), strx as usize)
            .to_string();
        if name.is_empty() {
            continue;
        }
        let defined = kind_bits != 0x0;
        // In an object `n_value` is an offset into the section `n_sect` names,
        // so it only becomes an address once the section's synthetic base is
        // added. In a linked image `n_value` is already the address.
        let addr = match (defined, object) {
            (false, _) => Addr::NULL,
            (true, false) => Addr(nvalue),
            (true, true) => {
                let s = sections.get(usize::from(nsect).wrapping_sub(1));
                Addr(s.map_or(0, |s| s.addr.0).wrapping_add(nvalue))
            }
        };
        // `N_SECT` does not say whether a symbol is code; the section it lives
        // in does. Without this, every function in an object file is reported
        // as data and `code.functions` finds nothing to call.
        let kind = if kind_bits == 0xe {
            match sections.get(usize::from(nsect).wrapping_sub(1)) {
                Some(s) if s.kind == SectionKind::Code => SymbolKind::Function,
                Some(_) => SymbolKind::Object,
                None => SymbolKind::Unknown,
            }
        } else {
            SymbolKind::Unknown
        };
        symbols.push(Symbol {
            name: name.clone(),
            addr,
            size: 0,
            kind,
            binding: if ext {
                SymbolBinding::Global
            } else {
                SymbolBinding::Local
            },
            section: None,
            defined,
            demangled: false,
        });
        if !defined && ext {
            imports.push(Import {
                name,
                library: None,
                ordinal: None,
                slot: None,
                kind: ImportKind::Normal,
            });
        } else if defined && ext {
            exports.push(Export {
                name,
                addr,
                ordinal: None,
                forward: None,
            });
        }
    }
    let _ = data;
    (symbols, imports, exports)
}

/// Read the program counter out of an `LC_UNIXTHREAD` command.
///
/// The command holds a thread state whose layout is the kernel's, not
/// Mach-O's: a flavour, a count of 32-bit words, then the registers. The
/// program counter sits at a fixed offset per flavour, and reading the wrong
/// one would point the analysis into the middle of a function.
fn read_thread_entry(
    r: &Reader<'_>,
    _data: &[u8],
    off: usize,
    cmdsize: usize,
    cputype: u32,
    _is64: bool,
) -> Option<u64> {
    let flavor = r.u32(off + 8).ok()?;
    let count = r.u32(off + 12).ok()? as usize;
    let state = off + 16;
    // The count is in 32-bit words and must fit in the command.
    if 16 + count.saturating_mul(4) > cmdsize {
        return None;
    }
    match (cputype, flavor) {
        // x86_THREAD_STATE64: 21 registers, rip is the 17th.
        (0x0100_0007, 4) => r.u64(state + 16 * 8).ok(),
        // ARM_THREAD_STATE64: x0..x28, fp, lr, sp, pc. pc follows sp.
        (0x0100_000c | 0x0200_000c, 6) => r.u64(state + 32 * 8).ok(),
        // ARM_THREAD_STATE: r0..r12, sp, lr, pc.
        (12, 1) => r.u32(state + 15 * 4).ok().map(u64::from),
        // x86_THREAD_STATE32: eax..edi, eip is the 11th word.
        (7, 1) => r.u32(state + 10 * 4).ok().map(u64::from),
        _ => None,
    }
}

/// Read `LC_FUNCTION_STARTS`.
///
/// The payload is a run of ULEB128 deltas from the previous function start,
/// beginning at the `__TEXT` base, terminated by zero. Every entry is a
/// function the compiler emitted, including the static ones the symbol table
/// may have dropped.
fn read_function_starts(
    data: &[u8],
    chunk_off: usize,
    size: usize,
    text_base: u64,
    text_end: u64,
) -> Vec<FunctionBound> {
    let Some(chunk) = data.get(chunk_off..chunk_off.saturating_add(size)) else {
        return Vec::new();
    };
    let mut pos = 0usize;
    let mut addr = text_base;
    let mut starts: Vec<u64> = Vec::new();
    while pos < chunk.len() && starts.len() < 1_000_000 {
        let Some(delta) = uleb(chunk, &mut pos) else {
            break;
        };
        if delta == 0 {
            break;
        }
        addr = addr.wrapping_add(delta);
        starts.push(addr);
    }
    let mut out = Vec::with_capacity(starts.len());
    for (i, start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(text_end).max(*start);
        out.push(FunctionBound {
            start: Addr(*start),
            end: Addr(end),
            source: "function_starts",
        });
    }
    out
}

/// Read the bound imports out of `LC_DYLD_CHAINED_FIXUPS`.
///
/// Modern arm64 macOS binaries carry no bind opcode stream: the pointers are
/// encoded in chains, and the only record of which slot holds which import is
/// the chain itself. Without this the import table of a current binary is
/// empty, and every call through the GOT is an anonymous indirect branch.
///
/// Only the 64-bit pointer formats are decoded. An arm64e image uses
/// authenticated pointers with a different layout, and guessing at that would
/// attribute calls to the wrong symbol.
fn read_chained_fixups(
    data: &[u8],
    base: usize,
    off: usize,
    size: usize,
    segs: &[Seg],
    dylibs: &[String],
) -> Vec<(String, Addr, Option<String>)> {
    let mut out = Vec::new();
    let Some(chunk) = data.get(base + off..base + off + size) else {
        return out;
    };
    let rd = |at: usize| -> Option<u32> {
        chunk
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let Some(starts_off) = rd(4) else { return out };
    let Some(imports_off) = rd(8) else { return out };
    let Some(symbols_off) = rd(12) else {
        return out;
    };
    let Some(imports_count) = rd(16) else {
        return out;
    };
    let Some(imports_format) = rd(20) else {
        return out;
    };

    // Import records, then their names in the symbol pool.
    let import_size = match imports_format {
        1 | 3 => 4usize,
        2 => 8,
        _ => return out,
    };
    let mut imports: Vec<(String, Option<String>)> = Vec::new();
    for i in 0..imports_count as usize {
        let at = imports_off as usize + i * import_size;
        let Some(name_off) = (match imports_format {
            1 => chunk
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f_ffff),
            2 => chunk
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f_ffff),
            3 => chunk
                .get(at..at + 8)
                .map(|b| u32::from_le_bytes([b[4], b[5], b[6], b[7]])),
            _ => None,
        }) else {
            break;
        };
        let ordinal = chunk
            .get(at..at + import_size)
            .map(|b| b[0] as usize)
            .unwrap_or(0);
        let name_at = symbols_off as usize + name_off as usize;
        let name = cstr(chunk, name_at);
        if name.is_empty() {
            continue;
        }
        imports.push((name, dylibs.get(ordinal.saturating_sub(1)).cloned()));
    }

    // Walks of the chain, segment by segment.
    let Some(seg_count) = rd(starts_off as usize) else {
        return out;
    };
    for s in 0..seg_count as usize {
        let Some(info_off) = rd(starts_off as usize + 4 + s * 4) else {
            continue;
        };
        if info_off == 0 {
            continue;
        }
        let at = starts_off as usize + info_off as usize;
        // `dyld_chained_starts_in_segment`: size (4), page_size (2),
        // pointer_format (2), segment_offset (8), max_valid_pointer (4),
        // page_count (2), then one page_start per page.
        let Some(page_size) = chunk
            .get(at + 4..at + 6)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
        else {
            continue;
        };
        let Some(format) = chunk
            .get(at + 6..at + 8)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
        else {
            continue;
        };
        let Some(page_count) = chunk
            .get(at + 20..at + 22)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
        else {
            continue;
        };
        // 1 = PTR_64, 2 = PTR_64_OFFSET; both use the same 64-bit layout.
        if !matches!(format, 1 | 2) {
            continue;
        }
        let Some(seg) = segs.get(s) else { continue };
        for page in 0..page_count as usize {
            let Some(start) = chunk
                .get(at + 22 + page * 2..at + 24 + page * 2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
            else {
                break;
            };
            if start == 0xffff {
                continue;
            }
            let mut cursor = seg.vmaddr + (page * page_size as usize) as u64 + u64::from(start);
            let mut guard = 0usize;
            while guard < 100_000 {
                guard += 1;
                if cursor < seg.vmaddr || cursor >= seg.vmaddr + seg.filesize {
                    break;
                }
                let file_at = (seg.fileoff + (cursor - seg.vmaddr)) as usize;
                // The chain pointer lives in the segment's bytes, not in the
                // fixups blob: `chunk` holds the table that describes the
                // chains, and the pointers themselves are in the image.
                let Some(raw) = data
                    .get(base + file_at..base + file_at + 8)
                    .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                else {
                    break;
                };
                let bind = raw >> 63 != 0;
                let next = (raw >> 51) & 0xfff;
                if bind {
                    let ordinal = (raw & 0xff_ffff) as usize;
                    if let Some((name, library)) = imports.get(ordinal) {
                        out.push((name.clone(), Addr(cursor), library.clone()));
                    }
                }
                if next == 0 {
                    break;
                }
                cursor = cursor.wrapping_add(next * 4);
            }
        }
    }
    out
}

fn cstr(data: &[u8], off: usize) -> String {
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

/// Walk the bind opcode stream and return `(symbol, slot, dylib ordinal)`.
fn read_binds(
    _r: &Reader<'_>,
    data: &[u8],
    base: usize,
    off: u32,
    size: u32,
    segs: &[Seg],
    _dylibs: &[String],
) -> Vec<(String, Addr, Option<usize>)> {
    let mut out = Vec::new();
    let start = base + off as usize;
    let end = (start + size as usize).min(data.len());
    if start >= end {
        return out;
    }
    let mut pos = start;
    let mut seg = 0usize;
    let mut seg_off = 0u64;
    let mut ordinal: Option<usize> = None;
    let mut symbol = String::new();
    while pos < end {
        let byte = data[pos];
        pos += 1;
        let op = byte & 0xf0;
        let imm = byte & 0x0f;
        match op {
            0x00 => break, // DONE
            0x10 => ordinal = Some(imm as usize),
            0x20 => {
                ordinal = uleb(data, &mut pos).map(|v| v as usize);
            }
            0x30 => {
                ordinal = Some(imm as usize);
                let _ = sleb(data, &mut pos);
            }
            0x40 => {
                let end_str = data[pos..end]
                    .iter()
                    .position(|b| *b == 0)
                    .map(|p| pos + p)
                    .unwrap_or(end);
                symbol = String::from_utf8_lossy(&data[pos..end_str]).into_owned();
                pos = end_str + 1;
            }
            0x50 => {}
            0x60 => {
                let _ = sleb(data, &mut pos);
            }
            0x70 => {
                seg = imm as usize;
                seg_off = uleb(data, &mut pos).unwrap_or(0);
            }
            0x80 => {
                seg_off = seg_off.wrapping_add(uleb(data, &mut pos).unwrap_or(0));
            }
            0x90 => {
                if let Some(s) = segs.get(seg)
                    && !symbol.is_empty()
                {
                    out.push((symbol.clone(), Addr(s.vmaddr + seg_off), ordinal));
                }
            }
            0xa0 => {
                if let Some(s) = segs.get(seg)
                    && !symbol.is_empty()
                {
                    out.push((symbol.clone(), Addr(s.vmaddr + seg_off), ordinal));
                }
                seg_off = seg_off.wrapping_add(uleb(data, &mut pos).unwrap_or(0));
            }
            0xb0 => {
                if let Some(s) = segs.get(seg)
                    && !symbol.is_empty()
                {
                    out.push((symbol.clone(), Addr(s.vmaddr + seg_off), ordinal));
                }
                seg_off = seg_off.wrapping_add(u64::from(imm) * 8);
            }
            0xc0 => {
                let count = uleb(data, &mut pos).unwrap_or(0);
                let skip = uleb(data, &mut pos).unwrap_or(0);
                if let Some(s) = segs.get(seg) {
                    for _ in 0..count.min(4096) {
                        if !symbol.is_empty() {
                            out.push((symbol.clone(), Addr(s.vmaddr + seg_off), ordinal));
                        }
                        seg_off = seg_off.wrapping_add(skip);
                    }
                }
            }
            _ => break,
        }
    }
    out
}

/// Walk the export trie.
fn read_exports(
    _r: &Reader<'_>,
    data: &[u8],
    base: usize,
    off: u32,
    size: u32,
    image_base: u64,
) -> Vec<Export> {
    let mut out = Vec::new();
    let start = base + off as usize;
    let end = (start + size as usize).min(data.len());
    if start >= end {
        return out;
    }
    let mut stack: Vec<(usize, String, usize)> = vec![(start, String::new(), 0)];
    let mut guard = 0usize;
    while let Some((node, prefix, depth)) = stack.pop() {
        guard += 1;
        if guard > 100_000 || depth > 64 || node >= end {
            break;
        }
        let mut pos = node;
        let Some(term_size) = uleb(data, &mut pos) else {
            continue;
        };
        if term_size > 0 {
            let term_end = pos + term_size as usize;
            if let Some(flags) = uleb(data, &mut pos) {
                let reexport = flags & 0x08 != 0;
                if reexport {
                    let _ = uleb(data, &mut pos);
                    // Skip the imported name.
                    while pos < term_end && data.get(pos).copied().unwrap_or(0) != 0 {
                        pos += 1;
                    }
                    out.push(Export {
                        name: prefix.clone(),
                        addr: Addr::NULL,
                        ordinal: None,
                        forward: Some("reexport".into()),
                    });
                } else if let Some(addr) = uleb(data, &mut pos) {
                    out.push(Export {
                        name: prefix.clone(),
                        addr: Addr(image_base.wrapping_add(addr)),
                        ordinal: None,
                        forward: None,
                    });
                }
            }
            pos = term_end;
        }
        if pos >= end {
            continue;
        }
        let children = data[pos] as usize;
        pos += 1;
        for _ in 0..children {
            let name_end = data[pos..end]
                .iter()
                .position(|b| *b == 0)
                .map(|p| pos + p)
                .unwrap_or(end);
            let edge = String::from_utf8_lossy(&data[pos..name_end]).into_owned();
            pos = name_end + 1;
            let Some(child_off) = uleb(data, &mut pos) else {
                break;
            };
            stack.push((
                start + child_off as usize,
                format!("{prefix}{edge}"),
                depth + 1,
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_macho64() -> Vec<u8> {
        // mach_header_64 + one LC_SEGMENT_64 with one section + LC_MAIN.
        let mut d = Vec::new();
        d.extend_from_slice(&MH_MAGIC_64.to_le_bytes());
        d.extend_from_slice(&0x0100_0007u32.to_le_bytes()); // x86_64
        d.extend_from_slice(&3u32.to_le_bytes()); // cpusubtype
        d.extend_from_slice(&2u32.to_le_bytes()); // MH_EXECUTE
        d.extend_from_slice(&2u32.to_le_bytes()); // ncmds
        d.extend_from_slice(&((72u32 + 24 + 24).to_le_bytes()));
        d.extend_from_slice(&0x0020_0000u32.to_le_bytes()); // PIE
        d.extend_from_slice(&0u32.to_le_bytes());
        // LC_SEGMENT_64 __TEXT
        d.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
        d.extend_from_slice(&152u32.to_le_bytes());
        d.extend_from_slice(b"__TEXT\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(&0x1_0000_0000u64.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&0u64.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&5u32.to_le_bytes()); // maxprot r-x
        d.extend_from_slice(&5u32.to_le_bytes()); // initprot r-x
        d.extend_from_slice(&1u32.to_le_bytes()); // nsects
        d.extend_from_slice(&0u32.to_le_bytes());
        // section_64 __text
        d.extend_from_slice(b"__text\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(b"__TEXT\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(&0x1_0000_0400u64.to_le_bytes());
        d.extend_from_slice(&0x100u64.to_le_bytes());
        d.extend_from_slice(&0x400u32.to_le_bytes());
        d.extend_from_slice(&4u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0x8000_0400u32.to_le_bytes()); // pure instructions
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        // LC_MAIN
        d.extend_from_slice(&LC_MAIN.to_le_bytes());
        d.extend_from_slice(&24u32.to_le_bytes());
        d.extend_from_slice(&0x400u64.to_le_bytes());
        d.extend_from_slice(&0u64.to_le_bytes());
        d.resize(0x500, 0);
        d
    }

    #[test]
    fn parses_a_minimal_executable() {
        let img = parse(&minimal_macho64(), "t", ImageId(0)).expect("parse");
        assert_eq!(img.format, Format::MachO);
        assert_eq!(img.arch, Arch::X86_64);
        assert_eq!(img.bits, 64);
        assert_eq!(img.entry, Addr(0x1_0000_0400));
        assert_eq!(img.segments.len(), 1);
        assert_eq!(img.sections.len(), 1);
        assert_eq!(img.sections[0].kind, SectionKind::Code);
        assert!(img.sections[0].perm.exec());
        assert_eq!(img.fingerprint.linkage, Linkage::Pie);
    }

    /// A 64-bit executable whose entry comes from `LC_UNIXTHREAD` and whose
    /// function list comes from `LC_FUNCTION_STARTS`.
    fn minimal_with_thread_and_starts() -> Vec<u8> {
        let thread_cmdsize = 16 + 42 * 4;
        let total = 72 + 152 + thread_cmdsize + 16;
        let mut d = Vec::new();
        d.extend_from_slice(&MH_MAGIC_64.to_le_bytes());
        d.extend_from_slice(&0x0100_0007u32.to_le_bytes()); // x86_64
        d.extend_from_slice(&3u32.to_le_bytes());
        d.extend_from_slice(&2u32.to_le_bytes()); // MH_EXECUTE
        d.extend_from_slice(&3u32.to_le_bytes()); // ncmds
        d.extend_from_slice(&(total as u32).to_le_bytes());
        d.extend_from_slice(&0x0020_0000u32.to_le_bytes()); // PIE
        d.extend_from_slice(&0u32.to_le_bytes());
        // LC_SEGMENT_64 __TEXT, one section, no content on disk.
        d.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
        d.extend_from_slice(&152u32.to_le_bytes());
        d.extend_from_slice(b"__TEXT\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(&0x1_0000_0000u64.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&0u64.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&5u32.to_le_bytes());
        d.extend_from_slice(&5u32.to_le_bytes());
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(b"__text\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(b"__TEXT\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(&0x1_0000_0400u64.to_le_bytes());
        d.extend_from_slice(&0x100u64.to_le_bytes());
        d.extend_from_slice(&0x400u32.to_le_bytes());
        d.extend_from_slice(&4u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0x8000_0400u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        // LC_UNIXTHREAD x86_THREAD_STATE64: rip is the 17th register.
        d.extend_from_slice(&LC_UNIXTHREAD.to_le_bytes());
        d.extend_from_slice(&(thread_cmdsize as u32).to_le_bytes());
        d.extend_from_slice(&4u32.to_le_bytes()); // x86_THREAD_STATE64
        d.extend_from_slice(&42u32.to_le_bytes()); // 32-bit word count
        for i in 0..21u64 {
            let value: u64 = if i == 16 { 0x1_0000_0420 } else { 0 };
            d.extend_from_slice(&value.to_le_bytes());
        }
        // LC_FUNCTION_STARTS: deltas 0x400 and 0x20 from the __TEXT base.
        d.extend_from_slice(&LC_FUNCTION_STARTS.to_le_bytes());
        d.extend_from_slice(&16u32.to_le_bytes());
        d.extend_from_slice(&0x500u32.to_le_bytes());
        d.extend_from_slice(&3u32.to_le_bytes());
        d.resize(0x510, 0);
        // ULEB128: 0x400, 0x20, 0.
        let mut pos = 0x500;
        for value in [0x400u32, 0x20, 0] {
            let mut v = value;
            loop {
                let byte = (v & 0x7f) as u8;
                v >>= 7;
                d[pos] = if v != 0 { byte | 0x80 } else { byte };
                pos += 1;
                if v == 0 {
                    break;
                }
            }
        }
        d
    }

    #[test]
    fn the_thread_state_names_the_entry_point() {
        let img = parse(&minimal_with_thread_and_starts(), "t", ImageId(0)).expect("parse");
        assert_eq!(img.entry, Addr(0x1_0000_0420));
    }

    #[test]
    fn function_starts_become_exact_bounds() {
        let img = parse(&minimal_with_thread_and_starts(), "t", ImageId(0)).expect("parse");
        let bounds: Vec<_> = img
            .function_bounds
            .iter()
            .filter(|b| b.source == "function_starts")
            .collect();
        assert_eq!(bounds.len(), 2, "two deltas, so two functions");
        assert_eq!(bounds[0].start, Addr(0x1_0000_0400));
        assert_eq!(bounds[0].end, Addr(0x1_0000_0420));
        assert_eq!(bounds[1].start, Addr(0x1_0000_0420));
        // The last function runs to the end of __TEXT when nothing follows it.
        assert_eq!(bounds[1].end, Addr(0x1_0000_1000));
    }

    /// A 64-bit arm64-style executable whose only record of an import is a
    /// chained fixup in `__DATA`.
    fn minimal_with_chained_fixups() -> Vec<u8> {
        const DYLIB: &[u8] = b"/usr/lib/libSystem.B.dylib\0";
        let dylib_cmdsize = (24 + DYLIB.len() + 7) & !7;
        let sizeofcmds = 72 + 152 + 152 + dylib_cmdsize + 16;
        let mut d = Vec::new();
        d.extend_from_slice(&MH_MAGIC_64.to_le_bytes());
        d.extend_from_slice(&0x0100_000cu32.to_le_bytes()); // arm64
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&2u32.to_le_bytes()); // MH_EXECUTE
        d.extend_from_slice(&5u32.to_le_bytes()); // ncmds
        d.extend_from_slice(&(sizeofcmds as u32).to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        // __PAGEZERO
        d.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
        d.extend_from_slice(&72u32.to_le_bytes());
        d.extend_from_slice(b"__PAGEZERO\0\0\0\0\0\0");
        d.extend_from_slice(&0u64.to_le_bytes());
        d.extend_from_slice(&0x1_0000_0000u64.to_le_bytes());
        d.extend_from_slice(&0u64.to_le_bytes());
        d.extend_from_slice(&0u64.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        // __TEXT with __text
        d.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
        d.extend_from_slice(&152u32.to_le_bytes());
        d.extend_from_slice(b"__TEXT\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(&0x1_0000_0000u64.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&0u64.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&5u32.to_le_bytes());
        d.extend_from_slice(&5u32.to_le_bytes());
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(b"__text\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(b"__TEXT\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(&0x1_0000_0400u64.to_le_bytes());
        d.extend_from_slice(&0x100u64.to_le_bytes());
        d.extend_from_slice(&0x400u32.to_le_bytes());
        d.extend_from_slice(&4u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0x8000_0400u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        // __DATA with __data
        d.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
        d.extend_from_slice(&152u32.to_le_bytes());
        d.extend_from_slice(b"__DATA\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(&0x1_0000_2000u64.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&0x2000u64.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&3u32.to_le_bytes());
        d.extend_from_slice(&3u32.to_le_bytes());
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(b"__data\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(b"__DATA\0\0\0\0\0\0\0\0\0\0");
        d.extend_from_slice(&0x1_0000_3000u64.to_le_bytes());
        d.extend_from_slice(&0x100u64.to_le_bytes());
        d.extend_from_slice(&0x3000u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        // LC_LOAD_DYLIB, so the ordinal has a library name.
        d.extend_from_slice(&LC_LOAD_DYLIB.to_le_bytes());
        d.extend_from_slice(&(dylib_cmdsize as u32).to_le_bytes());
        d.extend_from_slice(&24u32.to_le_bytes()); // name offset
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(DYLIB);
        d.resize(d.len() + (dylib_cmdsize - 24 - DYLIB.len()), 0);
        // LC_DYLD_CHAINED_FIXUPS pointing at 0x500.
        d.extend_from_slice(&LC_DYLD_CHAINED_FIXUPS.to_le_bytes());
        d.extend_from_slice(&16u32.to_le_bytes());
        d.extend_from_slice(&0x500u32.to_le_bytes());
        d.extend_from_slice(&0x100u32.to_le_bytes());
        d.resize(0x2018, 0);

        // The bind pointer: lib_ordinal 0, next 0, bind 1.
        let slot = 0x2010usize;
        let raw: u64 = 1 << 63;
        d[slot..slot + 8].copy_from_slice(&raw.to_le_bytes());

        // The fixups blob.
        let b = 0x500usize;
        let put32 = |d: &mut Vec<u8>, at: usize, v: u32| {
            d[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        put32(&mut d, b, 0); // fixups_version
        put32(&mut d, b + 4, 28); // starts_offset
        put32(&mut d, b + 8, 68); // imports_offset
        put32(&mut d, b + 12, 72); // symbols_offset
        put32(&mut d, b + 16, 1); // imports_count
        put32(&mut d, b + 20, 1); // imports_format = IMPORT
        put32(&mut d, b + 24, 0); // symbols_format
        // dyld_chained_starts_in_image
        put32(&mut d, b + 28, 3); // seg_count
        put32(&mut d, b + 32, 0); // __PAGEZERO
        put32(&mut d, b + 36, 0); // __TEXT
        put32(&mut d, b + 40, 16); // __DATA -> 28 + 16 = 44
        // dyld_chained_starts_in_segment for __DATA
        put32(&mut d, b + 44, 24); // size
        d[b + 48..b + 50].copy_from_slice(&0x1000u16.to_le_bytes()); // page_size
        d[b + 50..b + 52].copy_from_slice(&1u16.to_le_bytes()); // PTR_64
        d[b + 52..b + 60].copy_from_slice(&0x1_0000_2000u64.to_le_bytes());
        put32(&mut d, b + 60, 0); // max_valid_pointer
        d[b + 64..b + 66].copy_from_slice(&1u16.to_le_bytes()); // page_count
        d[b + 66..b + 68].copy_from_slice(&0x10u16.to_le_bytes()); // page_start
        // import record: lib_ordinal 1, weak 0, name_offset 1
        put32(&mut d, b + 68, 1);
        // symbol pool
        d[b + 72..b + 78].copy_from_slice(b"\0puts\0");
        d
    }

    #[test]
    fn chained_fixups_name_the_imported_symbol() {
        let img = parse(&minimal_with_chained_fixups(), "t", ImageId(0)).expect("parse");
        let puts = img
            .imports
            .iter()
            .find(|i| i.name == "puts")
            .unwrap_or_else(|| panic!("imports: {:?}", img.imports));
        assert_eq!(puts.slot, Some(Addr(0x1_0000_2010)));
        assert_eq!(puts.library.as_deref(), Some("/usr/lib/libSystem.B.dylib"));
    }

    #[test]
    fn truncated_input_is_an_error_not_a_panic() {
        let full = minimal_macho64();
        for cut in [0usize, 4, 20, 64, 100, 200] {
            let _ = parse(&full[..cut.min(full.len())], "t", ImageId(0));
        }
    }
}
