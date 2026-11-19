//! PE and COFF parsing.
//!
//! PE32 and PE32+ share almost everything except the width of `ImageBase`, the
//! size of the optional header, and the width of a thunk. Those three
//! differences are threaded through as a single `pe64` flag rather than
//! duplicated into two parsers, because the import and export directories are
//! byte-identical between the two.
//!
//! Two structures carry the import information and both must be read:
//! `OriginalFirstThunk` (the ILT) holds the names, and `FirstThunk` (the IAT)
//! holds the slots the loader patches. Reading only the IAT gives slot
//! addresses with no names; reading only the ILT gives names with no slots.
//! The pairing is positional, so a mismatch in either count corrupts every
//! entry after it — which is why the descriptor's thunk count is derived from
//! the table itself rather than from a header field.

use crate::reader::Reader;
use occ_core::{
    Addr, Arch, CoreError, Endian, Export, Fingerprint, Format, FunctionBound, Image, ImageId,
    Import, ImportKind, Linkage, Perm, RelocKind, Relocation, Result, Section, SectionKind,
    Segment, Symbol, SymbolBinding, SymbolKind,
};

/// `IMAGE_FILE_DLL`: the image is a library, not a program.
const IMAGE_FILE_DLL: u16 = 0x2000;
const PE32_MAGIC: u16 = 0x10b;
const PE32_PLUS_MAGIC: u16 = 0x20b;

// Data directory indices.
const DIR_EXPORT: usize = 0;
const DIR_IMPORT: usize = 1;
const DIR_EXCEPTION: usize = 3;
const DIR_BASERELOC: usize = 5;

// Section characteristics.
const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const IMAGE_SCN_MEM_DISCARDABLE: u32 = 0x0200_0000;

/// Section names used by known packers.
///
/// Kept here rather than in the fingerprint module because the decision is made
/// while walking the section table, and passing the names out and back in would
/// duplicate the walk.
const PACKER_SECTIONS: [&str; 17] = [
    "UPX0", "UPX1", "UPX2", ".UPX0", ".UPX1", ".aspack", ".adata", "ASPack", ".nsp0", ".nsp1",
    ".petite", ".MPRESS1", ".MPRESS2", ".themida", ".vmp0", ".vmp1", "PEC2",
];

struct Layout {
    pe64: bool,
    machine: u16,
    entry_rva: u32,
    image_base: u64,
    section_alignment: u32,
    size_of_image: u32,
    characteristics: u16,
    dirs: Vec<(u32, u32)>,
    sections: Vec<RawSection>,
}

#[derive(Clone)]
struct RawSection {
    name: String,
    virtual_address: u32,
    virtual_size: u32,
    raw_size: u32,
    raw_ptr: u32,
    characteristics: u32,
}

/// Parse a PE image.
pub fn parse(data: &[u8], path: &str, id: ImageId) -> Result<Image> {
    let r = Reader::new(data, Endian::Little);
    let l = parse_layout(&r, data)?;

    let mut warnings = Vec::new();
    let sections: Vec<Section> = l
        .sections
        .iter()
        .enumerate()
        .map(|(i, s)| Section {
            index: i + 1,
            name: s.name.clone(),
            addr: Addr(l.image_base + u64::from(s.virtual_address)),
            file_offset: u64::from(s.raw_ptr),
            size: u64::from(s.virtual_size.max(s.raw_size)),
            file_size: u64::from(s.raw_size),
            kind: classify_section(s),
            perm: Perm::of(
                s.characteristics & IMAGE_SCN_MEM_READ != 0,
                s.characteristics & IMAGE_SCN_MEM_WRITE != 0,
                s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0,
            ),
            // A header page is mapped but is not a section in the table.
        })
        .collect();

    // One segment covering the whole image. PE has no segment table; the loader
    // maps `SizeOfImage` bytes at `ImageBase` and applies per-section
    // protection, so a single range is the accurate model.
    let segments = vec![Segment {
        addr: Addr(l.image_base),
        size: u64::from(l.size_of_image),
        file_size: data.len() as u64,
        perm: Perm::of(true, false, false),
    }];

    let mut imports = read_imports(&r, &l, data, &mut warnings);
    let exports = read_exports(&r, &l, data, &mut warnings);
    let relocations = read_relocations(&r, &l, data);
    let rich = check_rich_header(data, &mut warnings);

    // Bind import slots to the IAT entries their descriptors name.
    for imp in &mut imports {
        if imp.slot.is_none()
            && let Some(rva) = iat_slot_of(&r, &l, &imp.name, data)
        {
            imp.slot = Some(Addr(l.image_base + u64::from(rva)));
        }
    }

    let symbols = imports
        .iter()
        .map(|i| Symbol {
            name: i.name.clone(),
            addr: i.slot.unwrap_or(Addr::NULL),
            size: 0,
            kind: SymbolKind::Function,
            binding: SymbolBinding::Global,
            section: None,
            defined: false,
            demangled: false,
        })
        .collect();

    let arch = map_machine(l.machine);
    let bits = if l.pe64 { 64 } else { 32 };
    if l.size_of_image == 0 {
        warnings.push("SizeOfImage is zero; the mapped range is unknown".into());
    }
    if !l.section_alignment.is_power_of_two() {
        warnings.push(format!(
            "SectionAlignment {} is not a power of two",
            l.section_alignment
        ));
    }
    let _ = rich;

    // The exception directory is the linker's own function list. On x64 and
    // ARM64 each entry states both ends; on ARM32 it states the start and a
    // unwind record, and the end is the next start.
    let function_bounds = pdata_bounds(data, &l);

    let mut fingerprint = Fingerprint::unknown(id);
    fingerprint.linkage = if l.characteristics & IMAGE_FILE_DLL != 0 {
        Linkage::Shared
    } else {
        Linkage::Executable
    };

    Ok(Image {
        id,
        path: path.to_string(),
        format: Format::Pe,
        arch,
        bits,
        endian: Endian::Little,
        entry: Addr(l.image_base + u64::from(l.entry_rva)),
        image_base: Addr(l.image_base),
        file_size: data.len() as u64,
        sha256: crate::sha256::hex_digest(data),
        segments,
        sections,
        symbols,
        imports,
        exports,
        relocations,
        debug_relocations: Vec::new(),
        fingerprint,
        interpreter: None,
        warnings,
        function_bounds,
        arm_modes: Default::default(),
        wasm_types: Default::default(),
        wasm_funcs: Default::default(),
    })
}

fn parse_layout(r: &Reader<'_>, data: &[u8]) -> Result<Layout> {
    if data.len() < 0x40 || &data[0..2] != b"MZ" {
        return Err(CoreError::Malformed {
            offset: 0,
            detail: "missing MZ signature".into(),
        });
    }
    let lfanew = r.u32(0x3c)? as usize;
    if lfanew + 24 > data.len() {
        return Err(CoreError::Malformed {
            offset: 0x3c,
            detail: format!("e_lfanew {lfanew:#x} points past end of file"),
        });
    }
    if &data[lfanew..lfanew + 4] != b"PE\0\0" {
        return Err(CoreError::Malformed {
            offset: lfanew as u64,
            detail: "missing PE signature".into(),
        });
    }

    let coff = lfanew + 4;
    let machine = r.u16(coff)?;
    let nsections = r.u16(coff + 2)? as usize;
    let opt_size = r.u16(coff + 16)? as usize;
    // IMAGE_FILE_DLL is the only bit here that changes an answer: a DLL is
    // loaded at a base the system picks, an EXE at its preferred one.
    let characteristics = r.u16(coff + 18)?;
    let opt = coff + 20;

    let magic = r.u16(opt)?;
    let pe64 = match magic {
        PE32_MAGIC => false,
        PE32_PLUS_MAGIC => true,
        other => {
            return Err(CoreError::Malformed {
                offset: opt as u64,
                detail: format!("unknown optional header magic {other:#x}"),
            });
        }
    };

    let entry_rva = r.u32(opt + 16)?;
    let (image_base, dir_off, nrva_off) = if pe64 {
        (r.u64(opt + 24)?, opt + 112, opt + 108)
    } else {
        (u64::from(r.u32(opt + 28)?), opt + 96, opt + 92)
    };
    let section_alignment = r.u32(opt + 32)?;
    let size_of_image = r.u32(opt + 56)?;

    let nrva = r.u32(nrva_off).unwrap_or(0) as usize;
    let nrva = nrva.min(16);
    let mut dirs = Vec::with_capacity(nrva);
    for i in 0..nrva {
        let off = dir_off + i * 8;
        dirs.push((r.u32(off)?, r.u32(off + 4)?));
    }

    let sec_off = opt + opt_size;
    let mut sections = Vec::with_capacity(nsections);
    for i in 0..nsections {
        let off = sec_off + i * 40;
        if off + 40 > data.len() {
            return Err(CoreError::Malformed {
                offset: off as u64,
                detail: format!("section header {i} is past end of file"),
            });
        }
        let raw = r.bytes(off, 8)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(8);
        sections.push(RawSection {
            name: String::from_utf8_lossy(&raw[..end]).into_owned(),
            virtual_size: r.u32(off + 8)?,
            virtual_address: r.u32(off + 12)?,
            raw_size: r.u32(off + 16)?,
            raw_ptr: r.u32(off + 20)?,
            characteristics: r.u32(off + 36)?,
        });
    }

    Ok(Layout {
        pe64,
        machine,
        entry_rva,
        image_base,
        section_alignment,
        size_of_image,
        characteristics,
        dirs,
        sections,
    })
}

/// Translate a relative virtual address into a file offset.
/// Function extents from the exception directory (`.pdata`).
///
/// x64 and ARM64 entries are `RUNTIME_FUNCTION` records of twelve bytes:
/// begin RVA, end RVA, unwind RVA. ARM32 records are eight bytes and carry no
/// end, so the next entry's start is this one's end — which is exact because
/// the linker emits the table sorted by address.
fn pdata_bounds(data: &[u8], l: &Layout) -> Vec<FunctionBound> {
    let Some(&(rva, size)) = l.dirs.get(DIR_EXCEPTION) else {
        return Vec::new();
    };
    if rva == 0 || size == 0 {
        return Vec::new();
    }
    let Some(base) = rva_to_offset(l, rva) else {
        return Vec::new();
    };
    let arm32 = l.machine == 0x01c0;
    let entry = if arm32 { 8usize } else { 12 };
    let count = (size as usize / entry).min(1_000_000);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + i * entry;
        let Some(begin) = data
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        else {
            break;
        };
        if begin == 0 {
            continue;
        }
        let end = if arm32 {
            0
        } else {
            data.get(at + 4..at + 8)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .unwrap_or(0)
        };
        out.push(FunctionBound {
            start: Addr(l.image_base + u64::from(begin)),
            end: Addr(l.image_base + u64::from(end)),
            source: "pdata",
        });
    }
    if arm32 {
        out.sort_by_key(|b| b.start.0);
        for i in 0..out.len() {
            let next = out.get(i + 1).map(|b| b.start).unwrap_or(Addr::NULL);
            if out[i].end.is_null() {
                out[i].end = next;
            }
        }
    }
    out
}

fn rva_to_offset(l: &Layout, rva: u32) -> Option<usize> {
    for s in &l.sections {
        let span = s.virtual_size.max(s.raw_size);
        if span == 0 {
            continue;
        }
        if rva >= s.virtual_address && rva < s.virtual_address.saturating_add(span) {
            let delta = rva - s.virtual_address;
            // Beyond `raw_size` the section is zero-filled; there is no byte in
            // the file to read.
            if delta >= s.raw_size {
                return None;
            }
            return Some(s.raw_ptr as usize + delta as usize);
        }
    }
    // Data directories can point into the header page, where RVA == offset.
    if (rva as usize) < 0x1000 {
        return Some(rva as usize);
    }
    None
}

fn read_imports(
    r: &Reader<'_>,
    l: &Layout,
    data: &[u8],
    warnings: &mut Vec<String>,
) -> Vec<Import> {
    let mut out = Vec::new();
    let Some(&(rva, size)) = l.dirs.get(DIR_IMPORT) else {
        return out;
    };
    if rva == 0 || size == 0 {
        return out;
    }
    let Some(base) = rva_to_offset(l, rva) else {
        warnings.push(format!(
            "import directory RVA {rva:#x} maps to no file bytes"
        ));
        return out;
    };

    let thunk_size = if l.pe64 { 8 } else { 4 };
    let ordinal_flag: u64 = if l.pe64 { 1 << 63 } else { 1 << 31 };

    for i in 0..4096 {
        let off = base + i * 20;
        if off + 20 > data.len() {
            warnings.push(format!("import descriptor {i} runs past end of file"));
            break;
        }
        let original_first_thunk = r.u32(off).unwrap_or(0);
        let name_rva = r.u32(off + 12).unwrap_or(0);
        let first_thunk = r.u32(off + 16).unwrap_or(0);
        if name_rva == 0 && first_thunk == 0 && original_first_thunk == 0 {
            break;
        }
        let library = rva_to_offset(l, name_rva)
            .and_then(|o| {
                let bytes = data.get(o..)?;
                let end = bytes.iter().position(|&b| b == 0)?;
                Some(String::from_utf8_lossy(&bytes[..end]).into_owned())
            })
            .unwrap_or_default();

        // The ILT holds the names; the IAT holds the slots. When
        // `OriginalFirstThunk` is zero the file was linked with the two tables
        // overlapping, and the IAT doubles as the name table — a layout some
        // packers produce deliberately.
        let name_table = if original_first_thunk != 0 {
            original_first_thunk
        } else {
            first_thunk
        };
        let Some(nt_off) = rva_to_offset(l, name_table) else {
            warnings.push(format!(
                "{library}: thunk table RVA {name_table:#x} maps to no file bytes"
            ));
            continue;
        };

        for k in 0..65536usize {
            let toff = nt_off + k * thunk_size;
            let value = if l.pe64 {
                match r.u64(toff) {
                    Ok(v) => v,
                    Err(_) => break,
                }
            } else {
                match r.u32(toff) {
                    Ok(v) => u64::from(v),
                    Err(_) => break,
                }
            };
            if value == 0 {
                break;
            }
            let iat_rva = first_thunk.saturating_add((k as u32) * thunk_size as u32);
            let slot = if first_thunk != 0 {
                Some(Addr(l.image_base + u64::from(iat_rva)))
            } else {
                None
            };

            if value & ordinal_flag != 0 {
                out.push(Import {
                    name: format!("#{}", value & 0xffff),
                    library: Some(library.clone()),
                    ordinal: Some((value & 0xffff) as u32),
                    slot,
                    kind: ImportKind::Normal,
                });
                continue;
            }
            // A name thunk holds an RVA to a `IMAGE_IMPORT_BY_NAME`: a 2-byte
            // hint followed by the NUL-terminated name.
            let name_rva = (value & 0x7fff_ffff) as u32;
            let Some(name_off) = rva_to_offset(l, name_rva) else {
                warnings.push(format!(
                    "{library}: import name RVA {name_rva:#x} is unmapped"
                ));
                continue;
            };
            let Some(bytes) = data.get(name_off + 2..) else {
                continue;
            };
            let Some(end) = bytes.iter().position(|&b| b == 0) else {
                continue;
            };
            let name = String::from_utf8_lossy(&bytes[..end]).into_owned();
            out.push(Import {
                name,
                library: Some(library.clone()),
                ordinal: None,
                slot,
                kind: ImportKind::Normal,
            });
        }
    }
    out
}

/// Find the IAT slot whose name matches, for imports whose descriptor was
/// unreadable.
fn iat_slot_of(r: &Reader<'_>, l: &Layout, name: &str, data: &[u8]) -> Option<u32> {
    let &(rva, size) = l.dirs.get(DIR_IMPORT)?;
    if rva == 0 || size == 0 {
        return None;
    }
    let base = rva_to_offset(l, rva)?;
    let thunk_size = if l.pe64 { 8 } else { 4 };
    let mut i = 0;
    loop {
        let off = base + i * 20;
        if off + 20 > data.len() {
            return None;
        }
        let original = r.u32(off).unwrap_or(0);
        let first = r.u32(off + 16).unwrap_or(0);
        if original == 0 && first == 0 {
            return None;
        }
        let nt = if original != 0 { original } else { first };
        let nt_off = rva_to_offset(l, nt)?;
        let mut k = 0usize;
        loop {
            let toff = nt_off + k * thunk_size;
            let value = if l.pe64 {
                r.u64(toff).ok()?
            } else {
                u64::from(r.u32(toff).ok()?)
            };
            if value == 0 {
                break;
            }
            let ordinal_flag: u64 = if l.pe64 { 1 << 63 } else { 1 << 31 };
            if value & ordinal_flag == 0 {
                let n = rva_to_offset(l, (value & 0x7fff_ffff) as u32)?;
                let bytes = data.get(n + 2..)?;
                let end = bytes.iter().position(|&b| b == 0)?;
                if &bytes[..end] == name.as_bytes() {
                    return Some(first + (k as u32) * thunk_size as u32);
                }
            }
            k += 1;
        }
        i += 1;
    }
}

fn read_exports(
    r: &Reader<'_>,
    l: &Layout,
    data: &[u8],
    warnings: &mut Vec<String>,
) -> Vec<Export> {
    let mut out = Vec::new();
    let Some(&(rva, size)) = l.dirs.get(DIR_EXPORT) else {
        return out;
    };
    if rva == 0 || size == 0 {
        return out;
    }
    let Some(base) = rva_to_offset(l, rva) else {
        warnings.push(format!(
            "export directory RVA {rva:#x} maps to no file bytes"
        ));
        return out;
    };
    if base + 40 > data.len() {
        warnings.push("export directory is truncated".into());
        return out;
    }
    let ordinal_base = r.u32(base + 16).unwrap_or(0);
    let nfuncs = r.u32(base + 20).unwrap_or(0);
    let nnames = r.u32(base + 24).unwrap_or(0);
    let funcs_rva = r.u32(base + 28).unwrap_or(0);
    let names_rva = r.u32(base + 32).unwrap_or(0);
    let ords_rva = r.u32(base + 36).unwrap_or(0);

    let (Some(funcs), Some(names), Some(ords)) = (
        rva_to_offset(l, funcs_rva),
        rva_to_offset(l, names_rva),
        rva_to_offset(l, ords_rva),
    ) else {
        warnings.push("export name/function tables are unmapped".into());
        return out;
    };

    for i in 0..nnames.min(65536) {
        let name_rva = r.u32(names + i as usize * 4).unwrap_or(0);
        let ord_idx = r.u16(ords + i as usize * 2).unwrap_or(0xffff);
        if ord_idx as u32 >= nfuncs {
            continue;
        }
        let func_rva = r.u32(funcs + ord_idx as usize * 4).unwrap_or(0);
        let name = rva_to_offset(l, name_rva)
            .and_then(|o| {
                let bytes = data.get(o..)?;
                let end = bytes.iter().position(|&b| b == 0)?;
                Some(String::from_utf8_lossy(&bytes[..end]).into_owned())
            })
            .unwrap_or_default();
        // A function RVA inside the export directory itself is a forwarder
        // string, not code.
        let forward = if func_rva >= rva && func_rva < rva.saturating_add(size) {
            rva_to_offset(l, func_rva).and_then(|o| {
                let bytes = data.get(o..)?;
                let end = bytes.iter().position(|&b| b == 0)?;
                Some(String::from_utf8_lossy(&bytes[..end]).into_owned())
            })
        } else {
            None
        };
        out.push(Export {
            name,
            addr: Addr(l.image_base + u64::from(func_rva)),
            ordinal: Some(ordinal_base + u32::from(ord_idx)),
            forward,
        });
    }
    out.sort_by_key(|e| e.addr);
    out
}

fn read_relocations(r: &Reader<'_>, l: &Layout, data: &[u8]) -> Vec<Relocation> {
    let mut out = Vec::new();
    let Some(&(rva, size)) = l.dirs.get(DIR_BASERELOC) else {
        return out;
    };
    if rva == 0 || size == 0 {
        return out;
    }
    let Some(mut off) = rva_to_offset(l, rva) else {
        return out;
    };
    let end = off.saturating_add(size as usize).min(data.len());
    let mut blocks = 0;
    while off + 8 <= end && blocks < 4096 {
        let page = r.u32(off).unwrap_or(0);
        let block_size = r.u32(off + 4).unwrap_or(0) as usize;
        if block_size < 8 || off + block_size > end {
            break;
        }
        let count = (block_size - 8) / 2;
        for k in 0..count {
            let entry = r.u16(off + 8 + k * 2).unwrap_or(0);
            let kind = entry >> 12;
            let delta = u32::from(entry & 0x0fff);
            // Type 0 is padding used to align a block, not a relocation.
            if kind == 0 {
                continue;
            }
            let target = u64::from(page) + u64::from(delta);
            let kind = match kind {
                3 => RelocKind::Absolute,       // HIGHLOW
                10 => RelocKind::ImageRelative, // DIR64
                1 => RelocKind::Other,          // HIGH
                2 => RelocKind::Other,          // LOW
                4 => RelocKind::Other,          // HIGHADJ
                _ => RelocKind::Other,
            };
            out.push(Relocation::bare(Addr(l.image_base + target), kind, None));
        }
        off += block_size;
        blocks += 1;
    }
    out
}

/// Check the Rich header for the zero-key tampering signature.
///
/// The Rich header is not documented and not needed for analysis, but its XOR
/// key is: every entry is XORed with a checksum of the DOS stub, and a key of
/// zero means the header was rewritten by a tool that did not recompute it.
fn check_rich_header(data: &[u8], warnings: &mut Vec<String>) -> bool {
    let limit = data.len().min(0x400);
    let Some(pos) = data[..limit].windows(4).position(|w| w == b"Rich") else {
        return false;
    };
    if pos < 8 {
        return false;
    }
    let key = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
    if key == 0 {
        warnings.push(
            "Rich header key is zero: the header was rewritten without recomputing the checksum"
                .into(),
        );
    }
    true
}

fn classify_section(s: &RawSection) -> SectionKind {
    if PACKER_SECTIONS
        .iter()
        .any(|p| s.name.trim_end_matches('\0').eq_ignore_ascii_case(p))
    {
        return SectionKind::Other;
    }
    if s.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA != 0
        && s.characteristics & IMAGE_SCN_CNT_CODE == 0
    {
        return SectionKind::Bss;
    }
    if s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0 {
        return SectionKind::Code;
    }
    let name = s.name.trim_end_matches('\0');
    if name.starts_with(".debug") || name == ".pdb" {
        return SectionKind::Debug;
    }
    if name == ".pdata" || name == ".xdata" || name == ".eh_frame" {
        return SectionKind::Unwind;
    }
    if name == ".rsrc" || name == ".reloc" || name == ".tls" {
        return SectionKind::Other;
    }
    if s.characteristics & IMAGE_SCN_MEM_DISCARDABLE != 0 {
        return SectionKind::Other;
    }
    let writable = s.characteristics & IMAGE_SCN_MEM_WRITE != 0;
    let readable = s.characteristics & IMAGE_SCN_MEM_READ != 0;
    if !writable && readable {
        return SectionKind::Rodata;
    }
    SectionKind::Data
}

fn map_machine(machine: u16) -> Arch {
    match machine {
        0x014c => Arch::X86,
        0x8664 => Arch::X86_64,
        0xaa64 => Arch::Aarch64,
        // IMAGE_FILE_MACHINE_ARMNT. The old 0x01c2/0x01c4 ARM and Thumb-only
        // values are not produced by any current toolchain; 0x01c0 is what
        // MSVC and LLVM emit for ARMv7.
        0x01c0 => Arch::Arm,
        0x5032 => Arch::RiscV32,
        0x5064 => Arch::RiscV64,
        _ => Arch::Other,
    }
}

/// Parse a standalone COFF object file.
///
/// An object has no optional header, no base address and no imports: it is a
/// bag of sections plus a symbol table that a linker will resolve later.
pub fn parse_coff(data: &[u8], path: &str, id: ImageId) -> Result<Image> {
    if data.len() < 20 {
        return Err(CoreError::Malformed {
            offset: 0,
            detail: "file is shorter than a COFF header".into(),
        });
    }
    let r = Reader::new(data, Endian::Little);
    let machine = r.u16(0)?;
    let nsections = r.u16(2)? as usize;
    let symptr = r.u32(8)? as usize;
    let nsyms = r.u32(12)? as usize;
    let opt_size = r.u16(16)? as usize;
    let sec_off = 20 + opt_size;

    let mut sections = Vec::with_capacity(nsections);
    for i in 0..nsections {
        let off = sec_off + i * 40;
        if off + 40 > data.len() {
            return Err(CoreError::Malformed {
                offset: off as u64,
                detail: format!("section header {i} is past end of file"),
            });
        }
        let raw = r.bytes(off, 8)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(8);
        let s = RawSection {
            name: String::from_utf8_lossy(&raw[..end]).into_owned(),
            virtual_size: r.u32(off + 8)?,
            virtual_address: r.u32(off + 12)?,
            raw_size: r.u32(off + 16)?,
            raw_ptr: r.u32(off + 20)?,
            characteristics: r.u32(off + 36)?,
        };
        sections.push(Section {
            index: i + 1,
            name: s.name.clone(),
            // Relocations in an object are not yet placed: `VirtualAddress` is
            // an offset within the section, not an address in any image. Using
            // the raw data offset keeps sections distinct and gives symbols
            // something to be rebased onto; see `read_coff_symbols` below.
            addr: Addr(u64::from(if s.raw_ptr != 0 {
                s.raw_ptr
            } else {
                s.virtual_address
            })),
            file_offset: u64::from(s.raw_ptr),
            size: u64::from(s.virtual_size.max(s.raw_size)),
            file_size: u64::from(s.raw_size),
            kind: classify_section(&s),
            perm: Perm::of(
                s.characteristics & IMAGE_SCN_MEM_READ != 0,
                s.characteristics & IMAGE_SCN_MEM_WRITE != 0,
                s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0,
            ),
        });
    }

    let mut symbols = Vec::new();
    if symptr > 0 && nsyms > 0 {
        let strtab_off = symptr + nsyms * 18;
        for i in 0..nsyms.min(200_000) {
            let off = symptr + i * 18;
            if off + 18 > data.len() {
                break;
            }
            let zeroes = r.u32(off).unwrap_or(0);
            let name = if zeroes == 0 {
                // Long name: offset into the string table, which follows the
                // symbol table and starts with a 4-byte length.
                let so = r.u32(off + 4).unwrap_or(0) as usize;
                r.str_at(data.get(strtab_off..).unwrap_or(&[]), so)
                    .to_string()
            } else {
                let raw = r.bytes(off, 8).unwrap_or(&[]);
                let end = raw.iter().position(|&b| b == 0).unwrap_or(8);
                String::from_utf8_lossy(&raw[..end]).into_owned()
            };
            if name.is_empty() {
                continue;
            }
            let value = r.u32(off + 8).unwrap_or(0);
            let secnum = r.i16(off + 12).unwrap_or(0);
            let kind = r.u16(off + 14).unwrap_or(0);
            let storage = r.u8(off + 16).unwrap_or(0);
            // `Value` is an offset into the section named by `SectionNumber`,
            // never an address: an object is linked at an unknown base.
            let addr = if secnum > 0 {
                let base = sections
                    .get((secnum - 1) as usize)
                    .map_or(0, |s| s.addr.0);
                Addr(base.wrapping_add(u64::from(value)))
            } else {
                Addr::NULL
            };
            symbols.push(Symbol {
                name: crate::demangle::clean_name(&name),
                addr,
                size: 0,
                kind: match kind {
                    0x20 => SymbolKind::Function,
                    _ => SymbolKind::Unknown,
                },
                binding: match storage {
                    2 => SymbolBinding::Global,
                    105 => SymbolBinding::Weak,
                    _ => SymbolBinding::Local,
                },
                section: if secnum > 0 {
                    sections.get((secnum - 1) as usize).map(|s| s.name.clone())
                } else {
                    None
                },
                defined: secnum > 0,
                demangled: false,
            });
        }
    }

    let entry = sections
        .iter()
        .find(|s| s.name == ".text")
        .map(|s| s.addr)
        .unwrap_or(Addr::NULL);

    Ok(Image {
        id,
        path: path.to_string(),
        format: Format::Coff,
        arch: map_machine(machine),
        bits: if matches!(machine, 0x8664 | 0xaa64 | 0x5064) {
            64
        } else {
            32
        },
        endian: Endian::Little,
        entry,
        image_base: Addr::NULL,
        file_size: data.len() as u64,
        sha256: crate::sha256::hex_digest(data),
        segments: Vec::new(),
        sections,
        symbols,
        imports: Vec::new(),
        exports: Vec::new(),
        relocations: Vec::new(),
        debug_relocations: Vec::new(),
        fingerprint: {
            let mut fp = Fingerprint::unknown(id);
            fp.linkage = Linkage::Relocatable;
            fp
        },
        interpreter: None,
        warnings: vec!["COFF object: addresses are section-relative until linked".into()],
        function_bounds: Vec::new(),
        arm_modes: Default::default(),
        wasm_types: Default::default(),
        wasm_funcs: Default::default(),
    })
}
