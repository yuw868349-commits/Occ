//! WebAssembly module parsing.
//!
//! WASM has no virtual address space, so addresses here are byte offsets into
//! the file. That is not a shortcut: an offset is the only stable identity a
//! function body has, and it is what a byte-level view can actually read.
//! Everything downstream — sections, function bodies, exports — is expressed in
//! offsets so a caller can hexdump what a symbol points at.
//!
//! What is recovered: the section table, imports (with kind and signature
//! index), exports with their targets, the function index space including
//! imported functions, every code body's offset and size, and names from the
//! `name` custom section when the toolchain emitted it. The instruction stream
//! is decoded by [`occ_decode::wasm`], which walks these bodies; this module
//! only reports where each one starts and ends.

use occ_core::{
    Addr, Arch, CoreError, Endian, Export, Fingerprint, Format, Image, ImageId, Import, ImportKind,
    Linkage, Perm, Result, Section, SectionKind, Symbol, SymbolBinding, SymbolKind, WasmFuncSig,
    WasmFuncType, WasmValueType,
};
use std::collections::HashMap;

/// Parse a WebAssembly module.
pub fn parse(data: &[u8], path: &str, id: ImageId) -> Result<Image> {
    if data.len() < 8 || &data[0..4] != b"\0asm" {
        return Err(CoreError::Malformed {
            offset: 0,
            detail: "missing WASM magic".into(),
        });
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != 1 {
        return Err(CoreError::Malformed {
            offset: 4,
            detail: format!("unsupported WASM version {version}"),
        });
    }

    let mut sections = Vec::new();
    let mut imports = Vec::new();
    let mut exports_raw: Vec<(String, u8, u32)> = Vec::new();
    let mut bodies: Vec<(u32, u64, u64)> = Vec::new(); // (func index, offset, size)
    let mut start_func: Option<u32> = None;
    let mut imported_funcs = 0u32;
    let mut defined_funcs = 0u32;
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut data_segments = 0u32;
    let mut memories = 0u32;
    let mut warnings = Vec::new();
    // The type section: one entry per function type, by type index.
    let mut func_types: Vec<WasmFuncType> = Vec::new();
    // The type index of every imported function, in import order.
    let mut imported_func_types: Vec<u32> = Vec::new();
    // The type index of every defined function, in definition order.
    let mut defined_func_types: Vec<u32> = Vec::new();

    let mut pos = 8usize;
    while pos < data.len() {
        let id_byte = data[pos];
        pos += 1;
        let Some(size) = leb_u32(data, &mut pos) else {
            break;
        };
        let body_start = pos;
        let body_end = body_start.saturating_add(size as usize).min(data.len());
        if body_end < body_start {
            break;
        }
        let name = section_name(id_byte);
        let kind = match id_byte {
            0 => SectionKind::Other,
            10 => SectionKind::Code,
            11 => SectionKind::Data,
            1 | 2 | 3 | 7 => SectionKind::Other,
            _ => SectionKind::Other,
        };
        sections.push(Section {
            index: sections.len() + 1,
            name: if id_byte == 0 {
                // A custom section's name is its first field.
                let mut p = body_start;
                match leb_u32(data, &mut p).and_then(|n| read_str(data, &mut p, n as usize)) {
                    Some(s) => format!("custom:{s}"),
                    None => "custom".into(),
                }
            } else {
                name.to_string()
            },
            addr: Addr(body_start as u64),
            file_offset: body_start as u64,
            size: size as u64,
            file_size: size as u64,
            kind,
            perm: Perm::R,
        });

        let mut p = body_start;
        match id_byte {
            0 => {
                // Custom sections are scanned for the name section.
                if let Some(len) = leb_u32(data, &mut p)
                    && let Some(name) = read_str(data, &mut p, len as usize)
                    && name == "name"
                {
                    parse_name_section(data, &mut p, body_end, &mut names);
                }
            }
            1 => {
                // The type section: one entry per function type, each of
                // them a form, a vector of parameter types and a vector of
                // result types. The types say what a call's arguments are
                // and what the body leaves on the stack, so they are kept
                // rather than counted.
                if let Some(count) = leb_u32(data, &mut p) {
                    for _ in 0..count.min(1_000_000) {
                        let Some(form) = data.get(p).copied() else {
                            break;
                        };
                        p += 1;
                        if form != 0x60 {
                            // Not a function type. The rest of the section
                            // cannot be read without knowing how long this
                            // one is, so the section stops here.
                            break;
                        }
                        let Some(param_count) = leb_u32(data, &mut p) else {
                            break;
                        };
                        let Some(params) = value_types(data, &mut p, param_count) else {
                            break;
                        };
                        let Some(result_count) = leb_u32(data, &mut p) else {
                            break;
                        };
                        let Some(results) = value_types(data, &mut p, result_count) else {
                            break;
                        };
                        func_types.push(WasmFuncType { params, results });
                    }
                }
            }
            2 => {
                if let Some(count) = leb_u32(data, &mut p) {
                    for _ in 0..count.min(100_000) {
                        let Some(mlen) = leb_u32(data, &mut p) else {
                            break;
                        };
                        let Some(module) = read_str(data, &mut p, mlen as usize) else {
                            break;
                        };
                        let Some(flen) = leb_u32(data, &mut p) else {
                            break;
                        };
                        let Some(field) = read_str(data, &mut p, flen as usize) else {
                            break;
                        };
                        let Some(kind) = data.get(p).copied() else {
                            break;
                        };
                        p += 1;
                        let imported_kind = match kind {
                            0 => {
                                let Some(type_idx) = leb_u32(data, &mut p) else {
                                    break;
                                };
                                imported_funcs += 1;
                                imported_func_types.push(type_idx);
                                ImportKind::Normal
                            }
                            1 => ImportKind::Data,
                            2 => {
                                let Some(_limits) = leb_u32(data, &mut p) else {
                                    break;
                                };
                                memories += 1;
                                ImportKind::Data
                            }
                            3 => ImportKind::Data,
                            _ => break,
                        };
                        imports.push(Import {
                            name: field,
                            library: Some(module),
                            ordinal: None,
                            slot: None,
                            kind: imported_kind,
                        });
                    }
                }
            }
            3 => {
                // The function section: one type index per defined function,
                // in definition order. It is what says which type each
                // function has, and the type is what says how many
                // parameters it takes.
                if let Some(count) = leb_u32(data, &mut p) {
                    defined_funcs = count;
                    for _ in 0..count.min(1_000_000) {
                        let Some(type_idx) = leb_u32(data, &mut p) else {
                            break;
                        };
                        defined_func_types.push(type_idx);
                    }
                }
            }
            5 => {
                if let Some(count) = leb_u32(data, &mut p) {
                    memories += count;
                }
            }
            7 => {
                if let Some(count) = leb_u32(data, &mut p) {
                    for _ in 0..count.min(100_000) {
                        let Some(nlen) = leb_u32(data, &mut p) else {
                            break;
                        };
                        let Some(name) = read_str(data, &mut p, nlen as usize) else {
                            break;
                        };
                        let Some(kind) = data.get(p).copied() else {
                            break;
                        };
                        p += 1;
                        let Some(index) = leb_u32(data, &mut p) else {
                            break;
                        };
                        exports_raw.push((name, kind, index));
                    }
                }
            }
            8 => {
                if let Some(idx) = leb_u32(data, &mut p) {
                    start_func = Some(idx);
                }
            }
            10 => {
                if let Some(count) = leb_u32(data, &mut p) {
                    for i in 0..count.min(1_000_000) {
                        let Some(body_size) = leb_u32(data, &mut p) else {
                            break;
                        };
                        let offset = p as u64;
                        bodies.push((imported_funcs + i, offset, body_size as u64));
                        p = p.saturating_add(body_size as usize);
                    }
                }
            }
            11 => {
                if let Some(count) = leb_u32(data, &mut p) {
                    data_segments = count;
                }
            }
            _ => {}
        }
        pos = body_end;
    }

    // Function symbols: body offset is the address.
    let mut symbols = Vec::new();
    for (index, offset, size) in &bodies {
        symbols.push(Symbol {
            name: names
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("func_{index}")),
            addr: Addr(*offset),
            size: *size,
            kind: SymbolKind::Function,
            binding: SymbolBinding::Global,
            section: Some("code".into()),
            defined: true,
            demangled: false,
        });
    }

    let mut exports = Vec::new();
    for (name, kind, index) in exports_raw {
        if kind == 0 {
            let addr = bodies
                .iter()
                .find(|(i, _, _)| *i == index)
                .map(|(_, o, _)| Addr(*o))
                .unwrap_or(Addr::NULL);
            exports.push(Export {
                name,
                addr,
                ordinal: None,
                forward: None,
            });
        }
    }

    let entry = start_func
        .and_then(|i| {
            bodies
                .iter()
                .find(|(idx, _, _)| *idx == i)
                .map(|(_, o, _)| Addr(*o))
        })
        .unwrap_or(Addr::NULL);

    if bodies.is_empty() {
        warnings.push("module contains no code bodies (library-only module)".into());
    }
    warnings.push(format!(
        "{} imported function(s), {} defined, {} exported, {} data segment(s), {} memories",
        imported_funcs,
        defined_funcs,
        exports.len(),
        data_segments,
        memories
    ));

    // The signature of every function, by function index: the imported ones
    // first, then the defined ones. This is the index space a `call`
    // names, so it is also what tells a call site how many arguments the
    // callee takes.
    let mut wasm_funcs: Vec<WasmFuncSig> =
        Vec::with_capacity(imported_func_types.len() + defined_func_types.len());
    for type_index in &imported_func_types {
        wasm_funcs.push(signature_of(*type_index, &func_types, None));
    }
    for (i, type_index) in defined_func_types.iter().enumerate() {
        let body = bodies.get(i).map(|(_, offset, _)| Addr(*offset));
        wasm_funcs.push(signature_of(*type_index, &func_types, body));
    }

    let mut fp = Fingerprint::unknown(id);
    fp.linkage = if imported_funcs > 0 {
        Linkage::Shared
    } else {
        Linkage::Static
    };
    fp.stripped = names.is_empty() && !symbols.is_empty();

    Ok(Image {
        id,
        path: path.to_string(),
        format: Format::Wasm,
        arch: Arch::Wasm,
        bits: 32,
        endian: Endian::Little,
        entry,
        image_base: Addr::NULL,
        file_size: data.len() as u64,
        sha256: crate::sha256::hex_digest(data),
        segments: Vec::new(),
        sections,
        resources: Vec::new(),
        symbols,
        imports,
        exports,
        relocations: Vec::new(),
        debug_relocations: Vec::new(),
        fingerprint: fp,
        interpreter: None,
        warnings,
        function_bounds: Vec::new(),
        arm_modes: Default::default(),
        wasm_types: func_types,
        wasm_funcs,
    })
}

/// The signature of a function whose type index is known.
///
/// A type index that names nothing is a module this parser could not read;
/// the counts fall back to zero rather than to a guess.
fn signature_of(type_index: u32, types: &[WasmFuncType], body: Option<Addr>) -> WasmFuncSig {
    let ty = types.get(type_index as usize);
    WasmFuncSig {
        type_index,
        params: ty.map(|t| t.params.len() as u32).unwrap_or(0),
        results: ty.map(|t| t.results.len() as u32).unwrap_or(0),
        body,
    }
}

/// Read a vector of value types.
fn value_types(data: &[u8], pos: &mut usize, count: u32) -> Option<Vec<WasmValueType>> {
    let mut out = Vec::with_capacity(count.min(1024) as usize);
    for _ in 0..count.min(1_000_000) {
        let byte = *data.get(*pos)?;
        *pos += 1;
        // A type this parser does not know is kept as a reference rather than
        // dropped: dropping it would shift every type after it.
        out.push(WasmValueType::from_byte(byte).unwrap_or(WasmValueType::ExternRef));
    }
    Some(out)
}

fn section_name(id: u8) -> &'static str {
    match id {
        0 => "custom",
        1 => "type",
        2 => "import",
        3 => "function",
        4 => "table",
        5 => "memory",
        6 => "global",
        7 => "export",
        8 => "start",
        9 => "element",
        10 => "code",
        11 => "data",
        12 => "datacount",
        _ => "unknown",
    }
}

fn leb_u32(data: &[u8], pos: &mut usize) -> Option<u32> {
    let mut result = 0u32;
    let mut shift = 0u32;
    loop {
        let b = *data.get(*pos)?;
        *pos += 1;
        result |= u32::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 35 {
            return None;
        }
    }
}

fn read_str(data: &[u8], pos: &mut usize, len: usize) -> Option<String> {
    let end = pos.checked_add(len)?;
    let slice = data.get(*pos..end)?;
    *pos = end;
    Some(String::from_utf8_lossy(slice).into_owned())
}

/// Parse the `name` custom section: subsection 1 maps function index to name.
fn parse_name_section(data: &[u8], pos: &mut usize, end: usize, names: &mut HashMap<u32, String>) {
    while *pos < end {
        let Some(id) = data.get(*pos).copied() else {
            return;
        };
        *pos += 1;
        let Some(size) = leb_u32(data, pos) else {
            return;
        };
        let sub_end = (*pos).saturating_add(size as usize).min(end);
        if id == 1 {
            let Some(count) = leb_u32(data, pos) else {
                return;
            };
            for _ in 0..count.min(1_000_000) {
                let Some(idx) = leb_u32(data, pos) else {
                    return;
                };
                let Some(len) = leb_u32(data, pos) else {
                    return;
                };
                let Some(name) = read_str(data, pos, len as usize) else {
                    return;
                };
                names.insert(idx, name);
                if *pos >= sub_end {
                    break;
                }
            }
        }
        *pos = sub_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"\0asm");
        d.extend_from_slice(&1u32.to_le_bytes());
        // type section: 1 type, () -> ()
        d.extend_from_slice(&[1, 4, 1, 0x60, 0, 0]);
        // function section: 1 function of type 0
        d.extend_from_slice(&[3, 2, 1, 0]);
        // export section: "run" func 0
        let mut ex = vec![1u8, 3, b'r', b'u', b'n', 0, 0];
        d.push(7);
        d.push(ex.len() as u8);
        d.append(&mut ex);
        // code section: one body of 4 bytes: 0 locals, end
        d.extend_from_slice(&[10, 6, 1, 4, 0, 0x0b]);
        // pad so the body has content
        d.push(0);
        d
    }

    #[test]
    fn parses_sections_and_exports() {
        let img = parse(&module(), "t.wasm", ImageId(0)).expect("parse");
        assert_eq!(img.format, Format::Wasm);
        assert_eq!(img.arch, Arch::Wasm);
        assert!(img.arch.is_disassemblable());
        assert!(img.sections.iter().any(|s| s.name == "code"));
        assert_eq!(img.exports.len(), 1);
        assert_eq!(img.exports[0].name, "run");
        assert_eq!(img.symbols.len(), 1);
        assert_eq!(img.symbols[0].name, "func_0");
        // No start section, so there is no entry point; the export still
        // resolves to the body offset, which is what a caller can hexdump.
        assert_eq!(img.entry, Addr::NULL);
        assert_eq!(img.exports[0].addr, img.symbols[0].addr);
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        assert!(parse(b"nope", "t", ImageId(0)).is_err());
        let mut d = module();
        d[4] = 2;
        assert!(parse(&d, "t", ImageId(0)).is_err());
    }

    #[test]
    fn truncated_module_does_not_panic() {
        let full = module();
        for cut in 0..full.len() {
            let _ = parse(&full[..cut], "t", ImageId(0));
        }
    }
}
