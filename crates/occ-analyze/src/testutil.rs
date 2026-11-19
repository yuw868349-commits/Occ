//! Shared fixtures for unit tests. Compiled only under `cfg(test)`.

use occ_core::*;

/// One readable data section at `0x1000`, backed by `data`.
pub fn image_with_data(data: &[u8]) -> Image {
    image_with_sections(vec![Section {
        index: 1,
        name: ".rodata".into(),
        addr: Addr(0x1000),
        file_offset: 0,
        size: data.len() as u64,
        file_size: data.len() as u64,
        kind: SectionKind::Rodata,
        perm: Perm::R,
    }])
}

/// An image carrying exactly the supplied sections.
pub fn image_with_sections(sections: Vec<Section>) -> Image {
    Image {
        id: ImageId(0),
        path: "test".into(),
        format: Format::Elf,
        arch: Arch::X86_64,
        bits: 64,
        endian: Endian::Little,
        entry: Addr(0x1000),
        image_base: Addr(0x1000),
        file_size: 0,
        sha256: String::new(),
        segments: Vec::new(),
        sections,
        resources: Vec::new(),
        symbols: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        relocations: Vec::new(),
        debug_relocations: Vec::new(),
        fingerprint: Fingerprint::unknown(ImageId(0)),
        interpreter: None,
        warnings: Vec::new(),
        function_bounds: Vec::new(),
        arm_modes: Default::default(),
        wasm_types: Default::default(),
        wasm_funcs: Default::default(),
    }
}

/// A code section at `0x1000` backed by `code`.
pub fn image_with_code(code: &[u8]) -> Image {
    image_with_sections(vec![Section {
        index: 1,
        name: ".text".into(),
        addr: Addr(0x1000),
        file_offset: 0,
        size: code.len() as u64,
        file_size: code.len() as u64,
        kind: SectionKind::Code,
        perm: Perm::R.union(Perm::X),
    }])
}
