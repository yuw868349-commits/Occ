//! DOS `.COM` images.
//!
//! A `.COM` has no container: the file *is* the image. DOS loads it at
//! offset `0x100` of a single segment and jumps to the first byte, so every
//! address in the file is `file_offset + 0x100`, the code is 16-bit real
//! mode, and the 64 KiB segment size is the only structural limit the
//! format has.
//!
//! Nothing in the bytes says "this is a COM file": the same bytes are a
//! valid sequence of 16-bit instructions in any file. The extension is
//! therefore part of the format definition, not a convenience — the
//! discriminator is "named `.com`, fits in one segment, and matches no other
//! container", which is exactly what DOS itself relies on.

use occ_core::{
    Addr, Arch, CoreError, Endian, Fingerprint, Format, Image, ImageId, Linkage, Perm, Result,
    Section, SectionKind,
};

/// The load address DOS gives a COM image.
pub const LOAD_ADDRESS: u64 = 0x100;

/// Largest image a COM file can be: one 64 KiB segment, minus the PSP.
pub const MAX_SIZE: usize = 0x1_0000 - LOAD_ADDRESS as usize;

/// True when `path` names a COM file and `data` could be one.
///
/// The size limit is not cosmetic: a file larger than the segment cannot be
/// loaded by DOS at `0x100`, so treating it as COM would decode addresses
/// that the format cannot represent.
pub fn looks_like_com(path: &str, data: &[u8]) -> bool {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    !data.is_empty() && ext.eq_ignore_ascii_case("com") && data.len() <= MAX_SIZE
}

/// Parse a COM image into the common [`Image`] shape.
pub fn parse(data: &[u8], path: &str, id: ImageId) -> Result<Image> {
    if data.len() > MAX_SIZE {
        return Err(CoreError::UnsupportedFormat(format!(
            "{path}: {} bytes exceeds the 64 KiB segment a COM image loads into",
            data.len()
        )));
    }
    let size = data.len() as u64;
    let mut fingerprint = Fingerprint::unknown(id);
    fingerprint.linkage = Linkage::Executable;

    Ok(Image {
        id,
        path: path.to_string(),
        format: Format::Com,
        arch: Arch::X86,
        bits: 16,
        endian: Endian::Little,
        entry: Addr(LOAD_ADDRESS),
        image_base: Addr(0),
        file_size: size,
        sha256: crate::sha256::hex_digest(data),
        segments: Vec::new(),
        sections: vec![Section {
            index: 1,
            // There are no sections in the format. One section names the
            // whole image so that every address-driven pass has a range to
            // work with; the name says what it is rather than inventing a
            // `.text` the file does not have.
            name: ".com".into(),
            addr: Addr(LOAD_ADDRESS),
            file_offset: 0,
            size,
            file_size: size,
            kind: SectionKind::Code,
            perm: Perm::R.union(Perm::X),
        }],
        symbols: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        relocations: Vec::new(),
        debug_relocations: Vec::new(),
        fingerprint,
        interpreter: None,
        warnings: vec![
            "COM images carry no header: all bytes are mapped at 0x100 and code and data share the segment, so a disassembly cannot tell them apart without following the code"
                .into(),
        ],
        function_bounds: Vec::new(),
        arm_modes: Default::default(),
        wasm_types: Default::default(),
        wasm_funcs: Default::default(),
    })
}
