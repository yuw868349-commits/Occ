//! Binary container parsing, with no external dependencies.
//!
//! One entry point, [`load`], and one output type, [`occ_core::Image`].
//! Format detection is by magic bytes, with one exception that the format
//! itself forces: a DOS `.COM` has no header at all, so its extension and
//! its size are the only discriminators that exist, and
//! [`com::looks_like_com`] applies both.
//!
//! Modules are split by container rather than by task, because the containers
//! share almost nothing: ELF has a section table and a program header table,
//! PE has a data-directory array, Mach-O has load commands, and WASM has a
//! section sequence. A unified reader would have to abstract away the very
//! differences each parser exists to handle.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod ar;
pub mod com;
pub mod demangle;
pub mod elf;
pub mod macho;
pub mod pe;
pub mod reader;
pub mod reloc;
pub mod sha256;
pub mod wasm;

use occ_core::{CoreError, Format, Image, ImageId, Result};

/// Identify the container format from the leading bytes.
///
/// Ordered from most to least specific. `MZ` is checked before bare COFF
/// because a DOS stub that carries a PE signature must not be mistaken for a
/// raw object file.
pub fn probe_format(data: &[u8]) -> Format {
    if data.len() >= 4 && &data[0..4] == b"\x7fELF" {
        return Format::Elf;
    }
    if data.len() >= 2 && &data[0..2] == b"MZ" {
        // A PE has an `e_lfanew` at 0x3c pointing to its signature.
        if data.len() >= 0x40 {
            let off = u32::from_le_bytes([data[0x3c], data[0x3d], data[0x3e], data[0x3f]]) as usize;
            if data.len() >= off.saturating_add(4) && &data[off..off + 4] == b"PE\0\0" {
                return Format::Pe;
            }
        }
        return Format::Coff;
    }
    if data.len() >= 4
        && matches!(
            data[0..4],
            [0xcf, 0xfa, 0xed, 0xfe] | [0xce, 0xfa, 0xed, 0xfe]
        )
    {
        return Format::MachO;
    }
    if data.len() >= 4 && data[0..4] == [0xca, 0xfe, 0xba, 0xbe] {
        return Format::MachOFat;
    }
    if data.len() >= 4 && &data[0..4] == b"\0asm" {
        return Format::Wasm;
    }
    if data.len() >= 8 && &data[0..8] == b"!<arch>\n" {
        return Format::Archive;
    }
    // A bare COFF object: machine type, then a plausible section count. This is
    // the weakest signal, so it is checked last.
    if data.len() >= 20 {
        let machine = u16::from_le_bytes([data[0], data[1]]);
        let sections = u16::from_le_bytes([data[2], data[3]]);
        if matches!(machine, 0x014c | 0x8664 | 0xaa64 | 0x01c0) && sections > 0 && sections < 96 {
            return Format::Coff;
        }
    }
    Format::Unknown
}

/// Parse `data` into an [`Image`].
///
/// `path` is recorded for display and is never opened: the caller has already
/// decided what bytes to analyse.
pub fn load(data: &[u8], path: &str, id: ImageId) -> Result<Image> {
    match probe_format(data) {
        Format::Elf => elf::parse(data, path, id),
        Format::Pe => pe::parse(data, path, id),
        Format::Coff => pe::parse_coff(data, path, id),
        Format::MachO | Format::MachOFat => macho::parse(data, path, id),
        Format::Wasm => wasm::parse(data, path, id),
        // A COM file has no leading bytes to recognise, so it is decided
        // last: anything a magic check claims is not one.
        Format::Unknown if com::looks_like_com(path, data) => com::parse(data, path, id),
        Format::Unknown => Err(CoreError::UnsupportedFormat(format!(
            "{path}: leading bytes match no known container"
        ))),
        other => Err(CoreError::UnsupportedFormat(format!(
            "{path}: {} is recognised but this build cannot parse it yet",
            other.label()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use occ_core::{Addr, Arch, Linkage};

    #[test]
    fn probe_distinguishes_elf_from_everything_else() {
        assert_eq!(probe_format(b"\x7fELF\x02\x01\x01\x00"), Format::Elf);
        assert_eq!(probe_format(b"\x00asm\x01\x00\x00\x00"), Format::Wasm);
        assert_eq!(probe_format(b"!<arch>\n"), Format::Archive);
        assert_eq!(probe_format(b"hello, world"), Format::Unknown);
        assert_eq!(probe_format(b""), Format::Unknown);
        assert_eq!(probe_format(b"\x7f"), Format::Unknown);
    }

    #[test]
    fn a_com_file_loads_at_one_hundred_and_decodes_as_sixteen_bit() {
        let bytes: Vec<u8> = vec![
            0xe9, 0x02, 0x00, // jmp +2
            0xb8, 0x34, 0x12, // mov ax, 0x1234
            0xcd, 0x21, // int 0x21
        ];
        let img = load(&bytes, "TEST.COM", ImageId(0)).expect("com should load");
        assert_eq!(img.format, Format::Com);
        assert_eq!(img.arch, Arch::X86);
        assert_eq!(img.bits, 16);
        assert_eq!(img.entry, Addr(0x100));
        assert_eq!(img.image_base, Addr(0));
        assert_eq!(img.sections.len(), 1);
        assert_eq!(img.sections[0].addr, Addr(0x100));
        assert_eq!(img.sections[0].size, bytes.len() as u64);
        assert_eq!(img.fingerprint.linkage, Linkage::Executable);
        assert!(img.warnings.iter().any(|w| w.contains("no header")));

        // The same bytes under a name that is not `.com` are not a COM
        // image: the format has no magic, so the name is the discriminator.
        assert!(load(&bytes, "TEST.BIN", ImageId(0)).is_err());
    }

    #[test]
    fn an_oversized_com_is_refused_rather_than_truncated() {
        let bytes = vec![0u8; 0x1_0000];
        assert!(!com::looks_like_com("BIG.COM", &bytes));
        assert!(load(&bytes, "BIG.COM", ImageId(0)).is_err());
    }

    #[test]
    fn a_dos_stub_without_a_pe_signature_is_coff() {
        let mut d = vec![0u8; 0x80];
        d[0] = b'M';
        d[1] = b'Z';
        // e_lfanew points at zeroed bytes, so there is no PE signature.
        d[0x3c..0x40].copy_from_slice(&0x40u32.to_le_bytes());
        assert_eq!(probe_format(&d), Format::Coff);
    }

    #[test]
    fn a_dos_stub_with_a_pe_signature_is_pe() {
        let mut d = vec![0u8; 0x100];
        d[0] = b'M';
        d[1] = b'Z';
        d[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        d[0x80..0x84].copy_from_slice(b"PE\0\0");
        assert_eq!(probe_format(&d), Format::Pe);
    }

    #[test]
    fn an_empty_wasm_module_loads_with_no_code() {
        // The container is now supported; a module with no sections is legal
        // and must load as an image with no functions rather than fail.
        let img = load(b"\x00asm\x01\x00\x00\x00", "m.wasm", ImageId(0)).expect("loads");
        assert_eq!(img.format, Format::Wasm);
        assert_eq!(img.arch, occ_core::Arch::Wasm);
        assert!(img.sections.is_empty());
    }

    #[test]
    fn an_unrecognised_container_names_itself() {
        let err = load(b"not a binary at all", "x", ImageId(0)).unwrap_err();
        assert!(err.to_string().contains("no known container"), "{err}");
    }
}
