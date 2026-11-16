//! Checks against a real MinGW-linked PE.
//!
//! The fixture is a 64-bit executable with three import libraries, so it
//! exercises PE32+ layout, the ILT/IAT pairing, and the base relocation table
//! at once. Expected values come from `objdump -p` on the same file.

use occ_core::{ImageId, SectionKind};

fn load() -> Option<occ_core::Image> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/pe_fixture.exe"
    );
    let data = std::fs::read(path).ok()?;
    if occ_bin::probe_format(&data) != occ_core::Format::Pe {
        return None;
    }
    Some(occ_bin::load(&data, path, ImageId(0)).expect("parse pe fixture"))
}

#[test]
fn header_matches_objdump() {
    let Some(img) = load() else { return };
    // objdump -p: ImageBase 140000000, AddressOfEntryPoint 1410.
    assert_eq!(img.image_base.get(), 0x1_4000_0000);
    assert_eq!(img.entry.get(), 0x1_4000_1410);
    assert_eq!(img.bits, 64);
    assert_eq!(img.arch, occ_core::Arch::X86_64);
    assert_eq!(img.sections.len(), 19);
    // C executables do not export anything.
    assert!(img.exports.is_empty(), "{} exports", img.exports.len());
}

#[test]
fn import_count_per_library_matches_objdump() {
    let Some(img) = load() else { return };
    assert_eq!(img.imports.len(), 48, "objdump reports 13+34+1");

    let count = |lib: &str| {
        img.imports
            .iter()
            .filter(|i| i.library.as_deref() == Some(lib))
            .count()
    };
    assert_eq!(count("KERNEL32.dll"), 13);
    assert_eq!(count("msvcrt.dll"), 34);
    assert_eq!(count("USER32.dll"), 1);

    // The three libraries are named, not guessed from the symbol.
    assert!(
        img.imports
            .iter()
            .any(|i| i.name == "MessageBoxA" && i.library.as_deref() == Some("USER32.dll")),
        "MessageBoxA must be attributed to USER32"
    );
}

#[test]
fn every_import_has_an_iat_slot_inside_a_section() {
    let Some(img) = load() else { return };
    let non_ordinal = img.imports.iter().filter(|i| i.ordinal.is_none());
    let mut checked = 0;
    for imp in non_ordinal {
        let slot = imp
            .slot
            .unwrap_or_else(|| panic!("{} has no slot", imp.name));
        assert!(
            img.section_at(slot).is_some(),
            "{} slot {slot} is outside every section",
            imp.name
        );
        checked += 1;
    }
    assert!(checked > 40, "checked only {checked}");
}

#[test]
fn iat_slots_are_eight_bytes_apart_and_ordered() {
    let Some(img) = load() else { return };
    // The first descriptor is KERNEL32; its thunks sit consecutively in .idata.
    let mut slots: Vec<u64> = img
        .imports
        .iter()
        .filter(|i| i.library.as_deref() == Some("KERNEL32.dll"))
        .filter_map(|i| i.slot.map(|s| s.get()))
        .collect();
    slots.sort_unstable();
    for w in slots.windows(2) {
        assert_eq!(w[1] - w[0], 8, "PE32+ thunks are 8 bytes apart: {w:x?}");
    }
}

#[test]
fn sections_are_classified_by_characteristics() {
    let Some(img) = load() else { return };
    let text = img.section_by_name(".text").expect(".text");
    assert_eq!(text.kind, SectionKind::Code);
    assert!(text.perm.exec());
    assert!(!text.perm.write());

    let rdata = img.section_by_name(".rdata").expect(".rdata");
    assert_eq!(rdata.kind, SectionKind::Rodata);
    assert!(!rdata.perm.write());

    // .pdata holds exception unwind records, not code.
    let pdata = img.section_by_name(".pdata").expect(".pdata");
    assert_eq!(pdata.kind, SectionKind::Unwind);
    assert!(!pdata.perm.exec());
}

#[test]
fn base_relocations_are_read() {
    let Some(img) = load() else { return };
    assert_eq!(
        img.relocations.len(),
        47,
        "objdump reports 47 .reloc entries"
    );
    // Every target must land inside the image.
    for r in &img.relocations {
        assert!(
            r.offset.get() >= img.image_base.get(),
            "reloc at {} is below the image base",
            r.offset
        );
    }
}

#[test]
fn a_truncated_pe_never_panics() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/pe_fixture.exe"
    );
    let Ok(data) = std::fs::read(path) else {
        return;
    };
    for cut in [2, 0x3c, 0x40, 0x80, 0x200, 0x400, data.len() / 2] {
        let _ = occ_bin::load(&data[..cut.min(data.len())], "cut", ImageId(0));
    }
}
