//! Checks against a real system binary.
//!
//! Every assertion here was cross-checked against `readelf` before being
//! written down. The numbers are not "whatever the parser produced": they are
//! the values `binutils` reports for the same file, so a regression in table
//! stride or offset shows up as a mismatch rather than as a silently different
//! but self-consistent result.
//!
//! Skipped when `/bin/ls` is absent or is not a dynamically linked ELF.

use occ_core::{ImageId, ImportKind, RelocKind, SectionKind};

fn load_ls() -> Option<occ_core::Image> {
    let data = std::fs::read("/bin/ls").ok()?;
    if occ_bin::probe_format(&data) != occ_core::Format::Elf {
        return None;
    }
    Some(occ_bin::load(&data, "/bin/ls", ImageId(0)).expect("parse /bin/ls"))
}

#[test]
fn header_fields_match_readelf() {
    let Some(img) = load_ls() else { return };
    assert_eq!(img.bits, 64);
    assert_eq!(img.arch, occ_core::Arch::X86_64);
    assert_eq!(img.sections.len(), 31);
    assert_eq!(img.entry.get(), 0x6d30);
    assert_eq!(img.image_base.get(), 0);
    assert_eq!(
        img.interpreter.as_deref(),
        Some("/lib64/ld-linux-x86-64.so.2")
    );
}

#[test]
fn relocation_count_matches_readelf() {
    let Some(img) = load_ls() else { return };
    // `readelf -r`: .rela.dyn 226 entries, .rela.plt 104.
    assert_eq!(img.relocations.len(), 330);
}

#[test]
fn first_plt_relocation_is_bound_to_the_right_symbol() {
    let Some(img) = load_ls() else { return };
    let r = img
        .relocations
        .iter()
        .find(|r| r.offset.get() == 0x22c50)
        .expect("PLT slot at 0x22c50");
    assert_eq!(r.kind, RelocKind::Got);
    assert_eq!(r.symbol.as_deref(), Some("__ctype_toupper_loc"));
}

#[test]
fn dynamic_imports_are_present_and_version_stripped() {
    let Some(img) = load_ls() else { return };
    assert!(img.imports.len() > 100, "got {}", img.imports.len());
    assert!(
        img.imports.iter().any(|i| i.name == "getenv"),
        "getenv missing"
    );
    assert!(
        !img.imports.iter().any(|i| i.name.contains('@')),
        "version suffix survived"
    );
    assert!(
        img.imports.iter().any(|i| i.kind == ImportKind::Data),
        "a data import such as __progname should be classified as data"
    );
}

#[test]
fn section_kinds_are_classified() {
    let Some(img) = load_ls() else { return };
    let text = img.section_by_name(".text").expect(".text");
    assert_eq!(text.kind, SectionKind::Code);
    assert!(text.perm.exec());

    let rodata = img.section_by_name(".rodata").expect(".rodata");
    assert_eq!(rodata.kind, SectionKind::Rodata);
    assert!(!rodata.perm.write());

    let dynsym = img.section_by_name(".dynsym").expect(".dynsym");
    assert_eq!(dynsym.kind, SectionKind::Other);

    let hash = img.section_by_name(".gnu.hash").expect(".gnu.hash");
    assert_eq!(hash.kind, SectionKind::Other);

    let bss = img.section_by_name(".bss").expect(".bss");
    assert_eq!(bss.kind, SectionKind::Bss);
    assert_eq!(bss.file_size, 0, ".bss occupies no file space");
}

#[test]
fn non_allocated_sections_have_no_address() {
    let Some(img) = load_ls() else { return };
    for s in &img.sections {
        if s.name.starts_with(".debug") || s.name == ".comment" {
            assert_eq!(
                s.addr.get(),
                0,
                "{} is not mapped and must not claim an address",
                s.name
            );
            assert_eq!(s.perm, occ_core::Perm::NONE);
        }
    }
}

#[test]
fn truncated_input_never_panics() {
    let Ok(data) = std::fs::read("/bin/ls") else {
        return;
    };
    for cut in [4, 16, 0x34, 0x40, 0x100, 0x1000, data.len() / 2] {
        let _ = occ_bin::load(&data[..cut.min(data.len())], "cut", ImageId(0));
    }
}

#[test]
fn eh_frame_states_function_extents() {
    let Some(img) = load_ls() else {
        eprintln!("no /bin/ls on this host");
        return;
    };
    let bounds: Vec<_> = img
        .function_bounds
        .iter()
        .filter(|b| b.source == "eh_frame")
        .collect();
    assert!(
        bounds.len() > 50,
        "a dynamically linked binary states its functions in .eh_frame, got {}",
        bounds.len()
    );
    for b in &bounds {
        assert!(b.end.0 > b.start.0, "{:#x} has no extent", b.start.0);
        // An FDE covers code. It may be the PLT rather than `.text` — the
        // dynamic linker needs unwind rows for the stubs too — so the check
        // is that the start lands in a section the loader may execute.
        let section = img
            .sections
            .iter()
            .filter(|s| s.perm.exec())
            .find(|s| b.start.0 >= s.addr.0 && b.start.0 < s.addr.0 + s.size);
        assert!(section.is_some(), "{:#x} is outside every code section", b.start.0);
    }
}
