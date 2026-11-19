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

/// Run `readelf` on the target and return its stdout.
///
/// The tests below used to carry the numbers from one machine's `/bin/ls`.
/// Those numbers are a property of the host's binutils build, not of this
/// parser, so a different distribution failed the suite while the parser was
/// correct. Reading the expected values from `readelf` at run time makes the
/// comparison portable and keeps it a comparison rather than a snapshot.
fn readelf(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("readelf")
        .args(args)
        .arg("/bin/ls")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

fn field<'a>(text: &'a str, label: &str) -> Option<&'a str> {
    text.lines()
        .find_map(|l| l.trim_start().strip_prefix(label))
        .map(|v| v.trim())
}

fn parse_hex(s: &str) -> Option<u64> {
    let s = s.trim();
    let s = s.split_whitespace().next().unwrap_or(s);
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

#[test]
fn header_fields_match_readelf() {
    let Some(img) = load_ls() else { return };
    let Some(header) = readelf(&["-h"]) else { return };

    let class = field(&header, "Class:").unwrap_or("");
    assert_eq!(img.bits, if class.contains("ELF64") { 64 } else { 32 });
    assert_eq!(
        img.arch,
        if img.bits == 64 {
            occ_core::Arch::X86_64
        } else {
            occ_core::Arch::X86
        },
        "class {class}"
    );

    let entry = parse_hex(field(&header, "Entry point address:").unwrap_or("")).expect("entry");
    assert_eq!(img.entry.get(), entry);

    let sections: usize = field(&header, "Number of section headers:")
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .expect("section count");
    assert_eq!(img.sections.len(), sections);

    // The interpreter is only printed by `-l`, and only for a dynamically
    // linked executable.
    if let Some(programs) = readelf(&["-l"]) {
        if let Some(line) = programs
            .lines()
            .find(|l| l.contains("Requesting program interpreter:"))
        {
            let expected = line
                .split("interpreter:")
                .nth(1)
                .map(|v| v.trim().trim_end_matches(']'))
                .unwrap_or("");
            assert_eq!(img.interpreter.as_deref(), Some(expected));
        }
    }

    // The image base is the lowest p_vaddr of a PT_LOAD, which `readelf -l`
    // prints in the first column of the LOAD rows.
    if let Some(programs) = readelf(&["-l"]) {
        let lowest = programs
            .lines()
            .skip_while(|l| !l.contains("Type"))
            .filter(|l| l.trim_start().starts_with("LOAD"))
            // `LOAD 0x<offset> 0x<vaddr> ...`: the image base is the vaddr,
            // not the file offset.
            .filter_map(|l| l.split_whitespace().nth(2))
            .filter_map(parse_hex)
            .min();
        if let Some(base) = lowest {
            assert_eq!(img.image_base.get(), base);
        }
    }
}

#[test]
fn relocation_count_matches_readelf() {
    let Some(img) = load_ls() else { return };
    let Some(text) = readelf(&["-r"]) else { return };
    // One data line per relocation: sixteen hex digits, then the addend or
    // type columns. Header lines ("Relocation section ...") and blank lines
    // do not start with an address.
    let expected = text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            if !(t.contains("R_X86_64_") || t.contains("R_386_")) {
                return false;
            }
            t.split_whitespace()
                .next()
                .is_some_and(|off| off.len() >= 8 && off.chars().all(|c| c.is_ascii_hexdigit()))
        })
        .count();
    assert_eq!(img.relocations.len(), expected);
}

#[test]
fn plt_relocations_are_bound_to_the_symbol_readelf_names() {
    let Some(img) = load_ls() else { return };
    let Some(text) = readelf(&["-r"]) else { return };
    // A JUMP_SLOT is the PLT's binding: the dynamic linker writes the
    // resolved address there. Pick the first one readelf reports and demand
    // the same offset, kind and name from the parser.
    let mut checked = 0;
    let mut in_plt = false;
    for line in text.lines() {
        if line.contains("Relocation section") {
            in_plt = line.contains(".rela.plt") || line.contains(".rel.plt");
            continue;
        }
        if !in_plt {
            continue;
        }
        let t = line.trim_start();
        let mut fields = t.split_whitespace();
        let Some(off_token) = fields.next() else { continue };
        if !off_token.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let offset = parse_hex(off_token).expect("offset");
        // `Offset Info Type Sym.Value Sym.Name + Addend`: the name carries a
        // version suffix (`getenv@GLIBC_2.2.5`), which this parser strips, so
        // compare against the part before the `@`.
        let name = t.split_whitespace().nth(4).unwrap_or("").split('@').next().unwrap_or("");
        let Some(r) = img.relocations.iter().find(|r| r.offset.get() == offset) else {
            panic!("parser has no relocation at {offset:#x}");
        };
        assert_eq!(r.symbol.as_deref(), Some(name), "at {offset:#x}");
        assert_eq!(r.kind, RelocKind::Got, "a PLT slot is a GOT write");
        checked += 1;
        if checked == 8 {
            break;
        }
    }
    assert!(checked > 0, "readelf reported no .rela.plt entries");
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
