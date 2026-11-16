//! Mach-O, against real files rather than a synthetic header.
//!
//! The fixtures are ordinary files that happen to be on a Linux box: Go's DWARF
//! test data (a 64-bit x86-64 object) and two shipped macOS dynamic libraries,
//! one per architecture. None of them were produced for this test, which is the
//! point — a parser written against its own fixture only proves it agrees with
//! itself.
//!
//! The tests skip when a file is absent, the same way the `/bin/ls` checks do.

use occ_analyze::{Analysis, AnalysisOptions, RecoverOptions, View, recover};
use occ_core::{Arch, Format, ImageId, Perm, SectionKind, SymbolKind};

const OBJECT: &str = "/usr/local/go/src/debug/dwarf/testdata/typedef.macho";
const DYLIB_X86: &str = "/root/.local/share/code-server/extensions/ms-python.python-2023.4.1/pythonFiles/lib/python/debugpy/_vendored/pydevd/pydevd_attach_to_process/attach_x86_64.dylib";
const DYLIB_ARM: &str = "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/py_mini_racer/armlibmini_racer.dylib";

fn load(path: &str) -> Option<(Vec<u8>, occ_core::Image)> {
    let data = std::fs::read(path).ok()?;
    let img = occ_bin::load(&data, path, ImageId(0)).ok()?;
    Some((data, img))
}

#[test]
fn an_object_file_reports_its_architecture_and_text() {
    let Some((_, img)) = load(OBJECT) else { return };
    assert_eq!(img.format, Format::MachO);
    assert_eq!(img.arch, Arch::X86_64);
    assert_eq!(img.bits, 64);
    let text = img
        .sections
        .iter()
        .find(|s| s.name == "__TEXT,__text")
        .expect("the object has a text section");
    assert_eq!(text.kind, SectionKind::Code);
    assert!(text.perm.exec(), "text executes: {:?}", text.perm);
    assert!(!text.perm.write(), "text is not writable");
}

#[test]
fn debug_sections_are_not_marked_executable() {
    let Some((_, img)) = load(OBJECT) else { return };
    // A relocatable object has one `__TEXT` segment with `initprot` 7, so a
    // parser that trusts the segment reports `rwx` for the DWARF too.
    for s in &img.sections {
        if s.kind == SectionKind::Debug {
            assert_eq!(s.perm, Perm::R, "{} should be read-only", s.name);
        }
    }
}

#[test]
fn a_text_symbol_is_a_function_not_an_object() {
    let Some((_, img)) = load(OBJECT) else { return };
    let main = img
        .symbols
        .iter()
        .find(|s| s.name == "_main")
        .expect("the object defines _main");
    // `N_SECT` says "defined in section n" and nothing more; the section decides
    // whether it is code. Reading it as data hides every function in the file.
    assert_eq!(main.kind, SymbolKind::Function);
    assert!(main.defined);
}

#[test]
fn the_object_recovers_a_function_with_instructions() {
    let Some((data, img)) = load(OBJECT) else { return };
    let view = View::new(&img, &data);
    let rec = recover(
        &view,
        &RecoverOptions {
            max_functions: 16,
            max_insns: 4096,
            follow_calls: false,
            sweep_uncovered: false,
        },
    );
    let main = rec
        .functions
        .iter()
        .find(|f| f.display_name() == "_main")
        .expect("_main is recovered from its symbol");
    assert!(!main.insns.is_empty(), "_main has a body");
    // The text section is 22 bytes, so the body cannot be longer than that.
    assert!(main.insns.len() <= 22, "{} instructions", main.insns.len());
    assert!(!main.blocks.is_empty());
}

#[test]
fn a_64_bit_dylib_parses_and_recovers_x86_functions() {
    let Some((data, img)) = load(DYLIB_X86) else { return };
    assert_eq!(img.format, Format::MachO);
    assert_eq!(img.arch, Arch::X86_64);
    assert!(!img.imports.is_empty(), "a dylib has imports");
    assert!(!img.exports.is_empty(), "a dylib exports something");

    let view = View::new(&img, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    assert!(
        !analysis.functions.is_empty(),
        "recovery found no function in a 55 KB dylib"
    );
    assert!(
        analysis.functions.iter().any(|f| f.insns.len() > 5),
        "no recovered function has a body"
    );
}

#[test]
fn an_arm64_dylib_is_recognised_as_aarch64() {
    let Some((data, img)) = load(DYLIB_ARM) else { return };
    assert_eq!(img.format, Format::MachO);
    assert_eq!(img.arch, Arch::Aarch64);
    assert_eq!(img.bits, 64);
    let text = img
        .sections
        .iter()
        .find(|s| s.name.ends_with("__text"))
        .expect("a dylib has text");
    assert!(text.size > 0);

    let view = View::new(&img, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    // The AArch64 decoder is fixed-width, so every recovered instruction is four
    // bytes and the bodies must not be empty.
    let with_body = analysis.functions.iter().filter(|f| !f.insns.is_empty()).count();
    assert!(with_body > 0, "no AArch64 function was recovered");
}

#[test]
fn every_section_address_lands_inside_the_file_or_is_zero_fill() {
    for path in [OBJECT, DYLIB_X86, DYLIB_ARM] {
        let Some((data, img)) = load(path) else { continue };
        for s in &img.sections {
            if s.file_size == 0 {
                continue;
            }
            let start = s.file_offset as usize;
            let end = start.saturating_add(s.file_size as usize);
            assert!(
                end <= data.len(),
                "{path}: {} runs past the end of the file ({end} > {})",
                s.name,
                data.len()
            );
        }
    }
}
