//! Relocatable objects, built by the system compilers at test time.
//!
//! An object is the case where a container's addresses mean nothing: `sh_addr`
//! is zero for every section and a symbol's value is an offset within one. A
//! loader that copies those numbers verbatim puts every section at address 0,
//! every function at address 0, and the whole file looks like a single unmapped
//! byte range. These tests compile one known source three ways and check that
//! both functions survive with their names and distinct addresses.
//!
//! Each test skips when its compiler is missing, the same way the `/bin/ls`
//! checks do.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, Format, Function, Image, SectionKind, SymbolKind};
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE: &str = r#"
int counter = 3;
static int helper(int x) { return x * 2 + counter; }
int entry_point(int a, int b) { return helper(a) + b; }
"#;

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("occ-objects-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    occ_sys::cleanup::remove_dir_at_exit(&dir);
    dir
}

/// Compile `SOURCE` with `tool` and the given flags.
///
/// `None` only when the compiler is absent. A compiler that is present but
/// fails is a failure: a fixture that silently skips leaves the feature
/// untested and the suite green.
fn compile(tool: &str, flags: &[&str], out: &str) -> Option<PathBuf> {
    let dir = scratch();
    let src = dir.join("unit.c");
    std::fs::write(&src, SOURCE).expect("write source");
    let obj = dir.join(out);
    let output = Command::new(tool)
        .args(flags)
        .arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .output()
        .ok()?;
    assert!(
        output.status.success(),
        "{tool} must compile the fixture:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(obj)
}

fn load(path: &Path) -> (Vec<u8>, Image) {
    let data = std::fs::read(path).expect("read object");
    let img = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse object");
    (data, img)
}

/// Every recovered function whose name matches, ignoring a leading underscore:
/// Mach-O prefixes C symbols with one, ELF and COFF do not.
fn named<'a>(analysis: &'a Analysis, name: &str) -> Vec<&'a Function> {
    analysis
        .functions
        .iter()
        .filter(|f| f.display_name().trim_start_matches('_') == name)
        .collect()
}

fn check(path: &Path, functions: &[&str]) -> (Image, usize) {
    let (data, img) = load(path);
    let view = View::new(&img, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    for name in functions {
        let found = named(&analysis, name);
        assert_eq!(
            found.len(),
            1,
            "{name} should be recovered once, got {:#?}",
            analysis
                .functions
                .iter()
                .map(|f| f.display_name())
                .collect::<Vec<_>>()
        );
        let f = found[0];
        assert!(!f.insns.is_empty(), "{name} has a body");
    }
    // Distinct addresses: the bug this guards against reported every function
    // at offset zero because symbol values were read as addresses.
    let mut addrs: Vec<Addr> = analysis
        .functions
        .iter()
        .filter(|f| functions.iter().any(|n| f.display_name().trim_start_matches('_') == *n))
        .map(|f| f.entry)
        .collect();
    addrs.sort_by_key(|a| a.0);
    addrs.dedup();
    assert_eq!(addrs.len(), functions.len(), "addresses collide: {addrs:?}");
    for a in &addrs {
        assert!(!a.is_null(), "function at the null address");
        assert!(view.is_exec(*a), "{a:?} is not inside executable bytes");
    }
    (img, analysis.functions.len())
}

#[test]
fn an_elf_object_yields_both_functions() {
    let Some(path) = compile("cc", &["-O0", "-fno-inline", "-g"], "unit.o") else {
        return;
    };
    let (img, total) = check(&path, &["entry_point", "helper"]);
    assert_eq!(img.format, Format::Elf);
    // No loadable segments: an object is not something the kernel maps, and
    // saying otherwise would imply a load address that does not exist.
    assert!(img.segments.is_empty(), "an object has no segments");
    assert!(total >= 2);
    assert!(
        img.symbols
            .iter()
            .any(|s| s.name == "helper" && s.kind == SymbolKind::Function),
        "the object's symbols include the static function"
    );
}

#[test]
fn a_coff_object_yields_its_functions() {
    let Some(path) = compile(
        "clang",
        &["--target=x86_64-pc-windows-msvc", "-O0"],
        "unit.obj",
    ) else {
        return;
    };
    let (img, _) = check(&path, &["entry_point", "helper"]);
    assert_eq!(img.format, Format::Coff);
    assert!(img.segments.is_empty(), "an object has no segments");
    // COFF marks code by section characteristics, not by the symbol kind, so
    // the section classification is what proves the text is readable.
    let text = img
        .sections
        .iter()
        .find(|s| s.name == ".text")
        .expect("a COFF object has .text");
    assert_eq!(text.kind, SectionKind::Code);
    assert!(text.perm.exec());
}

#[test]
fn a_macho_object_yields_its_functions() {
    let Some(path) = compile("clang", &["--target=x86_64-apple-macos11", "-O0"], "unit.mo") else {
        return;
    };
    let (img, total) = check(&path, &["entry_point", "helper"]);
    assert_eq!(img.format, Format::MachO);
    assert!(img.segments.is_empty(), "an object has no segments");
    assert!(total >= 2, "only {total} function(s) recovered");
    let text = img
        .sections
        .iter()
        .find(|s| s.name.ends_with("__text"))
        .expect("a Mach-O object has text");
    assert_eq!(text.kind, SectionKind::Code);
    // `vmaddr` is zero for an object; the address reported is the file offset,
    // which has to land inside the file or every read from it fails.
    assert!(
        text.addr.get() + text.size <= img.file_size,
        "text at {:#x} + {:#x} is past the {} byte file",
        text.addr.get(),
        text.size,
        img.file_size
    );
}
