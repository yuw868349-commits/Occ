//! A static, stripped executable still has a `main`.
//!
//! Nothing in the file names it: the symbol table is gone and there is no
//! dynamic symbol table to consult. The only record is the immediate the entry
//! point loads into the first argument register before calling the C runtime,
//! and a tool that does not read it reports "no main" for the one function a
//! reader always wants first.
//!
//! The tests compile a small program twice — once with symbols, once stripped —
//! and check that both answer with the same entry address. They skip when the
//! toolchain cannot produce a static binary, the same way the `/bin/ls` checks
//! skip when there is no `/bin/ls`.

use std::path::{Path, PathBuf};
use std::process::Command;

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, Image};

const SOURCE: &str = r#"
#include <stdio.h>
int helper(int x) { return x * 3 + 1; }
int main(int argc, char **argv) {
    int s = 0;
    for (int i = 0; i < argc; i++) s += helper(argv[i][0]);
    printf("%d\n", s);
    return 0;
}
"#;

struct Scratch(PathBuf);

impl Scratch {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch() -> Scratch {
    let dir = std::env::temp_dir().join(format!("occ-static-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch");
    Scratch(dir)
}

/// Compile the fixture, returning the path, or `None` when this toolchain
/// cannot link statically.
fn build(dir: &Path) -> Option<PathBuf> {
    let src = dir.join("unit.c");
    std::fs::write(&src, SOURCE).expect("write source");
    let exe = dir.join("unit");
    let out = Command::new("cc")
        .args(["-O1", "-static", "-o"])
        .arg(&exe)
        .arg(&src)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "skipping: cc cannot link statically here:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    Some(exe)
}

/// Strip a copy of `exe`. `None` when no stripper is installed.
fn stripped(exe: &Path, dir: &Path) -> Option<PathBuf> {
    let copy = dir.join("unit.stripped");
    std::fs::copy(exe, &copy).ok()?;
    for tool in ["llvm-strip", "strip"] {
        let out = Command::new(tool).arg("--strip-all").arg(&copy).output();
        if let Ok(out) = out
            && out.status.success()
        {
            return Some(copy);
        }
    }
    None
}

fn analyse(path: &Path) -> (Vec<u8>, Image, Analysis) {
    let data = std::fs::read(path).expect("read image");
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse image");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    (data, image, analysis)
}

fn main_entry(analysis: &Analysis) -> Option<Addr> {
    analysis
        .functions
        .iter()
        .find(|f| f.display_name() == "main")
        .map(|f| f.entry)
}

#[test]
fn a_stripped_static_binary_still_has_a_main() {
    let dir = scratch();
    let Some(exe) = build(dir.path()) else {
        return;
    };
    let (_d, _i, named) = analyse(&exe);
    let Some(expected) = main_entry(&named) else {
        panic!("the symbol table names main; recovery did not keep it");
    };
    let Some(stripped_exe) = stripped(&exe, dir.path()) else {
        eprintln!("skipping: no stripper installed");
        return;
    };
    let (_d, image, analysis) = analyse(&stripped_exe);

    // The premise of the test: the file really has no names left.
    assert!(
        image.symbols.iter().all(|s| s.name != "main"),
        "the stripped file still names main"
    );
    let found = main_entry(&analysis)
        .unwrap_or_else(|| {
            panic!(
                "main not identified; notes {:?}, functions {:#?}",
                analysis.notes,
                analysis
                    .functions
                    .iter()
                    .take(8)
                    .map(|f| f.display_name())
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(
        found, expected,
        "the stripped binary's main is at a different address than the named one"
    );
}

#[test]
fn stripping_does_not_halve_the_function_count() {
    // The report was that a stripped static sample recovers under half of its
    // functions. The number cannot match exactly — a stripped image has no
    // symbol to seed the bodies that are only reached indirectly — but the
    // shortfall has to be a minority, not most of the program.
    let dir = scratch();
    let Some(exe) = build(dir.path()) else {
        return;
    };
    let Some(stripped_exe) = stripped(&exe, dir.path()) else {
        eprintln!("skipping: no stripper installed");
        return;
    };
    let (_d, _i, named) = analyse(&exe);
    let (_d, _i, stripped_analysis) = analyse(&stripped_exe);
    let named_count = named.functions.iter().filter(|f| !f.insns.is_empty()).count();
    let stripped_count = stripped_analysis
        .functions
        .iter()
        .filter(|f| !f.insns.is_empty())
        .count();
    assert!(
        stripped_count * 2 >= named_count,
        "stripped recovery found {stripped_count} of {named_count} functions"
    );
    assert!(
        stripped_analysis.coverage.fraction() > 0.75,
        "only {:.1}% of code decoded after stripping",
        stripped_analysis.coverage.fraction() * 100.0
    );
}

#[test]
fn recovered_functions_never_overlap_and_sizes_are_accounted_for() {
    let dir = scratch();
    let Some(exe) = build(dir.path()) else {
        return;
    };
    let Some(stripped_exe) = stripped(&exe, dir.path()) else {
        eprintln!("skipping: no stripper installed");
        return;
    };
    let (_d, _i, analysis) = analyse(&stripped_exe);
    let mut sorted: Vec<_> = analysis.functions.iter().collect();
    sorted.sort_by_key(|f| f.entry.0);
    for f in &sorted {
        // Every instruction must lie inside the function's own range. Bytes
        // the range has and the instructions do not are cold blocks or a jump
        // table; bytes the instructions have and the range does not would be a
        // body that escapes its own boundary.
        for insn in &f.insns {
            assert!(
                insn.addr >= f.entry && insn.end() <= f.end(),
                "{} at {:#x}..{:#x} has an instruction at {:#x}",
                f.display_name(),
                f.entry.0,
                f.end().0,
                insn.addr.0
            );
        }
    }
    for pair in sorted.windows(2) {
        assert!(
            pair[0].end().0 <= pair[1].entry.0,
            "{} ends at {:#x}, past {} at {:#x}",
            pair[0].display_name(),
            pair[0].end().0,
            pair[1].display_name(),
            pair[1].entry.0
        );
    }
}
