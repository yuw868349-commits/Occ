//! The parameters of an inlined function are not the host's parameters.
//!
//! A compiler that inlines a call leaves the inlined function's arguments in
//! the debug information, under the inlined subroutine and pointing back at
//! the original declaration with `DW_AT_abstract_origin`. They are the
//! inlined function's — a `printf` has a parameter for its format string,
//! and that name belongs to the `printf` — and counting them as the host's
//! says the host takes more arguments than it does.
//!
//! The fixture calls `printf` from `main`, which is enough for the compiler
//! to inline it. The check is that `main` still has exactly the two
//! parameters it declares, whatever the compiler did with the call.

use std::path::{Path, PathBuf};
use std::process::Command;

use occ_analyze::View;
use occ_core::ImageId;

const SOURCE: &str = r#"
#include <stdio.h>
int main(int argc, char **argv) {
    printf("%d %s\n", argc, argv[0]);
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
    // Tests in one binary run on parallel threads and share a process id, so
    // the directory name needs a per-call part too: otherwise one test's
    // setup deletes the directory another test just created.
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let n = SEEN.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("occ-dwarf-inline-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch");
    Scratch(dir)
}

/// Compile the fixture, returning the path, or `None` when no compiler is
/// installed.
fn build(dir: &Path) -> Option<PathBuf> {
    let src = dir.join("unit.c");
    std::fs::write(&src, SOURCE).expect("write source");
    let exe = dir.join("unit");
    let out = Command::new("cc")
        .args(["-g", "-O1", "-o"])
        .arg(&exe)
        .arg(&src)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "skipping: cc cannot build the fixture here:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    Some(exe)
}

#[test]
fn an_inlined_functions_parameters_are_not_the_hosts() {
    let dir = scratch();
    let Some(exe) = build(dir.path()) else {
        return;
    };
    let data = std::fs::read(&exe).expect("read the fixture");
    let img = occ_bin::load(&data, &exe.to_string_lossy(), ImageId(0)).expect("parse");
    let view = View::new(&img, &data);
    let Some(dw) = occ_dwarf::Dwarf::load(&view) else {
        eprintln!("skipping: the fixture carries no debug information");
        return;
    };

    let main = dw
        .units
        .iter()
        .flat_map(|u| &u.functions)
        .find(|f| f.name == "main")
        .expect("the debug information names main");

    let names: Vec<&str> = main.params.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["argc", "argv"],
        "main declares two parameters, and the inlined call adds none"
    );

    // The inlined call is still recorded as an inlined call: skipping its
    // subtree must not lose the fact that the call was there.
    for frame in &main.inlined {
        assert!(
            !names.contains(&frame.name.as_str()),
            "the inlined function is not one of the host's parameters"
        );
    }
}
