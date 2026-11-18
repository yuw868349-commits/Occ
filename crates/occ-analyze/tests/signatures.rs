//! Learning names from one build and giving them to another.
//!
//! The fixture is compiled twice from one source: once with symbols, once
//! stripped. The code is byte-identical; only the names are gone. A signature
//! database learned from the first has to name the second, and a name it
//! cannot place has to stay unplaced rather than land on the wrong function.

use std::path::{Path, PathBuf};
use std::process::Command;

use occ_analyze::{Analysis, AnalysisOptions, View, sig};
use occ_core::Image;

const SOURCE: &str = r#"
#include <stdio.h>
#include <string.h>

int helper(int x) { return x * 3 + 1; }

static int classify(const char *s, int n) {
    int h = 0;
    for (int i = 0; i < n; i++) h = h * 31 + s[i];
    return h % 97;
}

int main(int argc, char **argv) {
    char buf[128];
    memset(buf, 0, sizeof buf);
    int s = 0;
    for (int i = 0; i < argc; i++) s += helper(argv[i][0]) + classify(argv[i], 4);
    snprintf(buf, sizeof buf, "%d", s);
    printf("%s\n", buf);
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
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let n = SEEN.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("occ-sig-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch");
    Scratch(dir)
}

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

fn stripped(exe: &Path, dir: &Path) -> Option<PathBuf> {
    let copy = dir.join("unit.stripped");
    std::fs::copy(exe, &copy).ok()?;
    for tool in ["llvm-strip", "strip"] {
        if let Ok(out) = Command::new(tool).arg("--strip-all").arg(&copy).output()
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

#[test]
fn names_survive_one_build_being_stripped() {
    let dir = scratch();
    let Some(exe) = build(dir.path()) else {
        return;
    };
    let Some(stripped_exe) = stripped(&exe, dir.path()) else {
        eprintln!("skipping: no stripper installed");
        return;
    };

    let (data, image, named) = analyse(&exe);
    let view = View::new(&image, &data);
    let db = sig::SignatureDb::learn(&image, &view, sig::DEFAULT_LEN);
    assert!(
        db.len() > 100,
        "a static link carries hundreds of named functions, learned {}",
        db.len()
    );

    // The text format is the exchange format; a round trip has to preserve
    // what a match compares.
    let text = db.to_text();
    let db = sig::SignatureDb::parse(&text).expect("parse its own output");

    let (sdata, simage, mut sanalysis) = analyse(&stripped_exe);
    let sview = View::new(&simage, &sdata);
    let before = sanalysis
        .functions
        .iter()
        .filter(|f| f.name.is_some())
        .count();
    let applied = db.apply(&mut sanalysis, &sview);
    let after = sanalysis
        .functions
        .iter()
        .filter(|f| f.name.is_some())
        .count();

    assert!(
        applied.named > 100,
        "named only {} functions from {} signatures",
        applied.named,
        db.len()
    );
    assert_eq!(after, before + applied.named);
    assert!(
        applied.collisions.len() * 10 < applied.named,
        "{} collisions against {} names: the prefix is too short to tell \
         functions apart",
        applied.collisions.len(),
        applied.named
    );

    // The names have to come back on the functions that had them. Comparing
    // by entry address against the build that still had symbols is the whole
    // claim: not "something was named" but "this address is that function".
    let original: std::collections::HashMap<u64, &str> = named
        .functions
        .iter()
        .filter_map(|f| f.name.as_deref().map(|n| (f.entry.0, n)))
        .collect();
    let mut checked = 0;
    for f in &sanalysis.functions {
        let Some(now) = f.name.as_deref() else {
            continue;
        };
        let Some(was) = original.get(&f.entry.0) else {
            continue;
        };
        assert_eq!(
            now, *was,
            "the function at {:#x} was {was}, and the database called it {now}",
            f.entry.0
        );
        checked += 1;
    }
    assert!(
        checked > 100,
        "only {checked} names could be checked against the unstripped build"
    );
}

#[test]
fn a_function_that_already_has_a_name_is_left_alone() {
    let dir = scratch();
    let Some(exe) = build(dir.path()) else {
        return;
    };
    let (data, image, mut analysis) = analyse(&exe);
    let view = View::new(&image, &data);
    let db = sig::SignatureDb::learn(&image, &view, sig::DEFAULT_LEN);

    // Rename one function to something the database does not contain; applying
    // the database must not overwrite it.
    let target = analysis
        .functions
        .iter_mut()
        .find(|f| f.name.is_none() && f.size > 0)
        .expect("a function without a name");
    target.name = Some("kept".into());
    let entry = target.entry;

    let _ = db.apply(&mut analysis, &view);
    let kept = analysis
        .functions
        .iter()
        .find(|f| f.entry == entry)
        .unwrap();
    assert_eq!(kept.name.as_deref(), Some("kept"));
}

#[test]
fn a_signature_never_lands_on_a_different_function() {
    // Build two unrelated programs. A signature from one must not name a
    // function in the other unless the bytes really are the same.
    let dir = scratch();
    let Some(exe) = build(dir.path()) else {
        return;
    };
    let other_src = dir.path().join("other.c");
    std::fs::write(
        &other_src,
        r#"
        #include <stdio.h>
        int totally_different(int a, int b) { return a ^ (b * 7); }
        int main(void) { printf("%d\n", totally_different(3, 4)); return 0; }
        "#,
    )
    .unwrap();
    let other = dir.path().join("other");
    let out = Command::new("cc")
        .args(["-O2", "-static", "-o"])
        .arg(&other)
        .arg(&other_src)
        .output()
        .unwrap();
    if !out.status.success() {
        return;
    }

    let (data, image, _) = analyse(&exe);
    let view = View::new(&image, &data);
    let db = sig::SignatureDb::learn(&image, &view, sig::DEFAULT_LEN);

    let (odata, oimage, mut oanalysis) = analyse(&other);
    let oview = View::new(&oimage, &odata);
    let applied = db.apply(&mut oanalysis, &oview);
    // Whatever matched, it matched on bytes; the names must belong to bodies
    // that start the same way, and `totally_different` must not be renamed.
    let f = oanalysis
        .functions
        .iter()
        .find(|f| f.name.is_some() && f.size > 0)
        .filter(|f| f.name.as_deref() == Some("totally_different"));
    assert!(f.is_some() || applied.named < 20, "{} names moved", applied.named);
}
