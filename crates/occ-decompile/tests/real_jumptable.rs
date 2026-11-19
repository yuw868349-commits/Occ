//! A `switch` compiled to a jump table, built by the system compiler at test
//! time.
//!
//! This is the case where a recursive-descent recovery loses most of a
//! function without saying so. The dispatch is an indirect jump through a
//! table; the case bodies sit after it and are reachable only through that
//! table. A recovery that stops at the indirect jump ends the function there,
//! reports a short body, and looks complete.
//!
//! The test compiles a known source and checks that the table is found, that
//! its entries point at recovered blocks, and that the pseudo-C carries a
//! `switch` with one case per entry.
//!
//! The case bodies write to a `volatile`, which keeps the compiler from
//! folding the whole switch into arithmetic: `case 0: return 10;` and friends
//! become a multiply and an add, and there is no table to find. The bodies
//! have to stay distinct for the test to test anything.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::Addr;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

const SOURCE: &str = r#"
volatile int sink;
/* One distinct callee per case. A single `tap(i)` helper is not enough: a
   compiler can fold seven calls to one function with different arguments into
   a byte table and one call, and then there is no dispatch table to recover.
   Distinct callees leave no option but a jump table, which is what this
   fixture is for. */
#define TAP(n) __attribute__((noinline)) static void tap##n(void) { sink += n * 13 + 1; }
TAP(0) TAP(1) TAP(2) TAP(3) TAP(4) TAP(5) TAP(6)
int classify(int c) {
    switch (c) {
    case 0: tap0(); return 10;
    case 1: tap1(); return 21;
    case 2: tap2(); return 32;
    case 3: tap3(); return 43;
    case 4: tap4(); return 54;
    case 5: tap5(); return 65;
    case 6: tap6(); return 76;
    default: sink = 9; return -1;
    }
}
int main(int argc, char **argv) { return classify(argc - 1); }
"#;

/// Counter behind the file name, so two tests compiling at the same time do
/// not fight over one path.
static SEQ: AtomicU32 = AtomicU32::new(0);

/// Compile `SOURCE` at `-O2` and return the binary.
///
/// `None` only when the compiler is absent. A compiler that is present but
/// fails is a failure, not a skip: a broken fixture that silently skips
/// leaves the feature untested and the suite green.
fn build() -> Option<PathBuf> {
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("occ-jumptable-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    occ_sys::cleanup::remove_dir_at_exit(&dir);
    let src = dir.join("unit.c");
    std::fs::write(&src, SOURCE).expect("write source");
    let out = dir.join("unit");
    let output = Command::new("cc")
        .args(["-O2", "-fno-inline", "-o"])
        .arg(&out)
        .arg(&src)
        .output()
        .ok()?;
    assert!(
        output.status.success(),
        "the fixture must compile:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(out)
}

fn analyze(path: &std::path::Path) -> (Analysis, occ_core::Image) {
    let data = std::fs::read(path).expect("read binary");
    let img = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse binary");
    let view = View::new(&img, &data);
    (Analysis::build(&view, &AnalysisOptions::default()), img)
}

#[test]
fn a_switch_compiled_to_a_table_is_recovered() {
    let Some(path) = build() else {
        return;
    };
    let (analysis, _) = analyze(&path);
    let f = analysis
        .functions
        .iter()
        .find(|f| f.display_name() == "classify")
        .expect("classify is recovered");

    assert_eq!(
        f.jump_tables.len(),
        1,
        "one dispatch table: {:#?}",
        f.jump_tables
    );
    let table = &f.jump_tables[0];

    // Seven cases: the compiler emits one entry per case, and the default is
    // reached by the range check that precedes the dispatch.
    assert!(
        table.count >= 7,
        "the table has {} entries, expected at least seven",
        table.count
    );
    assert_eq!(table.targets.len(), table.count as usize);
    assert!(
        table.index_reg.is_some(),
        "the dispatch names the register it indexes with"
    );

    // Every entry must land inside the function and on a block boundary: a
    // target that is not a block start means the case body was decoded but
    // never made reachable.
    let starts: Vec<Addr> = f.blocks.iter().map(|b| b.start).collect();
    for t in &table.targets {
        assert!(
            t.0 >= f.entry.0 && t.0 < f.end().0,
            "target {t:?} is outside the function {}..{}",
            f.entry,
            f.end()
        );
        assert!(
            starts.contains(t),
            "target {t:?} is not the start of a block: {starts:?}"
        );
    }
    // Distinct targets: a table whose entries repeat is data, not a dispatch.
    let mut unique = table.targets.clone();
    unique.sort_by_key(|a| a.0);
    unique.dedup();
    assert_eq!(unique.len(), table.targets.len(), "the table repeats a target");
}

#[test]
fn the_decompiled_switch_lists_every_case() {
    let Some(path) = build() else {
        return;
    };
    let (analysis, image) = analyze(&path);
    let f = analysis
        .functions
        .iter()
        .find(|f| f.display_name() == "classify")
        .expect("classify is recovered");
    let name_of = |a: Addr| {
        analysis
            .function_at(a)
            .map(|f| f.display_name())
            .unwrap_or_else(|| format!("sub_{:x}", a.get()))
    };
    let d = occ_decompile::decompile(
        &image,
        f,
        &occ_decompile::Options::default(),
        &name_of,
    );
    let code = occ_decompile::render::render(&d);

    assert!(
        code.contains("switch ("),
        "the pseudo-C has no switch:\n{code}"
    );
    for case in 0..7u64 {
        assert!(
            code.contains(&format!("case {case:#x}:")),
            "case {case} is missing:\n{code}"
        );
    }
    // The case bodies must carry their bodies, not just their labels.
    assert!(
        code.contains("0xa;") || code.contains("10;"),
        "a case body is missing:\n{code}"
    );
    assert_eq!(d.unstructured, 0, "the switch structured without a goto:\n{code}");
}
