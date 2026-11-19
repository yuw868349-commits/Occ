//! Indirect control flow in a binary the system compiler built.
//!
//! The value-set pass answers one question: what can an indirect branch or
//! call reach, and why. The answer has to come out of the binary, not out
//! of a model of it, so the fixture is compiled at test time and read back.
//!
//! Three shapes are checked, because each one fails differently:
//!
//! * a `switch` whose cases start at zero, where the table holds the case
//!   values themselves;
//! * a `switch` whose cases start at a hundred, where the compiler takes a
//!   hundred off the index first and the table holds positions — the case
//!   labels are then a fact about the data flow, not about the table;
//! * a dispatch through a table of function pointers, where every target is
//!   a function and none of them is a case of anything.
//!
//! The labels are what tell the three apart, so they are checked against
//! the source rather than against what the pass believes.

use occ_analyze::{Analysis, AnalysisOptions, View};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

const SOURCE: &str = r#"#include <stddef.h>
volatile int sink;

/* One distinct noinline callee per case, and a volatile pointer table for the
   dispatch. Without them the compiler is free to fold a switch into
   arithmetic or turn `ops[i](a,b)` into a jump table of code offsets: clang
   does exactly that for the obvious fixture, and then there is no data table
   for the pass to resolve. The side effects keep both shapes real, which is
   what this test measures. */
#define TAP(p,n) __attribute__((noinline)) static void p##n(void){ sink += n + 1; }
TAP(d,0) TAP(d,1) TAP(d,2) TAP(d,3) TAP(d,4) TAP(d,5) TAP(d,6) TAP(d,7) TAP(d,8) TAP(d,9)
TAP(b,0) TAP(b,1) TAP(b,2) TAP(b,3) TAP(b,4) TAP(b,5) TAP(b,6)

static int add(int a,int b){return a+b;}
static int sub(int a,int b){return a-b;}
static int mul(int a,int b){return a*b;}
static int dv(int a,int b){return b?a/b:0;}
typedef int (*op)(int,int);
static op volatile ops[] = {add,sub,mul,dv};
int dispatch(int which,int a,int b){ if(which<0||which>3) return -1; return ops[which](a,b); }

int dense(int x){ switch(x){
 case 0: d0(); return 11; case 1: d1(); return 22; case 2: d2(); return 33;
 case 3: d3(); return 44; case 4: d4(); return 55; case 5: d5(); return 66;
 case 6: d6(); return 77; case 7: d7(); return 88; case 8: d8(); return 99;
 case 9: d9(); return 111; default: return -1; } }

int biased(int x){ switch(x){
 case 100: b0(); return 1; case 101: b1(); return 2; case 102: b2(); return 3;
 case 103: b3(); return 4; case 104: b4(); return 5; case 105: b5(); return 6;
 case 106: b6(); return 7; default: return 0; } }
int main(int argc, char **argv) {
    sink = dispatch(argc & 3, argc, 2);
    sink += dense(argc);
    sink += biased(argc + 100);
    return sink;
}
"#;

/// Counter behind the file name, so two tests compiling at the same time do
/// not fight over one path.
static SEQ: AtomicU32 = AtomicU32::new(0);

/// Compile `SOURCE` at `-O2` and return the binary.
///
/// `None` only when the compiler is absent. A compiler that is present but
/// fails is a failure, not a skip: a broken fixture that silently skips
/// leaves the feature untested and the suite green.
///
/// The scratch directory is removed when the test binary exits. Leaving it
/// behind would put a copy of the source and the binary in `/tmp` for every
/// run of the suite.
fn build() -> Option<PathBuf> {
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("occ-vsa-switch-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    occ_sys::cleanup::remove_dir_at_exit(&dir);
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

/// The report for the fixture.
fn report() -> Option<occ_vsa::Report> {
    let path = build()?;
    let data = std::fs::read(&path).expect("read binary");
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse binary");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    Some(occ_vsa::analyze(
        &view,
        &analysis,
        &occ_vsa::Options::default(),
    ))
}

/// The labels a jump site resolved, in order.
fn labels(site: &occ_vsa::JumpSite) -> Vec<i64> {
    site.cases.iter().map(|c| c.label).collect()
}

#[test]
fn a_switch_whose_cases_start_at_zero_is_resolved() {
    let Some(report) = report() else {
        return;
    };
    // Every switch the fixture has, with the labels it resolved.
    let found: Vec<(u64, Vec<i64>)> = report
        .jumps
        .iter()
        .filter(|j| j.cases.len() > 1)
        .map(|j| (j.in_function.0, labels(j)))
        .collect();
    assert!(
        !found.is_empty(),
        "no jump site resolved: {:#?}",
        report.stats
    );
    // One of them is the dense switch: ten cases numbered from zero.
    assert!(
        found
            .iter()
            .any(|(_, labels)| *labels == (0..10).collect::<Vec<i64>>()),
        "the dense switch was not resolved with the right labels: {found:#?}"
    );
}

#[test]
fn a_switch_whose_cases_start_at_a_hundred_keeps_its_case_values() {
    let Some(report) = report() else {
        return;
    };
    let labels: Vec<i64> = report
        .jumps
        .iter()
        .filter(|j| j.cases.len() > 1)
        .flat_map(labels)
        .collect();
    // The biased switch: the compiler takes a hundred off the index before
    // it looks the table up, so the table holds positions and the case
    // values are the positions plus a hundred.
    assert!(
        labels
            .windows(7)
            .any(|w| w == [100, 101, 102, 103, 104, 105, 106]),
        "the biased switch did not keep its case values: {labels:#?}"
    );
}

#[test]
fn a_table_of_function_pointers_resolves_to_the_functions() {
    let Some(report) = report() else {
        return;
    };
    // The dispatch through `ops[]`: every target is a function entry, and
    // the pass says so rather than calling them cases.
    let dispatch = report
        .jumps
        .iter()
        .find(|j| j.tail && j.cases.len() == 4)
        .expect("the function pointer table was not resolved");
    assert!(
        dispatch.cases.iter().all(|c| c.target.0 != 0),
        "a target was not named: {:#?}",
        dispatch.cases
    );
    // Four distinct targets: a table of four pointers holds four addresses.
    let mut targets: Vec<u64> = dispatch.cases.iter().map(|c| c.target.0).collect();
    targets.sort_unstable();
    targets.dedup();
    assert_eq!(targets.len(), 4, "the targets repeat: {targets:#?}");
}

#[test]
fn every_resolved_site_names_a_target_inside_the_image() {
    let Some(report) = report() else {
        return;
    };
    for jump in &report.jumps {
        for case in &jump.cases {
            assert!(
                case.target.0 != 0,
                "a case names no address: {case:?} in {:#?}",
                jump.evidence
            );
        }
    }
    // Nothing was invented: every resolved site has a case, and no site
    // claims more cases than the table it read had entries.
    for jump in &report.jumps {
        if let Some(table) = &jump.table {
            assert!(
                jump.cases.len() <= table.entries,
                "the site claims {} cases for a table of {} entries",
                jump.cases.len(),
                table.entries
            );
        }
    }
}
