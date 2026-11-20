//! Traces of real programs, compiled at test time.
//!
//! Each fixture is a C program whose behaviour is stated in the test, so the
//! assertions are about observed facts rather than about the tracer's own
//! bookkeeping: the exit code the program returned, the function it called, the
//! byte it wrote, the block it skipped, the fault it raised.
//!
//! The tests skip when `cc` is missing, like the other real-file checks.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, ImageId};
use occ_dynamic::{End, RunConfig, correlate, function_blocks, run};
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE: &str = r#"
extern long write(int, const void *, unsigned long);

__attribute__((noinline)) static int leaf(int x) { return x + 1; }

int main(int argc, char **argv) {
    char c = (char)('a' + leaf(1));
    if (argc > 5) {
        c = 'z';
    }
    write(1, &c, 1);
    return 3;
}
"#;

const FAULT: &str = r#"
int main(void) {
    volatile int *p = (volatile int *)0;
    *p = 1;
    return 0;
}
"#;

const THREADED: &str = r#"
#include <pthread.h>

static void *work(void *p) { return p; }

int main(void) {
    pthread_t t;
    pthread_create(&t, 0, work, 0);
    pthread_join(t, 0);
    return 7;
}
"#;

/// The directory the fixtures are compiled into.
///
/// It is named after the process so two test runs cannot collide, and it is
/// removed when the test binary exits. Leaving it behind would fill `/tmp`
/// with a directory per test run, each holding a copy of every fixture
/// source and binary.
fn scratch() -> PathBuf {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let dir = std::env::temp_dir().join(format!("occ-run-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    occ_sys::cleanup::remove_dir_at_exit(&dir);
    ONCE.call_once(|| remove_scratch_on_exit(&dir));
    dir
}

/// Remove the scratch directory when the process ends.
///
/// `Drop` on a value held in a `static` does not run at exit, so the cleanup
/// is registered with `atexit` instead: the C runtime calls it while the
/// process is still single-threaded and the filesystem is still there.
fn remove_scratch_on_exit(dir: &Path) {
    unsafe extern "C" {
        fn atexit(f: extern "C" fn()) -> i32;
    }
    static DIR: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);
    extern "C" fn cleanup() {
        let taken = DIR.lock().ok().and_then(|mut g| g.take());
        if let Some(dir) = taken {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
    if let Ok(mut slot) = DIR.lock() {
        *slot = Some(dir.to_path_buf());
    }
    // SAFETY: `cleanup` is a plain `extern "C" fn` with no arguments, and
    // `atexit` takes exactly that.
    unsafe {
        atexit(cleanup);
    }
}

/// Compile `source` into a uniquely named binary.
///
/// The name carries a counter because the test harness runs these in parallel:
/// two tests writing the same path make `execve` fail with `ETXTBSY` while the
/// other one is running it, which is a flake, not a finding.
fn build(name: &str, source: &str) -> Option<PathBuf> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let dir = scratch();
    let unique = NEXT.fetch_add(1, Ordering::Relaxed);
    let src = dir.join(format!("{name}-{unique}.c"));
    std::fs::write(&src, source).expect("write source");
    let bin = dir.join(format!("{name}-{unique}"));
    // A compiler that is present but fails is a failure, not a skip: a broken
    // fixture would leave the feature untested and the suite green.
    // The stack protector is explicit: Ubuntu's gcc enables it by default and
    // clang does not, and the coverage test measures the canary failure block.
    // A fixture whose block count depends on which compiler is installed
    // tests the distribution, not the tracer.
    let output = Command::new("cc")
        .args([
            "-O0",
            "-fno-inline",
            "-g",
            "-pthread",
            "-fstack-protector-strong",
            "-o",
        ])
        .arg(&bin)
        .arg(&src)
        .output()
        .ok()?;
    assert!(
        output.status.success(),
        "the {name} fixture must compile:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(bin)
}

struct Loaded {
    data: Vec<u8>,
    image: occ_core::Image,
    bin: PathBuf,
}

fn load(bin: PathBuf) -> Loaded {
    let data = std::fs::read(&bin).expect("read fixture");
    let image = occ_bin::load(&data, &bin.display().to_string(), ImageId(0)).expect("parse");
    Loaded { data, image, bin }
}

fn fixture() -> Option<Loaded> {
    build("unit", SOURCE).map(load)
}

fn trace_of(l: &Loaded, cfg: &RunConfig) -> occ_dynamic::Trace {
    run(Path::new(&l.bin), &l.image, cfg).expect("trace")
}

#[test]
fn a_run_records_the_exit_code_and_the_steps() {
    let Some(l) = fixture() else { return };
    let t = trace_of(&l, &RunConfig::default());
    assert_eq!(t.end, End::Exited(3), "{}", t.end.label());
    assert!(!t.end.truncated());
    assert!(t.steps > 100, "only {} steps", t.steps);
    // Every step is either an address of the image or a count of one outside
    // it; nothing is dropped silently.
    assert_eq!(
        t.path_taken.len() as u64 + t.external_steps,
        t.steps,
        "{} recorded + {} external != {} steps",
        t.path_taken.len(),
        t.external_steps,
        t.steps
    );
    assert!(!t.path_taken.is_empty(), "the image ran");
    assert!(!t.threads.is_empty());
    assert_eq!(t.threads[0].tid, t.pid);
}

#[test]
fn the_trace_starts_at_the_entry_point_not_the_loader() {
    let Some(l) = fixture() else { return };
    let t = trace_of(&l, &RunConfig::default());
    assert_eq!(
        t.path_taken.first().copied(),
        Some(l.image.entry),
        "the first recorded instruction should be the entry point"
    );
}

#[test]
fn both_functions_are_covered_and_the_call_between_them_is_recorded() {
    let Some(l) = fixture() else { return };
    let view = View::new(&l.image, &l.data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let t = trace_of(&l, &RunConfig::default());
    let corr = correlate(&t, &l.image, &analysis);

    let main = analysis
        .function_by_name("main")
        .expect("main is recovered from its symbol");
    let leaf = analysis
        .function_by_name("leaf")
        .expect("leaf is recovered from its symbol");

    let hit = |entry: Addr| corr.functions.iter().find(|f| f.entry == entry);
    let main_hit = hit(main.entry).expect("main ran");
    let leaf_hit = hit(leaf.entry).expect("leaf ran");
    assert!(main_hit.insns > 0 && leaf_hit.insns > 0);
    assert!(leaf_hit.blocks >= 1);

    // The call from main into leaf, at an address inside main.
    let edge = corr
        .calls
        .iter()
        .flat_map(|c| c.targets.iter().map(move |t| (c, t)))
        .find(|(_, t)| t.in_image && t.addr == leaf.entry)
        .expect("the call into leaf is in the trace");
    assert!(
        edge.0.site.0 >= main.entry.0 && edge.0.site.0 < main.end().0,
        "call site {} is outside main [{}, {})",
        edge.0.site,
        main.entry,
        main.end()
    );
    assert!(!edge.0.indirect, "leaf is called directly");
}

#[test]
fn the_write_syscall_carries_its_descriptor_and_return() {
    let Some(l) = fixture() else { return };
    let t = trace_of(&l, &RunConfig::default());
    let write = t
        .syscalls
        .iter()
        .find(|s| s.name == "write")
        .expect("the program calls write");
    assert_eq!(write.nr, 1);
    assert_eq!(write.args[0], 1, "stdout");
    assert_eq!(write.args[2], 1, "one byte");
    assert!(write.returned);
    assert_eq!(write.ret, 1);
}

#[test]
fn a_branch_that_did_not_run_is_reported_as_not_run() {
    let Some(l) = fixture() else { return };
    let view = View::new(&l.image, &l.data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let main = analysis.function_by_name("main").expect("main");

    // argc is 1 in the first run and 7 in the second, so `argc > 5` decides one
    // block. Everything else in the function is identical, which makes the
    // difference between the two runs a measurement rather than a guess.
    let without = trace_of(&l, &RunConfig::default());
    let with = trace_of(
        &l,
        &RunConfig {
            args: vec!["1".into(); 6],
            ..RunConfig::default()
        },
    );
    let a = function_blocks(&without, main);
    let b = function_blocks(&with, main);
    assert_eq!(a.len(), b.len(), "same function, same blocks");
    assert!(a.len() >= 4, "main has {} block(s)", a.len());

    let unvisited: Vec<Addr> = a.iter().filter(|x| x.hits == 0).map(|x| x.start).collect();
    let entered: Vec<Addr> = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.hits == 0 && y.hits > 0)
        .map(|(x, _)| x.start)
        .collect();
    assert_eq!(
        entered.len(),
        1,
        "exactly one block should depend on argc: {a:?} vs {b:?}"
    );
    assert!(unvisited.contains(&entered[0]));
    // The other unvisited block is the stack-canary failure path, which neither
    // run reaches because the check passed. A trace that claimed it ran would be
    // reporting a program that crashed.
    assert_eq!(
        unvisited.len(),
        2,
        "the canary check and the argc branch: {a:?}"
    );
    assert!(
        b.iter()
            .any(|x| x.start == unvisited[0] || x.start == unvisited[1]),
        "the skipped blocks still exist in the second run"
    );
    for x in &b {
        if !unvisited.contains(&x.start) {
            continue;
        }
        if x.start == entered[0] {
            assert!(x.hits > 0);
        } else {
            assert_eq!(x.hits, 0, "the canary path must not run: {x:?}");
        }
    }
}

#[test]
fn a_step_limit_ends_the_run_and_says_so() {
    let Some(l) = fixture() else { return };
    let t = trace_of(
        &l,
        &RunConfig {
            max_steps: 20,
            ..RunConfig::default()
        },
    );
    assert_eq!(t.end, End::StepLimit);
    assert!(t.end.truncated(), "a cut run must report itself as cut");
    assert!(t.steps <= 21, "{} steps past a 20 step budget", t.steps);
}

#[test]
fn a_faulting_program_reports_the_signal_and_the_address() {
    let Some(l) = build("fault", FAULT).map(load) else {
        return;
    };
    let t = trace_of(&l, &RunConfig::default());
    assert_eq!(t.end, End::Signaled(11), "{}", t.end.label());
    let segv = t
        .signals
        .iter()
        .find(|s| s.signal == 11)
        .expect("SIGSEGV is recorded");
    assert_eq!(segv.fault, Some(0), "the fault address is the null page");
}

#[test]
fn a_program_that_dies_leaves_no_core_file_behind() {
    let Some(l) = build("fault", FAULT).map(load) else {
        return;
    };
    // The tracee runs in the directory the test binary was started from, which
    // is the crate directory. A core file there would land in the source tree
    // of the project being developed, and the tracer's report already says
    // what happened. The core-dump limit is lowered in the child before it
    // execs, so the file is never created.
    let before = core_files_in(&std::env::current_dir().expect("the current directory"));
    let t = trace_of(&l, &RunConfig::default());
    assert_eq!(t.end, End::Signaled(11), "{}", t.end.label());
    let after = core_files_in(&std::env::current_dir().expect("the current directory"));
    assert_eq!(
        before, after,
        "the tracee wrote a core file into the working directory"
    );
}

/// Core files in `dir`, by name.
///
/// The names are compared rather than the count so that a core file left by
/// something else running at the same time is not mistaken for one the tracee
/// wrote, and one the tracee wrote is not missed because a different one
/// disappeared.
fn core_files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read the directory")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("core"))
        .collect();
    names.sort();
    names
}

#[test]
fn a_run_does_not_hang_when_the_program_waits_for_input() {
    let Some(l) = build(
        "blocked",
        "extern long read(int, void *, unsigned long);\nint main(void) { char c; return (int)read(0, &c, 1); }\n",
    )
    .map(load)
    else {
        return;
    };
    // stdin is /dev/null, so the read returns 0 rather than blocking; the test
    // is that the call returns at all, which it would not if the tracer leaked
    // the parent's stdin into the tracee.
    let t = trace_of(&l, &RunConfig::default());
    assert!(matches!(t.end, End::Exited(0)), "{}", t.end.label());
}

#[test]
fn an_address_inside_an_instruction_is_an_overlap_not_an_orphan() {
    // Recovery and the CPU can disagree about where one instruction ends
    // without the function being absent. The trace recorded the byte the CPU
    // really executed; if that byte is inside an instruction the body already
    // recovered, the function must keep the hit and report an overlap, and the
    // address must not also appear as an orphan.
    let Some(l) = fixture() else { return };
    let view = View::new(&l.image, &l.data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let (entry, insn) = analysis
        .functions
        .iter()
        .find_map(|f| {
            f.insns
                .iter()
                .find(|i| i.len > 1)
                .map(|i| (f.entry, i.clone()))
        })
        .expect("a multi-byte instruction");
    let inside = Addr(insn.addr.0 + 1);
    let mut t = trace_of(&l, &RunConfig::default());
    // Keep only the synthetic byte: without the overlap path this address
    // would be an orphan, and the function would be listed as never executed.
    t.path_taken = vec![inside];
    let corr = correlate(&t, &l.image, &analysis);

    let hit = corr
        .functions
        .iter()
        .find(|h| h.entry == entry)
        .expect("the owning function keeps the hit");
    assert_eq!(hit.overlap, 1, "{hit:?}");
    assert_eq!(hit.hits, 1, "{hit:?}");
    let overlap = corr
        .overlaps
        .iter()
        .find(|o| o.addr == inside)
        .expect("the overlap is reported");
    assert_eq!(overlap.insn, insn.addr);
    assert_eq!(overlap.function, entry);
    assert!(
        !corr.orphans.iter().any(|o| o.addr == inside),
        "{:?}",
        corr.orphans
    );
    assert!(
        !corr.missed.iter().any(|(e, _)| *e == entry),
        "{:?}",
        corr.missed
    );
}

#[test]
fn a_second_thread_is_traced_and_does_not_deadlock_the_join() {
    let Some(l) = build("threaded", THREADED).map(load) else {
        return;
    };
    let t = trace_of(&l, &RunConfig::default());
    // A tracer that stepped one thread at a time would hang here: the join
    // blocks the parent, and the child never runs.
    assert_eq!(t.end, End::Exited(7), "{}", t.end.label());
    assert!(
        t.threads.len() >= 2,
        "only {} thread(s) traced",
        t.threads.len()
    );
    assert!(
        t.threads.iter().filter(|x| x.steps > 0).count() >= 2,
        "both threads should have run: {:?}",
        t.threads
    );
    assert!(
        t.threads.iter().any(|x| x.tid != t.pid),
        "no thread other than the initial one"
    );
}
