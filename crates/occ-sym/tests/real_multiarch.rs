//! The same function, compiled for three architectures, executed three ways.
//!
//! The decompiler's own suite checks that all three architectures lift the
//! same function. This one checks what the analyses built on top of it do
//! with that function: the symbolic executor, the constraint solver and the
//! taint tracker all consume the lowered program directly, so an
//! architecture that lifts but lifts *wrongly* still produces paths — just
//! the wrong ones.
//!
//! The check that matters is the answer. The source has four arms, and each
//! architecture has to pick the same arm for the same input. A lifter that
//! treats a signed comparison as an unsigned one, or that never sets its
//! flags, sends the input down a different arm, and the answer says so.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::Addr;
use occ_sym::{ExecConfig, Mode, Outcome, Source, TaintConfig, explore, trace};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

/// The body under test. Three branches, so four answers.
const SOURCE: &str = r#"
volatile int sink;
int classify(int x) {
    if (x < 0) { sink = 1; return -1; }
    if (x == 0) { sink = 2; return 0; }
    if (x < 10) { sink = 3; return 1; }
    sink = 4;
    return 2;
}
int main(int argc, char **argv) { return classify(argc - 1); }
"#;

/// The answers the source gives, by input.
const CASES: [(i64, i64); 5] = [
    (-100, -1),
    (-1, -1),
    (0, 0),
    (5, 1),
    (100, 2),
];

/// Counter behind the file name, so two tests compiling at the same time do
/// not fight over one path.
static SEQ: AtomicU32 = AtomicU32::new(0);

/// One target under test.
struct Target {
    /// Human name, used in assertion messages.
    name: &'static str,
    /// The compiler that produces this target.
    cc: &'static str,
    /// Extra flags the compiler needs, if any.
    extra: &'static [&'static str],
}

/// The targets. `cc` with no extra flags is the host, which is x86-64.
const TARGETS: [Target; 3] = [
    Target {
        name: "x86-64",
        cc: "cc",
        extra: &[],
    },
    Target {
        name: "aarch64",
        cc: "aarch64-linux-gnu-gcc",
        extra: &["-static"],
    },
    Target {
        name: "riscv64",
        cc: "riscv64-linux-gnu-gcc",
        extra: &["-static"],
    },
];

/// Compile `SOURCE` for one target and return the binary.
///
/// `None` only when the compiler is absent. A compiler that is present but
/// fails is a failure, not a skip: a broken fixture that silently skips
/// leaves the feature untested and the suite green.
fn build(target: &Target) -> Option<PathBuf> {
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "occ-multiarch-sym-{}-{}-{n}",
        std::process::id(),
        target.name
    ));
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    occ_sys::cleanup::remove_dir_at_exit(&dir);
    let src = dir.join("unit.c");
    std::fs::write(&src, SOURCE).expect("write source");
    let out = dir.join("unit");
    let output = Command::new(target.cc)
        .args(["-O2", "-fno-inline"])
        .args(target.extra)
        .args(["-o"])
        .arg(&out)
        .arg(&src)
        .output()
        .ok()?;
    assert!(
        output.status.success(),
        "the {} fixture must compile:\n{}",
        target.name,
        String::from_utf8_lossy(&output.stderr)
    );
    Some(out)
}

/// A loaded binary and everything the analyses need.
struct Loaded {
    analysis: Analysis,
    image: occ_core::Image,
    data: Vec<u8>,
}

fn load(path: &Path) -> Loaded {
    let data = std::fs::read(path).expect("read binary");
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse binary");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    Loaded {
        analysis,
        image,
        data,
    }
}

/// The `classify` function.
fn classify(loaded: &Loaded) -> occ_core::Function {
    loaded
        .analysis
        .functions
        .iter()
        .find(|f| f.display_name() == "classify")
        .expect("classify is recovered")
        .clone()
}

/// The resolver for call targets.
fn name_resolver(loaded: &Loaded) -> impl Fn(Addr) -> String + '_ {
    let analysis = &loaded.analysis;
    move |a: Addr| {
        analysis
            .function_at(a)
            .map(|f| f.display_name())
            .unwrap_or_else(|| format!("sub_{:x}", a.get()))
    }
}

/// Run the fixture for every target that has a compiler, and check each
/// result. A target without a compiler is skipped; a target that fails a
/// check fails the test.
fn for_each_built(mut check: impl FnMut(&Target, &Loaded, &occ_core::Function)) {
    for target in TARGETS.iter() {
        let Some(path) = build(target) else {
            eprintln!("skipping {}: no {}", target.name, target.cc);
            continue;
        };
        let loaded = load(&path);
        let f = classify(&loaded);
        check(target, &loaded, &f);
    }
}

/// The register the argument arrives in, in this architecture's spelling.
fn argument_register(loaded: &Loaded, f: &occ_core::Function) -> String {
    let name_of = name_resolver(loaded);
    let program = occ_decompile::build_program(
        &loaded.image,
        f,
        &occ_decompile::Options::default(),
        &name_of,
    );
    program.param_regs.first().cloned().unwrap_or_default()
}

/// The value a function that returns `int` hands back.
///
/// The three architectures do not agree on the high half of the return
/// register: x86-64 and AArch64 zero it when they write a 32-bit value,
/// RISC-V sign-extends it. The C type is the same in all three, so the
/// comparison is made on the low 32 bits read as a signed number — which is
/// what the source says the function returns.
fn returned_int(text: &str) -> Option<i64> {
    let t = text.trim();
    let (negative, digits) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t),
    };
    let digits = digits.strip_prefix("0x").unwrap_or(digits);
    let raw = u64::from_str_radix(digits, 16).ok()?;
    let value = if negative {
        raw.wrapping_neg()
    } else {
        raw
    };
    Some(value as u32 as i32 as i64)
}

#[test]
fn every_architecture_answers_the_same_way() {
    for_each_built(|target, loaded, f| {
        let name_of = name_resolver(loaded);
        let reg = argument_register(loaded, f);
        assert!(
            !reg.is_empty(),
            "{}: the function takes no argument",
            target.name
        );
        let view = View::new(&loaded.image, &loaded.data);
        let read_static = |at: Addr, bytes: u8| view.read_uint(at, usize::from(bytes));
        for (input, expected) in CASES {
            let cfg = ExecConfig {
                mode: Mode::Concrete,
                inputs: vec![(reg.clone(), input as u64)],
                ..ExecConfig::default()
            };
            let report = explore(
                &loaded.image,
                f,
                &occ_decompile::Options::default(),
                &name_of,
                &read_static,
                &loaded.analysis.functions,
                &cfg,
            );
            let returned: Vec<String> = report
                .paths
                .iter()
                .filter(|p| matches!(p.outcome, Outcome::Return))
                .filter_map(|p| p.ret.clone())
                .collect();
            assert_eq!(
                returned.len(),
                1,
                "{}: input {input} reached {} ends, not one:\n{returned:?}",
                target.name,
                returned.len()
            );
            let value = returned_int(&returned[0]).unwrap_or_else(|| {
                panic!("{}: the answer {returned:?} is not a number", target.name)
            });
            assert_eq!(
                value, expected,
                "{}: input {input} answered {value}, expected {expected}",
                target.name
            );
        }
    });
}

#[test]
fn every_architecture_explores_every_path() {
    for_each_built(|target, loaded, f| {
        let name_of = name_resolver(loaded);
        let view = View::new(&loaded.image, &loaded.data);
        let read_static = |at: Addr, bytes: u8| view.read_uint(at, usize::from(bytes));
        let report = explore(
            &loaded.image,
            f,
            &occ_decompile::Options::default(),
            &name_of,
            &read_static,
            &loaded.analysis.functions,
            &ExecConfig::default(),
        );
        let feasible = report.paths.iter().filter(|p| p.is_feasible()).count();
        assert_eq!(
            feasible,
            4,
            "{}: three tests make four paths:\n{:#?}",
            target.name,
            report.paths
        );
        assert!(
            report.lift_notes.is_empty(),
            "{}: the lifter had nothing to approximate: {:?}",
            target.name,
            report.lift_notes
        );
    });
}

#[test]
fn every_architecture_taints_the_global_store() {
    for_each_built(|target, loaded, f| {
        let name_of = name_resolver(loaded);
        let reg = argument_register(loaded, f);
        let report = trace(
            &loaded.image,
            f,
            &occ_decompile::Options::default(),
            &name_of,
            &[Source::Name(reg)],
            &TaintConfig::default(),
            &loaded.analysis.functions,
        );
        // The argument decides which arm runs, so it reaches every branch.
        assert!(
            !report.flows.is_empty(),
            "{}: the argument reached nothing at all",
            target.name
        );
        assert!(
            report.converged,
            "{}: the taint analysis did not converge: {:?}",
            target.name,
            report.notes
        );
        // A branch is one of the sinks, and every one of the three tests
        // reads the argument.
        let branches = report
            .flows
            .iter()
            .filter(|flow| flow.sink.contains("branch"))
            .count();
        assert_eq!(
            branches,
            3,
            "{}: the argument reaches all three tests:\n{:#?}",
            target.name,
            report.flows
        );
    });
}
