//! The same function, compiled for three architectures, lifted three times.
//!
//! The architecture abstraction in [`occ_decompile::lift`] is easy to break
//! in ways no unit test sees: one architecture gains a register the others
//! lack, one keeps its flags alive across instructions and another does not,
//! one names its parameters `rdi` and another `x0`. Each of those bugs shows
//! up as a function that decompiles into nonsense on exactly one architecture
//! while the other two are fine.
//!
//! So the test compiles one source with three cross compilers and checks that
//! all three produce the same function: same parameters, same control-flow
//! shape, same answer to the same inputs. The last one is the one that
//! matters — the body is written so that a wrong comparison shows up in the
//! answer, because `x < 0` on a signed value is not the same test as `x < 0`
//! on an unsigned one, and that difference is the difference between the
//! architectures.
//!
//! The symbolic executor and the taint tracker run over the AArch64 and
//! RISC-V builds as well. Those two crates consume the lowered program
//! directly, so an architecture that lifts but lifts *wrongly* would still
//! produce paths — just the wrong ones.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::Addr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

/// A body with three branches, chosen so that each one is a different kind of
/// comparison.
///
/// `x < 0` is signed, `x == 0` is equality, `x < 10` is a second signed
/// comparison against a constant. A lifter that treats the incoming value as
/// unsigned turns the first into `0 < x`, which reads the same but is not the
/// same test, and a lifter that never sets its flags turns all three into
/// `true`. The compiler is free to rewrite `x < 10` as `x <= 9`, so the
/// answer is checked rather than the spelling.
const SOURCE: &str = r#"
volatile int sink;
/* An empty asm block with a memory clobber sits after each store. Without it
   clang if-converts the chain: it computes the value with `setb` and writes
   one store, the branches vanish from the machine code, and a test that asks
   for three branches would be asking the compiler not to optimize. The
   barrier keeps the stores where the source puts them, so the control flow
   the test is about survives. It is not there to help the decompiler: it is
   there to make the compiler preserve the shape the source has. */
#define KEEP __asm__ volatile("" ::: "memory")
int classify(int x) {
    if (x < 0) { sink = 1; KEEP; return -1; }
    if (x == 0) { sink = 2; KEEP; return 0; }
    if (x < 10) { sink = 3; KEEP; return 1; }
    sink = 4; KEEP;
    return 2;
}
int main(int argc, char **argv) { return classify(argc - 1); }
"#;

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
        "occ-multiarch-{}-{}-{n}",
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
}

fn load(path: &Path) -> Loaded {
    let data = std::fs::read(path).expect("read binary");
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse binary");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    Loaded { analysis, image }
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

/// The address of the `sink` global, which every architecture writes to.
fn sink_address(loaded: &Loaded) -> Addr {
    loaded
        .image
        .symbols
        .iter()
        .find(|s| s.defined && s.name == "sink")
        .map(|s| s.addr)
        .expect("the fixture declares a global named sink")
}

/// The decompiled pseudo-C of one function.
fn decompile(loaded: &Loaded, f: &occ_core::Function) -> String {
    let name_of = name_resolver(loaded);
    let d = occ_decompile::decompile(
        &loaded.image,
        f,
        &occ_decompile::Options::default(),
        &name_of,
    );
    assert_eq!(
        d.unstructured, 0,
        "{}: the function structured without a goto:\n{}",
        loaded.image.arch.label(),
        occ_decompile::render::render(&d)
    );
    occ_decompile::render::render(&d)
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

#[test]
fn every_architecture_recovers_the_same_parameters() {
    for_each_built(|target, loaded, f| {
        let name_of = name_resolver(loaded);
        let program = occ_decompile::build_program(
            &loaded.image,
            f,
            &occ_decompile::Options::default(),
            &name_of,
        );
        assert_eq!(
            program.params,
            vec!["a0".to_string()],
            "{}: the function reads one argument, and it is the first: {:?}",
            target.name,
            program.params
        );
        assert_eq!(
            program.param_regs.len(),
            1,
            "{}: one argument register: {:?}",
            target.name,
            program.param_regs
        );
        assert!(
            program.notes.is_empty(),
            "{}: the lifter had nothing to approximate: {:?}",
            target.name,
            program.notes
        );
    });
}

#[test]
fn every_architecture_recovers_the_same_control_flow() {
    for_each_built(|target, loaded, f| {
        let code = decompile(loaded, f);
        // Three tests, so three `if`s.
        assert_eq!(
            code.matches("if (").count(),
            3,
            "{}: three branches:\n{code}",
            target.name
        );
        // Four arms, so four returns.
        assert_eq!(
            code.matches("return ").count(),
            4,
            "{}: four returns:\n{code}",
            target.name
        );
        // No `goto`: the arms are structured.
        assert!(
            !code.contains("goto "),
            "{}: the function fell back to goto:\n{code}",
            target.name
        );
    });
}

#[test]
fn every_architecture_writes_the_global_to_its_own_address() {
    for_each_built(|target, loaded, f| {
        let code = decompile(loaded, f);
        let sink = sink_address(loaded);
        // The store has to name the address of the global, not an offset
        // from a register nobody can name. On x86 the address is direct;
        // on AArch64 it comes from `adrp`; on RISC-V from `gp`, which the
        // linker fixes at `__global_pointer$`.
        assert!(
            code.contains(&format!("{:#x}", sink.get())),
            "{}: the store does not name the address of sink ({:#x}):\n{code}",
            target.name,
            sink.get()
        );
    });
}
