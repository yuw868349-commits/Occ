//! What the lifted text says about the machine underneath.
//!
//! Two properties are checked here, and both of them are the kind that fail
//! quietly: a decompiler that misreads a register still produces source that
//! compiles.
//!
//! * **The zero register is not the stack pointer.** AArch64 numbers both 31,
//!   so `csel w0, w2, wzr, eq` printed its false arm as `sp`, and a
//!   comparison against `xzr` compared against a register nobody had set.
//! * **The argument registers follow the binary's calling convention.**
//!   Windows x64 passes its first argument in `rcx`, System V in `rdi`. A
//!   Windows binary read with the System V order has every argument one or
//!   two places off, and the result still looks like a program.
//!
//! The rest of the file checks the operand forms that carry a whole
//! expression — a shift on an ALU operand, an extension on an index, a
//! post-indexed write-back — because each of them used to disappear and take
//! an instruction with it.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, Format};

/// A loaded fixture and everything the analyses need.
struct Loaded {
    image: occ_core::Image,
    analysis: Analysis,
}

fn load(name: &str) -> Loaded {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    let mut data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("the fixture parses");
    // Relocatable objects carry their addresses in relocation records, not
    // in the bytes, so the bytes are patched before anything reads them —
    // exactly what a session does when it opens one.
    if let Some(applied) = occ_bin::reloc::apply(&mut data, &image) {
        assert!(
            applied.out_of_range == 0,
            "{name}: a relocation did not fit its instruction: {applied:?}"
        );
    }
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    Loaded { image, analysis }
}

/// True when `code` contains `token` as a whole identifier.
///
/// A plain substring test would find `sp` inside `dispatch` and pass a
/// failure off as a success.
fn has_token(code: &str, token: &str) -> bool {
    code.split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|t| t == token)
}

impl Loaded {
    fn names(&self) -> impl Fn(Addr) -> String + '_ {
        let analysis = &self.analysis;
        move |a: Addr| {
            analysis
                .function_at(a)
                .map(|f| f.display_name())
                .unwrap_or_else(|| format!("sub_{:x}", a.get()))
        }
    }

    fn function(&self, name: &str) -> occ_core::Function {
        self.analysis
            .functions
            .iter()
            .find(|f| f.display_name() == name)
            .unwrap_or_else(|| panic!("{name} is recovered"))
            .clone()
    }

    fn program(&self, name: &str) -> occ_decompile::Program {
        let f = self.function(name);
        occ_decompile::build_program(
            &self.image,
            &f,
            &occ_decompile::Options::default(),
            &self.names(),
        )
    }

    fn code(&self, name: &str) -> String {
        let f = self.function(name);
        let d = occ_decompile::decompile(
            &self.image,
            &f,
            &occ_decompile::Options::default(),
            &self.names(),
        );
        occ_decompile::render::render(&d)
    }
}

#[test]
fn aarch64_zero_register_is_not_the_stack_pointer() {
    let loaded = load("fixture.aarch64.o");
    let code = loaded.code("dispatch");

    // `csinv w0, w2, wzr, eq` — the false arm is the complement of the zero
    // register, so all ones in 32 bits. It used to print as `sp`.
    assert!(
        code.contains("0xffffffff"),
        "the complement of the zero register is not all ones:\n{code}"
    );
    assert!(
        !has_token(&code, "sp"),
        "the zero register was lifted as the stack pointer:\n{code}"
    );
    // Every branch in this function reads a register, not a flag, so no
    // condition may be left unresolved.
    assert!(
        !code.contains("<condition>"),
        "a branch condition was not recovered:\n{code}"
    );
}

#[test]
fn a_tail_call_returns_what_the_callee_returns() {
    // `dispatch` ends one arm with `mov w0, w1; b ext` — a tail call. The
    // call's result is the function's result, so the arm has to return it.
    // Dropping it printed `return;` inside a function declared `u64`, which
    // said the arm returned nothing when it returned whatever `ext` did.
    let loaded = load("fixture.aarch64.o");
    let code = loaded.code("dispatch");
    assert!(
        code.contains("return sub_2000("),
        "the tail call's result is not returned:\n{code}"
    );
}

#[test]
fn aarch64_compare_and_branch_forms_are_lifted() {
    // `dispatch` uses `cbz` for one arm and `b.eq`/`b.gt`/`b.ne` for the
    // rest. Dropping the `cbz` arm left the function with three branches
    // and a hole where the fourth was.
    let loaded = load("fixture.aarch64.o");
    let code = loaded.code("dispatch");
    assert!(
        code.matches("if (").count() >= 3,
        "the switch has four arms, so at least three tests:\n{code}"
    );
    // The arm `cbz w3` guards returns the sum; `csinv` returns either the
    // difference or all ones.
    assert!(
        code.contains("if ((u32)x3_1 == 0)"),
        "the `cbz` arm is missing its condition:\n{code}"
    );
    assert!(
        code.matches("return").count() >= 4,
        "every arm returns:\n{code}"
    );
}

#[test]
fn aarch64_shifted_and_extended_operands_survive_lifting() {
    // `add x3, x2, w1, sxtw #2` is the end-pointer of the loop, and
    // `add w1, w0, w0, lsl #1` is `w0 * 3`. Both carry a modifier on their
    // third operand; an instruction that could not parse them vanished.
    let loaded = load("fixture.aarch64.o");
    let code = loaded.code("sum");
    assert!(
        !code.contains("unparsed"),
        "an instruction with a modified operand was dropped:\n{code}"
    );
    assert!(
        !code.contains("not lifted"),
        "an instruction was not lifted:\n{code}"
    );
    // The extension is a sign extension of the low word, so it reads as a
    // 64-bit value, and the shift stays outside it.
    // The register is the second parameter, so it prints under its
    // argument name; the extension and the shift are what the test is about.
    assert!(
        code.contains("(i64)a1 << 2"),
        "the `sxtw #2` operand lost its extension or its shift:\n{code}"
    );
    assert!(
        code.contains("(u32)x0_3 << 1"),
        "the `lsl #1` operand lost its shift:\n{code}"
    );
}

#[test]
fn aarch64_post_indexed_accesses_move_their_base() {
    // `ldr w1, [x2], #4` both loads and walks. Without the walk the loop
    // never advances and the lifted function is an infinite loop; without
    // the load the walked value is not there at all.
    let loaded = load("fixture.aarch64.o");
    let code = loaded.code("sum");
    assert!(
        code.contains("*(u32*)"),
        "the post-indexed load is missing:\n{code}"
    );
    assert!(
        code.contains("x2_2 = x2_2 + 4"),
        "the post-indexed write-back is missing:\n{code}"
    );
}

#[test]
fn aarch64_recovers_only_the_parameters_the_body_reads() {
    // `sum` takes a pointer and a count. The third parameter a previous
    // version invented was the index register of a `sxtw` operand that had
    // been dropped, read later as if it were incoming.
    let loaded = load("fixture.aarch64.o");
    let program = loaded.program("sum");
    assert_eq!(
        program.params,
        vec!["a0".to_string(), "a1".to_string()],
        "two arguments: {:?}",
        program.params
    );
    assert_eq!(program.param_regs, vec!["x0".to_string(), "x1".to_string()]);
}

#[test]
fn windows_x64_arguments_follow_the_windows_order() {
    let loaded = load("pe_fixture.exe");
    assert_eq!(loaded.image.format, Format::Pe);
    // Windows x64 passes four arguments in registers, in this order.
    let order = ["rcx", "rdx", "r8", "r9"];
    let mut widest = 0usize;
    for f in &loaded.analysis.functions {
        let program = occ_decompile::build_program(
            &loaded.image,
            f,
            &occ_decompile::Options::default(),
            &loaded.names(),
        );
        let mut last = None;
        for (i, reg) in program.param_regs.iter().enumerate() {
            let at = order.iter().position(|r| r == reg).unwrap_or_else(|| {
                panic!(
                    "{}: `{reg}` is not a Windows argument register",
                    f.display_name()
                )
            });
            assert!(
                last.is_none_or(|l: usize| at > l),
                "{}: arguments out of order: {:?}",
                f.display_name(),
                program.param_regs
            );
            last = Some(at);
            // A parameter's printed name is its position in the ABI order,
            // so the two have to agree.
            assert_eq!(
                program.params[i],
                format!("a{at}"),
                "{}: {} is argument {at}",
                f.display_name(),
                reg
            );
        }
        widest = widest.max(program.param_regs.len());
    }
    assert_eq!(
        widest, 4,
        "the fixture passes four arguments in registers somewhere"
    );
}

#[test]
fn windows_x64_takes_its_arguments_from_the_first_four_registers() {
    // A function that reads the fourth argument register is a four-argument
    // function; nothing may look for an argument in `rdi` or `rsi`, which
    // are ordinary saved registers under this convention.
    let loaded = load("pe_fixture.exe");
    let mut four = Vec::new();
    for f in &loaded.analysis.functions {
        let program = occ_decompile::build_program(
            &loaded.image,
            f,
            &occ_decompile::Options::default(),
            &loaded.names(),
        );
        if program.param_regs.len() == 4 {
            four.push(program.param_regs.clone());
        }
    }
    assert!(
        four.iter()
            .any(|r| r == &["rcx", "rdx", "r8", "r9"].iter().map(|s| s.to_string()).collect::<Vec<_>>()),
        "a four-argument function reads all four registers: {four:?}"
    );
}

#[test]
fn a_parameter_lists_every_position_up_to_the_last_one_used() {
    // A function that reads the second argument register takes two
    // arguments. Naming only the registers the body happens to touch gives
    // a one-parameter function whose single argument is `a1`, and a caller
    // reading that prototype passes its argument in `rcx` while the body
    // reads `rdx`.
    let loaded = load("pe_fixture.exe");
    let order = ["rcx", "rdx", "r8", "r9"];
    let mut holes = 0;
    for f in &loaded.analysis.functions {
        let program = occ_decompile::build_program(
            &loaded.image,
            f,
            &occ_decompile::Options::default(),
            &loaded.names(),
        );
        assert_eq!(
            program.params.len(),
            program.param_regs.len(),
            "{}: a parameter name per register",
            f.display_name()
        );
        for (i, name) in program.params.iter().enumerate() {
            assert_eq!(
                name,
                &format!("a{i}"),
                "{}: parameter {i} is named {name}",
                f.display_name()
            );
        }
        // The last parameter has to be the highest register the body reads,
        // and every register below it has to be listed too.
        if let Some(last) = program.param_regs.last() {
            let at = order.iter().position(|r| r == last).expect("an ABI register");
            assert_eq!(
                program.param_regs.len(),
                at + 1,
                "{}: reads {last} but lists {:?}",
                f.display_name(),
                program.param_regs
            );
            if at > 0 {
                holes += 1;
            }
        }
    }
    assert!(
        holes > 0,
        "the fixture has a function that reads a later argument without the first"
    );
}

/// A binary built from `source` with the host compiler, or `None` when there
/// is no compiler to build it with.
fn compile(name: &str, source: &str) -> Option<Loaded> {
    let dir = std::env::temp_dir().join(format!("occ-lift-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    // `load` reads the file into memory, so the directory is scratch that has
    // no reason to survive the run.
    occ_sys::cleanup::remove_dir_at_exit(&dir);
    let src = dir.join(format!("{name}.c"));
    let out = dir.join(name);
    std::fs::write(&src, source).ok()?;
    let status = std::process::Command::new("cc")
        .arg("-O1")
        .arg("-o")
        .arg(&out)
        .arg(&src)
        .output()
        .ok()?;
    if !status.status.success() {
        eprintln!(
            "skipping {name}: cc failed:\n{}",
            String::from_utf8_lossy(&status.stderr)
        );
        return None;
    }
    Some(load(out.to_str()?))
}

#[test]
fn a_call_through_a_register_names_the_value_it_uses() {
    // `call rax` reads whatever the register holds *at that point*. Carrying
    // the target as text meant it never passed through the renamer, so the
    // call printed `(*rax)` — the value from the function's first instruction
    // — however many times the body had written the register since.
    let Some(loaded) = compile(
        "fp",
        r#"
typedef int (*op)(int, int);
int probe(op f, int a) { int r = f(a, 1); if (r > 0) r += 2; return r; }
int twice(op f, int a) { return f(a, 1) + f(a, 2); }
int main(int argc, char **argv) { return probe(0, argc); }
"#,
    ) else {
        return;
    };
    for name in ["probe", "twice"] {
        let code = loaded.code(name);
        let call = code
            .lines()
            .find(|l| l.contains("(*"))
            .unwrap_or_else(|| panic!("{name}: no indirect call:\n{code}"));
        // A bare register name has no version; a value the body produced does.
        let inner = call
            .split("(*")
            .nth(1)
            .and_then(|r| r.split(')').next())
            .unwrap_or_default();
        assert!(
            inner.chars().any(|c| c.is_ascii_digit()),
            "{name}: the call target is not a value the body defined: {call}"
        );
    }
}

#[test]
fn a_function_that_returns_a_call_result_returns_it() {
    // The call defines the return register, so a body whose only value is the
    // callee's result has a value to return. Without that, the body printed
    // `return;` and the call's result silently disappeared.
    let Some(loaded) = compile(
        "ret",
        r#"
int target(int a) { return a * 3; }
int wrapper(int a) { return target(a); }
"#,
    ) else {
        return;
    };
    let code = loaded.code("wrapper");
    assert!(
        code.contains("return ") && !code.contains("return;"),
        "the wrapper returns the callee's value:\n{code}"
    );
}

#[test]
fn a_riscv_computed_jump_names_the_value_it_goes_through() {
    // `auipc t1, 0x2; jr t1` lifted to `goto *(<indirect target>)`: the block
    // had computed the address and the lifter threw away the register that
    // held it, so the one thing a reader needs — where the target came from —
    // was missing. The answer here is a constant because the address is a
    // link-time one; what matters is that it is the value the code computed
    // and not the word "unknown".
    let loaded = load("fixture.riscv64.o");
    let code = loaded.code("dispatch");
    assert!(
        !code.contains("indirect target"),
        "the computed jump has no target expression:\n{code}"
    );
    let goto = code
        .lines()
        .find(|l| l.contains("goto *("))
        .unwrap_or_else(|| panic!("the computed jump is not rendered:\n{code}"));
    assert!(
        !goto.contains("unknown"),
        "the target expression is not a value the body defined: {goto}"
    );
}
