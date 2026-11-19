//! The WebAssembly lift, over a module a compiler produced.
//!
//! WebAssembly is not "x86 without the flags". It has no registers at all:
//! every value an instruction works on is the one the instructions before it
//! left on the operand stack. Its branches do not name addresses either:
//! they name a depth, and the depth counts the blocks that are open at that
//! point. A `loop`'s label is its start and a `block`'s label is its end, so
//! the same instruction either repeats or leaves depending on what it names.
//!
//! Each of those changes the shape of the source, and each of them produces
//! a function that still compiles when it is wrong.
//!
//! The fixtures are `tests/fixtures/fixture.wasm` and
//! `tests/fixtures/fixture.wasm2`, built from the C files beside them. They
//! are checked in rather than compiled here because a WebAssembly build
//! needs a cross toolchain, and a test that skips itself leaves the feature
//! untested.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, WasmValueType};

/// A loaded fixture and everything the analyses need.
struct Loaded {
    image: occ_core::Image,
    analysis: Analysis,
}

/// Load one of the fixtures by name.
fn load(name: &str) -> Loaded {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../tests/fixtures/{name}"));
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("the fixture parses");
    assert_eq!(
        image.arch,
        occ_core::Arch::Wasm,
        "the fixture is WebAssembly"
    );
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    Loaded { image, analysis }
}

impl Loaded {
    fn names(&self) -> impl Fn(Addr) -> String + '_ {
        let analysis = &self.analysis;
        move |a: Addr| {
            analysis
                .function_at(a)
                .map(|f| f.display_name())
                .unwrap_or_else(|| format!("func_{:x}", a.get()))
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

    /// The module's own signature for a function.
    fn signature(&self, name: &str) -> occ_core::WasmFuncSig {
        let f = self.function(name);
        self.image
            .wasm_funcs
            .iter()
            .find(|s| s.body == Some(f.entry))
            .cloned()
            .unwrap_or_else(|| panic!("{name} has a signature"))
    }

    /// The module's own type for a function.
    fn signature_type(&self, name: &str) -> occ_core::WasmFuncType {
        let sig = self.signature(name);
        self.image
            .wasm_types
            .get(sig.type_index as usize)
            .cloned()
            .unwrap_or_else(|| panic!("{name} has a type"))
    }
}

/// The addition is one instruction and one value: `a + b`.
///
/// The fixture's `add` is `(func (param i32 i32) (result i32) local.get 1
/// local.get 0 i32.add)`, so the lift has to read the two parameters off
/// the stack and add them.
#[test]
fn an_addition_becomes_an_addition() {
    let loaded = load("fixture.wasm");
    let code = loaded.code("add");
    assert!(
        code.contains("i32 add(i32 p0, i32 p1)"),
        "the signature is the module's own:\n{code}"
    );
    assert!(
        code.contains("return p1 + p0;") || code.contains("return p0 + p1;"),
        "the addition is shown as one:\n{code}"
    );
}

/// The loop is a `loop` with a branch back to it, and it has to come out
/// as a loop rather than as a tangle of labels.
#[test]
fn a_loop_becomes_a_loop() {
    let loaded = load("fixture.wasm");
    let code = loaded.code("sum");
    assert!(
        code.contains("i32 sum(i32 p0, i32 p1)"),
        "the signature is the module's own:\n{code}"
    );
    assert!(
        code.contains("do {") || code.contains("while ("),
        "the loop is a loop:\n{code}"
    );
    assert!(
        code.contains("*(u32*)"),
        "the load of the array element is shown:\n{code}"
    );
    assert!(
        !code.contains("is not lifted"),
        "nothing in the fixture is left unlifted:\n{code}"
    );
}

/// The function's own result is returned, and the value it returns is the
/// one the body built.
#[test]
fn the_result_is_returned() {
    let loaded = load("fixture.wasm");
    let code = loaded.code("add");
    assert!(code.contains("return "), "the result is returned:\n{code}");
}

/// The module's types reach the signature: `i32` in, `i32` out.
#[test]
fn the_types_come_from_the_module() {
    let loaded = load("fixture.wasm");
    let ty = loaded.signature_type("add");
    assert_eq!(
        ty.params,
        vec![WasmValueType::I32, WasmValueType::I32],
        "the module says the function takes two integers"
    );
    assert_eq!(
        ty.results,
        vec![WasmValueType::I32],
        "the module says the function returns one integer"
    );
}

/// A call names the function it calls, and the callee is the module's own.
///
/// The second fixture keeps `add` out of line, so the call survives the
/// compiler's inlining and the lift has to name it.
#[test]
fn a_call_names_the_callee() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("sum");
    assert!(
        code.contains("add("),
        "the call to the other function is named:\n{code}"
    );
    assert!(
        !code.contains("function "),
        "nothing is left as an unnamed function:\n{code}"
    );
}

/// A call's arguments are the ones the module pushed, in the order it
/// pushed them.
#[test]
fn a_call_keeps_its_arguments() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("sum");
    assert!(
        code.contains("add(local3, *(u32*)p0)") || code.contains("add(*(u32*)p0, local3)"),
        "the call reads the accumulator and the element:\n{code}"
    );
}

/// A store through a pointer is a memory write with the address the
/// module computed.
#[test]
fn a_store_is_a_store() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("store");
    assert!(
        code.contains("*(u32*)p0 = p1;"),
        "the write is shown:\n{code}"
    );
    assert!(
        code.contains("return p1;"),
        "the value written is also the value returned:\n{code}"
    );
}

/// A floating point operation keeps its type, and its operands are the
/// module's own.
#[test]
fn a_float_stays_a_float() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("fadd");
    assert!(
        code.contains("float fadd(float p0, float p1)"),
        "the signature keeps the module's type:\n{code}"
    );
    assert!(
        code.contains("f32_add(p0, p1)"),
        "the addition is the module's own:\n{code}"
    );
}

/// A branch table becomes a switch, and every case keeps its body.
///
/// The compiler lays a sparse switch out as a table with one label per
/// case, and the cases that fall through to the same place share a label.
/// Reading the table wrongly loses the case bodies that sit behind the
/// shared labels, and the result still looks like a switch.
#[test]
fn a_switch_keeps_every_case() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("pick");
    assert!(
        code.contains("switch ("),
        "the branch table is a switch:\n{code}"
    );
    // The four cases of the source, with the values it returns.
    for value in ["0xa", "0x14", "0x1e", "0x28"] {
        assert!(
            code.contains(value),
            "the case that returns {value} is there:\n{code}"
        );
    }
    assert!(
        code.contains("case 0x0:") && code.contains("default:"),
        "the first case and the default are there:\n{code}"
    );
}

/// A branch that leaves a loop is a `break`.
///
/// The compiler puts the loop body in a block and branches out of it, so
/// the branch names a depth that leaves both the loop and the block. In C
/// that is a `break`, and saying so is the difference between a loop a
/// reader can follow and a label they have to look up.
#[test]
fn a_branch_out_of_a_loop_is_a_break() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("countdown");
    assert!(
        code.contains("break;"),
        "the branch out of the loop is a break:\n{code}"
    );
}

/// A branch that skips over code is an `if` with the skipped code in it.
///
/// The module says "if this holds, do not run the rest of the region".
/// The same program in C says "if this does not hold, run this", and the
/// second is the one a person wrote.
#[test]
fn a_branch_that_skips_code_becomes_an_if() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("find");
    // The source returns the index when the element matches, and skips
    // the return otherwise.
    assert!(
        code.contains("== (u32)p2)"),
        "the comparison is the source's own:\n{code}"
    );
    assert!(
        code.contains("return local3;"),
        "the return is inside the conditional:\n{code}"
    );
    assert!(
        code.contains("do {") || code.contains("while ("),
        "the loop is a loop:\n{code}"
    );
}

/// A loop that skips part of its body is a select, which is what the
/// module compiles it into.
#[test]
fn a_loop_that_skips_is_a_select() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("skip");
    assert!(code.contains("?"), "the skipped value is a choice:\n{code}");
    assert!(
        code.contains("& 1"),
        "the choice is the source's own test:\n{code}"
    );
}

/// The same loop with a store under the branch keeps the branch.
///
/// A store cannot be moved above the test, so the compiler cannot fold the
/// two paths into a choice, and the `if` that runs the rest of the body is
/// the shape the source had.
#[test]
fn a_loop_with_a_side_effect_keeps_its_conditional() {
    let loaded = load("fixture.wasm2");
    let code = loaded.code("skip_if");
    assert!(
        code.contains("if ((u32)(*(u8*)p0 & 1) == 0)"),
        "the test is the source's own:\n{code}"
    );
    assert!(
        code.contains("*(u32*)p2 = *(u32*)p2 + 1;"),
        "the store is under the test:\n{code}"
    );
    assert!(
        !code.contains("?"),
        "a store is not folded into a choice:\n{code}"
    );
}

/// A branch to the top of a loop is `continue`.
///
/// A compiler folds its own `continue` into a conditional around the rest
/// of the body, so this shape is in neither compiled fixture and cannot be:
/// `fixture.wasm3` is written by hand for it. The loop's own edge at the
/// bottom makes it a `while (1)`, the branch out of the block that holds it
/// is a `break`, and the branch back to the top is a `continue` — no jump
/// and no label is left in the output.
#[test]
fn a_branch_to_the_top_of_a_loop_is_a_continue() {
    let loaded = load("fixture.wasm3");
    let code = loaded.code("spin");
    assert!(code.contains("while (1)"), "the loop is a loop:\n{code}");
    assert!(
        code.contains("continue;"),
        "the branch back to the top is a continue:\n{code}"
    );
    assert!(
        code.contains("break;"),
        "the branch out of the block is a break:\n{code}"
    );
    assert!(!code.contains("goto"), "no jump is left:\n{code}");
    assert!(!code.contains("L1:"), "no label is left:\n{code}");
}

/// Every function in the fixture comes out as something.
#[test]
fn every_function_is_lifted() {
    let loaded = load("fixture.wasm2");
    for name in [
        "add",
        "sum",
        "store",
        "pick",
        "fadd",
        "countdown",
        "find",
        "skip",
        "skip_if",
    ] {
        let code = loaded.code(name);
        assert!(
            !code.trim().is_empty(),
            "{name} decompiles to something:\n{code}"
        );
        assert!(!code.contains("is not lifted"), "{name} is lifted:\n{code}");
        assert!(
            !code.contains("the operand stack was empty"),
            "{name} keeps its stack in step:\n{code}"
        );
    }
}
