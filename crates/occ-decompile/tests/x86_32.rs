//! The 32-bit x86 lift, over an object a compiler produced.
//!
//! The instruction set is the one the 64-bit lifter already reads; what
//! changes is where everything lives. The registers are half as wide and
//! have different names, the stack slots are four bytes instead of eight,
//! and every argument arrives on the stack instead of in a register. Each of
//! those is invisible in a unit test that hands the lifter text, and each of
//! them produces source that still compiles when it is wrong.
//!
//! The fixture is `tests/fixtures/fixture.x86-32.o`, built from the C file
//! beside it. It is checked in rather than compiled here because a 32-bit
//! build needs a toolchain most machines do not have installed, and a test
//! that skips itself leaves the feature untested.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::Addr;

/// A loaded fixture and everything the analyses need.
struct Loaded {
    image: occ_core::Image,
    analysis: Analysis,
}

fn load() -> Loaded {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/fixture.x86-32.o");
    let mut data =
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("the fixture parses");
    assert_eq!(image.arch, occ_core::Arch::X86, "the fixture is 32-bit x86");
    if let Some(applied) = occ_bin::reloc::apply(&mut data, &image) {
        assert_eq!(
            applied.out_of_range, 0,
            "a relocation did not fit its instruction: {applied:?}"
        );
    }
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
fn a_32_bit_function_is_lifted_rather_than_outlined() {
    let loaded = load();
    let code = loaded.code("add");
    // The outline path is the failure this whole module exists to remove.
    assert!(!code.contains("not lifted"), "{code}");
    assert!(!code.contains("body not lifted"), "{code}");
    // And the width is read from the image, not assumed: a 32-bit function
    // that returns `u64` tells its caller about an upper half that never
    // existed.
    assert!(!code.contains("u64"), "the word is four bytes wide:\n{code}");
    assert!(code.contains("u32 add("), "{code}");
}

#[test]
fn both_stack_arguments_become_parameters() {
    let loaded = load();
    let code = loaded.code("add");
    // `mov eax, [esp+8]; add eax, [esp+4]` reads the two arguments in the
    // opposite order from the one they were pushed in, and both of them are
    // parameters of the function rather than anonymous frames.
    assert_eq!(
        loaded.program("add").params,
        vec!["a0".to_string(), "a1".to_string()],
        "{code}"
    );
    assert!(code.contains("a0 + a1") || code.contains("a1 + a0"), "{code}");
    // A slot name in the body would mean the alias table never saw it.
    assert!(!code.contains("arg_4_u32"), "{code}");
    assert!(!code.contains("local_"), "{code}");
}

#[test]
fn a_pushed_argument_is_the_value_it_was_before_the_register_was_reused() {
    let loaded = load();
    // `mix` is `add(a * 3, b + 1)`, and the compiler computes the second
    // argument first and leaves it in `eax` while it computes the first in
    // the same register: `push eax; lea eax, [edx+edx*2]; push eax`. A lift
    // that pushes a bare register name gives the call the same value twice.
    let code = loaded.code("mix");
    assert!(code.contains("a1 + 1"), "the second argument:\n{code}");
    assert!(code.contains("a0 + a0 * 2"), "the first argument:\n{code}");
    let args = code
        .split_once('(')
        .and_then(|(_, rest)| rest.rsplit_once(')'))
        .map(|(a, _)| a.to_string())
        .unwrap_or_default();
    let (first, second) = args
        .split_once(", ")
        .unwrap_or_else(|| panic!("the call has two arguments:\n{code}"));
    assert_ne!(
        first, second,
        "the two arguments are different values:\n{code}"
    );
}

#[test]
fn a_function_that_passes_one_value_twice_still_pushes_it_twice() {
    let loaded = load();
    let code = loaded.code("twice");
    assert!(code.contains("(a0, a0)"), "{code}");
}

#[test]
fn a_call_inside_the_object_names_its_target() {
    let loaded = load();
    // `twice` is `add(x, x)`, and `add` is defined in the same object. Getting
    // the name back needs two things to line up: the relocation has to be
    // applied (it is `R_386_PC32`, whose addend is `-4` and has to be read as
    // signed), and the symbol's `st_value` has to be rebased onto the section
    // the loader placed. Either one wrong leaves `sub_44` — an address that
    // looks like a function and is not.
    let code = loaded.code("twice");
    assert!(code.contains("add(a0, a0)"), "{code}");
    assert!(
        !code.contains("sub_"),
        "a call to a symbol in this object is not anonymous:\n{code}"
    );
}

#[test]
fn the_branches_read_their_arguments_at_the_right_width() {
    let loaded = load();
    let code = loaded.code("classify");
    // `test eax, eax; js` is a signed test of a 32-bit value. Lifting it at
    // eight bytes reads a half of the register the comparison never looked
    // at, and `js` on that half is a different question.
    assert!(code.contains("(i32)") && code.contains("< 0"), "{code}");
    assert!(code.contains("<= 9"), "{code}");
    // Every arm writes the global before it returns, and a store with a
    // constant address is where the `ds:` prefix used to be lost.
    assert_eq!(
        code.matches("*(u32*)0x").count(),
        4,
        "one store per arm:\n{code}"
    );
    assert!(!code.contains("unparsed"), "{code}");
}

#[test]
fn a_global_is_read_at_the_width_of_the_register_it_lands_in() {
    let loaded = load();
    // `mov eax, ds:0x...` states no size: the destination register does. A
    // lift that defaults the missing size to the machine word reads four
    // bytes past the global and returns the next one along with it, which
    // still compiles and is still wrong.
    let code = loaded.code("read_sink");
    assert!(code.contains("*(u32*)"), "a four-byte load:\n{code}");
    assert!(!code.contains("u64"), "not a word-sized load:\n{code}");
    assert!(code.contains("0xf8"), "the global's address:\n{code}");
    // The same address read and written in one function has to keep one
    // width: two reads and a write, all four bytes.
    let bump = loaded.code("bump");
    assert_eq!(
        bump.matches("*(u32*)").count(),
        3,
        "read, write, read:\n{bump}"
    );
    assert!(!bump.contains("u64"), "{bump}");
}

#[test]
fn a_loop_over_a_stack_argument_stays_a_loop() {
    let loaded = load();
    let code = loaded.code("sum_to");
    assert!(code.contains("do {"), "{code}");
    assert!(code.contains("} while ("), "{code}");
    assert!(!code.contains("goto "), "{code}");
    // The counter is `ecx`, four bytes wide, and a loop that compares the
    // whole register never terminates when the compiler only wrote the low
    // half.
    assert!(code.contains("(u32)"), "{code}");
}

#[test]
fn a_callee_that_removes_its_arguments_states_the_arity() {
    let loaded = load();
    // `mov eax, [esp+4]; ret 8` reads one argument and removes two. The
    // count is in the instruction, and a function that ignores its second
    // argument still takes one: its callers push both.
    let program = loaded.program("only_first");
    assert_eq!(
        program.params,
        vec!["a0".to_string(), "a1".to_string()],
        "the epilogue states two arguments"
    );
    let code = loaded.code("only_first");
    assert!(code.contains("u32 only_first(u32 a0, u32 a1)"), "{code}");
    assert!(code.contains("return a0;"), "{code}");
    assert!(!code.contains("a1;"), "the body never reads it:\n{code}");
}

#[test]
fn a_frame_pointer_and_a_stack_pointer_name_the_same_slot() {
    let loaded = load();
    // `add` is frameless and reads `[esp+8]`; `mix` saves `ebp` and reads
    // `[esp+8]` after the push, which is the same four bytes one slot down.
    // Both have to name the same parameter, or one of the two functions
    // prints its arguments as anonymous frames.
    for name in ["add", "mix"] {
        let program = loaded.program(name);
        assert_eq!(
            program.param_regs.len(),
            program.params.len(),
            "{name}: every parameter has a location"
        );
        for (i, slot) in program.param_regs.iter().enumerate() {
            assert!(
                slot.ends_with("_u32"),
                "{name}: argument {i} is a four-byte slot, not {slot}"
            );
        }
    }
}
