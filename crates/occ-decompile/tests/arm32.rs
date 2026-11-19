//! The 32-bit ARM lift, over an object a compiler produced.
//!
//! A32 is not "x86 without the flags". Every instruction carries a condition
//! that may be `al`, the program counter is a general register that reads as
//! the address plus eight, a multiply can produce twice as many bits as its
//! factors, and a call is a branch with a link bit rather than a separate
//! instruction. Each of those changes the shape of the source, and each of
//! them produces a function that still compiles when it is wrong.
//!
//! The fixture is `tests/fixtures/fixture.arm32.o`, built from the C file
//! beside it. It is checked in rather than compiled here because an ARM
//! build needs a cross toolchain, and a test that skips itself leaves the
//! feature untested.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::Addr;

/// A loaded fixture and everything the analyses need.
struct Loaded {
    image: occ_core::Image,
    analysis: Analysis,
}

fn load() -> Loaded {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/fixture.arm32.o");
    let mut data =
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("the fixture parses");
    assert_eq!(image.arch, occ_core::Arch::Arm, "the fixture is 32-bit ARM");
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
fn the_arguments_arrive_in_registers() {
    let loaded = load();
    let code = loaded.code("add");
    // A32 passes its first four arguments in r0 to r3, and the body reads
    // two of them. Both are parameters of the function rather than
    // anonymous registers.
    assert_eq!(
        loaded.program("add").params,
        vec!["a0".to_string(), "a1".to_string()],
        "{code}"
    );
    assert!(code.contains("a0") && code.contains("a1"), "{code}");
}

#[test]
fn a_call_inside_the_object_names_its_target() {
    let loaded = load();
    // `twice` is `add(x, x)`, and `add` is defined in the same object.
    // Getting the name back needs the relocation to be applied — it is
    // `R_ARM_JUMP24`, whose addend is signed — and the symbol's value to
    // be rebased onto the section the loader placed.
    let code = loaded.code("twice");
    assert!(code.contains("add(a0, a0)"), "{code}");
    assert!(
        !code.contains("sub_"),
        "a call to a symbol in this object is not anonymous:\n{code}"
    );
}

#[test]
fn a_conditional_move_keeps_its_condition() {
    let loaded = load();
    let code = loaded.code("classify");
    // `movwlt r0, #1` computes a value on one path only. A lift that drops
    // the condition writes both values and lets the last one win, which is
    // a different function.
    assert!(code.contains("< 0xa"), "the comparison:\n{code}");
    assert!(code.contains("? 1 : 2"), "the conditional move:\n{code}");
    assert!(!code.contains("unparsed"), "{code}");
}

#[test]
fn a_conditional_store_happens_only_when_the_condition_holds() {
    let loaded = load();
    let code = loaded.code("set_if");
    // `streq r1, [r0]` is a store that happens on one path. The store has
    // to be inside the condition: a store that always happens writes the
    // global on every call.
    assert!(code.contains("if ("), "{code}");
    assert!(code.contains("*(u32*)"), "the store:\n{code}");
    assert!(
        code.contains("= 7") || code.contains("= r1"),
        "the value that is stored:\n{code}"
    );
}

#[test]
fn the_frame_a_function_builds_is_not_part_of_what_it_computes() {
    let loaded = load();
    // `push {r4, r5, r11, lr}` saves the registers the caller owns so
    // that the body can use them, and the matching `pop` puts them back.
    // Nothing inside the function reads the slots, and printing the
    // traffic says the function stores four values it never uses.
    let code = loaded.code("chain");
    assert!(!code.contains("frame"), "the frame is not a value:\n{code}");
    assert!(!code.contains("*(u32*)"), "the frame is not a store:\n{code}");
    // The two calls are still there, and the second one is not a tail
    // call: the value it returns is added to the first one's.
    assert!(code.contains("add(a0, 1)"), "{code}");
    assert!(code.contains("add(a0, 2)"), "{code}");
}

#[test]
fn a_double_word_load_reads_two_words() {
    let loaded = load();
    let code = loaded.code("sum_point");
    // `ldrd r0, r1, [r0]` reads two words and writes two registers. The
    // second read is at the next address, and the base is the pointer the
    // instruction was given — not the word the first read put in r0.
    assert_eq!(
        code.matches("*(u32*)").count(),
        2,
        "two four-byte reads:\n{code}"
    );
    assert!(code.contains("+ 4"), "the second word is at the next address:\n{code}");
}

#[test]
fn a_conditional_return_returns_on_one_path() {
    let loaded = load();
    let code = loaded.code("first");
    // `bxlt lr` is a return that happens on one path. Read as a branch
    // with no target it looks like a jump out of the function, and the
    // rest of the function then looks unreachable — which is how a closed
    // form of a loop loses its body.
    assert!(code.contains("return"), "{code}");
    assert!(
        code.contains("if ((i32)a1 < 2)"),
        "the guard that returns early:\n{code}"
    );
    assert!(code.contains("while"), "the loop is still there:\n{code}");
}

#[test]
fn a_counting_loop_ends_on_the_counter() {
    let loaded = load();
    let code = loaded.code("first");
    // The loop ends on `subs r1, r1, #1` followed by `bne`, so the
    // condition is the counter reaching zero. The machine writes it as the
    // negation of "is equal to zero", and the reader wants the comparison.
    assert!(code.contains("!= 0"), "{code}");
    assert!(!code.contains("!((u32)"), "the negation is folded:\n{code}");
}

#[test]
fn a_wide_product_keeps_both_halves() {
    let loaded = load();
    let code = loaded.code("wide_mul");
    // `umull r0, r1, r1, r0` writes the two halves of a sixty-four bit
    // product. The low half alone is not the product: `1 << 31` has a low
    // half of zero, and a function that returns only the low half returns
    // the wrong number.
    assert!(code.contains("u64 wide_mul("), "the type is wide:\n{code}");
    assert!(code.contains("*"), "the product:\n{code}");
    assert!(!code.contains("u32 wide_mul("), "{code}");
}

#[test]
fn a_signed_product_is_signed() {
    let loaded = load();
    let code = loaded.code("signed_wide");
    // `smull` is the same multiply read as a signed number, and its high
    // half is not the unsigned one's.
    assert!(code.contains("i64 signed_wide("), "{code}");
    assert!(code.contains("(i64)"), "the factors are widened signed:\n{code}");
}

#[test]
fn a_shift_says_which_bit_fell_off_the_end() {
    let loaded = load();
    let code = loaded.code("sum_to");
    // `lsrs r2, r2, #1` followed by `rrx r1, r1` moves a sixty-four bit
    // number one bit to the right. The carry out of the shift is the bit
    // that comes back in at the top, and the two halves of the number are
    // the two halves of one product.
    //
    // The carry is the bit that *fell off the end*, so it is the low bit of
    // the value before the shift. Reading it from the shifted value would
    // take the second bit of the number instead of the first, and the
    // product would lose its top bit: the shift's own result is dead here,
    // because only the flag it sets is used.
    assert!(code.contains("rcr("), "the rotate through the carry:\n{code}");
    assert!(
        code.contains("& 1) == 1"),
        "the carry is the bit that fell off the end:\n{code}"
    );
}

#[test]
fn a_call_through_a_pointer_passes_only_its_arguments() {
    let loaded = load();
    let code = loaded.code("dispatch");
    // `mov r3, r0; mov r0, r1; mov r1, r2; blx r3` passes two arguments
    // and keeps the pointer in r3. Counting r3 as a third argument invents
    // one.
    assert!(code.contains("r3(a1, a2)"), "{code}");
}

#[test]
fn a_global_is_read_at_the_width_of_the_register_it_lands_in() {
    let loaded = load();
    // `movw r0, #0; movt r0, #0; ldr r0, [r0]` reaches a global through
    // two halves of its address. The load is four bytes wide, because the
    // register it lands in is four bytes wide.
    let code = loaded.code("read_sink");
    assert!(code.contains("*(u32*)"), "a four-byte load:\n{code}");
    assert!(!code.contains("u64"), "not a word-sized load:\n{code}");
}
