//! The 32-bit ARM lift, in Thumb.
//!
//! Thumb is not A32 with shorter encodings. The condition lives in an `it`
//! instruction that governs the four after it rather than in each
//! instruction, the program counter is not a register at all in the
//! sixteen-bit set, and a good third of the instruction set is Thumb's
//! own: the compare-and-branch, the extends, the reversals, the parallel
//! additions, the exchanged ones and the checksums.
//!
//! The fixture is `tests/fixtures/fixture.thumb32.o`, built from the C file
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
        .join("../../tests/fixtures/fixture.thumb32.o");
    let mut data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
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
fn a_thumb_function_is_lifted_rather_than_outlined() {
    let loaded = load();
    let code = loaded.code("add");
    assert!(!code.contains("not lifted"), "{code}");
    assert!(!code.contains("body not lifted"), "{code}");
    assert!(code.contains("u32 add("), "{code}");
}

#[test]
fn the_arguments_arrive_in_registers() {
    let loaded = load();
    let code = loaded.code("add");
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
    let code = loaded.code("twice");
    assert!(code.contains("add(a0, a0)"), "{code}");
    assert!(
        !code.contains("sub_"),
        "a call to a symbol in this object is not anonymous:\n{code}"
    );
}

#[test]
fn an_it_block_keeps_its_condition() {
    let loaded = load();
    let code = loaded.code("either");
    // The choice is an `it` block with an else, and the second of the four
    // instructions it governs is the other arm. A lift that reads the
    // block as one condition turns a choice into a sequence.
    assert!(!code.contains("unparsed"), "{code}");
    assert!(code.contains("a0"), "{code}");
}

#[test]
fn a_compare_and_branch_keeps_its_condition() {
    let loaded = load();
    let code = loaded.code("nonzero");
    // `cbz` and `cbnz` are Thumb's own, and the condition is the value of
    // the register against zero rather than a condition code.
    assert!(!code.contains("unparsed"), "{code}");
}

#[test]
fn a_conditional_store_happens_on_one_path_only() {
    let loaded = load();
    let code = loaded.code("set_if");
    // `if (x == 3) sink = 7;` is an `it` block around one store. A lift
    // that drops the condition writes the global on every path.
    assert!(code.contains("?"), "the store is conditional:\n{code}");
    assert!(!code.contains("unparsed"), "{code}");
}

#[test]
fn the_high_registers_are_read() {
    let loaded = load();
    let code = loaded.code("high");
    // Thumb has two banks, and `r8` to `r15` need the wide form. A lift
    // that reads `r8` as an unknown register loses the value.
    assert!(!code.contains("unparsed"), "{code}");
    assert!(!code.contains("r8"), "the register is named:\n{code}");
}

#[test]
fn an_extension_is_part_of_the_instruction() {
    let loaded = load();
    let code = loaded.code("widened");
    // A byte and a half word, sign- and zero-extended. Reading these as
    // plain moves loses the half of the value the instruction threw away.
    assert!(!code.contains("unparsed"), "{code}");
}

// ---------------------------------------------------------------------------
// The instructions that are Thumb's own.
// ---------------------------------------------------------------------------

#[test]
fn a_parallel_addition_is_not_an_ordinary_addition() {
    let loaded = load();
    let code = loaded.code("parallel_add8");
    // Each lane of the word is worked on by itself, and the carry of one
    // lane never reaches the next. An ordinary addition says something
    // different about every lane but the last.
    assert!(!code.contains("unparsed"), "{code}");
    assert!(
        code.contains("padd8") || code.contains("sadd8"),
        "the parallel addition keeps its name:\n{code}"
    );
}

#[test]
fn a_parallel_saturating_addition_keeps_its_name() {
    let loaded = load();
    let code = loaded.code("parallel_qadd16");
    assert!(!code.contains("unparsed"), "{code}");
    assert!(
        code.contains("qadd16"),
        "the saturating form keeps its name:\n{code}"
    );
}

#[test]
fn a_parallel_halving_addition_keeps_its_name() {
    let loaded = load();
    let code = loaded.code("parallel_shadd8");
    assert!(!code.contains("unparsed"), "{code}");
    assert!(
        code.contains("shadd8"),
        "the halving form keeps its name:\n{code}"
    );
}

#[test]
fn a_parallel_unsigned_halving_subtraction_keeps_its_name() {
    let loaded = load();
    let code = loaded.code("parallel_uhsub16");
    assert!(!code.contains("unparsed"), "{code}");
    assert!(
        code.contains("uhsub16"),
        "the unsigned halving form keeps its name:\n{code}"
    );
}

#[test]
fn an_exchanged_addition_keeps_its_name() {
    let loaded = load();
    let code = loaded.code("exchange_sasx");
    assert!(!code.contains("unparsed"), "{code}");
    assert!(
        code.contains("sasx") || code.contains("exchange"),
        "the exchanged form keeps its name:\n{code}"
    );
}

#[test]
fn an_exchanged_unsigned_halving_addition_keeps_its_name() {
    let loaded = load();
    let code = loaded.code("exchange_uhasx");
    assert!(!code.contains("unparsed"), "{code}");
    assert!(
        code.contains("uhasx") || code.contains("exchange"),
        "the exchanged form keeps its name:\n{code}"
    );
}

#[test]
fn a_reversal_is_one_operation() {
    let loaded = load();
    // Each reversal throws away something, and the operation is named
    // for what it does rather than for the instruction that does it.
    for (name, wanted) in [
        ("reverse_bytes", "rev_bytes"),
        ("reverse_halves", "rev_halves"),
        ("reverse_half_signed", "rev_halves_signed"),
        ("reverse_bits", "rev_bits"),
    ] {
        let code = loaded.code(name);
        assert!(!code.contains("unparsed"), "{name}:\n{code}");
        assert!(
            code.contains(wanted),
            "{name} is the {wanted} operation:\n{code}"
        );
    }
}

#[test]
fn an_extended_addition_says_what_it_extends() {
    let loaded = load();
    // The extension is part of the instruction, and the lifted source
    // says so: the half word of the second operand is read as a signed
    // number before it is added.
    let code = loaded.code("extended_add_signed");
    assert!(!code.contains("unparsed"), "{code}");
    assert!(
        code.contains("(i32)") || code.contains("sext"),
        "the extension is visible:\n{code}"
    );

    // The half word form works on both halves of the word at once, and
    // it is a parallel operation: the carry of one half never reaches
    // the other.
    let code = loaded.code("extended_add_halves");
    assert!(!code.contains("unparsed"), "{code}");
    assert!(
        code.contains("extend_add"),
        "the parallel extension is visible:\n{code}"
    );
}

#[test]
fn a_saturating_addition_is_one_operation() {
    let loaded = load();
    for (name, wanted) in [
        ("saturating_add", "sat_add"),
        ("saturating_add_double", "sat_add"),
    ] {
        let code = loaded.code(name);
        assert!(!code.contains("unparsed"), "{name}:\n{code}");
        assert!(
            code.contains(wanted),
            "{name} is the {wanted} operation:\n{code}"
        );
    }
}

#[test]
fn a_doubled_saturating_addition_adds_twice_the_second_operand() {
    let loaded = load();
    let code = loaded.code("saturating_add_double");
    // The doubled form adds twice the second operand, and the value it
    // adds is doubled before the sum is taken.
    assert!(code.contains("a1 + a1"), "the operand is doubled:\n{code}");
}
