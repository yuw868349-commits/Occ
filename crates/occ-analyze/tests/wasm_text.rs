//! Instruction-level view of a real WebAssembly module.
//!
//! The fixture is `tests/fixtures/fixture.wasm.c` compiled with
//! `clang --target=wasm32 -O1 -nostdlib -Wl,--no-entry -Wl,--export-all`. It has
//! a straight-line function, a loop with a back edge, a memory load and a nested
//! block, so every part of the body decoder is exercised by real compiler output
//! rather than by bytes written to match the decoder.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, Arch, Format, ImageId};

fn fixture() -> (Vec<u8>, occ_core::Image) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/fixture.wasm"
    );
    let data = std::fs::read(path).expect("fixture present");
    let img = occ_bin::load(&data, path, ImageId(0)).expect("parses");
    (data, img)
}

fn functions() -> Vec<occ_core::Function> {
    let (data, img) = fixture();
    let view = View::new(&img, &data);
    Analysis::build(&view, &AnalysisOptions::default()).functions
}

#[test]
fn loads_as_wasm() {
    let (_, img) = fixture();
    assert_eq!(img.format, Format::Wasm);
    assert_eq!(img.arch, Arch::Wasm);
}

#[test]
fn named_functions_are_recovered() {
    let fns = functions();
    let names: Vec<String> = fns.iter().map(|f| f.display_name()).collect();
    assert!(names.contains(&"add".to_string()), "{names:?}");
    assert!(names.contains(&"sum".to_string()), "{names:?}");
}

#[test]
fn instruction_offsets_are_contiguous() {
    // Every byte of a body belongs to exactly one instruction, so the
    // instruction after one must start where that one ended. A wrong immediate
    // length shows up here as a gap, which is the failure mode a byte-level view
    // cannot afford.
    for f in functions() {
        for pair in f.insns.windows(2) {
            assert_eq!(
                pair[0].end(),
                pair[1].addr,
                "gap after {} in {}",
                pair[0].text,
                f.display_name()
            );
        }
        let last = f.insns.last().expect("non-empty");
        assert_eq!(
            last.end(),
            f.entry.add(f.size),
            "{} does not end at its declared size",
            f.display_name()
        );
    }
}

#[test]
fn a_straight_line_function_decodes_operand_for_operand() {
    let fns = functions();
    let add = fns
        .iter()
        .find(|f| f.display_name() == "add")
        .expect("add is recovered");
    let text: Vec<String> = add.insns.iter().map(|i| i.text.clone()).collect();
    assert_eq!(
        text,
        vec!["local.get 1", "local.get 0", "i32.add", "end"],
        "clang -O1 emits the operands in this order"
    );
}

#[test]
fn a_loop_back_edge_points_at_the_loop_body() {
    let fns = functions();
    let sum = fns
        .iter()
        .find(|f| f.display_name() == "sum")
        .expect("sum is recovered");
    assert!(sum.blocks.len() >= 4, "the loop needs several blocks");
    // The first `br_if` guards the empty-range case and jumps forward; the one
    // closing the loop jumps back. Picking the backward one is the point: a
    // forward branch resolved backwards would still look plausible.
    let back = sum
        .insns
        .iter()
        .find(|i| {
            i.mnemonic.as_ref() == "br_if"
                && matches!(i.target, Some(occ_core::CallTarget::Direct { to }) if to < i.addr)
        })
        .expect("the loop has a backward conditional branch");
    let Some(occ_core::CallTarget::Direct { to }) = back.target else {
        panic!("br_if must name its target");
    };
    assert!(
        sum.blocks.iter().any(|b| b.start == to),
        "a branch target must start a block"
    );
}

#[test]
fn every_branch_target_starts_a_block() {
    for f in functions() {
        for insn in &f.insns {
            let Some(occ_core::CallTarget::Direct { to }) = insn.target else {
                continue;
            };
            if !insn.flow.terminates_block() {
                continue;
            }
            assert!(
                f.blocks.iter().any(|b| b.start == to),
                "{} in {} targets {to}, which is not a block start",
                insn.text,
                f.display_name()
            );
        }
    }
}

#[test]
fn a_memory_operand_keeps_its_width_and_offset() {
    let fns = functions();
    let sum = fns
        .iter()
        .find(|f| f.display_name() == "sum")
        .expect("sum is recovered");
    let load = sum
        .insns
        .iter()
        .find(|i| i.mnemonic.as_ref() == "i32.load")
        .expect("the loop reads through the pointer");
    assert_eq!(load.text, "i32.load align=2 offset=0");
    assert_eq!(load.len, 3, "opcode plus two one-byte LEB immediates");
}

#[test]
fn an_address_inside_a_body_names_its_function() {
    let (data, img) = fixture();
    let view = View::new(&img, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let f = analysis
        .function_by_name("sum")
        .expect("sum is recovered");
    let at = f.entry.add(u64::from(f.insns[2].len));
    assert_eq!(
        analysis.function_covering(at).map(|f| f.display_name()),
        Some("sum".to_string())
    );
    assert!(at.0 > 0);
}

#[test]
fn a_truncated_module_does_not_panic() {
    let (data, img) = fixture();
    for cut in (0..data.len()).step_by(7) {
        let slice = &data[..cut];
        let Ok(truncated) = occ_bin::load(slice, "cut", ImageId(0)) else {
            continue;
        };
        let view = View::new(&truncated, slice);
        let _ = Analysis::build(&view, &AnalysisOptions::default());
    }
    assert!(img.sections.iter().any(|s| s.name == "code"));
}

#[test]
fn entry_is_null_without_a_start_section() {
    let (_, img) = fixture();
    assert_eq!(img.entry, Addr::NULL);
}
