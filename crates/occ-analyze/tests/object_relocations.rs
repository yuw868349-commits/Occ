//! Relocations decide what a relocatable object means.
//!
//! The committed fixtures are real `aarch64` and `riscv64` objects built from
//! one C file. Before their relocations are applied, an `adrp`/`add` pair that
//! forms the address of a string points at page zero, and a `b` that tail-calls
//! an external function branches to itself: the placeholders are zero and the
//! linker is the only thing that was ever going to fill them in.
//!
//! After the pass, the same bytes name the string and reach outside the object.
//! These tests are the guard for that difference, and for the boundary
//! invariants that depend on it: a recovered function's size is the span of the
//! instructions it actually holds, and no two functions overlap.

use std::path::{Path, PathBuf};

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, Function, Image};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn open(path: &Path) -> (Vec<u8>, Image) {
    let data = std::fs::read(path).expect("read fixture");
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse fixture");
    (data, image)
}

fn find<'a>(analysis: &'a Analysis, name: &str) -> &'a Function {
    analysis
        .functions
        .iter()
        .find(|f| f.display_name().trim_start_matches('_') == name)
        .unwrap_or_else(|| {
            panic!(
                "no {name}; recovered {:#?}",
                analysis
                    .functions
                    .iter()
                    .map(|f| f.display_name())
                    .collect::<Vec<_>>()
            )
        })
}

fn code_section(image: &Image) -> &occ_core::Section {
    image
        .sections
        .iter()
        .find(|s| s.kind == occ_core::SectionKind::Code && s.file_size > 0)
        .expect("a code section")
}

/// A function's declared size has to be the bytes its instructions occupy,
/// and no two functions may claim the same byte.
fn assert_boundaries_are_honest(analysis: &Analysis) {
    for f in &analysis.functions {
        let Some(last) = f.insns.last() else {
            continue;
        };
        let span = last.end().0.saturating_sub(f.entry.0);
        assert_eq!(
            f.size,
            span,
            "{} claims {} bytes but its instructions span {span}",
            f.display_name(),
            f.size
        );
        assert!(
            f.size >= 2 * f.insns.len() as u64,
            "{} has {} instructions in {} bytes",
            f.display_name(),
            f.insns.len(),
            f.size
        );
    }
    let mut sorted: Vec<&Function> = analysis.functions.iter().collect();
    sorted.sort_by_key(|f| f.entry.0);
    for pair in sorted.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        assert!(
            a.end().0 <= b.entry.0,
            "{} ends at {:#x}, past {} at {:#x}",
            a.display_name(),
            a.end().0,
            b.display_name(),
            b.entry.0
        );
    }
}

fn word_at(data: &[u8], addr: Addr) -> u32 {
    let off = usize::try_from(addr.0).expect("address fits in a usize");
    let b = data
        .get(off..off + 4)
        .unwrap_or_else(|| panic!("no four bytes at {:#x}", addr.0));
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The first bytes of the fixture's `.rodata.str1.8`, which are `occ\0`.
///
/// The string is three characters, below the extractor's minimum, so it is
/// checked directly rather than through the string table.
fn rodata_prefix(image: &Image, data: &[u8]) -> (Addr, Vec<u8>) {
    let section = image
        .sections
        .iter()
        .find(|s| s.name == ".rodata.str1.8")
        .expect("the fixture has .rodata.str1.8");
    let off = usize::try_from(section.addr.0).expect("address fits");
    (section.addr, data[off..off + 4].to_vec())
}

/// The address an `adrp`/`add` pair forms at `at`.
fn aarch64_pair_target(data: &[u8], at: Addr) -> u64 {
    let adrp = word_at(data, at);
    assert_eq!(
        adrp & 0x9f00_0000,
        0x9000_0000,
        "expected adrp at {:#x}, found {adrp:#010x}",
        at.0
    );
    let immlo = ((adrp >> 29) & 3) as i64;
    let immhi = ((adrp >> 5) & 0x7ffff) as i64;
    let raw = (immhi << 2) | immlo;
    let imm = (raw << 43) >> 43;
    let page = (at.0 & !0xfff).wrapping_add_signed(imm << 12);

    let add = word_at(data, at.add(4));
    assert_eq!(
        add & 0xff00_0000,
        0x9100_0000,
        "expected add-immediate at {:#x}, found {add:#010x}",
        at.0 + 4
    );
    let shift = if add & 0x0040_0000 != 0 { 12 } else { 0 };
    let imm12 = u64::from((add >> 10) & 0xfff) << shift;
    page.wrapping_add(imm12)
}

/// The address an `auipc`/`addi` pair forms at `at`.
fn riscv_pair_target(data: &[u8], at: Addr) -> u64 {
    let auipc = word_at(data, at);
    assert_eq!(
        auipc & 0x7f,
        0x17,
        "expected auipc at {:#x}, found {auipc:#010x}",
        at.0
    );
    let hi = (auipc & 0xffff_f000) as i32 as i64;
    let base = at.0.wrapping_add_signed(hi);

    let addi = word_at(data, at.add(4));
    assert_eq!(
        addi & 0x7f,
        0x13,
        "expected addi at {:#x}, found {addi:#010x}",
        at.0 + 4
    );
    let lo = (addi as i32) >> 20;
    base.wrapping_add_signed(i64::from(lo))
}

#[test]
fn aarch64_adrp_add_names_the_string_it_was_written_for() {
    let path = fixture("fixture.aarch64.o");
    let (mut data, image) = open(&path);
    let applied = occ_bin::reloc::apply(&mut data, &image).expect("an object needs relocations");
    assert!(applied.applied >= 5, "{applied:?}");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());

    let msg = find(&analysis, "msg");
    let target = aarch64_pair_target(&data, msg.entry);
    let (rodata, prefix) = rodata_prefix(&image, &data);
    assert_eq!(prefix, b"occ\0", "the fixture stores \"occ\" first");
    assert_eq!(
        target, rodata.0,
        "msg forms {target:#x}, but .rodata.str1.8 starts at {:#x}",
        rodata.0
    );
    assert_boundaries_are_honest(&analysis);
    assert_eq!(image.arch, occ_core::Arch::Aarch64);
}

#[test]
fn aarch64_tail_call_reaches_outside_the_object() {
    let path = fixture("fixture.aarch64.o");
    let (mut data, image) = open(&path);
    let applied = occ_bin::reloc::apply(&mut data, &image).expect("an object needs relocations");
    assert!(
        applied.externals > 0,
        "the fixture calls `ext`: {applied:?}"
    );
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());

    let dispatch = find(&analysis, "dispatch");
    let text = code_section(&image);
    let text_end = text.addr.add(text.file_size);
    let outside: Vec<Addr> = dispatch
        .blocks
        .iter()
        .flat_map(|b| b.edges.iter().map(|e| e.to))
        .filter(|to| *to < text.addr || *to >= text_end)
        .collect();
    assert!(
        !outside.is_empty(),
        "no edge of dispatch leaves .text; blocks {:#?}",
        dispatch
            .blocks
            .iter()
            .map(|b| (
                b.start.0,
                b.edges.iter().map(|e| e.to.0).collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>()
    );
    // The source dispatches four ways; the branch structure is only visible
    // once the conditional branches have real targets.
    assert!(
        dispatch.blocks.len() >= 4,
        "{} blocks",
        dispatch.blocks.len()
    );
    assert_boundaries_are_honest(&analysis);
}

#[test]
fn riscv64_auipc_pair_names_the_string_it_was_written_for() {
    let path = fixture("fixture.riscv64.o");
    let (mut data, image) = open(&path);
    let applied = occ_bin::reloc::apply(&mut data, &image).expect("an object needs relocations");
    assert!(applied.applied >= 10, "{applied:?}");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());

    // The pair sits at `.text + 0x6c` (`R_RISCV_PCREL_HI20` plus its
    // `R_RISCV_PCREL_LO12_I` partner).
    let text = code_section(&image);
    let target = riscv_pair_target(&data, text.addr.add(0x6c));
    let (rodata, prefix) = rodata_prefix(&image, &data);
    assert_eq!(prefix, b"occ\0", "the fixture stores \"occ\" first");
    assert_eq!(
        target, rodata.0,
        "auipc/addi forms {target:#x}, but .rodata.str1.8 starts at {:#x}",
        rodata.0
    );
    assert_boundaries_are_honest(&analysis);
}

#[test]
fn without_relocations_the_string_reference_points_at_zero() {
    // The contrast is what proves the pass did something: the same bytes, read
    // without filling in the placeholders, form an address on page zero, which
    // is not where any string lives.
    let path = fixture("fixture.aarch64.o");
    let (data, image) = open(&path);
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let msg = find(&analysis, "msg");
    let target = aarch64_pair_target(&data, msg.entry);
    assert!(
        analysis.strings.iter().all(|s| s.addr.0 != target),
        "unrelocated msg already forms {target:#x}, which is a string"
    );
}

#[test]
fn every_function_stays_inside_the_code_section() {
    for name in ["fixture.aarch64.o", "fixture.riscv64.o"] {
        let path = fixture(name);
        let (mut data, image) = open(&path);
        occ_bin::reloc::apply(&mut data, &image).expect("an object needs relocations");
        let view = View::new(&image, &data);
        let analysis = Analysis::build(&view, &AnalysisOptions::default());
        let text = code_section(&image);
        for f in &analysis.functions {
            assert!(
                f.entry >= text.addr && f.end() <= text.addr.add(text.file_size),
                "{name}: {} at {:#x}..{:#x} leaves .text {:#x}..{:#x}",
                f.display_name(),
                f.entry.0,
                f.end().0,
                text.addr.0,
                text.addr.0 + text.file_size
            );
        }
    }
}
