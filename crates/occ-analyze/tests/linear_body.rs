//! Every recovered function must be one linear instruction stream.
//!
//! Recovery follows several seeds. Before this invariant was enforced a
//! branch target could land inside an instruction another seed had already
//! decoded, and `code.disasm` returned both — one byte, two instructions.
//! Blocks indexed the de-duplicated list while the decompiler sliced the
//! original one, so the two disagreed about which statements were in a block.
//! The check is cheap and catches the whole class.

use occ_analyze::{Analysis, AnalysisOptions, View};
use std::path::Path;

fn check(path: &Path) {
    let data = std::fs::read(path).expect("read fixture");
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse fixture");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let mut checked = 0usize;
    for f in &analysis.functions {
        assert!(
            f.is_linear(),
            "{}: function {} is not a linear stream",
            path.display(),
            f.display_name()
        );
        let mut sum = 0u64;
        for insn in &f.insns {
            sum += u64::from(insn.len);
        }
        if f.size > 0 {
            assert!(
                sum <= f.size,
                "{}: {} claims {} bytes but its instructions total {}",
                path.display(),
                f.display_name(),
                f.size,
                sum
            );
        }
        // Block slices are indices into the same list the decompiler reads.
        for b in &f.blocks {
            assert!(
                b.insn_start < b.insn_end && b.insn_end <= f.insns.len(),
                "{}: {} has a block outside its instruction list",
                path.display(),
                f.display_name()
            );
            let first = &f.insns[b.insn_start];
            let last = &f.insns[b.insn_end - 1];
            assert_eq!(first.addr, b.start, "{}: block start", f.display_name());
            assert_eq!(last.end(), b.end, "{}: block end", f.display_name());
        }
        checked += 1;
    }
    assert!(checked > 0, "{}: no functions recovered", path.display());
}

#[test]
fn a_real_elf_body_is_linear() {
    let path = Path::new("/bin/ls");
    if !path.exists() {
        eprintln!("no /bin/ls on this host; skipping");
        return;
    }
    check(path);
}
