//! How much structure the decompiler recovers from a whole binary.
//!
//! `cargo run -p occ-decompile --example structure_stats -- /bin/ls`
//!
//! Two numbers matter and neither is visible function by function. The first
//! is how often the structuring pass gives up and prints `goto`: a function
//! that needs it is still correct, but the reader pays for every one. The
//! second is how many bytes inside a recovered function no instruction covers
//! — jump tables, alignment padding, and stretches the descent never followed.
//! A body that jumps over such a stretch is where a listing quietly stops
//! being complete.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, ImageId};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/bin/ls".into());
    let data = std::fs::read(&path).expect("read");
    let image = occ_bin::load(&data, &path, ImageId(0)).expect("parse");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());

    let names = |a: Addr| {
        analysis
            .function_at(a)
            .map(|f| f.display_name())
            .unwrap_or_else(|| format!("sub_{:x}", a.get()))
    };

    let (mut functions, mut lines, mut with_goto, mut gotos) = (0usize, 0usize, 0usize, 0usize);
    let (mut insn_bytes, mut gap_bytes) = (0u64, 0u64);
    let mut worst: Vec<(usize, String)> = Vec::new();
    let mut widest: Vec<(u64, String)> = Vec::new();
    for f in &analysis.functions {
        let d = occ_decompile::decompile(&image, f, &occ_decompile::Options::default(), &names);
        let code = occ_decompile::render::render(&d);
        let n = code.matches("goto ").count();
        functions += 1;
        lines += code.lines().count();
        if n > 0 {
            with_goto += 1;
            gotos += n;
            worst.push((n, f.display_name()));
        }
        let used: u64 = f.insns.iter().map(|i| u64::from(i.len)).sum();
        insn_bytes += used;
        gap_bytes += f.size.saturating_sub(used);
        widest.push((f.size.saturating_sub(used), f.display_name()));
    }
    let cov = analysis.coverage;
    println!(
        "{path}: {} functions, {lines} lines of pseudo-C",
        cov.functions
    );
    println!(
        "  decoded {} of {} code byte(s) = {:.1}%; {} byte(s) inside a function are not instructions",
        cov.decoded_bytes,
        cov.code_bytes,
        f64::from(cov.fraction()) * 100.0,
        gap_bytes
    );
    println!(
        "  {with_goto} of {functions} function(s) need goto ({:.1}%), {gotos} in total",
        f64::from(with_goto as u32) * 100.0 / functions.max(1) as f64
    );
    println!("  {insn_bytes} instruction byte(s), {} blocks", cov.blocks);
    worst.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
    for (n, name) in worst.iter().take(4) {
        println!("  {n} goto(s) in {name}");
    }
    widest.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
    for (n, name) in widest.iter().take(4) {
        if *n > 0 {
            println!("  {n} non-instruction byte(s) in {name}");
        }
    }
}
