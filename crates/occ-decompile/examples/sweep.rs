//! Decompile every recovered function and summarise what came out.
//!
//! `cargo run -p occ-decompile --example sweep -- <file> [limit]`
//!
//! Used as a regression net: a change to lifting, SSA or structuring that makes
//! a function render empty, fall back to `goto`, or panic shows up as a count
//! that moved. The default limit is small enough to read.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, ImageId};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: sweep <file> [limit]");
    let limit: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);

    let data = std::fs::read(&path).expect("read");
    let img = occ_bin::load(&data, &path, ImageId(0)).expect("parse");
    let view = View::new(&img, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let names = |addr: Addr| -> String {
        if let Some(f) = analysis.function_at(addr) {
            return f.display_name();
        }
        if let Some(s) = img.symbol_at(addr) {
            return s.name.clone();
        }
        format!("sub_{:x}", addr.0)
    };

    let opts = occ_decompile::Options::default();
    let mut total = 0usize;
    let mut empty = Vec::new();
    let mut unstructured: Vec<(String, usize)> = Vec::new();
    let mut noted = 0usize;
    let mut lines_total = 0usize;
    for f in analysis.functions.iter().take(limit) {
        total += 1;
        let d = occ_decompile::decompile(&img, f, &opts, &names);
        let text = occ_decompile::render::render(&d);
        let lines = text.lines().count();
        lines_total += lines;
        // A PLT stub is a jump to an import; it has no body of its own, and
        // counting it would drown the real finding.
        if lines <= 2 && !f.display_name().ends_with("@plt") {
            empty.push(f.display_name());
        }
        if d.unstructured > 0 {
            unstructured.push((f.display_name(), d.unstructured));
        }
        if !d.notes.is_empty() {
            noted += 1;
        }
    }
    unstructured.sort_by_key(|(_, n)| std::cmp::Reverse(*n));

    println!("{path}: {total} function(s), {lines_total} line(s)");
    println!("  empty bodies: {}", empty.len());
    for name in empty.iter().take(20) {
        println!("    {name}");
    }
    println!("  functions with goto fallbacks: {}", unstructured.len());
    for (name, n) in unstructured.iter().take(10) {
        println!("    {name}: {n}");
    }
    println!("  functions with notes: {noted}");
}
