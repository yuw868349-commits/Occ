//! Decompile one function from a real binary.
//!
//! `cargo run -p occ-decompile --example decompile -- <file> <name-or-addr>`

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, ImageId};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: decompile <file> <name-or-addr>");
    let want = args.next().unwrap_or_else(|| "main".into());
    let data = std::fs::read(&path).expect("read");
    let img = occ_bin::load(&data, &path, ImageId(0)).expect("parse");
    let view = View::new(&img, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());

    let func = occ_core::addr::parse_addr(&want)
        .and_then(|a| analysis.function_covering(a))
        .or_else(|| analysis.function_by_name(&want))
        .expect("function not found");

    if std::env::var("OCC_DEBUG").is_ok() {
        eprintln!("function {} entry={} blocks={} insns={}", func.display_name(), func.entry, func.blocks.len(), func.insns.len());
        for b in &func.blocks {
            eprintln!("  block {:#x} insns {}..{}", b.start.0, b.insn_start, b.insn_end);
            for i in &func.insns[b.insn_start..b.insn_end] {
                eprintln!("      {}", i.text);
            }
        }
    }

    let names = |addr: Addr| -> String {
        if let Some(f) = analysis.function_at(addr) {
            return f.display_name();
        }
        if let Some(s) = img.symbol_at(addr) {
            return s.name.clone();
        }
        for imp in &img.imports {
            if imp.slot == Some(addr) {
                return format!("{}@plt", imp.name);
            }
        }
        format!("sub_{:x}", addr.0)
    };
    let d = occ_decompile::decompile(&img, func, &occ_decompile::Options::default(), &names);
    print!("{}", occ_decompile::render::render(&d));
    if d.unstructured > 0 {
        eprintln!("\n{} block(s) needed goto", d.unstructured);
    }
    for n in &d.notes {
        eprintln!("note: {n}");
    }
}
