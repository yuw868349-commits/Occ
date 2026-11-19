//! Report which instructions the lifter still does not model.
//!
//! `cargo run -p occ-decompile --example lift_coverage -- <file>`
//!
//! Every function in the binary is decompiled, and the instructions that
//! came out as "not lifted" or "unparsed" are counted by mnemonic. The
//! list is the work list: the mnemonics at the top are the ones that
//! cost a reader the most, because they are the ones a compiler emits
//! most often.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, ImageId};
use std::collections::BTreeMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: lift_coverage <file>");
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
        for imp in &img.imports {
            if imp.slot == Some(addr) {
                return format!("{}@plt", imp.name);
            }
        }
        format!("sub_{:x}", addr.0)
    };

    let mut missing: BTreeMap<String, usize> = BTreeMap::new();
    let mut total = 0usize;
    for func in &analysis.functions {
        let d = occ_decompile::decompile(&img, func, &occ_decompile::Options::default(), &names);
        let text = occ_decompile::render::render(&d);
        for line in text.lines() {
            let Some(rest) = line.trim().strip_prefix("// ") else {
                continue;
            };
            let Some(mnemonic) = rest.split(' ').next() else {
                continue;
            };
            if !rest.contains("not lifted") && !rest.contains("unparsed") {
                continue;
            }
            *missing.entry(mnemonic.to_string()).or_default() += 1;
            total += 1;
        }
    }

    let mut rows: Vec<(&String, &usize)> = missing.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!("{} instruction(s) not modelled:", total);
    for (mnemonic, count) in rows {
        println!("  {count:>6}  {mnemonic}");
    }
}
