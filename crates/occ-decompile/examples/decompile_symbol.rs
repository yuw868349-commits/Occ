//! Decompile one symbol from a shared library.
//!
//! Usage: `cargo run -p occ-decompile --example decompile_symbol -- <library> <symbol>`
//!
//! A library is the harshest input available: the code is hand written and
//! heavily optimised, so the control-flow graph has shapes a compiler never
//! produces. Running this over `libc.so.6` is how the decompiler's behaviour
//! on real code is checked — in particular that it finishes instead of
//! running out of stack.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::Addr;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        "/lib/x86_64-linux-gnu/libc.so.6".to_string()
    });
    let symbol = args.next().unwrap_or_else(|| "warn".to_string());
    let data = std::fs::read(&path).expect("read library");
    let img = occ_bin::load(&data, &path, occ_core::ImageId(0)).expect("parse");
    let view = View::new(&img, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let f = analysis
        .functions
        .iter()
        .find(|f| f.display_name() == symbol)
        .expect("symbol");
    let name_of = |a: Addr| {
        analysis
            .function_at(a)
            .map(|f| f.display_name())
            .unwrap_or_else(|| format!("sub_{:x}", a.get()))
    };
    let d = occ_decompile::decompile(&img, f, &occ_decompile::Options::default(), &name_of);
    println!("// {symbol}: {} nodes, {} unstructured", d.body.len(), d.unstructured);
    for note in &d.notes {
        println!("// note: {note}");
    }
    println!("{}", occ_decompile::render::render(&d));
}
