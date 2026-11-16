use occ_analyze::analysis::{Analysis, AnalysisOptions};
use occ_analyze::view::View;
use occ_core::ImageId;

fn main() {
    let path = std::env::args().nth(1).expect("usage: summary <binary>");
    let data = std::fs::read(&path).unwrap();
    let img = occ_bin::load(&data, &path, ImageId(0)).unwrap();
    println!("{}", img.headline());
    let v = View::new(&img, &data);
    let t = std::time::Instant::now();
    let a = Analysis::build(&v, &AnalysisOptions::default());
    println!("built in {:?}", t.elapsed());
    println!("{}", a.summary());
    println!("strings {} xrefs {}", a.strings.len(), a.xrefs.len());
    println!("compiler {:?} lang {:?} link {:?} stripped {}", a.fingerprint.compiler, a.fingerprint.language, a.fingerprint.linkage, a.fingerprint.stripped);
    println!("thunks {}", a.thunks.len());
    for n in &a.notes { println!("note: {n}"); }
    for f in a.largest(8) {
        println!("  {:>6} insns at {} {} callers={} callees={}", f.insns.len(), f.entry, f.display_name(), f.callers.len(), f.callees.len());
    }
    let roots = a.roots();
    println!("roots {}", roots.len());
    for s in a.strings.iter().take(3) { println!("str {} {:?}", s.addr, s.text); }
    if let Some(f) = a.function_by_name("main") {
        println!("main: {} insns, blocks {}, string refs {}", f.insns.len(), f.blocks.len(), f.string_refs.len());
        for i in f.insns.iter().take(6) { println!("   {} {}", i.addr, i.text); }
    }
}
