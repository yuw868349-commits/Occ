//! Time each analysis stage on one file, so a hang can be attributed.
//!
//! `Analysis::build` runs five passes in sequence. When a malformed input makes
//! one of them take minutes, the useful question is which one, and a profiler
//! answers it with more ceremony than this needs.
//!
//! Usage: `stage_profile <file>`

use std::time::Instant;

use occ_analyze::view::View;
use occ_analyze::{RecoverOptions, StringOptions, entropy, fingerprint, recover, scan_strings};
use occ_core::ImageId;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: stage_profile <file>");
        std::process::exit(2);
    };
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("read: {e}");
            std::process::exit(1);
        }
    };
    let t = Instant::now();
    let image = match occ_bin::load(&data, &path, ImageId(0)) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("load: {e}");
            std::process::exit(1);
        }
    };
    println!("load          {:>10.3}ms  {} sections", t.elapsed().as_secs_f64() * 1e3, image.sections.len());

    let view = View::new(&image, &data);

    let t = Instant::now();
    let rec = recover(&view, &RecoverOptions::default());
    println!(
        "recover       {:>10.3}ms  {} functions, {} insns, {} xrefs",
        t.elapsed().as_secs_f64() * 1e3,
        rec.functions.len(),
        rec.functions.iter().map(|f| f.insns.len()).sum::<usize>(),
        rec.xrefs.len()
    );

    let t = Instant::now();
    let strs = scan_strings(&view, &StringOptions::default());
    println!("strings       {:>10.3}ms  {} found", t.elapsed().as_secs_f64() * 1e3, strs.len());

    let t = Instant::now();
    let fp = fingerprint::fingerprint(&image, &view);
    println!("fingerprint   {:>10.3}ms  {:?}", t.elapsed().as_secs_f64() * 1e3, fp.compiler);

    let t = Instant::now();
    let e = entropy::per_section(&view);
    println!("entropy       {:>10.3}ms  {} sections", t.elapsed().as_secs_f64() * 1e3, e.len());
}
