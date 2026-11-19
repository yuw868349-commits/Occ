//! Run the value-set pass over a file and print what it resolved.
//!
//! `report <file> [--calls] [--jumps] [--quiet]` -- the counters always print,
//! the sites print unless `--quiet`.

use occ_analyze::{Analysis, AnalysisOptions, View};

fn main() {
    let mut path: Option<String> = None;
    let mut show_calls = false;
    let mut show_jumps = false;
    let mut quiet = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--calls" => show_calls = true,
            "--jumps" => show_jumps = true,
            "--quiet" => quiet = true,
            "--functions" => {}
            other if other.starts_with("--") => {}
            other => path = Some(other.to_string()),
        }
    }
    let Some(path) = path else {
        eprintln!("usage: report <file> [--calls] [--jumps] [--quiet]");
        return;
    };
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{path}: {e}");
            return;
        }
    };
    let image = match occ_bin::load(&data, &path, occ_core::ImageId(0)) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{path}: {e}");
            return;
        }
    };
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let report = occ_vsa::analyze(&view, &analysis, &occ_vsa::Options::default());
    let stats = &report.stats;
    println!(
        "{}: {} functions, {} blocks",
        path, stats.functions, stats.blocks
    );
    println!(
        "calls      {:>6} sites, {:>6} resolved ({:.1}%)",
        stats.call_sites,
        stats.calls_resolved,
        stats.call_rate() * 100.0
    );
    println!(
        "jumps      {:>6} sites, {:>6} resolved ({:.1}%)",
        stats.jump_sites,
        stats.jumps_resolved,
        stats.jump_rate() * 100.0
    );
    println!(
        "tables     {:>6} read, {} two-level; syntactic resolver found {}",
        stats.tables, stats.two_level, stats.syntactic_tables
    );
    if quiet {
        return;
    }
    if std::env::args().any(|a| a == "--functions") {
        for f in &analysis.functions {
            println!(
                "fn    {:>10} {:>6} insns  {}",
                format!("{:#x}", f.entry.0),
                f.insns.len(),
                f.display_name()
            );
        }
    }
    for call in &report.calls {
        if !show_calls && !show_jumps {
            // Both are printed when neither was asked for.
        }
        println!(
            "call  {:>10} in {:>10} via {:<5} -> {:?}  ({})",
            format!("{:#x}", call.at.0),
            format!("{:#x}", call.in_function.0),
            call.via,
            call.targets
                .iter()
                .map(|t| format!("{:#x}", t.0))
                .collect::<Vec<_>>(),
            call.evidence
        );
    }
    for jump in &report.jumps {
        println!(
            "jump  {:>10} in {:>10} via {:<5} {} cases{}  ({})",
            format!("{:#x}", jump.at.0),
            format!("{:#x}", jump.in_function.0),
            jump.via,
            jump.cases.len(),
            if jump.two_level { " two-level" } else { "" },
            jump.evidence
        );
        for case in &jump.cases {
            println!("        case {:>6} -> {:#x}", case.label, case.target.0);
        }
    }
}
