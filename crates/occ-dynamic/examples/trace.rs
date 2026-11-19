//! Manual probe: `cargo run -p occ-dynamic --example trace -- /bin/true`
//!
//! Prints what one run observed, without going through the session or the MCP
//! layer. Useful for checking a trace by eye on a binary whose behaviour is
//! known, which a test cannot do for every case.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::ImageId;
use occ_dynamic::{RunConfig, correlate, run};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: trace <program> [args...]");
    let rest: Vec<String> = args.collect();
    let data = std::fs::read(&path).expect("read");
    let img = occ_bin::load(&data, &path, ImageId(0)).expect("parse");
    let view = View::new(&img, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());

    let cfg = RunConfig {
        args: rest,
        max_steps: 300_000,
        max_millis: 8_000,
        ..RunConfig::default()
    };
    let trace = run(std::path::Path::new(&path), &img, &cfg).expect("trace");
    let corr = correlate(&trace, &img, &analysis);

    println!(
        "{path}: {} steps in {} ms, {}, base {:#x}, delta {}",
        trace.steps,
        trace.millis,
        trace.end.label(),
        trace.base,
        trace.delta
    );
    println!(
        "  {} distinct instruction(s) in the image, {} step(s) outside it",
        corr.executed_in_image, corr.external
    );
    println!(
        "  {} function(s) hit, {} missed, {} orphan(s), {} call site(s)",
        corr.functions.iter().filter(|f| f.hits > 0).count(),
        corr.missed.len(),
        corr.orphans.len(),
        corr.calls.len()
    );
    for f in corr.functions.iter().filter(|f| f.hits > 0).take(12) {
        println!(
            "  {:<28} {} {}/{} insns, {}/{} blocks, {} hits",
            f.name, f.entry, f.insns, f.total, f.blocks, f.blocks_total, f.hits
        );
    }
    for c in corr
        .calls
        .iter()
        .filter(|c| c.indirect || c.targets.len() > 1)
        .take(8)
    {
        let targets: Vec<String> = c
            .targets
            .iter()
            .take(4)
            .map(|t| {
                if t.in_image {
                    format!("{}x{}", t.addr, t.count)
                } else {
                    format!("{}(external)x{}", t.addr, t.count)
                }
            })
            .collect();
        println!(
            "  {} {} in {} -> {}",
            if c.indirect { "indirect" } else { "multi" },
            c.site,
            c.from.clone().unwrap_or_else(|| "?".into()),
            targets.join(", ")
        );
    }
    for s in trace.syscalls.iter().take(12) {
        println!(
            "  syscall {}({:#x}, {:#x}, {:#x}) = {} {}",
            s.name,
            s.args[0],
            s.args[1],
            s.args[2],
            s.ret,
            s.paths.join(" ")
        );
    }
    if let Some(f) = analysis.function_by_name("main") {
        println!("  blocks of main:");
        for b in occ_dynamic::function_blocks(&trace, f) {
            println!(
                "    {}..{} {} insns, {} hits",
                b.start, b.end, b.insns, b.hits
            );
        }
    }
    for n in &trace.notes {
        println!("  note: {n}");
    }
}
