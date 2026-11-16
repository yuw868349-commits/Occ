//! Dump what the DWARF reader recovered from a real binary.
//!
//! `cargo run -p occ-dwarf --example probe -- /path/to/binary [addr ...]`
//!
//! With addresses given, each one is resolved to `file:line` and to the
//! enclosing debug function, which is what the MCP tools expose.

use occ_analyze::View;
use occ_core::{Addr, ImageId};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: dwarf_probe <binary> [addr ...]");
            return;
        }
    };
    let want: Vec<u64> = args
        .filter_map(|a| {
            let t = a.trim_start_matches("0x");
            u64::from_str_radix(t, 16).ok()
        })
        .collect();

    let data = std::fs::read(&path).unwrap();
    let img = occ_bin::load(&data, &path, ImageId(0)).unwrap();
    let view = View::new(&img, &data);
    let Some(dw) = occ_dwarf::Dwarf::load(&view) else {
        println!("no debug info in {path}");
        return;
    };

    println!("units           {}", dw.units.len());
    println!("functions       {}", dw.function_count());
    println!("types           {}", dw.types().count());
    println!("files           {}", dw.source_files().len());
    for n in &dw.notes {
        println!("note            {n}");
    }
    for (i, u) in dw.units.iter().enumerate() {
        let lines: usize = u.lines.rows.len();
        println!(
            "unit {i}: v? addr_size? off={:#x} name={:?} producer={:?} funcs={} types={} rows={}",
            u.offset,
            u.name,
            u.producer,
            u.functions.len(),
            u.types.len(),
            lines
        );
    }

    println!("\n-- functions --");
    for u in &dw.units {
        for f in u.functions.iter().take(12) {
            println!(
                "{:#010x}..{:#010x} {:40} decl={:?} params={} vars={}",
                f.low_pc.0,
                f.high_pc.0,
                f.name,
                f.decl_file.as_deref(),
                f.params.len(),
                f.locals.len()
            );
            for p in f.params.iter().take(8) {
                println!(
                    "    param {:<20} {:?} ranges={} at_entry={:?}",
                    p.name,
                    p.type_name,
                    p.ranges.len(),
                    p.location_at(f.low_pc)
                );
                for r in p.ranges.iter().take(3) {
                    println!("        {:#x}..{:#x} {}", r.start.0, r.end.0, r.expr);
                }
            }
            for v in f.locals.iter().take(8) {
                println!(
                    "    var   {:<20} {:?} ranges={} at_entry={:?}",
                    v.name,
                    v.type_name,
                    v.ranges.len(),
                    v.location_at(f.low_pc)
                );
            }
        }
    }

    if !want.is_empty() {
        println!("\n-- lookups --");
        for a in want {
            let a = Addr(a);
            let line = dw.line_for(a);
            let func = dw.function_for(a).map(|f| f.name.as_str());
            println!(
                "{:#010x} line={:<32} func={:?}",
                a.0,
                line.map(|(f, l, c)| format!("{f}:{l}:{c}"))
                    .unwrap_or_else(|| "-".into()),
                func
            );
        }
    }
}
