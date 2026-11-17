//! Print the statements the value-set pass walks for one function.
//!
//! `dump <file> <entry>` -- every block of the function that starts at
//! `<entry>`, in address order, with the values left in the registers the
//! block writes. This is the view the pass has: when a jump resolves to
//! nothing, the reason is usually one statement that lost a value.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, Function};
use occ_decompile::{Program, build_program};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(path), Some(entry)) = (args.first(), args.get(1)) else {
        eprintln!("usage: dump <file> <entry>");
        return;
    };
    let entry = match u64::from_str_radix(entry.trim_start_matches("0x"), 16) {
        Ok(v) => Addr(v),
        Err(e) => {
            eprintln!("{entry}: {e}");
            return;
        }
    };
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{path}: {e}");
            return;
        }
    };
    let image = match occ_bin::load(&data, path, occ_core::ImageId(0)) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{path}: {e}");
            return;
        }
    };
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let Some(func): Option<&Function> = analysis.functions.iter().find(|f| f.entry == entry) else {
        eprintln!("no function at {:#x}", entry.0);
        return;
    };
    let program = build_program(
        view.image(),
        func,
        &occ_decompile::Options {
            max_blocks: 256,
            keep_comments: false,
            keep_dead: true,
            debug: None,
        },
        &|a: Addr| format!("sub_{:x}", a.0),
    );
    for t in &func.jump_tables {
        println!(
            "jumptable at {:#x} base {:#x} width {} relative {} count {} index {:?} first {:?}",
            t.at.0, t.base.0, t.entry_bytes, t.relative, t.count, t.index_reg, t.first_case
        );
        println!("  targets {:?}", t.targets.iter().map(|a| format!("{:#x}", a.0)).collect::<Vec<_>>());
    }
    for bb in &func.blocks {
        println!(
            "bb {:#x}..{:#x} insns {}..{}",
            bb.start.0, bb.end.0, bb.insn_start, bb.insn_end
        );
    }
    for insn in &func.insns {
        println!(
            "insn {:#x}  {:<28} {:?}{}",
            insn.addr.0,
            insn.text,
            insn.flow,
            if func.blocks.iter().any(|b| b.start == insn.addr) {
                "  <- block"
            } else {
                ""
            }
        );
    }
    print(&program);
}

fn print(program: &Program) {
    for (id, block) in program.blocks.iter().enumerate() {
        let preds: Vec<String> = program.preds[id]
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>();
        println!(
            "block {id} at {:#x} <- [{}]",
            block.addr.0,
            preds.join(", ")
        );
        for stmt in &block.stmts {
            println!("  {stmt:?}");
        }
        println!("  term {:?}", block.term);
    }
}
