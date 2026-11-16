//! Disassemble every code section of a file, for cross-checking with objdump.
//!
//! `objdump -d` walks every section marked as code, not just `.text`, and the
//! cross-check is only meaningful if both sides walk the same ground. This
//! example therefore starts at the beginning of each code section in turn,
//! decoding each one as a continuous stream.
//!
//! With `--symbols` the walk also starts at every function entry the symbol
//! table names. That matters because `objdump` starts a new listing at every
//! symbol it recognises: the padding between two functions is never decoded
//! as code by `objdump`, and a walk that only went from section to section
//! would decode it and be one instruction out of step for the rest of the
//! file. Decoding from the entries as well puts both sides on the same
//! instruction boundaries.
//!
//! The output is one line per instruction, in address order, so that two
//! runs can be compared line by line.
//!
//! `cargo run --release -p occ-decode --example disasm_all -- <file> [--tsv|--symbols]`

use std::collections::{BTreeMap, BTreeSet};

use occ_core::{Addr, ImageId};
use occ_decode::Decoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: disasm_all <file> [--tsv|--symbols]");
    let flags: Vec<String> = args.collect();
    let tsv = flags.iter().any(|f| f == "--tsv");
    let symbols = flags.iter().any(|f| f == "--symbols");

    let data = std::fs::read(&path).expect("read");
    let img = occ_bin::load(&data, &path, ImageId(0)).expect("parse");
    let dec = Decoder::new(img.bits).expect("decoder");

    // Every instruction that was decoded, by address.
    //
    // The symbol walk runs first, because it is the one that knows where
    // the instruction boundaries really are: it starts at every function
    // entry, so it never reads the padding between two functions as code.
    // The linear walk then fills in whatever the symbol table does not
    // name, and it does not overwrite what the symbol walk found.
    let mut insns: BTreeMap<u64, (u32, String)> = BTreeMap::new();

    if symbols {
        entries(&img, &data, &dec, &mut insns);
    }
    linear(&img, &data, &dec, &mut insns);

    if tsv {
        for (addr, (len, mnemonic)) in &insns {
            // The mnemonic field, not the text: the text carries the
            // prefixes (`lock`, `rep`) that objdump prints in front of the
            // name as well, and comparing them would report every locked
            // instruction as a difference.
            println!("{addr:08x}\t{len}\t{mnemonic}");
        }
    } else {
        for (addr, (_, mnemonic)) in &insns {
            println!("{addr:08x}\t{mnemonic}");
        }
    }
}

/// Decode each code section as one continuous stream, from its first byte.
///
/// The walk restarts at every function entry the symbol table names. That
/// is what `objdump` does, and it matters: the padding between two
/// functions is not code, and a walk that read it as code would be one
/// instruction out of step for the rest of the section.
fn linear(
    img: &occ_core::Image,
    data: &[u8],
    dec: &Decoder,
    insns: &mut BTreeMap<u64, (u32, String)>,
) {
    // The function entries, in address order, so that the walk can find
    // the next one without searching the whole symbol table each time.
    let mut entries: Vec<u64> = img
        .symbols
        .iter()
        .filter(|s| {
            s.defined
                && s.addr.get() != 0
                && matches!(s.kind, occ_core::SymbolKind::Function)
        })
        .map(|s| s.addr.get())
        .collect();
    entries.sort_unstable();
    entries.dedup();

    for sec in &img.sections {
        if !sec.is_code() && !sec.perm.exec() {
            continue;
        }
        if sec.file_size == 0 {
            continue;
        }
        let mut pc = sec.addr.get();
        let end = pc + sec.file_size;
        // The entry the walk is heading for, if the next one is still ahead.
        let mut next = entries.partition_point(|&e| e < pc);
        while pc < end {
            // A function entry that lies inside the instruction about to be
            // decoded is where the code really starts, so the walk jumps
            // there instead of reading the padding as code.
            let mut restart = None;
            for &e in &entries[next..] {
                if e >= end {
                    break;
                }
                if e <= pc {
                    continue;
                }
                restart = Some(e);
                break;
            }
            let Some(bytes) = bytes_at(img, data, pc) else { break };
            let Some(d) = dec.decode(Addr(pc), bytes) else {
                break;
            };
            let len = u64::from(d.insn.len).max(1);
            // The entry has to be inside this instruction for the jump to
            // be the right one: an entry further ahead is reached by
            // walking to it.
            if let Some(e) = restart {
                if e < pc + len {
                    pc = e;
                    next = entries.partition_point(|&x| x < pc);
                    continue;
                }
            }
            insns
                .entry(pc)
                .or_insert_with(|| (d.insn.len, d.insn.mnemonic.to_string()));
            pc += len;
            next = entries.partition_point(|&x| x < pc);
        }
    }
}

/// Decode from the function entries the symbol table names.
///
/// The walk follows the direct calls and jumps it finds, so the code that
/// only the entry points reach is decoded as well.
fn entries(
    img: &occ_core::Image,
    data: &[u8],
    dec: &Decoder,
    insns: &mut BTreeMap<u64, (u32, String)>,
) {
    let mut todo: Vec<Addr> = Vec::new();
    let mut queued: BTreeSet<Addr> = BTreeSet::new();

    for sym in &img.symbols {
        if !sym.defined || sym.addr.get() == 0 {
            continue;
        }
        if !matches!(sym.kind, occ_core::SymbolKind::Function) {
            continue;
        }
        if queued.insert(sym.addr) {
            todo.push(sym.addr);
        }
    }

    while let Some(start) = todo.pop() {
        let mut pc = start.get();
        loop {
            if !queued.insert(Addr(pc)) {
                break;
            }
            let Some(bytes) = bytes_at(img, data, pc) else { break };
            let Some(d) = dec.decode(Addr(pc), bytes) else { break };
            insns
                .entry(pc)
                .or_insert_with(|| (d.insn.len, d.insn.mnemonic.to_string()));
            // A direct call or jump names another place to decode from.
            if let Some(occ_core::CallTarget::Direct { to }) = d.insn.target {
                if queued.insert(to) {
                    todo.push(to);
                }
            }
            let len = u64::from(d.insn.len).max(1);
            pc += len;
        }
    }
}

/// The bytes of the section that holds `addr`, starting at `addr`.
fn bytes_at<'a>(
    img: &occ_core::Image,
    data: &'a [u8],
    addr: u64,
) -> Option<&'a [u8]> {
    for sec in &img.sections {
        let start = sec.addr.get();
        let end = start + sec.file_size;
        if addr < start || addr >= end {
            continue;
        }
        let off = sec.file_offset as usize;
        let rel = (addr - start) as usize;
        return data.get(off + rel..off + sec.file_size as usize);
    }
    None
}
