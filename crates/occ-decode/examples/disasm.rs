//! Disassemble a byte range from a real file, for cross-checking with objdump.

use occ_core::{Addr, ImageId};
use occ_decode::Decoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: disasm <file> [start] [count] [--tsv]");
    let start = args
        .next()
        .and_then(|s| occ_core::addr::parse_addr(&s))
        .map(|a| a.get())
        .unwrap_or(0);
    let count: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    // `--tsv` prints one machine-readable row per instruction: the address,
    // the length in bytes, and the text. The cross-check against objdump
    // needs the length as a number, not as a count of printed bytes.
    let tsv = args.next().as_deref() == Some("--tsv");

    let data = std::fs::read(&path).expect("read");
    let img = occ_bin::load(&data, &path, ImageId(0)).expect("parse");
    let sec = img
        .section_at(Addr(start))
        .or_else(|| img.section_by_name(".text"))
        .expect("no .text");
    let off = sec.file_offset as usize;
    let bytes = data.get(off..off + sec.file_size as usize).unwrap_or(&[]);

    let dec = Decoder::new(img.bits).expect("decoder");
    let mut pc = if start == 0 { sec.addr.get() } else { start };
    let mut shown = 0;
    while shown < count {
        let rel = (pc - sec.addr.get()) as usize;
        let Some(slice) = bytes.get(rel..) else { break };
        let Some(d) = dec.decode(Addr(pc), slice) else {
            break;
        };
        if tsv {
            // The mnemonic alone: objdump's Intel syntax writes the operand
            // size in upper case, and comparing the whole text would report
            // `byte ptr` against `BYTE PTR` as a difference.
            let mnemonic = d.insn.text.split_whitespace().next().unwrap_or("");
            println!("{pc:08x}\t{}\t{mnemonic}", d.insn.len);
        } else {
            println!("{:8x}  {:<26} {}", pc, hex(d.insn.encoded()), d.insn.text);
        }
        pc += u64::from(d.insn.len);
        shown += 1;
    }
}

fn hex(b: &[u8]) -> String {
    b.iter()
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}
