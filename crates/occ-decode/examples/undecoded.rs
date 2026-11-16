//! Report the byte sequences the decoder gives up on.
//!
//! `cargo run -p occ-decode --example undecoded -- <file>`
//!
//! A byte sequence the decoder cannot read is a hole in the analysis: every
//! instruction after it is read at the wrong offset, and every address
//! after it is wrong. The report says where the holes are and what they
//! look like, so the opcode that caused them can be found.

use occ_core::ImageId;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: undecoded <file>");
    let data = std::fs::read(&path).expect("read");
    let img = occ_bin::load(&data, &path, ImageId(0)).expect("parse");
    let dec = occ_decode::AnyDecoder::new(img.arch, img.bits).expect("decoder");
    let mut holes: std::collections::BTreeMap<String, usize> = Default::default();
    let mut shown = 0usize;
    for s in img.code_sections() {
        let off = s.file_offset as usize;
        let len = s.file_size as usize;
        if off + len > data.len() {
            continue;
        }
        let mut bytes = &data[off..off + len];
        let mut addr = s.addr;
        while !bytes.is_empty() {
            match dec.decode(addr, bytes) {
                Some(d) => {
                    let l = d.insn.len as usize;
                    if l == 0 {
                        break;
                    }
                    if d.insn.mnemonic.as_ref() == ".byte" {
                        let key = format!("{:02x?}", &bytes[..bytes.len().min(8)]);
                        *holes.entry(key.clone()).or_default() += 1;
                        if shown < 40 {
                            eprintln!("{:#x}: {key}", addr.get());
                            shown += 1;
                        }
                    }
                    addr = addr.add(l as u64);
                    bytes = &bytes[l..];
                }
                None => {
                    let key = format!("{:02x?}", &bytes[..bytes.len().min(8)]);
                    *holes.entry(key.clone()).or_default() += 1;
                    if shown < 40 {
                        eprintln!("{:#x}: {key}", addr.get());
                        shown += 1;
                    }
                    // Step over the byte and try again: one bad byte does
                    // not mean the rest of the section is unreadable.
                    addr = addr.add(1);
                    bytes = &bytes[1..];
                }
            }
        }
    }
    println!("{} distinct undecodable sequences:", holes.len());
    let mut rows: Vec<(&String, &usize)> = holes.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1));
    for (bytes, count) in rows.iter().take(40) {
        println!("  {count:>6}  {bytes}");
    }
}
