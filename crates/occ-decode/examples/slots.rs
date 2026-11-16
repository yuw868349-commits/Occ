//! Decode a buffer of fixed-size slots, one instruction per slot.
//!
//! The opcode tables in this crate were built by laying encodings out in
//! slots sixteen bytes wide and asking a disassembler what each one means.
//! Checking the decoder against them needs the same layout on this side:
//! an encoding the decoder rejects must not be able to move the ones after
//! it, or one wrong length would be reported as hundreds.
//!
//! The output is one line per slot: the bytes it holds, the text the
//! decoder produced, and the length it claims. A slot the decoder cannot
//! start on prints `(bad)`.
//!
//! `cargo run --release -p occ-decode --example slots -- <file> [slot]`

use occ_core::addr::Addr;
use occ_decode::x86::Decoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: slots <file> [slot]");
    let slot: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let data = std::fs::read(&path).expect("read");
    let decoder = Decoder::new(64).expect("a decoder");
    for (n, chunk) in data.chunks(slot).enumerate() {
        let text = match decoder.decode(Addr((n * slot) as u64), chunk) {
            Some(d) => d.insn.text.to_string(),
            None => "(bad)".to_string(),
        };
        let hex: String = chunk.iter().map(|b| format!("{b:02x}")).collect();
        println!("{hex}\t{text}");
    }
}
