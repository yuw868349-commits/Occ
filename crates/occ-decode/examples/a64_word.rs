//! Decode A64 words given as hexadecimal on standard input.
//!
//! One word per line, `xxxxxxxx` or any spelling the hex digits survive. The
//! output is the word followed by the rendered instruction, so that a run can
//! be compared line by line with a reference disassembler. This exists to check
//! the decoder against the assembler that produced the encodings, not to be a
//! user-facing tool.

use std::io::{self, BufRead};

use occ_core::{Addr, Arch};
use occ_decode::AnyDecoder;

fn main() {
    let decoder = AnyDecoder::new(Arch::Aarch64, 64).expect("aarch64 decoder");
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.expect("read");
        let hex: String = line.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        if hex.len() != 8 {
            continue;
        }
        let word = u32::from_str_radix(&hex, 16).expect("eight hex digits");
        let bytes = word.to_le_bytes();
        match decoder.decode(Addr(0), &bytes) {
            Some(d) => println!("{hex}  {}", d.insn.text),
            None => println!("{hex}  <none>"),
        }
    }
}
