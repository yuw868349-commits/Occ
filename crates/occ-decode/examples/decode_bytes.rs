//! Decode the encodings named on the command line and print the result.
//!
//! The encodings are given as hexadecimal byte strings. Each one is
//! decoded on its own, and the result is printed in the same shape the
//! disassembler uses, so that the two can be compared line by line.

use occ_decode::x86::Decoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let bits: u8 = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let decoder = Decoder::new(bits).expect("a decoder");
    for arg in args {
        let hex: String = arg.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        let mut bytes = Vec::with_capacity(hex.len() / 2);
        let raw = hex.as_bytes();
        let mut i = 0;
        while i + 1 < raw.len() {
            let pair = std::str::from_utf8(&raw[i..i + 2]).expect("hex");
            bytes.push(u8::from_str_radix(pair, 16).expect("hex"));
            i += 2;
        }
        match decoder.decode(occ_core::addr::Addr(0), &bytes) {
            Some(d) => println!("{:<28} -> {}", arg, d.insn.text),
            None => println!("{:<28} -> <undecodable>", arg),
        }
    }
}
