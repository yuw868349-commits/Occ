//! Decode the Thumb encodings named on the input and print the result.
//!
//! Each line of the input is a pair of hexadecimal halfwords: the first
//! and the second.  A 16-bit instruction names only the first, and the
//! second is then zero.  The result is printed in the same shape the
//! disassembler uses, so that it can be compared line by line against
//! another decoder.
//!
//! The address the instruction is read at matters.  A branch's destination
//! is an absolute address, so reading the same encoding at two addresses
//! gives two different destinations, and a comparison against another
//! disassembler is only a comparison of the architecture if both read it at
//! the same address.  A third field on the line gives that address; without
//! one the instruction is read at zero.

use occ_decode::arm::{Decoder, Mode};

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = match args.next().as_deref() {
        Some("arm") => Mode::Arm,
        _ => Mode::Thumb,
    };
    let decoder = Decoder::with_mode(mode);
    let mut input = String::new();
    use std::io::Read;
    let _ = std::io::stdin().read_to_string(&mut input);
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut halves = line.split_whitespace();
        let first = halves.next().and_then(|s| u16::from_str_radix(s, 16).ok());
        let second = halves.next().and_then(|s| u16::from_str_radix(s, 16).ok());
        let (first, second) = match (first, second) {
            (Some(a), b) => (a, b.unwrap_or(0)),
            _ => continue,
        };
        // The address is optional, and defaults to zero.
        let at = halves
            .next()
            .and_then(|s| u64::from_str_radix(s, 16).ok())
            .unwrap_or(0);
        let mut bytes = Vec::with_capacity(4);
        bytes.extend_from_slice(&first.to_le_bytes());
        bytes.extend_from_slice(&second.to_le_bytes());
        match decoder.decode(occ_core::addr::Addr(at), &bytes) {
            Some(d) => println!("{first:04x} {second:04x} {}", d.insn.text),
            None => println!("{first:04x} {second:04x} <undecodable>"),
        }
    }
}
