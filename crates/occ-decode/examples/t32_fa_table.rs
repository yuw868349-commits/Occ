//! Print what the decoder makes of every encoding of the T32 `0xFA` class.
//!
//! The class is the one whose first halfword is `1111 1010`: the
//! register shifts, the extends, the parallel additions and
//! subtractions, the exchanged ones, the checksums and the shared
//! space.  It is read from the second halfword as well, so a mistake
//! in one field shows up as a different instruction rather than as a
//! wrong operand, and the only way to find it is to read every
//! encoding of the class.
//!
//!     cargo run -p occ-decode --example t32_fa_table -- [--rn 3]
//!
//! The output is one line per encoding, in the same shape the
//! reference disassembler's output is turned into, so that the two
//! can be compared line by line.

use occ_core::Addr;

fn main() {
    let mut rn = 3u32;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--rn" {
            rn = args.next().unwrap().parse().unwrap();
        }
    }

    let dec = occ_decode::arm::Decoder::with_mode(occ_decode::arm::Mode::Thumb);
    // The class is the one whose first halfword is `1111 1010` *and*
    // whose second halfword is `1111 xxxx`: the two together are what
    // pick the class out, so the second halfword only runs from
    // `0xf000` up.
    for op in 0..16u32 {
        let hw1 = 0xfa00 | (op << 4) | rn;
        for hw2 in 0xf000..0x10000u32 {
            let word = (hw2 << 16) | hw1;
            let bytes = word.to_le_bytes();
            let decoded = dec.decode(Addr::from(0), &bytes).unwrap();
            println!("{hw1:#06x} {hw2:#06x}  {}", decoded.insn.text);
        }
    }
}
