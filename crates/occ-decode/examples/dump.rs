//! Print the disassembly of an image, choosing the decoder by address.
//!
//! ARM is the one architecture whose instruction set changes inside a file,
//! so the decoder is chosen by the address rather than by the image: the file
//! itself says which set is in force where, through the mapping symbols.
//!
//!     cargo run -p occ-decode --example dump -- image

use occ_core::{Addr, ArmMode};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(&path).unwrap();
    let img = occ_bin::load(&data, &path, occ_core::ImageId(0)).unwrap();
    let dec = occ_decode::AnyDecoder::new(img.arch, img.bits).unwrap();
    for s in img.code_sections() {
        let off = s.file_offset as usize;
        let len = s.file_size as usize;
        let mut bytes = &data[off..off + len];
        let mut addr = s.addr;
        let mut list: Vec<occ_core::Instruction> = Vec::new();
        while !bytes.is_empty() {
            let Some(d) = decode(&img, &dec, addr, bytes) else {
                // A byte the decoder will not read is not a hole in the
                // listing: it is the end of what this pass can say. The
                // section may hold several functions, and the next one
                // starts at an address the image names.
                break;
            };
            let l = d.insn.len as usize;
            addr = addr.add(l as u64);
            bytes = &bytes[l..];
            list.push(d.insn);
        }
        // The conditions an `it` block imposes are only known once the
        // instructions around it have been read.
        occ_decode::arm::apply_it_conditions(&mut list);
        for insn in &list {
            println!(
                "{:8x}: {:<44} {}",
                insn.addr.get(),
                insn.text,
                hex(insn)
            );
        }
    }
}

/// Decode one instruction, in the instruction set the image states.
fn decode(
    img: &occ_core::Image,
    dec: &occ_decode::AnyDecoder,
    addr: Addr,
    bytes: &[u8],
) -> Option<occ_decode::Decoded> {
    let occ_decode::AnyDecoder::Arm(_) = dec else {
        return dec.decode(addr, bytes);
    };
    match img.arm_mode_at(addr).unwrap_or(ArmMode::A32) {
        ArmMode::T32 => occ_decode::arm::Decoder::with_mode(occ_decode::arm::Mode::Thumb)
            .decode(addr, bytes),
        ArmMode::A32 => occ_decode::arm::Decoder::with_mode(occ_decode::arm::Mode::Arm)
            .decode(addr, bytes),
        ArmMode::Data => None,
    }
}

fn hex(i: &occ_core::Instruction) -> String {
    i.encoded()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join("")
}
