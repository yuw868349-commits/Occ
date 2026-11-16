//! Disassembly of a real function, checked against `objdump`.
//!
//! The expected sequence is what `objdump -d --start-address=0x6d30 /bin/ls`
//! prints. Comparing text is not enough on its own: a length error keeps the
//! first instruction right and corrupts everything after it, so the test also
//! checks that the walk consumes exactly the bytes objdump consumed.

use occ_core::{Addr, ImageId};
use occ_decode::Decoder;

struct Listing {
    img: occ_core::Image,
    bytes: Vec<u8>,
}

fn ls() -> Option<Listing> {
    let data = std::fs::read("/bin/ls").ok()?;
    if occ_bin::probe_format(&data) != occ_core::Format::Elf {
        return None;
    }
    let img = occ_bin::load(&data, "/bin/ls", ImageId(0)).expect("parse");
    let text = img.section_by_name(".text")?;
    let off = text.file_offset as usize;
    let bytes = data.get(off..off + text.file_size as usize)?.to_vec();
    Some(Listing { img, bytes })
}

#[test]
fn entry_point_matches_objdump_instruction_for_instruction() {
    let Some(l) = ls() else { return };
    let dec = Decoder::new(l.img.bits).expect("decoder");
    let text = l.img.section_by_name(".text").expect(".text");

    // (address, expected mnemonic, expected length)
    let want: &[(u64, &str, u8)] = &[
        (0x6d30, "endbr64", 4),
        (0x6d34, "xor", 2),
        (0x6d36, "mov", 3),
        (0x6d39, "pop", 1),
        (0x6d3a, "mov", 3),
        (0x6d3d, "and", 4),
        (0x6d41, "push", 1),
        (0x6d42, "push", 1),
        (0x6d43, "xor", 3),
        (0x6d46, "xor", 2),
        (0x6d48, "lea", 7),
        (0x6d4f, "call", 6),
        (0x6d55, "hlt", 1),
    ];

    for (addr, mnemonic, len) in want {
        let rel = (*addr - text.addr.get()) as usize;
        let d = dec.decode(Addr(*addr), &l.bytes[rel..]).expect("decode");
        assert_eq!(
            d.insn.mnemonic.as_ref(),
            *mnemonic,
            "at {addr:#x}: {}",
            d.insn.text
        );
        assert_eq!(d.insn.len, u32::from(*len), "at {addr:#x}: {}", d.insn.text);
    }
}

#[test]
fn entry_point_renders_the_same_text_as_objdump() {
    let Some(l) = ls() else { return };
    let dec = Decoder::new(l.img.bits).expect("decoder");
    let text = l.img.section_by_name(".text").expect(".text");

    let cases: &[(u64, &str)] = &[
        (0x6d30, "endbr64"),
        (0x6d34, "xor ebp, ebp"),
        (0x6d36, "mov r9, rdx"),
        (0x6d3a, "mov rdx, rsp"),
        (0x6d41, "push rax"),
        (0x6d43, "xor r8d, r8d"),
        (0x6d46, "xor ecx, ecx"),
    ];
    for (addr, want) in cases {
        let rel = (*addr - text.addr.get()) as usize;
        let d = dec.decode(Addr(*addr), &l.bytes[rel..]).expect("decode");
        assert_eq!(&d.insn.text, want, "at {addr:#x}");
    }
}

#[test]
fn lea_and_call_resolve_their_rip_relative_targets() {
    let Some(l) = ls() else { return };
    let dec = Decoder::new(l.img.bits).expect("decoder");
    let text = l.img.section_by_name(".text").expect(".text");

    // lea rdi, [rip-0x1faf] at 0x6d48 -> 0x4da0, which is inside .rodata.
    let rel = (0x6d48 - text.addr.get()) as usize;
    let d = dec.decode(Addr(0x6d48), &l.bytes[rel..]).expect("decode");
    let target = d.rip_target.expect("resolved");
    assert_eq!(target.get(), 0x4da0);
    assert!(
        l.img.section_at(target).is_some(),
        "lea target {target} must land in a section"
    );

    // call qword ptr [rip+0x1c243] at 0x6d4f -> 0x22f98, a GOT slot that
    // .rela.plt binds to __libc_start_main.
    let rel = (0x6d4f - text.addr.get()) as usize;
    let d = dec.decode(Addr(0x6d4f), &l.bytes[rel..]).expect("decode");
    let slot = d.rip_target.expect("resolved");
    assert_eq!(slot.get(), 0x22f98);
    let bound = l
        .img
        .relocations
        .iter()
        .find(|r| r.offset == slot)
        .and_then(|r| r.symbol.clone());
    assert_eq!(
        bound.as_deref(),
        Some("__libc_start_main"),
        "the call slot must bind to the libc entry"
    );
}

#[test]
fn a_long_walk_stays_aligned() {
    let Some(l) = ls() else { return };
    let dec = Decoder::new(l.img.bits).expect("decoder");
    let text = l.img.section_by_name(".text").expect(".text");

    // Walk 4000 instructions linearly. The point is not that linear sweep finds
    // real code boundaries, but that the decoder never stalls, never returns a
    // zero length, and never runs past the section.
    let mut pc = text.addr.get();
    let end = text.addr.get() + text.size;
    let mut n = 0u32;
    while pc < end && n < 4000 {
        let rel = (pc - text.addr.get()) as usize;
        let d = dec.decode(Addr(pc), &l.bytes[rel..]).expect("decode");
        assert!(d.insn.len >= 1, "zero-length instruction at {pc:#x}");
        assert!(d.insn.len <= 15, "over-long instruction at {pc:#x}");
        pc += u64::from(d.insn.len);
        n += 1;
    }
    assert!(n >= 4000, "walk stalled after {n} instructions");
    assert!(pc <= end, "walk ran past the section");
}
