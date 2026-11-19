//! Cross-checks the AArch64 decoder against a real compiler output.
//!
//! The fixture is `fixture.c` compiled with `aarch64-linux-gnu-gcc -O2 -c`.
//! Golden values below were read from `objdump -d` of that object, so a
//! regression in the decoder shows up as a mismatch against the reference
//! disassembler rather than against this tool's own earlier output.

use occ_core::{Addr, Arch, CallTarget, FlowKind, ImageId};
use occ_decode::AnyDecoder;

fn fixture() -> (Vec<u8>, occ_core::Image) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/fixture.aarch64.o"
    );
    let data = std::fs::read(path).expect("fixture present");
    let img = occ_bin::load(&data, path, ImageId(0)).expect("parses");
    (data, img)
}

#[test]
fn loads_as_aarch64() {
    let (_, img) = fixture();
    assert_eq!(img.arch, Arch::Aarch64);
    assert_eq!(img.bits, 64);
}

#[test]
fn every_word_in_text_decodes() {
    let (data, img) = fixture();
    let dec = AnyDecoder::new(img.arch, img.bits).unwrap();
    let text = img.section_by_name(".text").unwrap();
    let off = text.file_offset as usize;
    let mut bytes = &data[off..off + text.file_size as usize];
    let mut addr = text.addr;
    let mut unknown = Vec::new();
    while !bytes.is_empty() {
        let d = dec.decode(addr, bytes).unwrap();
        if d.insn.mnemonic.as_ref() == ".word" {
            unknown.push(addr);
        }
        let len = d.insn.len as usize;
        addr = addr.add(len as u64);
        bytes = &bytes[len..];
    }
    assert!(unknown.is_empty(), "undecoded words at {unknown:?}");
}

#[test]
fn matches_objdump_on_known_instructions() {
    let (_, img) = fixture();
    let dec = AnyDecoder::new(img.arch, img.bits).unwrap();
    let text = img.section_by_name(".text").unwrap();
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/fixture.aarch64.o"
    ))
    .unwrap();
    let base = text.file_offset as usize;
    let at = |a: u64| {
        dec.decode(Addr(a), &data[base + a as usize..])
            .unwrap()
            .insn
    };

    let d = at(0);
    assert_eq!(d.text, "cmp w1, #0x0");
    let d = at(4);
    assert_eq!(d.text, "b.le 0x40");
    assert_eq!(d.flow, FlowKind::ConditionalBranch);
    assert_eq!(d.target, Some(CallTarget::Direct { to: Addr(0x40) }));

    let d = at(0x10);
    assert_eq!(d.text, "add x3, x2, w1, sxtw #2");
    assert_eq!(d.text.split_whitespace().next(), Some("add"));

    let d = at(0x18);
    assert_eq!(d.text, "ldr w1, [x2], #4");

    let d = at(0x24);
    assert_eq!(d.text, "b.ne 0x18");

    let d = at(0x34);
    assert_eq!(d.text, "sub w1, w0, w1");
    let d = at(0x38);
    assert_eq!(d.text, "csel w0, w1, w0, gt");

    let d = at(0x3c);
    assert_eq!(d.text, "ret");
    assert_eq!(d.flow, FlowKind::Return);

    let d = at(0x8c);
    assert_eq!(d.text, "mul w1, w1, w2");
}
