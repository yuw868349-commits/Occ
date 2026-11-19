//! Cross-checks the RISC-V decoder against `riscv64-linux-gnu-gcc -O2`.

use occ_core::{Addr, Arch, CallTarget, FlowKind, ImageId};
use occ_decode::AnyDecoder;

fn load() -> (Vec<u8>, occ_core::Image) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/fixture.riscv64.o"
    );
    let data = std::fs::read(path).expect("fixture present");
    let img = occ_bin::load(&data, path, ImageId(0)).expect("parses");
    (data, img)
}

#[test]
fn loads_as_riscv64() {
    let (_, img) = load();
    assert_eq!(img.arch, Arch::RiscV64);
}

#[test]
fn every_unit_in_text_decodes() {
    let (data, img) = load();
    let dec = AnyDecoder::new(img.arch, img.bits).unwrap();
    let text = img.section_by_name(".text").unwrap();
    let mut bytes = &data[text.file_offset as usize..][..text.file_size as usize];
    let mut addr = text.addr;
    let mut unknown = Vec::new();
    let mut compressed = 0usize;
    while !bytes.is_empty() {
        let d = dec.decode(addr, bytes).unwrap();
        if d.insn.mnemonic.as_ref() == ".word" {
            unknown.push(addr);
        }
        if d.insn.len == 2 {
            compressed += 1;
        }
        let len = d.insn.len as usize;
        addr = addr.add(len as u64);
        bytes = &bytes[len..];
    }
    assert!(unknown.is_empty(), "undecoded units at {unknown:?}");
    assert!(compressed > 10, "compressed instructions should be present");
}

#[test]
fn matches_objdump_on_known_instructions() {
    let (data, img) = load();
    let dec = AnyDecoder::new(img.arch, img.bits).unwrap();
    let text = img.section_by_name(".text").unwrap();
    let base = text.file_offset as usize;
    let at = |a: u64| {
        dec.decode(Addr(a), &data[base + a as usize..])
            .unwrap()
            .insn
    };

    let d = at(0);
    assert_eq!(d.text, "bge zero, a1, 0x2a");
    assert_eq!(d.flow, FlowKind::ConditionalBranch);

    let d = at(4);
    assert_eq!(d.text, "slli a1, a1, 2");
    assert_eq!(d.len, 2);

    let d = at(0xe);
    assert_eq!(d.text, "lw a4, 0(a5)");
    assert_eq!(d.len, 2);

    let d = at(0x12);
    assert_eq!(d.text, "addw a0, a4");

    let d = at(0x26);
    assert_eq!(d.text, "subw a0, a4");

    let d = at(0x28);
    // `jr ra` and `ret` are the same encoding. The decoder reports `ret`
    // because the instruction returns: the jump target is the return
    // address, which the caller set, and a function that ends in `ret` is a
    // fact the control-flow analysis needs. objdump prints the jump form
    // because it is disassembling, not analysing.
    assert_eq!(d.text, "ret");
    assert_eq!(d.flow, FlowKind::Return);

    let d = at(0x5c);
    assert_eq!(d.text, "mulw a1, a1, a2");

    let d = at(0x50);
    assert_eq!(d.text, "auipc t1, 0x0");
}

#[test]
fn ret_is_recognised() {
    let dec = AnyDecoder::new(Arch::RiscV64, 64).unwrap();
    let d = dec.decode(Addr(0x1000), &[0x67, 0x80, 0x00, 0x00]).unwrap();
    assert_eq!(d.insn.text, "ret");
    assert_eq!(d.insn.flow, FlowKind::Return);
    assert_eq!(
        dec.decode(Addr(0), &[0x6f, 0x00, 0x80, 0x00])
            .unwrap()
            .insn
            .target,
        Some(CallTarget::Direct { to: Addr(0x8) })
    );
}
