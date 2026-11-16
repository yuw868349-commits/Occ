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
        while !bytes.is_empty() {
            let Some(d) = dec.decode(addr, bytes) else { break };
            println!("{:8x}: {:<40} {}", d.insn.addr.get(), d.insn.text, hex(&d.insn));
            let l = d.insn.len as usize;
            addr = addr.add(l as u64);
            bytes = &bytes[l..];
        }
    }
}
fn hex(i: &occ_core::Instruction) -> String {
    i.encoded().iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join("")
}
