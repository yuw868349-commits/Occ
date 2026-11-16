fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(&path).unwrap();
    let img = occ_bin::load(&data, &path, occ_core::ImageId(0)).unwrap();
    println!("{} arch={:?} bits={} sections={}", img.path, img.arch, img.bits, img.sections.len());
    let dec = occ_decode::AnyDecoder::new(img.arch, img.bits).unwrap();
    for s in img.code_sections() {
        let off = s.file_offset as usize;
        let len = s.file_size as usize;
        if off + len > data.len() { continue; }
        let mut bytes = &data[off..off + len];
        let mut addr = s.addr;
        let mut bad = 0usize; let mut n = 0usize;
        while !bytes.is_empty() {
            let Some(d) = dec.decode(addr, bytes) else { break };
            let l = d.insn.len as usize;
            if d.insn.mnemonic.as_ref() == ".word" { bad += 1; }
            n += 1;
            addr = addr.add(l as u64);
            bytes = &bytes[l..];
        }
        println!("{} {:?} addr={} size={} insns={} word={} ({:.1}%)", s.name, s.kind, s.addr, s.size, n, bad, 100.0*bad as f64/n as f64);
    }
}
