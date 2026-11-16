fn main() {
    let path = "/bin/ls";
    let data = std::fs::read(path).unwrap();
    let img = occ_bin::load(&data, path, occ_core::ImageId(0)).unwrap();
    println!("imports {} sections with plt:", img.imports.len());
    for i in img.imports.iter().take(6) { println!("  {:?} slot={:?} lib={:?} kind={:?}", i.name, i.slot, i.library, i.kind); }
    for s in img.sections.iter().filter(|s| s.name.contains("plt")) { println!("  sec {} addr={} size={}", s.name, s.addr, s.size); }
    println!("func symbols {}", img.symbols.iter().filter(|s| s.kind==occ_core::SymbolKind::Function).count());
    let f = img.symbol_at(occ_core::Addr(0x4d84));
    println!("symbol@4d84 = {:?}", f.map(|s| (&s.name, s.size)));
    for a in [0x4d84u64, 0x174a0, 0xe430] { println!("{:x} -> {:?}", a, img.symbol_at(occ_core::Addr(a)).map(|s| (&s.name, s.size))); }
}
