fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(&path).unwrap();
    let img = occ_bin::load(&data, &path, occ_core::ImageId(0)).unwrap();
    println!("arch={:?} bits={} entry={:#x}", img.arch, img.bits, img.entry.get());
    for s in &img.symbols {
        println!("  sym {:?} kind={:?} bind={:?} addr={:#x}", s.name, s.kind, s.binding, s.addr.get());
    }
}
