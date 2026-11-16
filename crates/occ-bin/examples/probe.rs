//! Print what the parser sees, for cross-checking against `readelf`/`objdump`.

use occ_core::ImageId;

fn main() {
    let path = std::env::args().nth(1).expect("usage: probe <file>");
    let data = std::fs::read(&path).expect("read");
    let img = occ_bin::load(&data, &path, ImageId(0)).expect("parse");
    println!(
        "format={} arch={} bits={} endian={:?}",
        img.format.label(),
        img.arch.label(),
        img.bits,
        img.endian
    );
    println!(
        "entry={} base={} size={}",
        img.entry, img.image_base, img.file_size
    );
    println!(
        "sections={} segments={} symbols={} imports={} exports={} relocs={}",
        img.sections.len(),
        img.segments.len(),
        img.symbols.len(),
        img.imports.len(),
        img.exports.len(),
        img.relocations.len()
    );
    println!("interp={:?}", img.interpreter);
    println!("-- sections --");
    for s in img.sections.iter().take(8) {
        println!(
            "  [{}] {:<16} addr={} size={} kind={:?} perm={}",
            s.index, s.name, s.addr, s.size, s.kind, s.perm
        );
    }
    println!("-- first symbols --");
    for s in img.symbols.iter().filter(|s| !s.name.is_empty()).take(8) {
        println!(
            "  {:<28} {} size={} kind={:?} bind={:?} def={}",
            s.name, s.addr, s.size, s.kind, s.binding, s.defined
        );
    }
    println!("-- imports --");
    for i in img.imports.iter().take(60) {
        println!("  {:<24} {:?}", i.name, i.library);
    }
    println!("-- relocs --");
    for r in img.relocations.iter().take(6) {
        println!(
            "  {} {:?} {:?} addend={:?}",
            r.offset, r.kind, r.symbol, r.addend
        );
    }
}
