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

/// One instruction as `objdump` prints it.
struct Row {
    addr: u64,
    len: u8,
    mnemonic: String,
    text: String,
    /// The absolute target objdump resolves for a rip-relative operand.
    target: Option<u64>,
}

/// Ask objdump for the bytes it would disassemble at the entry point.
///
/// Hard-coding this machine's `/bin/ls` addresses made the test a snapshot of
/// one distribution's libc, not a check of the decoder. Deriving the listing
/// at run time keeps the test a differential comparison on any host that has
/// binutils, which is the point of the whole file.
fn objdump(entry: u64, span: u64) -> Option<Vec<Row>> {
    let start = format!("--start-address={entry:#x}");
    let stop = format!("--stop-address={:#x}", entry + span);
    let out = std::process::Command::new("objdump")
        .args(["-d", "-M", "intel", &start, &stop, "/bin/ls"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let mut rows: Vec<Row> = Vec::new();
    for line in text.lines() {
        let Some((addr_part, rest)) = line.trim().split_once(':') else {
            continue;
        };
        let Ok(addr) = u64::from_str_radix(addr_part, 16) else {
            continue;
        };
        // A wrapped instruction ends with bare bytes and no second tab.
        let rest = rest.trim_start();
        let (bytes_part, text_part) = rest.split_once('\t').unwrap_or((rest, ""));
        let bytes: Vec<&str> = bytes_part.split_whitespace().collect();
        if bytes.is_empty() || !bytes.iter().all(|b| u8::from_str_radix(b, 16).is_ok()) {
            continue;
        }
        let text_part = text_part.trim();
        // A long instruction is wrapped: the continuation lines repeat the
        // address of their first byte but carry no mnemonic. They are the rest
        // of the previous instruction, not an instruction of their own.
        if text_part.is_empty() {
            if let Some(last) = rows.last_mut() {
                last.len += bytes.len() as u8;
            }
            continue;
        }
        let mnemonic = text_part.split_whitespace().next().unwrap_or("");
        let target = text_part
            .split('#')
            .nth(1)
            .and_then(|c| c.split_whitespace().next())
            .and_then(parse_hex);
        rows.push(Row {
            addr,
            len: bytes.len() as u8,
            mnemonic: mnemonic.to_string(),
            text: text_part.to_string(),
            target,
        });
    }
    Some(rows)
}

fn parse_hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

/// objdump appends the operand size to the mnemonic (`retq`, `movl`); this
/// decoder prints the bare form. Treat `mov` and `movl` as the same mnemonic
/// while still failing on a genuinely different operation.
fn mnemonic_eq(ours: &str, theirs: &str) -> bool {
    if ours == theirs {
        return true;
    }
    match theirs.strip_suffix(['q', 'l', 'w', 'b']) {
        Some(stem) => stem == ours,
        None => false,
    }
}

#[test]
fn entry_point_matches_objdump_instruction_for_instruction() {
    let Some(l) = ls() else { return };
    let Some(rows) = objdump(l.img.entry.get(), 0x100) else {
        return;
    };
    let dec = Decoder::new(l.img.bits).expect("decoder");
    let text = l.img.section_by_name(".text").expect(".text");

    assert!(!rows.is_empty(), "objdump produced no listing");
    let mut checked = 0;
    for row in &rows {
        if row.addr < text.addr.get() || row.addr >= text.addr.get() + text.size {
            continue;
        }
        let rel = (row.addr - text.addr.get()) as usize;
        let bytes = &l.bytes[rel..];
        let d = dec.decode(Addr(row.addr), bytes).expect("decode");
        assert_eq!(
            d.insn.len,
            u32::from(row.len),
            "at {:#x}: {}",
            row.addr,
            d.insn.text
        );
        // objdump prints prefixes as words of their own (`cs nop ...`,
        // `lock push ...`); accept the mnemonic wherever it appears in the
        // prefix chain.
        assert!(
            row.text
                .split_whitespace()
                .any(|t| mnemonic_eq(d.insn.mnemonic.as_ref(), t)),
            "at {:#x}: ours {} vs objdump {}",
            row.addr,
            d.insn.mnemonic,
            row.mnemonic
        );
        checked += 1;
        if checked == 24 {
            break;
        }
    }
    assert!(checked >= 8, "only {checked} instructions compared");
}

#[test]
fn entry_point_renders_the_same_operands_as_objdump() {
    let Some(l) = ls() else { return };
    let Some(rows) = objdump(l.img.entry.get(), 0x100) else {
        return;
    };
    let dec = Decoder::new(l.img.bits).expect("decoder");
    let text = l.img.section_by_name(".text").expect(".text");

    let mut checked = 0;
    for row in &rows {
        if row.addr < text.addr.get() || row.addr >= text.addr.get() + text.size {
            continue;
        }
        // Memory operands are rendered differently by each tool (`rip+0x1f`
        // versus `rip + 0x1f`, optional `ptr`), so compare only register and
        // immediate forms, where the syntax is unambiguous.
        if row.text.contains('[') || row.text.contains('#') {
            continue;
        }
        let rel = (row.addr - text.addr.get()) as usize;
        let d = dec.decode(Addr(row.addr), &l.bytes[rel..]).expect("decode");
        let ours: String = d.insn.text.chars().filter(|c| !c.is_whitespace()).collect();
        let theirs: String = row.text.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(ours, theirs, "at {:#x}", row.addr);
        checked += 1;
        if checked == 6 {
            break;
        }
    }
    assert!(checked >= 2, "only {checked} instructions compared");
}

#[test]
fn rip_relative_operands_resolve_to_the_target_objdump_prints() {
    let Some(l) = ls() else { return };
    let Some(rows) = objdump(l.img.entry.get(), 0x100) else {
        return;
    };
    let dec = Decoder::new(l.img.bits).expect("decoder");
    let text = l.img.section_by_name(".text").expect(".text");

    let mut checked = 0;
    for row in &rows {
        let Some(target) = row.target else { continue };
        if row.addr < text.addr.get() || row.addr >= text.addr.get() + text.size {
            continue;
        }
        let rel = (row.addr - text.addr.get()) as usize;
        let d = dec.decode(Addr(row.addr), &l.bytes[rel..]).expect("decode");
        let ours = d
            .rip_target
            .or_else(|| d.insn.mem_reads.first().copied())
            .unwrap_or_else(|| panic!("at {:#x}: no resolved target in {}", row.addr, d.insn.text));
        assert_eq!(ours.get(), target, "at {:#x}: {}", row.addr, d.insn.text);
        checked += 1;
        if checked == 3 {
            break;
        }
    }
    assert!(checked >= 1, "objdump resolved no rip-relative operand");
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
