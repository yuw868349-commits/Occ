//! Applying relocations to relocatable objects.
//!
//! An `ET_REL` object has no load address. Every section starts at zero, every
//! reference to another section is a placeholder, and a relocation entry says
//! how to fill that placeholder once a linker picks addresses. Reading the
//! placeholders as if they were values produces a file that no linker ever
//! emitted: a `call` whose displacement is zero lands on the next instruction,
//! a jump table is a run of zeroes, and every function appears to fall through
//! into the next one.
//!
//! This module rewrites those placeholders in place, using the synthetic
//! addresses the loader assigned (each section at its file offset, see
//! [`occ_core::Section`]). It operates on the copy of the file the session
//! keeps for analysis; the file on disk is untouched, and
//! [`occ_core::Image::sha256`] still identifies the original bytes.
//!
//! Symbols not defined in the object — `puts`, `memcpy` — have no in-file
//! target. Each is given a distinct address just past the end of the file, so
//! a call to one stays outside every section and is reported as a call out of
//! the image instead of being mistaken for a branch into the next function.

use std::collections::HashMap;

use occ_core::{Addr, Arch, ArmMode, Endian, Image, Linkage, Relocation};

/// What applying relocations did.
#[derive(Debug, Default, Clone)]
pub struct Applied {
    /// Relocations whose bytes were rewritten.
    pub applied: usize,
    /// Relocations in debug sections whose bytes were rewritten.
    pub debug_applied: usize,
    /// Relocations left alone because this build cannot encode their effect
    /// (GOT slots, TLS offsets, debug-only arithmetic).
    pub deferred: usize,
    /// Relocations whose target does not fit the instruction's immediate.
    pub out_of_range: usize,
    /// Symbols defined outside the object that were given a synthetic address.
    pub externals: usize,
    /// Notes worth carrying into the open response.
    pub notes: Vec<String>,
}

/// True when the image is one whose placeholders still need filling in.
pub fn needed(image: &Image) -> bool {
    image.fingerprint.linkage == Linkage::Relocatable
        && !(image.relocations.is_empty() && image.debug_relocations.is_empty())
}

/// Apply every relocation this build understands.
///
/// Returns `None` when the image needs no relocation pass, so a caller can tell
/// "nothing to do" from "did nothing".
pub fn apply(data: &mut [u8], image: &Image) -> Option<Applied> {
    if !needed(image) {
        return None;
    }
    let mut report = Applied::default();
    let endian = image.endian;

    // Section index to (synthetic address, file offset, readable length).
    let mut sections: HashMap<usize, (u64, u64, u64)> = HashMap::new();
    for s in &image.sections {
        sections.insert(s.index, (s.addr.0, s.file_offset, s.file_size));
    }

    // A symbol the object does not define is resolved by a linker that is not
    // here. The address has to be one no section covers, so that a call to it
    // is not read as a call into the following function; keeping one address
    // per name also keeps two calls to `puts` from looking like two callees.
    let external_base = align_up(data.len() as u64 + 0x1000, 0x1000);
    let mut external: HashMap<&str, u64> = HashMap::new();
    let mut unsupported: Vec<u32> = Vec::new();

    // RISC-V splits one PC-relative address across two instructions and emits a
    // second relocation that names the first instruction rather than the
    // target. The delta computed for the high half has to be kept so its low
    // half can reuse it.
    let mut pcrel_hi: HashMap<u64, i64> = HashMap::new();

    for r in &image.relocations {
        // `ALIGN` and `RELAX` are linker instructions, not value patches.
        if is_linker_hint(image.arch, r.raw_type) {
            continue;
        }
        let Some(section_index) = r.section else {
            report.deferred += 1;
            continue;
        };
        let Some(&(sec_addr, sec_file, sec_size)) = sections.get(&section_index) else {
            report.deferred += 1;
            continue;
        };
        if r.offset.0 >= sec_size {
            // A relocation past the end of the bytes it claims to patch is a
            // malformed table, not a place to write.
            report.deferred += 1;
            continue;
        }
        let place = sec_addr.wrapping_add(r.offset.0);
        let off = (sec_file as usize).wrapping_add(r.offset.0 as usize);
        if off >= data.len() {
            report.deferred += 1;
            continue;
        }

        let value = match r.symbol_addr {
            Some(a) => Some(a.0),
            None => match r.symbol.as_deref().filter(|n| !n.is_empty()) {
                Some(name) => {
                    let next = external_base + (external.len() as u64) * 16;
                    let slot = *external.entry(name).or_insert(next);
                    Some(slot)
                }
                None => None,
            },
        };
        let value = match value {
            Some(v) => v,
            None => {
                report.deferred += 1;
                continue;
            }
        };

        let thumb = image.arm_mode_at(Addr(place)) == Some(ArmMode::T32);
        let outcome = patch_one(
            data,
            image.arch,
            r,
            place,
            off,
            value,
            endian,
            &mut pcrel_hi,
            thumb,
        );
        match outcome {
            Outcome::Patched => report.applied += 1,
            Outcome::Deferred => {
                report.deferred += 1;
                if !unsupported.contains(&r.raw_type) {
                    unsupported.push(r.raw_type);
                }
            }
            Outcome::OutOfRange => report.out_of_range += 1,
        }
    }

    // Debug tables carry placeholders of their own: a `.debug_info` full of
    // zero string offsets and zero code pointers is what an unlinked object
    // holds, and reading it that way names every function after the first
    // string in the file. Their offsets are file offsets, and a relocatable
    // object's section address is its file offset, so the same arithmetic
    // applies; they get their own counter because a reader asking about code
    // should not see them.
    for r in &image.debug_relocations {
        let Some(section_index) = r.section else {
            report.deferred += 1;
            continue;
        };
        let Some(&(sec_addr, sec_file, sec_size)) = sections.get(&section_index) else {
            report.deferred += 1;
            continue;
        };
        if r.offset.0 >= sec_size {
            report.deferred += 1;
            continue;
        }
        let place = sec_addr.wrapping_add(r.offset.0);
        let off = (sec_file as usize).wrapping_add(r.offset.0 as usize);
        if off >= data.len() {
            report.deferred += 1;
            continue;
        }
        let value = match r.symbol_addr {
            Some(a) => Some(a.0),
            None => match r.symbol.as_deref().filter(|n| !n.is_empty()) {
                Some(name) => {
                    let next = external_base + (external.len() as u64) * 16;
                    Some(*external.entry(name).or_insert(next))
                }
                None => None,
            },
        };
        let Some(value) = value else {
            report.deferred += 1;
            continue;
        };
        // A DWARF reference into another debug section is an offset from that
        // section's start, not an address. The symbol's synthetic address is
        // the section's file offset, so subtracting it back out is what makes
        // the string tables readable. A reference into an allocated section —
        // `DW_AT_low_pc` — is a real address and keeps the value.
        let value = match r.symbol_addr {
            Some(a) => debug_section_offset(image, a.0).unwrap_or(value),
            None => value,
        };
        let thumb = image.arm_mode_at(Addr(place)) == Some(ArmMode::T32);
        match patch_one(
            data,
            image.arch,
            r,
            place,
            off,
            value,
            endian,
            &mut pcrel_hi,
            thumb,
        ) {
            Outcome::Patched => report.debug_applied += 1,
            Outcome::Deferred => {
                report.deferred += 1;
                if !unsupported.contains(&r.raw_type) {
                    unsupported.push(r.raw_type);
                }
            }
            Outcome::OutOfRange => report.out_of_range += 1,
        }
    }
    if report.debug_applied > 0 {
        report.notes.push(format!(
            "applied {} relocation(s) to debug sections so names and code pointers in the DWARF resolve",
            report.debug_applied
        ));
    }

    report.externals = external.len();
    if report.applied > 0 {
        report.notes.push(format!(
            "applied {} relocation(s) to a relocatable object so branch, call and table targets resolve",
            report.applied
        ));
    }
    if report.externals > 0 {
        // The names matter: a call to a synthetic address reads as
        // `sub_2000` in the decompiled body, and this is the only place
        // that says which function it stands for.
        let mut named: Vec<(&str, u64)> = external.iter().map(|(n, a)| (*n, *a)).collect();
        named.sort_by_key(|(_, addr)| *addr);
        let shown: Vec<String> = named
            .iter()
            .take(16)
            .map(|(name, addr)| format!("{name}@{addr:#x}"))
            .collect();
        let more = named.len().saturating_sub(shown.len());
        report.notes.push(format!(
            "{} symbol(s) are defined outside the object; their calls land at synthetic addresses past the end of the image: {}{}",
            report.externals,
            shown.join(", "),
            if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            }
        ));
    }
    if report.out_of_range > 0 {
        report.notes.push(format!(
            "{} relocation(s) name a target too far for the instruction encoding and were left unpatched",
            report.out_of_range
        ));
    }
    if !unsupported.is_empty() {
        unsupported.sort_unstable();
        let list: Vec<String> = unsupported.iter().map(|t| t.to_string()).collect();
        report.notes.push(format!(
            "{} relocation(s) of type(s) {} have no encoder here and were left unpatched",
            report.deferred,
            list.join(", ")
        ));
    }
    Some(report)
}

enum Outcome {
    Patched,
    Deferred,
    OutOfRange,
}

/// Relocations that carry no value of their own.
///
/// `ALIGN` and `RELAX` are instructions to the linker. `ADD*`/`SUB*`/`SET*` are
/// one half of a two-entry computation that describes an unwind or debug table:
/// neither entry names an address on its own, and the tables they build are not
/// code, so applying one side would write a number that means nothing.
/// Turn a synthetic address that points into a debug section back into the
/// section-relative offset DWARF stores there.
fn debug_section_offset(image: &Image, addr: u64) -> Option<u64> {
    image
        .sections
        .iter()
        .find(|s| {
            s.name.starts_with(".debug")
                && addr >= s.addr.0
                && addr < s.addr.0 + s.file_size
        })
        .map(|s| addr - s.addr.0)
}

fn is_linker_hint(arch: Arch, raw: u32) -> bool {
    match arch {
        Arch::RiscV32 | Arch::RiscV64 => matches!(raw, 33..=40 | 43 | 51 | 52..=54),
        _ => false,
    }
}

fn align_up(v: u64, to: u64) -> u64 {
    v.div_ceil(to) * to
}

#[allow(clippy::too_many_arguments)]
fn patch_one(
    data: &mut [u8],
    arch: Arch,
    r: &Relocation,
    place: u64,
    off: usize,
    value: u64,
    endian: Endian,
    pcrel_hi: &mut HashMap<u64, i64>,
    thumb: bool,
) -> Outcome {
    // `SHT_REL` keeps the addend in the bytes being relocated. Each encoder
    // below knows how wide its addend is; until it reads it, assume zero and
    // let the encoder add the in-place value.
    let addend = r.addend.unwrap_or_default();
    match arch {
        Arch::X86_64 => patch_x86_64(data, r.raw_type, place, off, value, addend, endian),
        Arch::X86 => patch_x86(data, r.raw_type, place, off, value, addend, endian),
        Arch::Arm => patch_arm(
            data,
            r.raw_type,
            place,
            off,
            value,
            addend,
            endian,
            thumb,
        ),
        Arch::Aarch64 => patch_aarch64(data, r.raw_type, place, off, value, addend, endian),
        Arch::RiscV32 | Arch::RiscV64 => patch_riscv(
            data, r, place, off, value, addend, endian, pcrel_hi,
        ),
        _ => Outcome::Deferred,
    }
}

// ---------------------------------------------------------------------------
// integer access

fn get_u32(d: &[u8], off: usize, e: Endian) -> Option<u32> {
    let b = d.get(off..off + 4)?;
    Some(match e {
        Endian::Little => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        Endian::Big => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
    })
}

fn get_u16(d: &[u8], off: usize, e: Endian) -> Option<u16> {
    let b = d.get(off..off + 2)?;
    Some(match e {
        Endian::Little => u16::from_le_bytes([b[0], b[1]]),
        Endian::Big => u16::from_be_bytes([b[0], b[1]]),
    })
}

fn put_u16(d: &mut [u8], off: usize, v: u16, e: Endian) -> bool {
    let Some(b) = d.get_mut(off..off + 2) else {
        return false;
    };
    let bytes = match e {
        Endian::Little => v.to_le_bytes(),
        Endian::Big => v.to_be_bytes(),
    };
    b.copy_from_slice(&bytes);
    true
}

fn put_u32(d: &mut [u8], off: usize, v: u32, e: Endian) -> bool {
    let Some(b) = d.get_mut(off..off + 4) else {
        return false;
    };
    let bytes = match e {
        Endian::Little => v.to_le_bytes(),
        Endian::Big => v.to_be_bytes(),
    };
    b.copy_from_slice(&bytes);
    true
}

fn put_u64(d: &mut [u8], off: usize, v: u64, e: Endian) -> bool {
    let Some(b) = d.get_mut(off..off + 8) else {
        return false;
    };
    let bytes = match e {
        Endian::Little => v.to_le_bytes(),
        Endian::Big => v.to_be_bytes(),
    };
    b.copy_from_slice(&bytes);
    true
}

/// Signed difference `S + A - P`, computed without wrapping surprises.
fn delta(value: u64, addend: i64, place: u64) -> i128 {
    i128::from(value) + i128::from(addend) - i128::from(place)
}

// ---------------------------------------------------------------------------
// x86-64

fn patch_x86_64(
    data: &mut [u8],
    raw: u32,
    place: u64,
    off: usize,
    value: u64,
    addend: i64,
    endian: Endian,
) -> Outcome {
    match raw {
        // R_X86_64_64: an absolute 64-bit pointer.
        1 => {
            let v = value.wrapping_add(addend as u64);
            if put_u64(data, off, v, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_X86_64_PC32, R_X86_64_PLT32: a 32-bit displacement from the field.
        2 | 4 => {
            let d = delta(value, addend, place);
            if d < i128::from(i32::MIN) || d > i128::from(i32::MAX) {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, d as i32 as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_X86_64_32, R_X86_64_32S: a 32-bit absolute.
        10 | 11 => {
            let v = value.wrapping_add(addend as u64);
            let fits = if raw == 11 {
                // A signed 32-bit absolute is sign-extended into the full
                // register, so the value itself has to already fit in `i32`.
                // Addresses here are non-negative, which leaves the positive
                // half of the range.
                v <= u64::from(i32::MAX as u32)
            } else {
                v <= u64::from(u32::MAX)
            };
            if !fits {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, v as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_X86_64_PC64: a 64-bit displacement.
        24 => {
            let d = delta(value, addend, place);
            if put_u64(data, off, d as i64 as u64, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        _ => Outcome::Deferred,
    }
}

// ---------------------------------------------------------------------------
// i386

fn patch_x86(
    data: &mut [u8],
    raw: u32,
    place: u64,
    off: usize,
    value: u64,
    addend: i64,
    endian: Endian,
) -> Outcome {
    match raw {
        // R_386_32: absolute 32-bit. `SHT_REL`, so the addend is in the field.
        1 => {
            let inplace = get_u32(data, off, endian).map(i64::from).unwrap_or(0);
            let v = i128::from(value) + i128::from(addend) + i128::from(inplace);
            if v < 0 || v > i128::from(u32::MAX) {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, v as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_386_PC32: displacement from the field.
        //
        // The addend is in the field, and it is *signed*: a `call` writes
        // `-4`, because the displacement is measured from the end of the
        // four bytes rather than their start. Reading those four bytes as an
        // unsigned number turns the addend into 4294967292, and every call in
        // an unlinked 32-bit object then lands four gigabytes away and is
        // reported as out of range.
        2 => {
            let inplace = get_u32(data, off, endian)
                .map(|v| i64::from(v as i32))
                .unwrap_or(0);
            let d = delta(value, addend.wrapping_add(inplace), place);
            if d < i128::from(i32::MIN) || d > i128::from(i32::MAX) {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, d as i32 as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        _ => Outcome::Deferred,
    }
}

// ---------------------------------------------------------------------------
// ARM32

/// Patch one ARM32 relocation.
///
/// ARM's `SHT_REL` addend for a branch is `-8`, because the architecture
/// reads the program counter as the instruction's address plus eight. The
/// assembler writes that `-8` into the displacement field, so reading the
/// field back as an unsigned number and adding it puts every branch eight
/// bytes past where it belongs — or, for a call, four gigabytes away.
#[allow(clippy::too_many_arguments)]
fn patch_arm(
    data: &mut [u8],
    raw: u32,
    place: u64,
    off: usize,
    value: u64,
    addend: i64,
    endian: Endian,
    thumb: bool,
) -> Outcome {
    match raw {
        // R_ARM_ABS32: an absolute pointer, which is what a literal pool and
        // a jump table are made of.
        2 => {
            let v = value.wrapping_add(addend as u64);
            if put_u32(data, off, v as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_ARM_REL32.
        3 => {
            let d = delta(value, addend, place);
            if d < i128::from(i32::MIN) || d > i128::from(i32::MAX) {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, d as i32 as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_ARM_PC24, R_ARM_CALL, R_ARM_JUMP24: a signed 26-bit word
        // displacement carrying a 24-bit field.
        1 | 28 | 29 => {
            let Some(word) = get_u32(data, off, endian) else {
                return Outcome::Deferred;
            };
            // The addend is the sign-extended field, and it is already `-8`
            // for a branch the assembler emitted.
            let inplace = sext_arm(bits32(word, 23, 0) << 2, 26);
            let d = i128::from(value) + i128::from(addend) + i128::from(inplace) - i128::from(place);
            if !(-(1i128 << 25)..(1i128 << 25)).contains(&d) {
                return Outcome::OutOfRange;
            }
            if d & 3 != 0 {
                return Outcome::Deferred;
            }
            let field = ((d >> 2) as u32) & 0x00ff_ffff;
            let patched = (word & 0xff00_0000) | field;
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_ARM_THM_CALL, R_ARM_THM_JUMP24: the 26-bit displacement split
        // across two halfwords, with two of its bits inverted.
        10 | 30 => patch_thumb_branch(data, place, off, value, addend, endian),
        // R_ARM_PREL31: `.ARM.exidx` entries, a 31-bit displacement that
        // keeps its top bit as a flag.
        42 => {
            let d = delta(value, addend, place) as i32;
            let word = get_u32(data, off, endian).unwrap_or(0);
            let patched = (word & 0x8000_0000) | (d as u32 & 0x7fff_ffff);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_ARM_MOVW_ABS_NC / R_ARM_MOVT_ABS and their PC-relative forms,
        // and the four Thumb spellings of the same four.
        43..=50 => {
            let v = match raw {
                45 | 46 | 49 | 50 => delta(value, addend, place) as u64,
                _ => value.wrapping_add(addend as u64),
            };
            let imm16 = if matches!(raw, 43 | 45 | 47 | 49) {
                v as u16
            } else {
                (v >> 16) as u16
            };
            // The Thumb spellings are 47 to 50, and they are the only ones
            // that reach a Thumb instruction.  A file that states the
            // instruction set of the region says the same thing, and the
            // two are used together: a Thumb file that names the A32
            // spelling is still a Thumb file.
            if thumb || matches!(raw, 47..=50) {
                return patch_thumb_mov_imm(data, off, imm16, endian);
            }
            let Some(word) = get_u32(data, off, endian) else {
                return Outcome::Deferred;
            };
            // `movw Rd, #imm16`: the four high bits sit at 19:16 and the low
            // twelve at 11:0, with the opcode and register in between.
            let patched = (word & 0xfff0_f000)
                | ((u32::from(imm16) & 0xf000) << 4)
                | u32::from(imm16 & 0x0fff);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_ARM_V4BX: asks a linker to rewrite `bx` for pre-ARMv4T cores.
        // The instruction is already what a modern core wants.
        40 => Outcome::Patched,
        _ => Outcome::Deferred,
    }
}

fn bits32(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

fn sext_arm(v: u32, n: u32) -> i64 {
    let shift = 64 - n;
    ((v as i64) << shift) >> shift
}

/// Patch a Thumb `movw` or `movt`.
///
/// The two instruction sets spell the same relocation the same way and put
/// the immediate in different places.  A32 keeps it in one word: the top
/// four bits at 19:16 and the low twelve at 11:0.  T32 splits the word in
/// two halfwords, and the immediate is in four pieces — the top four bits
/// at 3:0 of the first halfword, the next bit at 10, and the low eleven
/// split between the two halves of the second.
///
/// The halfword that says which of the two it is — `movw` or `movt` — is
/// bit 7 of the first halfword, and it is left alone: the relocation says
/// which half of the address to write, and the instruction says which
/// half of the register to write it into.
fn patch_thumb_mov_imm(
    data: &mut [u8],
    off: usize,
    imm16: u16,
    endian: Endian,
) -> Outcome {
    let Some(mut hw1) = get_u16(data, off, endian) else {
        return Outcome::Deferred;
    };
    let Some(mut hw2) = get_u16(data, off + 2, endian) else {
        return Outcome::Deferred;
    };
    let imm16 = u32::from(imm16);
    // The top four bits, the next bit, the next three and the low eight.
    let imm4 = ((imm16 >> 12) & 0xf) as u16;
    let i = ((imm16 >> 11) & 1) as u16;
    let imm3 = ((imm16 >> 8) & 0x7) as u16;
    let imm8 = (imm16 & 0xff) as u16;
    // The opcode, the register and the half that says which half of the
    // address this is stay where they are.
    hw1 = (hw1 & 0xfbf0) | (i << 10) | imm4;
    hw2 = (hw2 & 0x8f00) | (imm3 << 12) | imm8;
    if put_u16(data, off, hw1, endian) && put_u16(data, off + 2, hw2, endian) {
        Outcome::Patched
    } else {
        Outcome::Deferred
    }
}

/// Patch a Thumb `bl`/`b.w`.
///
/// The displacement is spread over both halfwords and two of its bits are
/// stored inverted, so the reader has to un-invert them before the value is
/// an offset and invert them again before it is an encoding. Getting the
/// inversion wrong produces a branch that is right about half the time.
fn patch_thumb_branch(
    data: &mut [u8],
    place: u64,
    off: usize,
    value: u64,
    addend: i64,
    endian: Endian,
) -> Outcome {
    let Some(mut hw1) = get_u16(data, off, endian) else {
        return Outcome::Deferred;
    };
    let Some(mut hw2) = get_u16(data, off + 2, endian) else {
        return Outcome::Deferred;
    };
    let s = (hw1 >> 10) & 1;
    let j1 = (hw2 >> 13) & 1;
    let j2 = (hw2 >> 11) & 1;
    let i1 = (!(j1 ^ s)) & 1;
    let i2 = (!(j2 ^ s)) & 1;
    let mut a = u32::from(s);
    a = (a << 10) | u32::from(hw1 & 0x3ff);
    a = (a << 1) | u32::from(i1);
    a = (a << 1) | u32::from(i2);
    a = (a << 11) | u32::from(hw2 & 0x7ff);
    a <<= 1;
    let inplace = sext_arm(a, 25);
    let d = i128::from(value) + i128::from(addend) + i128::from(inplace) - i128::from(place);
    if !(-(1i128 << 24)..(1i128 << 24)).contains(&d) {
        return Outcome::OutOfRange;
    }
    let x = d as i64;
    let s = ((x >> 24) & 1) as u16;
    let i1 = ((x >> 23) & 1) as u16;
    let i2 = ((x >> 22) & 1) as u16;
    let imm10 = ((x >> 12) & 0x3ff) as u16;
    let imm11 = ((x >> 1) & 0x7ff) as u16;
    // J1 = NOT(I1 XOR S), the inverse of how the addend was read.
    let j1 = (!(i1 ^ s)) & 1;
    let j2 = (!(i2 ^ s)) & 1;
    hw1 = (hw1 & 0xf800) | (s << 10) | imm10;
    hw2 = (hw2 & 0xd000) | (j1 << 13) | (j2 << 11) | imm11;
    if put_u16(data, off, hw1, endian) && put_u16(data, off + 2, hw2, endian) {
        Outcome::Patched
    } else {
        Outcome::Deferred
    }
}

// ---------------------------------------------------------------------------
// AArch64

/// Patch a 21-bit ADR/ADRP immediate: `immlo` in bits 30:29, `immhi` in 23:5.
fn put_adr_imm(word: u32, imm: i64) -> u32 {
    let immlo = (imm as u64 & 0x3) as u32;
    let immhi = ((imm >> 2) as u64 & 0x7ffff) as u32;
    (word & !((0x3 << 29) | (0x7ffff << 5))) | (immlo << 29) | (immhi << 5)
}

fn patch_aarch64(
    data: &mut [u8],
    raw: u32,
    place: u64,
    off: usize,
    value: u64,
    addend: i64,
    endian: Endian,
) -> Outcome {
    let Some(word) = get_u32(data, off, endian) else {
        return Outcome::Deferred;
    };
    match raw {
        // R_AARCH64_ABS64 / ABS32: absolute data pointers, which is what a
        // jump table and a string-pointer array are made of.
        257 => {
            let v = value.wrapping_add(addend as u64);
            if put_u64(data, off, v, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        258 => {
            let v = delta(value, addend, 0);
            if v < 0 || v > i128::from(u32::MAX) {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, v as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_AARCH64_PREL32 / PREL64: an explicit displacement.
        261 => {
            let d = delta(value, addend, place);
            if d < i128::from(i32::MIN) || d > i128::from(i32::MAX) {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, d as i32 as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        260 => {
            let d = delta(value, addend, place);
            if put_u64(data, off, d as i64 as u64, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // MOVW_UABS_G0..G3: the four 16-bit slices of an absolute address,
        // written by `movz`/`movk`.
        263..=269 => {
            let shift = match raw {
                263 | 264 => 0,
                265 | 266 => 16,
                267 | 268 => 32,
                269 => 48,
                _ => 0,
            };
            let v = value.wrapping_add(addend as u64);
            let slice = ((v >> shift) & 0xffff) as u32;
            let patched = (word & !(0xffff << 5)) | (slice << 5);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_AARCH64_LD_PREL_LO19: a literal load, `imm19` in bits 23:5.
        273 => {
            let d = delta(value, addend, place);
            if d % 4 != 0 {
                return Outcome::OutOfRange;
            }
            let imm = d >> 2;
            if !(-(1 << 18)..(1 << 18)).contains(&imm) {
                return Outcome::OutOfRange;
            }
            let patched = (word & !(0x7ffff << 5)) | (((imm as u32) & 0x7ffff) << 5);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_AARCH64_ADR_PREL_LO21: page-less `adr`.
        274 => {
            let d = delta(value, addend, place);
            if !(-(1 << 20)..(1 << 20)).contains(&d) {
                return Outcome::OutOfRange;
            }
            let patched = put_adr_imm(word, d as i64);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_AARCH64_ADR_PREL_PG_HI21(_NC): `adrp`, page-granular.
        275 | 276 => {
            let target_page = (value.wrapping_add(addend as u64)) & !0xfff;
            let d = i128::from(target_page) - i128::from(place & !0xfff);
            let pages = d >> 12;
            if !(-(1 << 20)..(1 << 20)).contains(&pages) {
                return Outcome::OutOfRange;
            }
            let patched = put_adr_imm(word, pages as i64);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_AARCH64_ADD_ABS_LO12_NC: the low 12 bits of an address.
        277 => {
            let imm = (value.wrapping_add(addend as u64) & 0xfff) as u32;
            let patched = (word & !(0xfff << 10)) | (imm << 10);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_AARCH64_LDST{8,16,32,64,128}_ABS_LO12_NC: the low 12 bits scaled by
        // the access size, because the instruction's immediate counts elements.
        278 | 284 | 285 | 286 | 299 => {
            let scale = match raw {
                278 => 0,
                284 => 1,
                285 => 2,
                286 => 3,
                299 => 4,
                _ => 0,
            };
            let imm = ((value.wrapping_add(addend as u64) & 0xfff) >> scale) as u32;
            let patched = (word & !(0xfff << 10)) | (imm << 10);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_AARCH64_TSTBR14 / CONDBR19: `tbz`/`b.cond`.
        279 => {
            let d = delta(value, addend, place);
            if d % 4 != 0 {
                return Outcome::OutOfRange;
            }
            let imm = d >> 2;
            if !(-(1 << 13)..(1 << 13)).contains(&imm) {
                return Outcome::OutOfRange;
            }
            let patched = (word & !(0x3fff << 5)) | (((imm as u32) & 0x3fff) << 5);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        280 => {
            let d = delta(value, addend, place);
            if d % 4 != 0 {
                return Outcome::OutOfRange;
            }
            let imm = d >> 2;
            if !(-(1 << 18)..(1 << 18)).contains(&imm) {
                return Outcome::OutOfRange;
            }
            let patched = (word & !(0x7ffff << 5)) | (((imm as u32) & 0x7ffff) << 5);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_AARCH64_JUMP26 / CALL26: `b`/`bl`.
        282 | 283 => {
            let d = delta(value, addend, place);
            if d % 4 != 0 {
                return Outcome::OutOfRange;
            }
            let imm = d >> 2;
            if !(-(1 << 25)..(1 << 25)).contains(&imm) {
                return Outcome::OutOfRange;
            }
            let patched = (word & !0x03ff_ffff) | ((imm as u32) & 0x03ff_ffff);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        _ => Outcome::Deferred,
    }
}

// ---------------------------------------------------------------------------
// RISC-V

const RV_BRANCH: u32 = 16;
const RV_JAL: u32 = 17;
const RV_CALL: u32 = 18;
const RV_CALL_PLT: u32 = 19;
const RV_PCREL_HI20: u32 = 23;
const RV_PCREL_LO12_I: u32 = 24;
const RV_PCREL_LO12_S: u32 = 25;
const RV_HI20: u32 = 26;
const RV_LO12_I: u32 = 27;
const RV_LO12_S: u32 = 28;
const RV_RVC_BRANCH: u32 = 44;
const RV_RVC_JUMP: u32 = 45;
const RV_32_PCREL: u32 = 57;

fn riscv_b(imm: i64) -> Option<u32> {
    if imm & 1 != 0 || !(-(1 << 12)..(1 << 12)).contains(&imm) {
        return None;
    }
    let u = imm as u32;
    Some(
        (((u >> 12) & 1) << 31)
            | (((u >> 5) & 0x3f) << 25)
            | (((u >> 1) & 0xf) << 8)
            | (((u >> 11) & 1) << 7),
    )
}

fn riscv_j(imm: i64) -> Option<u32> {
    if imm & 1 != 0 || !(-(1 << 20)..(1 << 20)).contains(&imm) {
        return None;
    }
    let u = imm as u32;
    Some(
        (((u >> 20) & 1) << 31)
            | (((u >> 1) & 0x3ff) << 21)
            | (((u >> 11) & 1) << 20)
            | (((u >> 12) & 0xff) << 12),
    )
}

/// CB-format immediate (`c.beqz`/`c.bnez`), a 9-bit signed multiple of two.
fn rvc_cb(imm: i64) -> Option<u32> {
    if imm & 1 != 0 || !(-(1 << 8)..(1 << 8)).contains(&imm) {
        return None;
    }
    let u = imm as u32;
    Some(
        (((u >> 8) & 1) << 12)
            | (((u >> 3) & 0x3) << 10)
            | (((u >> 6) & 0x3) << 5)
            | (((u >> 1) & 0x3) << 3)
            | (((u >> 5) & 1) << 2),
    )
}

/// CJ-format immediate (`c.j`/`c.jal`), an 11-bit signed multiple of two.
fn rvc_cj(imm: i64) -> Option<u32> {
    if imm & 1 != 0 || !(-(1 << 11)..(1 << 11)).contains(&imm) {
        return None;
    }
    let u = imm as u32;
    Some(
        (((u >> 11) & 1) << 12)
            | (((u >> 4) & 1) << 11)
            | (((u >> 8) & 0x3) << 9)
            | (((u >> 10) & 1) << 8)
            | (((u >> 6) & 1) << 7)
            | (((u >> 7) & 1) << 6)
            | (((u >> 1) & 0x7) << 3)
            | (((u >> 5) & 1) << 2),
    )
}

fn riscv_hi20(d: i128) -> Option<i64> {
    let hi = (d + 0x800) >> 12;
    if !(-(1 << 19)..(1 << 19)).contains(&hi) {
        return None;
    }
    Some(hi as i64)
}

#[allow(clippy::too_many_arguments)]
fn patch_riscv(
    data: &mut [u8],
    r: &Relocation,
    place: u64,
    off: usize,
    value: u64,
    addend: i64,
    endian: Endian,
    pcrel_hi: &mut HashMap<u64, i64>,
) -> Outcome {
    let raw = r.raw_type;
    let Some(word) = get_u32(data, off, endian) else {
        return Outcome::Deferred;
    };
    match raw {
        // R_RISCV_64 / R_RISCV_32: absolute pointers, jump tables, string
        // arrays.
        2 => {
            let v = value.wrapping_add(addend as u64);
            if put_u64(data, off, v, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        1 => {
            let v = delta(value, addend, 0);
            if v < 0 || v > i128::from(u32::MAX) {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, v as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        RV_BRANCH => {
            let d = delta(value, addend, place);
            let Some(enc) = riscv_b(d as i64) else {
                return Outcome::OutOfRange;
            };
            let patched = (word & !0xfe00_0f80) | enc;
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        RV_JAL => {
            let d = delta(value, addend, place);
            let Some(enc) = riscv_j(d as i64) else {
                return Outcome::OutOfRange;
            };
            let patched = (word & !0xffff_f000) | enc;
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_RISCV_CALL(_PLT): an `auipc`/`jalr` pair that together reach the
        // whole address space. Both halves belong to one address.
        RV_CALL | RV_CALL_PLT => {
            let d = delta(value, addend, place);
            let Some(hi) = riscv_hi20(d) else {
                return Outcome::OutOfRange;
            };
            let lo = d - (i128::from(hi) << 12);
            if !(-2048..=2047).contains(&lo) {
                return Outcome::OutOfRange;
            }
            let auipc = (word & 0xfff) | ((hi as u32 & 0xfffff) << 12);
            if !put_u32(data, off, auipc, endian) {
                return Outcome::Deferred;
            }
            let Some(jalr) = get_u32(data, off + 4, endian) else {
                return Outcome::Deferred;
            };
            let jalr = (jalr & !(0xfff << 20)) | (((lo as i32 as u32) & 0xfff) << 20);
            if put_u32(data, off + 4, jalr, endian) {
                pcrel_hi.insert(place, d as i64);
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        // R_RISCV_PCREL_HI20: the high half of an `auipc` pair whose low half
        // arrives as a separate relocation naming this instruction.
        RV_PCREL_HI20 => {
            let d = delta(value, addend, place);
            let Some(hi) = riscv_hi20(d) else {
                return Outcome::OutOfRange;
            };
            let patched = (word & 0xfff) | ((hi as u32 & 0xfffff) << 12);
            if !put_u32(data, off, patched, endian) {
                return Outcome::Deferred;
            }
            pcrel_hi.insert(place, d as i64);
            Outcome::Patched
        }
        // R_RISCV_PCREL_LO12_I/S: the low half of an `auipc` pair. The symbol
        // names the `auipc`, not the target, so the delta comes from the map.
        RV_PCREL_LO12_I | RV_PCREL_LO12_S => {
            let Some(label) = r.symbol_addr.map(|a| a.0) else {
                return Outcome::Deferred;
            };
            let Some(&d) = pcrel_hi.get(&label) else {
                return Outcome::Deferred;
            };
            let Some(hi) = riscv_hi20(i128::from(d)) else {
                return Outcome::OutOfRange;
            };
            let lo = i128::from(d) - (i128::from(hi) << 12);
            patch_riscv_lo(data, off, word, raw, lo as i64, endian)
        }
        RV_HI20 => {
            let d = delta(value, addend, 0);
            let Some(hi) = riscv_hi20(d) else {
                return Outcome::OutOfRange;
            };
            let patched = (word & 0xfff) | ((hi as u32 & 0xfffff) << 12);
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        RV_LO12_I | RV_LO12_S => {
            let d = delta(value, addend, 0);
            patch_riscv_lo(data, off, word, raw, (d & 0xfff) as i64, endian)
        }
        RV_32_PCREL => {
            let d = delta(value, addend, place);
            if d < i128::from(i32::MIN) || d > i128::from(i32::MAX) {
                return Outcome::OutOfRange;
            }
            if put_u32(data, off, d as i32 as u32, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        RV_RVC_BRANCH => {
            let d = delta(value, addend, place);
            let Some(enc) = rvc_cb(d as i64) else {
                return Outcome::OutOfRange;
            };
            let patched = (word & !0x1c7c) | enc;
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        RV_RVC_JUMP => {
            let d = delta(value, addend, place);
            let Some(enc) = rvc_cj(d as i64) else {
                return Outcome::OutOfRange;
            };
            let patched = (word & !0x1ffc) | enc;
            if put_u32(data, off, patched, endian) {
                Outcome::Patched
            } else {
                Outcome::Deferred
            }
        }
        _ => Outcome::Deferred,
    }
}

/// I-format immediate in bits 31:20, S-format split across 31:25 and 11:7.
fn patch_riscv_lo(
    data: &mut [u8],
    off: usize,
    word: u32,
    kind: u32,
    lo: i64,
    endian: Endian,
) -> Outcome {
    let bits = (lo as u32) & 0xfff;
    let patched = if kind == RV_LO12_S || kind == RV_PCREL_LO12_S {
        (word & !((0x7f << 25) | (0x1f << 7))) | (((bits >> 5) & 0x7f) << 25) | ((bits & 0x1f) << 7)
    } else {
        (word & !(0xfff << 20)) | (bits << 20)
    };
    if put_u32(data, off, patched, endian) {
        Outcome::Patched
    } else {
        Outcome::Deferred
    }
}

/// Count the relocation types this build can apply, for the open response.
pub fn encodable(arch: Arch, raw: u32) -> bool {
    match arch {
        Arch::X86_64 => matches!(raw, 1 | 2 | 4 | 10 | 11 | 24),
        Arch::X86 => matches!(raw, 1 | 2),
        Arch::Aarch64 => matches!(
            raw,
            257 | 258 | 260 | 261 | 263..=269 | 273..=283 | 284 | 285 | 286 | 299
        ),
        Arch::RiscV32 | Arch::RiscV64 => matches!(
            raw,
            1 | 2 | RV_BRANCH | RV_JAL | RV_CALL | RV_CALL_PLT | RV_PCREL_HI20
                | RV_PCREL_LO12_I | RV_PCREL_LO12_S | RV_HI20 | RV_LO12_I | RV_LO12_S
                | RV_32_PCREL | RV_RVC_BRANCH | RV_RVC_JUMP
        ),
        _ => false,
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use occ_core::{Addr, RelocKind};

    /// A `call` in an unlinked 32-bit object states its own addend.
    ///
    /// The displacement is measured from the end of the four-byte field, so
    /// the assembler writes `-4` into it and the relocation has nothing else
    /// to go on. Reading that field as unsigned puts the target four
    /// gigabytes before the call, and every call in every 32-bit object is
    /// then refused as out of range — which is exactly what happened before
    /// the field was read as signed.
    #[test]
    fn a_32_bit_call_relocation_reads_its_addend_as_signed() {
        // `call rel32` at 0x1000 to a target at 0x1000: the field holds -4.
        let mut data = vec![0u8; 0x1010];
        data[0x1004..0x1008].copy_from_slice(&(-4i32).to_le_bytes());
        let outcome = patch_x86(&mut data, 2, 0x1004, 0x1004, 0x1000, 0, Endian::Little);
        assert!(
            matches!(outcome, Outcome::Patched),
            "the displacement fits"
        );
        let written = i32::from_le_bytes(data[0x1004..0x1008].try_into().unwrap());
        assert_eq!(written, -8, "target minus the end of the field");
    }

    #[test]
    fn adr_immediate_round_trips() {
        for imm in [0i64, 4, -4, 0x1234, -0x1234] {
            let w = put_adr_imm(0x9000_0000, imm);
            let lo = ((w >> 29) & 0x3) as i64;
            let hi = ((w >> 5) & 0x7ffff) as i64;
            let raw = (hi << 2) | lo;
            let got = if raw & (1 << 20) != 0 {
                raw - (1 << 21)
            } else {
                raw
            };
            assert_eq!(got, imm, "adr immediate {imm:#x}");
        }
    }

    #[test]
    fn riscv_branch_immediate_round_trips() {
        for imm in [0i64, 2, -2, 0x800, -0x800, 0xffe, -0xffe] {
            let enc = riscv_b(imm).expect("fits");
            let back = ((((enc >> 31) & 1) as i64) << 12)
                | ((((enc >> 25) & 0x3f) as i64) << 5)
                | ((((enc >> 8) & 0xf) as i64) << 1)
                | ((((enc >> 7) & 1) as i64) << 11);
            let signed = if back & (1 << 12) != 0 { back - (1 << 13) } else { back };
            assert_eq!(signed, imm, "b-type immediate {imm:#x}");
        }
    }

    #[test]
    fn riscv_jump_immediate_round_trips() {
        for imm in [0i64, 2, -2, 0x7fe, -0x7fe, 0xffffe, -0xffffe] {
            let enc = riscv_j(imm).expect("fits");
            let back = ((((enc >> 31) & 1) as i64) << 20)
                | ((((enc >> 21) & 0x3ff) as i64) << 1)
                | ((((enc >> 20) & 1) as i64) << 11)
                | ((((enc >> 12) & 0xff) as i64) << 12);
            let signed = if back & (1 << 20) != 0 { back - (1 << 21) } else { back };
            assert_eq!(signed, imm, "j-type immediate {imm:#x}");
        }
    }

    #[test]
    fn compressed_branch_immediate_round_trips() {
        for imm in [0i64, 2, -2, 0xfc, -0xfc, 0xfe, -0xfe] {
            let enc = rvc_cb(imm).expect("fits");
            let back = ((((enc >> 12) & 1) as i64) << 8)
                | ((((enc >> 10) & 0x3) as i64) << 3)
                | ((((enc >> 5) & 0x3) as i64) << 6)
                | ((((enc >> 3) & 0x3) as i64) << 1)
                | ((((enc >> 2) & 1) as i64) << 5);
            let signed = if back & (1 << 8) != 0 { back - (1 << 9) } else { back };
            assert_eq!(signed, imm, "cb immediate {imm:#x}");
        }
    }

    #[test]
    fn compressed_jump_immediate_round_trips() {
        for imm in [0i64, 2, -2, 0x7fe, -0x7fe] {
            let enc = rvc_cj(imm).expect("fits");
            let back = ((((enc >> 12) & 1) as i64) << 11)
                | ((((enc >> 11) & 1) as i64) << 4)
                | ((((enc >> 9) & 0x3) as i64) << 8)
                | ((((enc >> 8) & 1) as i64) << 10)
                | ((((enc >> 7) & 1) as i64) << 6)
                | ((((enc >> 6) & 1) as i64) << 7)
                | ((((enc >> 3) & 0x7) as i64) << 1)
                | ((((enc >> 2) & 1) as i64) << 5);
            let signed = if back & (1 << 11) != 0 { back - (1 << 12) } else { back };
            assert_eq!(signed, imm, "cj immediate {imm:#x}");
        }
    }

    #[test]
    fn x86_pc32_writes_the_displacement() {
        let mut data = vec![0u8; 16];
        let r = Relocation::bare(Addr(4), RelocKind::Relative, None);
        let r = Relocation {
            raw_type: 2,
            section: Some(1),
            ..r
        };
        let outcome = patch_one(
            &mut data,
            Arch::X86_64,
            &r,
            0x4010,
            4,
            0x4020,
            Endian::Little,
            &mut HashMap::new(),
            false,
        );
        assert!(matches!(outcome, Outcome::Patched));
        assert_eq!(get_u32(&data, 4, Endian::Little).unwrap(), 0x10);
    }
}
