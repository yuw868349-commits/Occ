//! Borland published-method tables.
//!
//! A VCL form resource names its event handlers as strings (`OnClick =
//! Button1Click`), and the class's published-method table is what turns that
//! name into an address at run time. The table is data, so a disassembler sees
//! no call to the handler and a string search sees the name only inside the
//! form resource, with no pointer to it.
//!
//! Each entry is a fixed record:
//!
//! ```text
//! u16 size          ; 7 + length of the name
//! u32 address       ; the method's code
//! u8  name_len
//! u8  name[name_len]
//! ```
//!
//! The record is self-describing enough to find by scanning: the size must
//! equal `7 + name_len`, the address must land in executable bytes, and the
//! name must be an identifier. Those three checks together are what separate
//! a real entry from the surrounding RTTI, which is full of small integers and
//! pointers.

use crate::view::View;
use occ_core::Addr;
use occ_core::Image;

/// One published method: the name a form refers to and the code behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedMethod {
    /// Method name as the form resource spells it.
    pub name: String,
    /// Entry point of the method.
    pub addr: Addr,
}

/// Scan the image's writable sections for published-method records.
///
/// The scan is deliberately limited to `data`-like sections: a Borland
/// compiler emits the table next to the class's VMT, never in `.text`, and
/// scanning executable bytes would turn instruction immediates into methods.
pub fn published_methods(view: &View, image: &Image) -> Vec<PublishedMethod> {
    let mut out = Vec::new();
    for section in &image.sections {
        // Delphi emits published methods into CODE, Borland C++ into DATA;
        // both are tables, not instructions, and the address check keeps an
        // instruction immediate from becoming a method.
        if !matches!(
            section.kind,
            occ_core::SectionKind::Code
                | occ_core::SectionKind::Data
                | occ_core::SectionKind::Rodata
        ) || section.file_size == 0
        {
            continue;
        }
        let Some(bytes) = view.section_data(section) else {
            continue;
        };
        let mut i = 0usize;
        while i + 7 <= bytes.len() {
            let size = u16::from_le_bytes([bytes[i], bytes[i + 1]]) as usize;

            if !(7..=96).contains(&size) {
                i += 1;
                continue;
            }
            let addr = u32::from_le_bytes([bytes[i + 2], bytes[i + 3], bytes[i + 4], bytes[i + 5]]);
            let name_len = bytes[i + 6] as usize;
            if name_len == 0 || size != 7 + name_len || i + size > bytes.len() {
                i += 1;
                continue;
            }
            let name_bytes = &bytes[i + 7..i + 7 + name_len];
            if !name_bytes
                .iter()
                .all(|b| b.is_ascii_graphic() || *b == b'_')
            {
                i += 1;
                continue;
            }
            let Ok(name) = std::str::from_utf8(name_bytes) else {
                i += 1;
                continue;
            };
            if !name.as_bytes()[0].is_ascii_alphabetic() && name.as_bytes()[0] != b'_' {
                i += 1;
                continue;
            }
            let target = Addr(u64::from(addr));
            let executable = image
                .sections
                .iter()
                .any(|s| s.is_code() && target.0 >= s.addr.0 && target.0 < s.end().0);
            if !executable {
                i += 1;
                continue;
            }
            out.push(PublishedMethod {
                name: name.to_string(),
                addr: target,
            });
            i += size;
        }
    }
    out.sort_by(|a, b| a.addr.cmp(&b.addr).then(a.name.cmp(&b.name)));
    out.dedup_by(|a, b| a.addr == b.addr && a.name == b.name);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::image_with_sections;
    use occ_core::{Perm, Section, SectionKind};

    fn section(name: &str, addr: u64, size: u64, kind: SectionKind, perm: Perm) -> Section {
        Section {
            index: 1,
            name: name.into(),
            addr: Addr(addr),
            file_offset: 0,
            size,
            file_size: size,
            kind,
            perm,
        }
    }

    #[test]
    fn a_size_check_rejects_the_surrounding_rtti() {
        let mut data = vec![0u8; 64];
        // A real record: size 7 + 12, address 0x1000, name Button1Click.
        let at = 8usize;
        data[at..at + 2].copy_from_slice(&19u16.to_le_bytes());
        data[at + 2..at + 6].copy_from_slice(&0x1000u32.to_le_bytes());
        data[at + 6] = 12;
        data[at + 7..at + 19].copy_from_slice(b"Button1Click");
        // A near miss: the size does not match the name length.
        data[32..34].copy_from_slice(&20u16.to_le_bytes());
        data[34..38].copy_from_slice(&0x1000u32.to_le_bytes());
        data[38] = 12;
        data[39..51].copy_from_slice(b"Button2Click");
        let img = image_with_sections(vec![
            section(
                ".text",
                0x1000,
                16,
                SectionKind::Code,
                Perm::R.union(Perm::X),
            ),
            section(
                ".data",
                0x2000,
                64,
                SectionKind::Data,
                Perm::of(true, true, false),
            ),
        ]);
        let view = View::new(&img, &data);
        let found = published_methods(&view, &img);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "Button1Click");
        assert_eq!(found[0].addr, Addr(0x1000));
    }
}
