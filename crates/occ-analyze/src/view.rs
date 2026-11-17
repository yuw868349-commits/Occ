//! Address-space access over a parsed image.
//!
//! Every question this crate answers — "what bytes are at this address", "is
//! this address executable", "where does this string end" — goes through
//! [`View`]. Keeping the translation in one place is what stops the classic
//! bug of reading a virtual address as a file offset: the two coincide only
//! for the first `PT_LOAD` of a simple executable, so a mistake still produces
//! plausible bytes and a wrong answer.
//!
//! Relocatable objects (`.o`, `.obj`) have no load address: every section
//! starts at zero. For those, addresses are reported as file offsets, which
//! keeps them unique and self-consistent. [`View::synthetic`] records that
//! choice so the caller can say so instead of implying the kernel would map
//! the file there.

use occ_core::{Addr, Image, Perm, Section};

/// One mapped span, resolved once at construction.
#[derive(Clone, Debug)]
struct Span {
    start: Addr,
    end: Addr,
    file_off: usize,
    section: usize,
}

/// A read-only window over the image's bytes.
pub struct View<'a> {
    image: &'a Image,
    data: &'a [u8],
    spans: Vec<Span>,
    /// True when addresses were synthesised from file offsets because the
    /// image is relocatable.
    synthetic: bool,
}

impl<'a> View<'a> {
    /// Build a view. Spans are clipped to the file: a section whose declared
    /// size runs past EOF is truncated here rather than producing a read error
    /// in the middle of an unrelated query.
    pub fn new(image: &'a Image, data: &'a [u8]) -> View<'a> {
        let relocatable = image.segments.is_empty();
        let mut spans = Vec::with_capacity(image.sections.len());
        for (i, s) in image.sections.iter().enumerate() {
            if s.file_size == 0 {
                continue;
            }
            let (start, file_off) = if !s.addr.is_null() {
                (s.addr, s.file_offset as usize)
            } else if relocatable {
                (Addr(s.file_offset), s.file_offset as usize)
            } else {
                // A non-allocated section in a loadable image has no address.
                continue;
            };
            let avail = data.len().saturating_sub(file_off) as u64;
            let len = s.file_size.min(avail);
            if len == 0 {
                continue;
            }
            spans.push(Span {
                start,
                end: start.add(len),
                file_off,
                section: i,
            });
        }
        spans.sort_by_key(|s| s.start.0);
        View {
            image,
            data,
            spans,
            synthetic: relocatable,
        }
    }

    /// The image this view reads.
    pub fn image(&self) -> &'a Image {
        self.image
    }

    /// Raw file bytes.
    pub fn file(&self) -> &'a [u8] {
        self.data
    }

    /// True when virtual addresses are synthesised from file offsets.
    pub const fn synthetic(&self) -> bool {
        self.synthetic
    }

    /// Bytes covering `[addr, addr+len)`, or `None` when any part is unmapped.
    ///
    /// Partial reads are refused on purpose: a caller that asks for 16 bytes
    /// and gets 3 would decode a truncated instruction and be told it is
    /// invalid, which is the wrong diagnosis. `None` means "not present".
    pub fn bytes(&self, addr: Addr, len: usize) -> Option<&'a [u8]> {
        if len == 0 {
            return Some(&[]);
        }
        let span = self.span_at(addr)?;
        if addr.0.checked_add(len as u64)? > span.end.0 {
            return None;
        }
        let delta = (addr.0 - span.start.0) as usize;
        let off = span.file_off + delta;
        Some(&self.data[off..off + len])
    }

    /// Bytes from `addr` to the end of its containing section.
    pub fn bytes_to_end(&self, addr: Addr) -> Option<&'a [u8]> {
        let span = self.span_at(addr)?;
        let delta = (addr.0 - span.start.0) as usize;
        Some(&self.data[span.file_off + delta..span.file_off + (span.end.0 - span.start.0) as usize])
    }

    /// Contents of a section, addressed by file offset rather than by virtual
    /// address.
    ///
    /// Sections that are not mapped — `.debug_*`, `.comment`, symbol tables —
    /// have no address and are therefore absent from the span map, but their
    /// bytes are still in the file. Anything that consumes metadata rather
    /// than code has to come through here.
    pub fn section_data(&self, section: &Section) -> Option<&'a [u8]> {
        if section.file_size == 0 {
            return None;
        }
        let off = usize::try_from(section.file_offset).ok()?;
        let len = usize::try_from(section.file_size).ok()?;
        let end = off.checked_add(len)?.min(self.data.len());
        if off >= end {
            return None;
        }
        self.data.get(off..end)
    }

    /// Contents of the section with this exact name.
    pub fn section_data_named(&self, name: &str) -> Option<&'a [u8]> {
        self.section_data(self.image.section_by_name(name)?)
    }

    fn span_at(&self, addr: Addr) -> Option<&Span> {
        // Binary search: images have thousands of sections at most, but this
        // is called once per instruction during recovery, so the linear scan
        // would dominate.
        let idx = self.spans.partition_point(|s| s.start <= addr);
        let span = self.spans.get(idx.checked_sub(1)?)?;
        if addr.0 < span.end.0 {
            Some(span)
        } else {
            None
        }
    }

    /// Section containing `addr`.
    pub fn section_at(&self, addr: Addr) -> Option<&'a Section> {
        self.span_at(addr).map(|s| &self.image.sections[s.section])
    }

    /// Permissions in effect at `addr`, from the section when possible and
    /// from the segment map otherwise.
    pub fn perm_at(&self, addr: Addr) -> Perm {
        if let Some(s) = self.section_at(addr)
            && !s.perm.is_empty()
        {
            return s.perm;
        }
        self.image.perm_at(addr)
    }

    /// True when `addr` holds executable bytes.
    pub fn is_exec(&self, addr: Addr) -> bool {
        self.perm_at(addr).exec()
    }

    /// True when `addr` lies inside any mapped span.
    pub fn is_mapped(&self, addr: Addr) -> bool {
        self.span_at(addr).is_some()
    }

    /// Read a NUL-terminated string of at most `limit` bytes.
    ///
    /// Returns `None` when the first byte is not printable ASCII, which
    /// distinguishes "a string starts here" from "this address happens to be
    /// mapped".
    pub fn cstr(&self, addr: Addr, limit: usize) -> Option<String> {
        let bytes = self.bytes_to_end(addr)?;
        let end = bytes
            .iter()
            .take(limit)
            .position(|b| *b == 0)
            .unwrap_or(bytes.len().min(limit));
        let slice = &bytes[..end];
        if slice.is_empty() || !slice.iter().all(|b| (0x20..0x7f).contains(b) || *b == b'\t') {
            return None;
        }
        String::from_utf8(slice.to_vec()).ok()
    }

    /// Little-endian unsigned read of `n` bytes, where `n` is 1, 2, 4 or 8.
    pub fn read_uint(&self, addr: Addr, n: usize) -> Option<u64> {
        let b = self.bytes(addr, n)?;
        let mut v = 0u64;
        for (i, byte) in b.iter().enumerate() {
            v |= u64::from(*byte) << (8 * i);
        }
        Some(v)
    }

    /// Pointer-sized unsigned read honouring the image's endianness.
    pub fn read_ptr(&self, addr: Addr) -> Option<u64> {
        let n = usize::from(self.image.bits / 8);
        let b = self.bytes(addr, n)?;
        let mut v = 0u64;
        match self.image.endian {
            occ_core::Endian::Little => {
                for (i, byte) in b.iter().enumerate() {
                    v |= u64::from(*byte) << (8 * i);
                }
            }
            occ_core::Endian::Big => {
                for byte in b {
                    v = (v << 8) | u64::from(*byte);
                }
            }
        }
        Some(v)
    }

    /// Display name for an address: exact symbol, then covering symbol, then
    /// `sub_<hex>`.
    pub fn name_at(&self, addr: Addr) -> String {
        if let Some(s) = self.image.symbol_at(addr) {
            return s.name.clone();
        }
        if let Some(s) = self.image.symbol_covering(addr) {
            return format!("{}+{:#x}", s.name, addr.0 - s.addr.0);
        }
        format!("sub_{:x}", addr.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use occ_core::*;

    fn image_with(sections: Vec<Section>, segments: Vec<Segment>) -> Image {
        Image {
            id: ImageId(0),
            path: "t".into(),
            format: Format::Elf,
            arch: Arch::X86_64,
            bits: 64,
            endian: Endian::Little,
            entry: Addr(0x1000),
            image_base: Addr(0x1000),
            file_size: 0,
            sha256: String::new(),
            segments,
            sections,
            symbols: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            relocations: Vec::new(),
            fingerprint: Fingerprint::unknown(ImageId(0)),
            interpreter: None,
            warnings: Vec::new(),
            function_bounds: Vec::new(),
        }
    }

    fn section(name: &str, addr: u64, off: u64, size: u64, exec: bool) -> Section {
        Section {
            index: 1,
            name: name.into(),
            addr: Addr(addr),
            file_offset: off,
            size,
            file_size: size,
            kind: if exec { SectionKind::Code } else { SectionKind::Data },
            perm: Perm::of(true, !exec, exec),
        }
    }

    #[test]
    fn reads_through_section_offsets() {
        let data: Vec<u8> = (0..64u8).collect();
        let img = image_with(vec![section(".text", 0x1000, 16, 16, true)], Vec::new());
        let v = View::new(&img, &data);
        assert_eq!(v.bytes(Addr(0x1000), 4).unwrap(), &[16, 17, 18, 19]);
        assert_eq!(v.bytes(Addr(0x100f), 1).unwrap(), &[31]);
        assert!(v.bytes(Addr(0x1010), 1).is_none());
        assert!(v.bytes(Addr(0x0fff), 1).is_none());
    }

    #[test]
    fn refuses_partial_reads() {
        let data: Vec<u8> = vec![0; 32];
        let img = image_with(vec![section(".text", 0x1000, 0, 8, true)], Vec::new());
        let v = View::new(&img, &data);
        assert!(v.bytes(Addr(0x1004), 8).is_none());
        assert_eq!(v.bytes(Addr(0x1004), 4).unwrap().len(), 4);
    }

    #[test]
    fn relocatable_objects_use_file_offsets() {
        let data = vec![0u8; 64];
        let img = image_with(vec![section(".text", 0, 16, 16, true)], Vec::new());
        let v = View::new(&img, &data);
        assert!(v.synthetic());
        assert_eq!(v.bytes(Addr(16), 4).unwrap().len(), 4);
    }

    #[test]
    fn cstr_stops_at_nul_and_rejects_binary() {
        let mut data = vec![0u8; 32];
        data[0..6].copy_from_slice(b"hello\0");
        data[8..12].copy_from_slice(&[0xff, 0x00, 0x01, 0x02]);
        let img = image_with(vec![section(".rodata", 0x2000, 0, 32, false)], Vec::new());
        let v = View::new(&img, &data);
        assert_eq!(v.cstr(Addr(0x2000), 64).unwrap(), "hello");
        assert!(v.cstr(Addr(0x2008), 64).is_none());
    }

    #[test]
    fn pointer_read_honours_width() {
        let mut data = vec![0u8; 16];
        data[0..8].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        let img = image_with(vec![section(".data", 0x3000, 0, 16, false)], Vec::new());
        let v = View::new(&img, &data);
        assert_eq!(v.read_ptr(Addr(0x3000)), Some(0x1122_3344_5566_7788));
    }
}
