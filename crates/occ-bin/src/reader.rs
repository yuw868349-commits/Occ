//! Bounds-checked, endian-aware reads over a file image.
//!
//! Every container parser in this crate reads through this type. The point is
//! not abstraction: it is that every offset in a binary is attacker-controlled,
//! and a parser that slices directly eventually panics on a truncated or
//! hand-crafted file. Here a bad offset produces
//! `Malformed { offset, detail }`, which the MCP layer turns into a structured
//! error instead of a crashed server.
//!
//! Reads are explicit about width, so a struct field cannot silently be read at
//! the wrong size.

use occ_core::{CoreError, Endian, Result};

/// A view over bytes with a fixed byte order.
#[derive(Clone, Copy)]
pub struct Reader<'a> {
    data: &'a [u8],
    endian: Endian,
}

impl<'a> Reader<'a> {
    /// Wrap `data`, reading multi-byte values in `endian` order.
    pub fn new(data: &'a [u8], endian: Endian) -> Self {
        Reader { data, endian }
    }

    /// The underlying bytes.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Total length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True for an empty file.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Byte order in force.
    pub fn endian(&self) -> Endian {
        self.endian
    }

    fn need(&self, offset: usize, len: usize) -> Result<&'a [u8]> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| CoreError::Malformed {
                offset: offset as u64,
                detail: format!("read of {len} bytes overflows the address space"),
            })?;
        self.data
            .get(offset..end)
            .ok_or_else(|| CoreError::Malformed {
                offset: offset as u64,
                detail: format!(
                    "read of {len} bytes past end of file (file is {} bytes)",
                    self.data.len()
                ),
            })
    }

    /// One byte.
    pub fn u8(&self, offset: usize) -> Result<u8> {
        Ok(self.need(offset, 1)?[0])
    }

    /// Two bytes.
    pub fn u16(&self, offset: usize) -> Result<u16> {
        let b = self.need(offset, 2)?;
        let a = [b[0], b[1]];
        Ok(match self.endian {
            Endian::Little => u16::from_le_bytes(a),
            Endian::Big => u16::from_be_bytes(a),
        })
    }

    /// Four bytes.
    pub fn u32(&self, offset: usize) -> Result<u32> {
        let b = self.need(offset, 4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Ok(match self.endian {
            Endian::Little => u32::from_le_bytes(a),
            Endian::Big => u32::from_be_bytes(a),
        })
    }

    /// Eight bytes.
    pub fn u64(&self, offset: usize) -> Result<u64> {
        let b = self.need(offset, 8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(match self.endian {
            Endian::Little => u64::from_le_bytes(a),
            Endian::Big => u64::from_be_bytes(a),
        })
    }

    /// Four bytes as a signed value.
    pub fn i32(&self, offset: usize) -> Result<i32> {
        Ok(self.u32(offset)? as i32)
    }

    /// Two bytes as a signed value.
    pub fn i16(&self, offset: usize) -> Result<i16> {
        Ok(self.u16(offset)? as i16)
    }

    /// Eight bytes as a signed value.
    pub fn i64(&self, offset: usize) -> Result<i64> {
        Ok(self.u64(offset)? as i64)
    }

    /// A byte range.
    pub fn bytes(&self, offset: usize, len: usize) -> Result<&'a [u8]> {
        self.need(offset, len)
    }

    /// A range when it fits, `None` when it does not.
    ///
    /// For optional structures where absence is not an error: a section with a
    /// file offset past the end simply has no bytes, it does not corrupt the
    /// file.
    pub fn bytes_opt(&self, offset: usize, len: usize) -> Option<&'a [u8]> {
        self.data.get(offset..offset.checked_add(len)?)
    }

    /// A NUL-terminated string at `offset`.
    ///
    /// Invalid UTF-8 is replaced rather than rejected: symbol names in the wild
    /// contain stray bytes, and refusing to load an entire binary because one
    /// name is mis-encoded would be the wrong trade.
    pub fn cstr(&self, offset: usize) -> Result<&'a str> {
        if offset >= self.data.len() {
            return Err(CoreError::Malformed {
                offset: offset as u64,
                detail: format!("string offset past end of file ({} bytes)", self.data.len()),
            });
        }
        let rest = &self.data[offset..];
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        Ok(std::str::from_utf8(&rest[..end]).unwrap_or("<invalid utf-8>"))
    }

    /// A string from a dedicated string table.
    ///
    /// Returns an empty string for an out-of-range offset. String table
    /// indices of 0 are legal and mean "no name", so this mirrors that
    /// distinction rather than turning it into an error.
    pub fn str_at(&self, table: &'a [u8], offset: usize) -> &'a str {
        if offset >= table.len() {
            return "";
        }
        let rest = &table[offset..];
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        std::str::from_utf8(&rest[..end]).unwrap_or("<invalid utf-8>")
    }

    /// A fixed-size array at `offset`, as a `[u8; N]`.
    pub fn array<const N: usize>(&self, offset: usize) -> Result<[u8; N]> {
        let b = self.need(offset, N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(b);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_follow_byte_order() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let le = Reader::new(&data, Endian::Little);
        let be = Reader::new(&data, Endian::Big);
        assert_eq!(le.u16(0).unwrap(), 0x0201);
        assert_eq!(be.u16(0).unwrap(), 0x0102);
        assert_eq!(le.u32(0).unwrap(), 0x04030201);
        assert_eq!(be.u32(0).unwrap(), 0x01020304);
        assert_eq!(le.u64(0).unwrap(), 0x0807060504030201);
    }

    #[test]
    fn reads_past_the_end_are_errors_not_panics() {
        let data = [1u8, 2, 3];
        let r = Reader::new(&data, Endian::Little);
        assert!(r.u32(0).is_err());
        assert!(r.u32(2).is_err());
        assert!(r.u64(usize::MAX - 4).is_err());
        let err = r.u32(1).unwrap_err();
        assert!(err.to_string().contains("past end of file"), "{err}");
    }

    #[test]
    fn optional_ranges_return_none_quietly() {
        let data = [1u8, 2, 3];
        let r = Reader::new(&data, Endian::Little);
        assert_eq!(r.bytes_opt(1, 2), Some(&data[1..3]));
        assert_eq!(r.bytes_opt(1, 3), None);
        assert_eq!(r.bytes_opt(usize::MAX, 1), None);
    }

    #[test]
    fn strings_stop_at_nul() {
        let data = b"hello\0world\0";
        let r = Reader::new(data, Endian::Little);
        assert_eq!(r.cstr(0).unwrap(), "hello");
        assert_eq!(r.cstr(6).unwrap(), "world");
        // Offset 12 is one past the final NUL, so there is no byte there at
        // all. An empty name is offset-0-in-the-table, not "past the end".
        assert!(r.cstr(12).is_err());
    }

    #[test]
    fn string_table_offsets_out_of_range_are_empty() {
        let table = b"\0puts\0";
        let r = Reader::new(table, Endian::Little);
        assert_eq!(r.str_at(table, 1), "puts");
        assert_eq!(r.str_at(table, 0), "");
        assert_eq!(r.str_at(table, 99), "");
    }

    #[test]
    fn invalid_utf8_in_a_name_does_not_fail_the_read() {
        let data = b"\xff\xfe\0";
        let r = Reader::new(data, Endian::Little);
        assert_eq!(r.cstr(0).unwrap(), "<invalid utf-8>");
    }

    #[test]
    fn signed_reads_preserve_negative_values() {
        let data = (-2i32).to_le_bytes();
        let r = Reader::new(&data, Endian::Little);
        assert_eq!(r.i32(0).unwrap(), -2);
    }
}
