//! The tracee's address space, as `/proc/<pid>/maps` reports it.
//!
//! A PIE executable is linked at address zero; the kernel picks the real base
//! at exec time and nothing in the file records it. Every address a trace
//! produces is therefore in a coordinate system the image does not share, and
//! the mapping table is the only translation between them.
//!
//! This is read with `std::fs`, not `ptrace`: the kernel exposes it as text and
//! parsing text is cheaper than a second syscall interface, and the text is
//! already the format every other tool agrees on.

use occ_core::{CoreError, Perm, Result};

/// One mapped range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Map {
    /// First address in the range.
    pub start: u64,
    /// One past the last address.
    pub end: u64,
    /// Access bits.
    pub perm: Perm,
    /// Offset into the mapped file.
    pub offset: u64,
    /// Inode of the mapped file, zero for anonymous memory.
    pub inode: u64,
    /// Path as the kernel printed it, without the ` (deleted)` suffix.
    pub path: Option<String>,
}

impl Map {
    /// Size in bytes.
    pub const fn size(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// True when `addr` falls inside.
    pub const fn contains(&self, addr: u64) -> bool {
        addr >= self.start && addr < self.end
    }
}

/// Read the mapping table for `pid`.
pub fn read(pid: i32) -> Result<Vec<Map>> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/maps")).map_err(|e| CoreError::Io {
        path: format!("/proc/{pid}/maps"),
        source: e,
    })?;
    Ok(parse(&text))
}

/// Parse the text form. Split out so it can be tested without a live process.
pub fn parse(text: &str) -> Vec<Map> {
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(m) = parse_line(line) {
            out.push(m);
        }
    }
    out
}

fn parse_line(line: &str) -> Option<Map> {
    let mut fields = line.splitn(6, ' ');
    let range = fields.next()?;
    let perms = fields.next()?;
    let offset = fields.next()?;
    let _dev = fields.next()?;
    let inode = fields.next()?;
    // The pathname is the rest of the line and may contain spaces; `splitn`
    // keeps it whole, `trim_start` drops the column padding.
    let path = fields.next().map(str::trim_start).filter(|p| !p.is_empty());

    let (start, end) = range.split_once('-')?;
    let start = u64::from_str_radix(start, 16).ok()?;
    let end = u64::from_str_radix(end, 16).ok()?;
    if end <= start {
        return None;
    }
    let bits = perms.as_bytes();
    if bits.len() < 3 {
        return None;
    }
    Some(Map {
        start,
        end,
        perm: Perm::of(bits[0] == b'r', bits[1] == b'w', bits[2] == b'x'),
        offset: u64::from_str_radix(offset, 16).ok()?,
        inode: inode.parse().unwrap_or(0),
        path: path.map(|p| p.trim_end_matches(" (deleted)").to_string()),
    })
}

/// First address at which `path` is mapped.
///
/// This is the load base: the segment the linker placed at file offset zero
/// lands here, so `runtime - base + image_base` is the address the image
/// itself uses.
pub fn base_of(maps: &[Map], path: &str) -> Option<u64> {
    let want = std::fs::canonicalize(path)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.to_string());
    maps.iter()
        .filter(|m| m.offset == 0)
        .filter(|m| {
            m.path.as_deref().is_some_and(|p| {
                p == want || p == path || p.ends_with(&format!("/{}", basename(path)))
            })
        })
        .map(|m| m.start)
        .min()
}

/// Last path component, for matching a mapping against a relative path.
fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The mapping containing `addr`.
pub fn map_at(maps: &[Map], addr: u64) -> Option<&Map> {
    maps.iter().find(|m| m.contains(addr))
}

/// Bounds of the main thread's stack.
pub fn stack(maps: &[Map]) -> Option<(u64, u64)> {
    maps.iter()
        .find(|m| m.path.as_deref() == Some("[stack]"))
        .map(|m| (m.start, m.end))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
555555554000-555555555000 r--p 00000000 08:01 131 /usr/bin/cat
555555555000-555555556000 r-xp 00001000 08:01 131 /usr/bin/cat
555555556000-555555557000 r--p 00002000 08:01 131 /usr/bin/cat
555555557000-555555558000 r--p 00002000 08:01 131 /usr/bin/cat
555555558000-55555555a000 rw-p 00003000 08:01 131 /usr/bin/cat
7ffffffde000-7ffffffff000 rw-p 00000000 00:00 0                          [stack]
7fffffff0000-7fffffff1000 r--p 00000000 00:00 0                          [vvar]
";

    #[test]
    fn maps_parse_into_ranges_with_permissions() {
        let maps = parse(SAMPLE);
        assert_eq!(maps.len(), 7);
        assert_eq!(maps[0].start, 0x5555_5555_4000);
        assert_eq!(maps[0].end, 0x5555_5555_5000);
        assert!(maps[0].perm.read() && !maps[0].perm.exec());
        assert!(maps[1].perm.exec());
        assert_eq!(maps[1].offset, 0x1000);
        assert_eq!(maps[0].path.as_deref(), Some("/usr/bin/cat"));
    }

    #[test]
    fn the_stack_is_found_by_its_kernel_name() {
        let maps = parse(SAMPLE);
        assert_eq!(stack(&maps), Some((0x7fff_fffd_e000, 0x7fff_ffff_f000)));
    }

    #[test]
    fn the_base_is_the_mapping_that_starts_at_file_offset_zero() {
        let maps = parse(SAMPLE);
        assert_eq!(base_of(&maps, "/usr/bin/cat"), Some(0x5555_5555_4000));
        // A path the table does not mention has no base, which is how the
        // caller learns the tracee is not the image it is holding.
        assert_eq!(base_of(&maps, "/usr/bin/other"), None);
    }

    #[test]
    fn a_deleted_binary_still_matches() {
        let maps = parse("400000-401000 r-xp 00000000 08:01 9 /tmp/x (deleted)\n");
        assert_eq!(maps[0].path.as_deref(), Some("/tmp/x"));
    }

    #[test]
    fn a_truncated_line_is_skipped_not_guessed() {
        assert!(parse("not a mapping\n").is_empty());
        assert!(
            parse("1000-900 rw-p 0 0 0\n").is_empty(),
            "end before start"
        );
        assert!(parse("").is_empty());
    }
}
