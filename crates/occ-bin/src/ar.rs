//! `ar` archives, the container behind `.a` and `.lib`.
//!
//! An archive is a sequence of members, each with a sixty-byte header, and
//! nothing else: no index that a reader depends on, no compression, no
//! nesting. What makes it worth a module is the naming, which has three
//! dialects in circulation.
//!
//! * **GNU**, the common one. A short name ends with `/`, a long one is
//!   written as `/` plus an offset into a separate member called `//`, and a
//!   member called `/` holds the symbol index.
//! * **BSD**. A long name is stored inside the member's own data, and the
//!   header says `#1/` plus the name's length.
//! * **Windows**. `.lib` files are `ar` with two extra member kinds, `/` and
//!   `//` as above, and a second index member whose name is `/` followed by
//!   nothing at all.
//!
//! The two index members carry no code and are not returned. Everything else
//! is returned in file order, with the name resolved and the size honoured,
//! so the caller can hand each one to the object parser unchanged.

use occ_core::{CoreError, Result};

/// The global header every `ar` file starts with.
const MAGIC: &[u8; 8] = b"!<arch>\n";

/// One member header is fixed at sixty bytes.
const HEADER_LEN: usize = 60;

/// One object stored in an archive.
#[derive(Debug)]
pub struct Member<'a> {
    /// The name as recorded, without the GNU terminator or the BSD length.
    ///
    /// It is not a path: `ar` allows any byte except `/` and newline, and a
    /// long-name table entry is terminated by either.
    pub name: String,
    /// The member's bytes, exactly as stored, with any BSD name removed.
    pub data: &'a [u8],
}

/// Split an archive into its members.
///
/// The symbol indexes are dropped, because they name no code and every
/// question this tool answers is about code. A member whose header is
/// inconsistent stops the walk with the offset — a truncated archive is a
/// damaged file, and reporting half of it as if it were whole is worse than
/// saying where it broke.
pub fn members(data: &[u8]) -> Result<Vec<Member<'_>>> {
    if data.len() < MAGIC.len() || &data[..MAGIC.len()] != MAGIC {
        return Err(CoreError::UnsupportedFormat(
            "ar: the file does not start with the !<arch> magic".into(),
        ));
    }

    let mut out = Vec::new();
    // The long-name table, when the archive has one. It is a member of its
    // own and must be read before the members that refer into it, which is
    // why it is kept rather than skipped.
    let mut long_names: Option<&[u8]> = None;
    let mut pos = MAGIC.len();

    while pos < data.len() {
        // A trailing newline after the last member is legal and common.
        if data.len() - pos < HEADER_LEN {
            if data[pos..].iter().all(|b| *b == b'\n') {
                break;
            }
            return Err(CoreError::Malformed {
                offset: pos as u64,
                detail: format!(
                    "a member header needs {HEADER_LEN} bytes but only {} remain",
                    data.len() - pos
                ),
            });
        }

        let head = &data[pos..pos + HEADER_LEN];
        if &head[58..60] != b"`\n" {
            return Err(CoreError::Malformed {
                offset: (pos + 58) as u64,
                detail: "the member header does not end with the two magic bytes".into(),
            });
        }

        let raw_name = field(&head[0..16]);
        let size = decimal(&head[48..58], pos + 48)?;
        let body_start = pos + HEADER_LEN;
        let body_end = match body_start.checked_add(size) {
            Some(end) if end <= data.len() => end,
            _ => {
                return Err(CoreError::Malformed {
                    offset: pos as u64,
                    detail: format!(
                        "the member claims {size} bytes but the file ends at {}",
                        data.len()
                    ),
                });
            }
        };
        let body = &data[body_start..body_end];

        if raw_name == "/" {
            // The GNU symbol index. It names symbols, not objects.
            pos = align(body_end);
            continue;
        }
        if raw_name == "//" {
            long_names = Some(body);
            pos = align(body_end);
            continue;
        }

        let (name, member) = resolve_name(raw_name, body, long_names, pos)?;
        out.push(Member {
            name,
            data: member,
        });
        pos = align(body_end);
    }

    Ok(out)
}

/// The next member starts on an even offset.
fn align(end: usize) -> usize {
    end + (end & 1)
}

/// A header field, with the padding spaces removed.
fn field(bytes: &[u8]) -> &str {
    let text = std::str::from_utf8(bytes).unwrap_or("");
    text.trim_end_matches(' ')
}

/// A header field holding a decimal number.
fn decimal(bytes: &[u8], offset: usize) -> Result<usize> {
    let text = field(bytes);
    text.trim()
        .parse::<usize>()
        .map_err(|_| CoreError::Malformed {
            offset: offset as u64,
            detail: format!("expected a decimal size, found {text:?}"),
        })
}

/// Work out a member's name and where its data starts.
///
/// The three dialects differ only here, so this is the one function that has
/// to know about all of them.
fn resolve_name<'a>(
    raw: &str,
    body: &'a [u8],
    long_names: Option<&[u8]>,
    offset: usize,
) -> Result<(String, &'a [u8])> {
    // BSD: the name is the first `#1/N` bytes of the member itself.
    if let Some(len) = raw.strip_prefix("#1/") {
        let len: usize = len.parse().map_err(|_| CoreError::Malformed {
            offset: offset as u64,
            detail: format!("the member name length {len:?} is not a number"),
        })?;
        if len > body.len() {
            return Err(CoreError::Malformed {
                offset: (offset + HEADER_LEN) as u64,
                detail: format!("the member name claims {len} bytes but the member is {} long", body.len()),
            });
        }
        let name = String::from_utf8_lossy(&body[..len])
            .trim_end_matches('\0')
            .to_string();
        return Ok((name, &body[len..]));
    }

    // GNU: `/` followed by a decimal offset into the `//` member.
    if let Some(digits) = raw.strip_prefix('/') {
        if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
            let at: usize = digits.parse().map_err(|_| CoreError::Malformed {
                offset: offset as u64,
                detail: format!("the long-name offset {digits:?} is not a number"),
            })?;
            let table = long_names.ok_or_else(|| CoreError::Malformed {
                offset: offset as u64,
                detail: "the member refers to a long-name table the archive does not have".into(),
            })?;
            if at >= table.len() {
                return Err(CoreError::Malformed {
                    offset: offset as u64,
                    detail: format!("the long-name offset {at} is past the {}-byte table", table.len()),
                });
            }
            let rest = &table[at..];
            let end = rest
                .iter()
                .position(|b| *b == b'/' || *b == b'\n')
                .unwrap_or(rest.len());
            return Ok((String::from_utf8_lossy(&rest[..end]).to_string(), body));
        }
    }

    // A plain name, terminated by `/` in the GNU dialect.
    Ok((raw.trim_end_matches('/').to_string(), body))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a member with a GNU header.
    fn member(name: &str, data: &[u8]) -> Vec<u8> {
        let mut head = [b' '; HEADER_LEN];
        head[..name.len()].copy_from_slice(name.as_bytes());
        let size = format!("{}", data.len());
        head[48..48 + size.len()].copy_from_slice(size.as_bytes());
        head[58] = b'`';
        head[59] = b'\n';
        let mut out = head.to_vec();
        out.extend_from_slice(data);
        if data.len() & 1 == 1 {
            out.push(b'\n');
        }
        out
    }

    fn archive(parts: &[Vec<u8>]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        for p in parts {
            out.extend_from_slice(p);
        }
        out
    }

    #[test]
    fn a_short_name_loses_its_terminator() {
        let data = archive(&[member("first.o/", b"AAAA"), member("second.o/", b"BBBB")]);
        let found = members(&data).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].name, "first.o");
        assert_eq!(found[0].data, b"AAAA");
        assert_eq!(found[1].name, "second.o");
    }

    #[test]
    fn an_odd_member_is_followed_by_a_pad_byte() {
        // Five bytes of data, one byte of padding, then a member that must
        // still be found at the even offset.
        let data = archive(&[member("odd.o/", b"AAAAA"), member("next.o/", b"B")]);
        let found = members(&data).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].data, b"AAAAA");
        assert_eq!(found[1].name, "next.o");
        assert_eq!(found[1].data, b"B");
    }

    #[test]
    fn the_symbol_index_is_not_a_member() {
        let data = archive(&[member("/", b"\0\0\0\x01"), member("real.o/", b"CC")]);
        let found = members(&data).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "real.o");
    }

    #[test]
    fn a_long_name_is_read_out_of_the_table() {
        let table = b"a_very_long_object_name.o/\nanother_one.o/\n";
        // The second entry starts after the first name and its terminator.
        let second = "a_very_long_object_name.o/\n".len();
        let data = archive(&[
            member("//", table),
            member("/0", b"DD"),
            member(&format!("/{second}"), b"EE"),
        ]);
        let found = members(&data).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].name, "a_very_long_object_name.o");
        assert_eq!(found[1].name, "another_one.o");
    }

    #[test]
    fn a_bsd_name_is_taken_from_the_member_data() {
        // `#1/8` means the first eight bytes of the member are its name.
        let mut body = b"short.o\0".to_vec();
        body.extend_from_slice(b"FF");
        let mut head = [b' '; HEADER_LEN];
        head[..4].copy_from_slice(b"#1/8");
        let size = format!("{}", body.len());
        head[48..48 + size.len()].copy_from_slice(size.as_bytes());
        head[58] = b'`';
        head[59] = b'\n';
        let mut whole = head.to_vec();
        whole.extend_from_slice(&body);

        let data = archive(&[whole]);
        let found = members(&data).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "short.o");
        assert_eq!(found[0].data, b"FF");
    }

    #[test]
    fn a_member_that_runs_past_the_file_is_an_error_with_an_offset() {
        let mut data = archive(&[member("cut.o/", b"AAAA")]);
        data.truncate(data.len() - 2);
        let err = members(&data).unwrap_err();
        match err {
            CoreError::Malformed { offset, .. } => assert_eq!(offset, 8),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_bad_header_magic_is_an_error() {
        let data = archive(&[member("bad.o/", b"A")]);
        let mut broken = data.clone();
        let at = 8 + 58;
        broken[at] = b'X';
        assert!(members(&broken).is_err());
    }
}
