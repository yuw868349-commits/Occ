//! Location lists: where a variable lives at a given address.
//!
//! Optimised code moves variables between registers and stack slots, so
//! `DW_AT_location` stops being a single expression and becomes a list of
//! `[start, end) -> expression` entries. Two encodings exist: `.debug_loc`
//! (DWARF 2-4) and `.debug_loclists` (DWARF 5, with a compact opcode form).
//!
//! The output is normalised to [`LocRange`] so callers never branch on
//! version.

use occ_core::Addr;

use crate::{describe_expr, read_u16, read_u32, read_u64, read_u8, uleb};

/// One address range and the expression that holds the variable there.
#[derive(Clone, Debug)]
pub struct LocRange {
    /// First address covered.
    pub start: Addr,
    /// One past the last covered address.
    pub end: Addr,
    /// Rendered DWARF expression.
    pub expr: String,
}

impl LocRange {
    /// True when `addr` falls inside this range.
    pub fn covers(&self, addr: Addr) -> bool {
        addr.0 >= self.start.0 && addr.0 < self.end.0
    }
}

fn read_addr(data: &[u8], pos: &mut usize, size: u8) -> Option<u64> {
    let n = usize::from(size);
    let b = data.get(*pos..*pos + n)?;
    let mut v = 0u64;
    for (i, byte) in b.iter().enumerate() {
        v |= u64::from(*byte) << (8 * i);
    }
    *pos += n;
    Some(v)
}

/// `.debug_loc` stores the expression length as a 16-bit integer.
fn read_expr_u16(data: &[u8], pos: &mut usize) -> Option<String> {
    let len = usize::from(read_u16(data, pos)?);
    let b = data.get(*pos..*pos + len)?;
    *pos += len;
    Some(describe_expr(b))
}

/// `.debug_loclists` switched to ULEB128 for the same field.
fn read_expr_uleb(data: &[u8], pos: &mut usize) -> Option<String> {
    let len = usize::try_from(uleb(data, pos)?).ok()?;
    let b = data.get(*pos..*pos + len)?;
    *pos += len;
    Some(describe_expr(b))
}

/// Parse a `.debug_loc` list starting at `off`.
///
/// The list ends at a pair of zero addresses; `0xff..ff` introduces a base
/// address that later offset pairs are relative to.
pub fn parse_debug_loc(data: &[u8], off: usize, addr_size: u8, cu_base: u64) -> Vec<LocRange> {
    let mut out = Vec::new();
    let mut pos = off;
    let mut base = cu_base;
    let size = addr_size.max(4);
    let max = if size == 8 { u64::MAX } else { u64::from(u32::MAX) };
    let mut guard = 0;
    while pos < data.len() && guard < 1_000_000 {
        guard += 1;
        let Some(begin) = read_addr(data, &mut pos, size) else { break };
        let Some(end) = read_addr(data, &mut pos, size) else { break };
        if begin == 0 && end == 0 {
            break;
        }
        if begin == max {
            base = end;
            continue;
        }
        let Some(expr) = read_expr_u16(data, &mut pos) else { break };
        let (start, stop) = (base.wrapping_add(begin), base.wrapping_add(end));
        out.push(LocRange {
            start: Addr(start),
            end: Addr(stop),
            expr,
        });
    }
    out
}

/// Parse a `.debug_loclists` list starting at `off`.
///
/// `addr_sec` backs `DW_LLE_*x` entries that index `.debug_addr`; pass `None`
/// when the section is absent, in which case those entries are skipped.
pub fn parse_debug_loclists(
    data: &[u8],
    off: usize,
    addr_size: u8,
    addr_sec: Option<&[u8]>,
    cu_base: u64,
) -> Vec<LocRange> {
    let mut out = Vec::new();
    let mut pos = off;
    let mut base = cu_base;
    let addr_of = |index: u64| -> Option<u64> {
        let sec = addr_sec?;
        let width = usize::from(addr_size.max(4));
        let at = usize::try_from(index).ok()?.checked_mul(width)?;
        let mut v = 0u64;
        for (i, byte) in sec.get(at..at + width)?.iter().enumerate() {
            v |= u64::from(*byte) << (8 * i);
        }
        Some(v)
    };
    let mut guard = 0;
    while pos < data.len() && guard < 1_000_000 {
        guard += 1;
        let Some(kind) = read_u8(data, &mut pos) else { break };
        match kind {
            0x00 => break, // end_of_list
            0x01 => {
                let Some(i) = uleb(data, &mut pos) else { break };
                if let Some(a) = addr_of(i) {
                    base = a;
                }
            }
            0x02 => {
                let (Some(a), Some(b)) = (uleb(data, &mut pos), uleb(data, &mut pos)) else {
                    break;
                };
                let Some(expr) = read_expr_uleb(data, &mut pos) else { break };
                if let (Some(start), Some(end)) = (addr_of(a), addr_of(b)) {
                    out.push(LocRange {
                        start: Addr(start),
                        end: Addr(end),
                        expr,
                    });
                }
            }
            0x03 => {
                let (Some(a), Some(len)) = (uleb(data, &mut pos), uleb(data, &mut pos)) else {
                    break;
                };
                let Some(expr) = read_expr_uleb(data, &mut pos) else { break };
                if let Some(start) = addr_of(a) {
                    out.push(LocRange {
                        start: Addr(start),
                        end: Addr(start.wrapping_add(len)),
                        expr,
                    });
                }
            }
            0x04 => {
                let (Some(a), Some(b)) = (uleb(data, &mut pos), uleb(data, &mut pos)) else {
                    break;
                };
                let Some(expr) = read_expr_uleb(data, &mut pos) else { break };
                out.push(LocRange {
                    start: Addr(base.wrapping_add(a)),
                    end: Addr(base.wrapping_add(b)),
                    expr,
                });
            }
            0x05 => {
                let Some(expr) = read_expr_uleb(data, &mut pos) else { break };
                out.push(LocRange {
                    start: Addr(0),
                    end: Addr(u64::MAX),
                    expr,
                });
            }
            0x06 => {
                let Some(a) = read_addr(data, &mut pos, addr_size) else { break };
                base = a;
            }
            0x07 => {
                let (Some(a), Some(b)) = (
                    read_addr(data, &mut pos, addr_size),
                    read_addr(data, &mut pos, addr_size),
                ) else {
                    break;
                };
                let Some(expr) = read_expr_uleb(data, &mut pos) else { break };
                out.push(LocRange {
                    start: Addr(a),
                    end: Addr(b),
                    expr,
                });
            }
            0x08 => {
                let Some(a) = read_addr(data, &mut pos, addr_size) else { break };
                let Some(len) = uleb(data, &mut pos) else { break };
                let Some(expr) = read_expr_uleb(data, &mut pos) else { break };
                out.push(LocRange {
                    start: Addr(a),
                    end: Addr(a.wrapping_add(len)),
                    expr,
                });
            }
            _ => break,
        }
    }
    out
}

/// Parse the offset table at the start of `.debug_loclists`, returning the
/// absolute offset for entry `index`. A value past the table is already an
/// absolute section offset, which is what `DW_FORM_sec_offset` supplies.
pub fn loclists_offset(data: &[u8], index: u64) -> Option<usize> {
    let mut pos = 0usize;
    let len = read_u32(data, &mut pos)?;
    if len == 0xffff_ffff {
        read_u64(data, &mut pos)?;
    }
    let version = read_u16(data, &mut pos)?;
    if version < 5 {
        return None;
    }
    let _addr_size = read_u8(data, &mut pos)?;
    let _seg = read_u8(data, &mut pos)?;
    let count = read_u32(data, &mut pos)?;
    if index >= u64::from(count) {
        return usize::try_from(index).ok();
    }
    let at = pos.checked_add(usize::try_from(index).ok()?.checked_mul(4)?)?;
    let mut p = at;
    usize::try_from(read_u32(data, &mut p)?).ok()
}
