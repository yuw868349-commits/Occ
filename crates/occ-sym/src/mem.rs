//! A sparse memory for the executor.
//!
//! Cells are keyed by address and hold either bytes or a symbolic value. A load
//! succeeds only when the bytes it needs are covered; a partial overlap reports
//! nothing rather than a guess, which is what keeps the executor from inventing
//! a value the program never wrote.
//!
//! Two limits are deliberate. A store of a symbolic value into a symbolic
//! address cannot be tracked, and neither can a store to a cell a later store
//! partially overwrites; both surface as a missing read, and the caller turns
//! that into a note. Modelling them would mean array theory, which this
//! executor does not have and does not pretend to.

use std::collections::BTreeMap;

use occ_decompile::ir::{BinOp, CastKind, sign_extend, truncate};
use occ_core::Addr;

use crate::value::Value;

/// One memory cell.
#[derive(Clone, Debug)]
enum Cell {
    /// Bytes exactly as written, little-endian.
    Bytes(Vec<u8>),
    /// A symbolic value written at this address.
    Sym {
        /// Width of the write.
        bytes: u8,
        /// The value written.
        value: Value,
    },
}

/// Sparse little-endian memory.
#[derive(Clone, Debug, Default)]
pub struct Memory {
    cells: BTreeMap<u64, Cell>,
}

impl Memory {
    /// Empty memory.
    pub fn new() -> Memory {
        Memory::default()
    }

    /// Number of cells currently held.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// True when nothing was written.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Initialise bytes at an address, as a caller would when setting up an
    /// argument buffer.
    pub fn write_bytes(&mut self, at: Addr, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let addr = at.get();
        let end = addr.wrapping_add(data.len() as u64);
        let dead: Vec<u64> = self
            .cells
            .iter()
            .filter(|(start, cell)| **start < end && *start + cell_len(Some(cell)) > addr)
            .map(|(start, _)| *start)
            .collect();
        for key in dead {
            self.cells.remove(&key);
        }
        // One cell per byte keeps a later partial read well defined, and the
        // number of bytes a caller initialises by hand is small.
        for (i, b) in data.iter().enumerate() {
            let a = addr.wrapping_add(i as u64);
            self.cells.insert(a, Cell::Bytes(vec![*b]));
        }
    }

    /// Store a word.
    pub fn store(&mut self, addr: u64, bytes: u8, value: &Value) {
        let len = u64::from(bytes.max(1));
        let end = addr.wrapping_add(len);
        let mut dead = Vec::new();
        for start in self.cells.keys() {
            if *start >= end {
                break;
            }
            if *start + cell_len(self.cells.get(start)) > addr {
                dead.push(*start);
            }
        }
        // `retain` cannot see the key order requirement above, so drop the
        // overlaps explicitly.
        for key in dead {
            self.cells.remove(&key);
        }
        match value {
            Value::Const(v) => {
                let raw = v.to_le_bytes();
                let chunk: Vec<u8> = raw[..usize::from(bytes.clamp(1, 8))].to_vec();
                self.cells.insert(addr, Cell::Bytes(chunk));
            }
            Value::Sym(e) => {
                self.cells.insert(
                    addr,
                    Cell::Sym {
                        bytes: bytes.clamp(1, 8),
                        value: Value::Sym(e.clone()),
                    },
                );
            }
        }
    }

    /// Load a word, sign-extending when asked.
    ///
    /// `None` means the bytes are not all known.
    pub fn load(&self, addr: u64, bytes: u8, signed: bool) -> Option<Value> {
        let bytes = bytes.clamp(1, 8);
        if let Some(cell) = self.cells.get(&addr) {
            match cell {
                Cell::Sym {
                    bytes: stored,
                    value,
                } if *stored == bytes => return Some(narrow(value, bytes, signed)),
                Cell::Sym { bytes: stored, value } if *stored > bytes => {
                    let cut = Value::cast(CastKind::Trunc, value, *stored, 8);
                    return Some(narrow(&cut, bytes, signed));
                }
                Cell::Bytes(data) if data.len() >= usize::from(bytes) => {
                    let mut raw = [0u8; 8];
                    raw[..usize::from(bytes)]
                        .copy_from_slice(&data[..usize::from(bytes)]);
                    return Some(Value::Const(assemble(raw, bytes, signed)));
                }
                _ => {}
            }
        }
        // Assemble from single bytes when every one of them is known.
        let mut raw = [0u8; 8];
        for (i, slot) in raw.iter_mut().enumerate().take(usize::from(bytes)) {
            match self.cells.get(&addr.wrapping_add(i as u64)) {
                Some(Cell::Bytes(data)) if data.len() == 1 => *slot = data[0],
                _ => return None,
            }
        }
        Some(Value::Const(assemble(raw, bytes, signed)))
    }

    /// Addresses with a cell, in order.
    pub fn addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.cells.keys().copied()
    }
}

fn cell_len(cell: Option<&Cell>) -> u64 {
    match cell {
        Some(Cell::Bytes(d)) => d.len() as u64,
        Some(Cell::Sym { bytes, .. }) => u64::from(*bytes),
        None => 0,
    }
}
fn assemble(raw: [u8; 8], bytes: u8, signed: bool) -> u64 {
    let mut v = u64::from_le_bytes(raw);
    v = truncate(v, usize::from(bytes));
    if signed {
        sign_extend(v, bytes) as u64
    } else {
        v
    }
}

fn narrow(v: &Value, bytes: u8, signed: bool) -> Value {
    if bytes >= 8 {
        return v.clone();
    }
    match v {
        Value::Const(c) => {
            let cut = truncate(*c, usize::from(bytes));
            Value::Const(if signed {
                sign_extend(cut, bytes) as u64
            } else {
                cut
            })
        }
        Value::Sym(_) if signed => Value::cast(CastKind::Sext, v, bytes, 8),
        Value::Sym(_) => {
            let mask = if bytes >= 8 {
                u64::MAX
            } else {
                (1u64 << (u32::from(bytes) * 8)) - 1
            };
            Value::bin(BinOp::And, v, &Value::Const(mask), 8)
        }
    }
}

/// Read an unsigned word of `bytes` width from an expression's address, for
/// callers that already resolved the address to a constant.
pub fn read_const(mem: &Memory, addr: u64, bytes: u8, signed: bool) -> Option<Value> {
    mem.load(addr, bytes, signed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_read_back_at_the_same_width() {
        let mut m = Memory::new();
        m.store(0x1000, 4, &Value::Const(0x1122_3344));
        assert_eq!(m.load(0x1000, 4, false), Some(Value::Const(0x1122_3344)));
        assert_eq!(m.load(0x1000, 2, false), Some(Value::Const(0x3344)));
        assert_eq!(m.load(0x1000, 1, false), Some(Value::Const(0x44)));
    }

    #[test]
    fn signed_loads_sign_extend() {
        let mut m = Memory::new();
        m.store(0x2000, 1, &Value::Const(0xff));
        assert_eq!(
            m.load(0x2000, 1, true),
            Some(Value::Const(0xffff_ffff_ffff_ffff))
        );
    }

    #[test]
    fn a_partial_overwrite_is_not_reported_as_a_value() {
        let mut m = Memory::new();
        m.store(0x3000, 8, &Value::Const(1));
        m.store(0x3004, 4, &Value::Const(2));
        // The first eight bytes are no longer one value, and guessing which
        // half survived is exactly the kind of invention this avoids.
        assert_eq!(m.load(0x3000, 8, false), None);
        assert_eq!(m.load(0x3004, 4, false), Some(Value::Const(2)));
    }

    #[test]
    fn initialised_bytes_are_visible_byte_wise() {
        let mut m = Memory::new();
        m.write_bytes(Addr(0x4000), &[0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(m.load(0x4000, 4, false), Some(Value::Const(0xefbe_adde)));
    }
}
