//! The `.debug_line` state machine.
//!
//! Line tables are a program, not a table: opcodes advance a state machine and
//! emit rows. Two versions matter — DWARF 2-4, where file names are a list of
//! NUL-terminated strings, and DWARF 5, where the directory and file tables are
//! themselves form-encoded and can reference `.debug_line_str`. Both are parsed
//! here and normalised to one row shape, so the caller never branches on
//! version.
//!
//! Rows with address zero are dropped. Compilers emit them as markers
//! (`DW_LNE_set_address` to 0 at the start of a section); keeping them would
//! make every lookup at a low address return "line 0 of the first file".

use occ_core::Addr;

use crate::{cstr, read_u16, read_u32, read_u64, read_u8, sleb, uleb, Value};

/// Line number table content types (DWARF 5).
const DW_LNCT_PATH: u64 = 0x1;
const DW_LNCT_DIR_INDEX: u64 = 0x2;

/// One emitted row.
#[derive(Clone, Debug)]
pub struct LineRow {
    /// Address the row starts at.
    pub addr: Addr,
    /// Source file, as recorded (directory joined when the compiler supplied
    /// one).
    pub file: String,
    /// Line number, 1-based.
    pub line: u32,
    /// Column, 0 when unknown.
    pub column: u32,
    /// True when this row begins a statement.
    pub is_stmt: bool,
    /// True for the terminator row of a sequence.
    pub end_sequence: bool,
}

/// A parsed line program.
#[derive(Clone, Debug, Default)]
pub struct LineTable {
    /// File names, index 0 first.
    pub files: Vec<String>,
    /// Rows in program order.
    pub rows: Vec<LineRow>,
}

impl LineTable {
    /// Row covering `addr`: the last row at or below it, within the same
    /// sequence.
    ///
    /// A row spans `[row.addr, next.addr)`, so an `end_sequence` row is a
    /// fence: `addr` at or past it lies outside the sequence and has no
    /// source line, even if an earlier row shares the same address.
    pub fn row_for(&self, addr: Addr) -> Option<&LineRow> {
        let idx = self.rows.partition_point(|r| r.addr <= addr);
        if idx == 0 {
            return None;
        }
        let row = &self.rows[idx - 1];
        if row.end_sequence {
            return None;
        }
        Some(row)
    }

    /// Source location as `file:line`.
    pub fn location(&self, addr: Addr) -> Option<String> {
        self.row_for(addr)
            .map(|r| format!("{}:{}", r.file, r.line))
    }
}

/// Parse a line program at `offset` in `.debug_line`.
pub(crate) fn parse(
    data: &[u8],
    offset: usize,
    _cu_version: u16,
    _addr_size: u8,
    line_str: &[u8],
    comp_dir: Option<&str>,
) -> LineTable {
    let mut out = LineTable::default();
    if data.is_empty() || offset >= data.len() {
        return out;
    }
    let mut pos = offset;
    let Some(unit_len) = read_u32(data, &mut pos) else {
        return out;
    };
    let (unit_len, _is64) = if unit_len == 0xffff_ffff {
        match read_u64(data, &mut pos) {
            Some(l) => (l as usize, true),
            None => return out,
        }
    } else {
        (unit_len as usize, false)
    };
    let unit_end = pos.saturating_add(unit_len).min(data.len());
    let Some(version) = read_u16(data, &mut pos) else {
        return out;
    };
    let (addr_size, header_len) = if version >= 5 {
        let Some(a) = read_u8(data, &mut pos) else { return out };
        let _seg = read_u8(data, &mut pos);
        let Some(h) = read_u32(data, &mut pos) else { return out };
        (a, h as usize)
    } else {
        (_addr_size, match read_u32(data, &mut pos) {
            Some(h) => h as usize,
            None => return out,
        })
    };
    let header_end = pos.saturating_add(header_len).min(unit_end);
    let Some(min_inst) = read_u8(data, &mut pos) else { return out };
    if version >= 4 {
        let _max_ops = read_u8(data, &mut pos);
    }
    let Some(default_is_stmt) = read_u8(data, &mut pos) else { return out };
    // Signed: DWARF stores line_base as an sbyte, and GCC emits -5 whenever
    // line_range is 14.
    let Some(line_base) = read_u8(data, &mut pos).map(|b| b as i8) else { return out };
    let Some(line_range) = read_u8(data, &mut pos) else { return out };
    let Some(opcode_base) = read_u8(data, &mut pos) else { return out };
    let mut std_lengths = Vec::with_capacity(usize::from(opcode_base));
    std_lengths.push(0u8);
    for _ in 1..opcode_base {
        match read_u8(data, &mut pos) {
            Some(v) => std_lengths.push(v),
            None => return out,
        }
    }

    let mut dirs: Vec<String> = Vec::new();
    if version >= 5 {
        // Directory table.
        let Some(fmt_count) = read_u8(data, &mut pos) else { return out };
        let mut formats = Vec::new();
        for _ in 0..fmt_count {
            let Some(ct) = uleb(data, &mut pos) else { return out };
            let Some(form) = uleb(data, &mut pos) else { return out };
            formats.push((ct, form));
        }
        let Some(count) = uleb(data, &mut pos) else { return out };
        for _ in 0..count {
            let mut path = String::new();
            for (ct, form) in &formats {
                match read_entry_form(data, &mut pos, *form, line_str) {
                    Some(Value::Str(s)) if *ct == DW_LNCT_PATH => path = s,
                    Some(_) => {}
                    None => return out,
                }
            }
            dirs.push(path);
        }
        // A table without entry 0 still resolves relative names against the
        // compilation directory.
        if dirs.first().is_none_or(|d| d.is_empty())
            && let Some(cd) = comp_dir
        {
            if dirs.is_empty() {
                dirs.push(cd.to_string());
            } else {
                dirs[0] = cd.to_string();
            }
        }
        // File table.
        let Some(fmt_count) = read_u8(data, &mut pos) else { return out };
        let mut formats = Vec::new();
        for _ in 0..fmt_count {
            let Some(ct) = uleb(data, &mut pos) else { return out };
            let Some(form) = uleb(data, &mut pos) else { return out };
            formats.push((ct, form));
        }
        let Some(count) = uleb(data, &mut pos) else { return out };
        for _ in 0..count {
            let mut name = String::new();
            let mut dir_index = 0usize;
            for (ct, form) in &formats {
                match read_entry_form(data, &mut pos, *form, line_str) {
                    Some(Value::Str(s)) if *ct == DW_LNCT_PATH => name = s,
                    Some(Value::Uint(n)) if *ct == DW_LNCT_DIR_INDEX => dir_index = n as usize,
                    Some(_) => {}
                    None => return out,
                }
            }
            out.files.push(join_dir(dirs.get(dir_index), &name));
        }
    } else {
        // include_directories
        loop {
            match cstr(data, pos) {
                // The empty string terminates the table; step over it so the
                // file-name table starts at the right byte.
                s if s.is_empty() => {
                    pos += 1;
                    break;
                }
                s => {
                    let len = s.len();
                    dirs.push(s);
                    pos += len + 1;
                }
            }
            if pos >= header_end {
                break;
            }
        }
        // file_names
        loop {
            if pos >= header_end {
                break;
            }
            let name = cstr(data, pos);
            if name.is_empty() {
                break;
            }
            pos += name.len() + 1;
            // DWARF 2-4 directory indices are 1-based, and index 0 means the
            // compilation directory rather than the first table entry.
            let raw = uleb(data, &mut pos).unwrap_or(0) as usize;
            let _mtime = uleb(data, &mut pos);
            let _size = uleb(data, &mut pos);
            let dir = if raw == 0 {
                comp_dir.map(str::to_string)
            } else {
                dirs.get(raw - 1).cloned()
            };
            out.files.push(join_dir(dir.as_ref(), &name));
        }
    }
    pos = header_end;

    // State machine.
    let mut address = 0u64;
    let mut file = 1u64;
    let mut line = 1i64;
    let mut column = 0u64;
    let mut is_stmt = default_is_stmt != 0;
    let mut end_sequence = false;
    let mut guard = 0usize;

    while pos < unit_end && guard < 5_000_000 {
        guard += 1;
        let Some(op) = read_u8(data, &mut pos) else { break };
        if op >= opcode_base {
            let adjusted = u64::from(op - opcode_base);
            let lr = u64::from(line_range.max(1));
            address = address.wrapping_add((adjusted / lr) * u64::from(min_inst.max(1)));
            line += i64::from(line_base) + (adjusted % lr) as i64;
            emit(&mut out, address, file, line, column, is_stmt, end_sequence, version);
            continue;
        }
        match op {
            0 => {
                let Some(len) = uleb(data, &mut pos) else { break };
                let sub_end = pos.saturating_add(len as usize);
                let Some(sub) = read_u8(data, &mut pos) else { break };
                match sub {
                    1 => {
                        emit(&mut out, address, file, line, column, is_stmt, true, version);
                        address = 0;
                        line = 1;
                        column = 0;
                        file = 1;
                        end_sequence = false;
                        is_stmt = default_is_stmt != 0;
                    }
                    2 => {
                        let n = usize::from(addr_size.max(1));
                        let Some(b) = data.get(pos..pos + n) else { break };
                        let mut v = 0u64;
                        for (i, byte) in b.iter().enumerate() {
                            v |= u64::from(*byte) << (8 * i);
                        }
                        address = v;
                    }
                    3 => {
                        // define_file (DWARF 2-4)
                        let name = cstr(data, pos);
                        pos += name.len() + 1;
                        let raw = uleb(data, &mut pos).unwrap_or(0) as usize;
                        let _ = uleb(data, &mut pos);
                        let _ = uleb(data, &mut pos);
                        let dir = if raw == 0 {
                            comp_dir.map(str::to_string)
                        } else {
                            dirs.get(raw - 1).cloned()
                        };
                        out.files.push(join_dir(dir.as_ref(), &name));
                    }
                    4 => {
                        let _ = uleb(data, &mut pos);
                    }
                    _ => {}
                }
                pos = sub_end;
            }
            1 => {
                emit(&mut out, address, file, line, column, is_stmt, end_sequence, version);
            }
            2 => {
                let adv = uleb(data, &mut pos).unwrap_or(0);
                address = address.wrapping_add(adv * u64::from(min_inst.max(1)));
            }
            3 => {
                line += sleb(data, &mut pos).unwrap_or(0);
            }
            4 => {
                file = uleb(data, &mut pos).unwrap_or(1);
            }
            5 => {
                column = uleb(data, &mut pos).unwrap_or(0);
            }
            6 => is_stmt = !is_stmt,
            7 => {}
            8 => {
                let lr = u64::from(line_range.max(1));
                let adjusted = 255u64.saturating_sub(u64::from(opcode_base));
                address = address.wrapping_add((adjusted / lr) * u64::from(min_inst.max(1)));
            }
            9 => {
                let Some(adv) = read_u16(data, &mut pos) else { break };
                address = address.wrapping_add(u64::from(adv));
            }
            10 => {}
            11 => {}
            12 => {
                let _ = uleb(data, &mut pos);
            }
            other => {
                let n = std_lengths
                    .get(other as usize)
                    .copied()
                    .unwrap_or(0);
                for _ in 0..n {
                    let _ = uleb(data, &mut pos);
                }
            }
        }
    }
    // Sequences are not guaranteed to be in address order: GCC emits hot code
    // first and cold code after, but the line program follows the order the
    // functions were compiled in. The lookup is a binary search, so sort once
    // here. The sort is stable, which keeps an end_sequence row ahead of the
    // first row of the sequence that starts at the same address.
    out.rows.sort_by_key(|r| r.addr.0);
    out
}

#[allow(clippy::too_many_arguments)]
fn emit(
    out: &mut LineTable,
    address: u64,
    file: u64,
    line: i64,
    column: u64,
    is_stmt: bool,
    end_sequence: bool,
    version: u16,
) {
    if address == 0 {
        return;
    }
    // DWARF 4 file indices are 1-based; DWARF 5 made them 0-based.
    let idx = if version >= 5 { file as usize } else { (file as usize).saturating_sub(1) };
    let name = out
        .files
        .get(idx)
        .cloned()
        .or_else(|| out.files.first().cloned())
        .unwrap_or_default();
    out.rows.push(LineRow {
        addr: Addr(address),
        file: name,
        line: u32::try_from(line.max(0)).unwrap_or(0),
        column: column as u32,
        is_stmt,
        end_sequence,
    });
}

fn join_dir(dir: Option<&String>, name: &str) -> String {
    match dir {
        Some(d) if !d.is_empty() && !name.starts_with('/') => format!("{d}/{name}"),
        _ => name.to_string(),
    }
}


fn read_entry_form(data: &[u8], pos: &mut usize, form: u64, line_str: &[u8]) -> Option<Value> {
    Some(match form {
        0x08 => Value::Str(cstr(line_str, read_u32(data, pos)? as usize)), // string
        0x1f => Value::Str(cstr(line_str, read_u32(data, pos)? as usize)), // line_strp
        0x0b => Value::Uint(u64::from(*data.get(*pos)?)),                  // data1
        0x05 => Value::Uint(u64::from(read_u16(data, pos)?)),              // data2
        0x06 => Value::Uint(u64::from(read_u32(data, pos)?)),              // data4
        0x07 => Value::Uint(read_u64(data, pos)?),                         // data8
        0x0f => Value::Uint(uleb(data, pos)?),                             // udata
        0x1a => Value::Uint(uleb(data, pos)?),                             // strx
        0x19 => Value::Flag(true),                                         // flag_present
        0x0c => {
            *pos += 1;
            Value::Flag(true)
        }
        0x1e => {
            // DW_LNCT_MD5: the checksum of the file, sixteen bytes of it.
            // It is not a name, but its bytes are part of the table and a
            // reader that stopped here lost the whole line program.
            let b = data.get(*pos..*pos + 16)?.to_vec();
            *pos += 16;
            Value::Block(b)
        }
        0x0d => Value::Sint(sleb(data, pos)?), // sdata
        _ => return None,
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A DWARF 2 line program with one file, one row at 0x1000 line 10.
    fn program() -> Vec<u8> {
        let mut body = vec![1u8]; // min_inst_length
        body.push(1u8); // default_is_stmt
        body.push(1i8 as u8); // line_base = 1
        body.push(1u8); // line_range
        body.push(10u8); // opcode_base (9 standard + 1)
        body.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1]); // std lengths
        body.push(0); // empty include dirs
        body.extend_from_slice(b"a.c\0");
        body.push(0); // dir index
        body.push(0);
        body.push(0);
        body.push(0); // end of files

        let mut ops = Vec::new();
        // DW_LNE_set_address 0x1000
        ops.extend_from_slice(&[0, 9, 2]);
        ops.extend_from_slice(&0x1000u64.to_le_bytes()[..8]);
        // DW_LNS_advance_line +9
        ops.extend_from_slice(&[3, 9]);
        // DW_LNS_copy
        ops.push(1);
        // DW_LNS_advance_pc +4, so the sequence covers [0x1000, 0x1004)
        ops.extend_from_slice(&[2, 4]);
        // DW_LNE_end_sequence
        ops.extend_from_slice(&[0, 1, 1]);

        let mut out = Vec::new();
        let total = 2 + 4 + body.len() + ops.len();
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out.extend_from_slice(&ops);
        out
    }

    #[test]
    fn decodes_a_minimal_program() {
        let data = program();
        let t = parse(&data, 0, 2, 8, &[], None);
        assert_eq!(t.files, vec!["a.c".to_string()]);
        assert!(!t.rows.is_empty());
        let row = t.row_for(Addr(0x1000)).expect("row");
        assert_eq!(row.line, 10);
        assert_eq!(row.file, "a.c");
        assert!(t.rows.last().unwrap().end_sequence);
    }

    #[test]
    fn lookup_outside_the_range_returns_none() {
        let data = program();
        let t = parse(&data, 0, 2, 8, &[], None);
        assert!(t.row_for(Addr(0xfff)).is_none());
    }

    #[test]
    fn lookup_past_the_sequence_end_returns_none() {
        let data = program();
        let t = parse(&data, 0, 2, 8, &[], None);
        assert_eq!(t.row_for(Addr(0x1003)).expect("inside").line, 10);
        // 0x1004 is the end_sequence address: one past the last byte.
        assert!(t.row_for(Addr(0x1004)).is_none());
        assert!(t.row_for(Addr(0x2000)).is_none());
    }

}
