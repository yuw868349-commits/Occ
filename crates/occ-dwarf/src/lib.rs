//! DWARF debugging information.
//!
//! Two questions justify this crate: "which source line is this instruction
//! from" and "what is this variable called, and what type does it have". Both
//! are answered from `.debug_line` and `.debug_info`, and both change an
//! analysis from guessing into reading.
//!
//! Scope is DWARF 2 through 5, no split DWARF, no compressed (`.zdebug_*`)
//! sections. Within that scope the unit header, abbreviation table, all the
//! forms compilers actually emit (including `DW_FORM_addrx`, `strx` and
//! `implicit_const`), the line-number program and both location-list encodings
//! are implemented. A variable whose location cannot be determined is left
//! without one rather than guessed: a wrong variable location is worse than a
//! missing one, because it looks like a finding.


#![deny(unsafe_code)]
#![deny(missing_docs)]

mod line;

pub mod locs;

pub use line::{LineRow, LineTable};
pub use locs::LocRange;

use std::collections::HashMap;

use occ_analyze::view::View;
use occ_core::Addr;

/// A parsed attribute value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Unsigned integer.
    Uint(u64),
    /// Signed integer.
    Sint(i64),
    /// String, already resolved through its string table.
    Str(String),
    /// Address, already relocated against the image base.
    Addr(u64),
    /// Reference to another DIE, as an absolute section offset.
    Ref(u64),
    /// Boolean.
    Flag(bool),
    /// Raw block, used for location and frame-base expressions.
    Block(Vec<u8>),
    /// Value that could not be decoded; recorded so the caller knows why.
    Unresolved(&'static str),
}

impl Value {
    /// Unsigned view of the value, when it has one.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint(v) | Value::Addr(v) | Value::Ref(v) => Some(*v),
            Value::Sint(v) => u64::try_from(*v).ok(),
            Value::Flag(b) => Some(u64::from(*b)),
            _ => None,
        }
    }

    /// String view of the value, when it has one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One variable or parameter.
#[derive(Clone, Debug)]
pub struct Variable {
    /// Declared name.
    pub name: String,
    /// Resolved type name, following typedefs and qualifiers.
    pub type_name: Option<String>,
    /// Location expression, or the first entry of a location list.
    pub location: Option<String>,
    /// Address ranges mapping to expressions, when the compiler used a list.
    pub ranges: Vec<locs::LocRange>,
    /// Declaration line in the source file.
    pub decl_line: Option<u32>,
    /// True for a formal parameter rather than a local.
    pub is_param: bool,
}

impl Variable {
    /// Expression holding this variable at `addr`.
    ///
    /// A single-location variable returns its expression for any address; a
    /// list returns the entry that covers `addr`, so a caller inspecting a
    /// function at one instruction gets the right register.
    pub fn location_at(&self, addr: Addr) -> Option<&str> {
        if self.ranges.is_empty() {
            return self.location.as_deref();
        }
        self.ranges
            .iter()
            .find(|r| r.covers(addr))
            .map(|r| r.expr.as_str())
    }
}

/// One inlined call site inside a function.
///
/// An optimised build folds `sum()` into its caller; the address still belongs
/// to the caller, but the source line belongs to `sum`. These frames are how
/// that is expressed.
#[derive(Clone, Debug)]
pub struct InlineFrame {
    /// Name of the inlined function.
    pub name: String,
    /// First address of the inlined body.
    pub low_pc: Addr,
    /// One past the last address.
    pub high_pc: Addr,
    /// Source file that contains the call.
    pub call_file: Option<String>,
    /// Line of the call site.
    pub call_line: Option<u32>,
}

/// One function recovered from debug info.
#[derive(Clone, Debug)]
pub struct DwarfFunction {
    /// Name as written in the source.
    pub name: String,
    /// Mangled or linkage name, when present.
    pub linkage_name: Option<String>,
    /// First address.
    pub low_pc: Addr,
    /// One past the last address. Zero when the range was expressed as a list
    /// this crate does not evaluate.
    pub high_pc: Addr,
    /// Source file of the declaration.
    pub decl_file: Option<String>,
    /// Declaration line.
    pub decl_line: Option<u32>,
    /// Declared return type, when the subprogram states one.
    pub return_type: Option<String>,
    /// Frame base as DWARF expresses it.
    ///
    /// The usual value is `call_frame_cfa`, which is what lets a `fbreg`
    /// location be converted to a decompiler slot. `reg6` says the frame base
    /// is a register, and a consumer that assumes the CFA would put every
    /// local at the wrong address.
    pub frame_base: Option<String>,
    /// Parameters, in declaration order.
    pub params: Vec<Variable>,
    /// Local variables.
    pub locals: Vec<Variable>,
    /// Inlined calls whose bodies lie inside this function.
    pub inlined: Vec<InlineFrame>,
}

impl DwarfFunction {
    /// Size in bytes, or zero when unknown.
    pub fn size(&self) -> u64 {
        self.high_pc.0.saturating_sub(self.low_pc.0)
    }
}

/// One struct/class/union member.
#[derive(Clone, Debug)]
pub struct Member {
    /// Member name.
    pub name: String,
    /// Type name.
    pub type_name: Option<String>,
    /// Byte offset within the aggregate.
    pub offset: u64,
    /// Size in bytes, when known.
    pub size: Option<u64>,
}

/// One type.
#[derive(Clone, Debug)]
pub struct DwarfType {
    /// Name, possibly synthesised for anonymous types.
    pub name: String,
    /// Kind: base, pointer, struct, union, class, enum, array, typedef, function.
    pub kind: String,
    /// Size in bytes.
    pub byte_size: Option<u64>,
    /// Members for aggregates.
    pub members: Vec<Member>,
    /// Element type for arrays and pointee for pointers.
    pub target: Option<String>,
}

/// One compilation unit.
#[derive(Clone, Debug)]
pub struct Unit {
    /// Offset of the unit header in `.debug_info`.
    pub offset: u64,
    /// Producer string, which names the compiler.
    pub producer: Option<String>,
    /// Source language constant (`DW_LANG_*`).
    pub language: Option<u64>,
    /// Compilation directory.
    pub comp_dir: Option<String>,
    /// Name of the primary source file.
    pub name: Option<String>,
    /// Functions declared in this unit.
    pub functions: Vec<DwarfFunction>,
    /// Types declared in this unit.
    pub types: Vec<DwarfType>,
    /// Line table, when the unit has one.
    pub lines: LineTable,
}

/// Everything read from one image's debug sections.
#[derive(Clone, Debug, Default)]
pub struct Dwarf {
    /// Compilation units.
    pub units: Vec<Unit>,
    /// Notes about what could not be read.
    pub notes: Vec<String>,
}

impl Dwarf {
    /// Load debug info from an image, or return `None` when there is none.
    pub fn load(view: &View) -> Option<Dwarf> {
        let image = view.image();
        let info = section_bytes(view, ".debug_info")?;
        let abbrev = section_bytes(view, ".debug_abbrev")?;
        let str_sec = section_bytes(view, ".debug_str").unwrap_or(&[]);
        let line_str = section_bytes(view, ".debug_line_str").unwrap_or(&[]);
        let line_sec = section_bytes(view, ".debug_line").unwrap_or(&[]);
        let str_offsets = section_bytes(view, ".debug_str_offsets").unwrap_or(&[]);
        let addr_sec = section_bytes(view, ".debug_addr").unwrap_or(&[]);
        let loc_sec = section_bytes(view, ".debug_loclists")
            .or_else(|| section_bytes(view, ".debug_loc"))
            .unwrap_or(&[]);

        let mut notes = Vec::new();
        for s in &image.sections {
            if s.name.starts_with(".zdebug_") {
                notes.push(format!(
                    "{} is compressed; compressed debug sections are not read",
                    s.name
                ));
            }
        }

        let mut units = Vec::new();
        let mut pos = 0usize;
        while pos + 11 < info.len() {
            let unit_start = pos;
            let Some(len) = read_u32(info, &mut pos) else { break };
            let (unit_len, is64) = if len == 0xffff_ffff {
                let Some(l) = read_u64(info, &mut pos) else { break };
                (l as usize, true)
            } else {
                (len as usize, false)
            };
            let unit_end = pos.saturating_add(unit_len);
            if unit_end > info.len() || unit_len == 0 {
                notes.push(format!(
                    "unit at {unit_start:#x} claims {unit_len} bytes past the section end"
                ));
                break;
            }
            let Some(version) = read_u16(info, &mut pos) else { break };
            let (unit_type, addr_size, abbrev_off) = if version >= 5 {
                let Some(ut) = read_u8(info, &mut pos) else { break };
                let Some(as_) = read_u8(info, &mut pos) else { break };
                let Some(ab) = read_u32(info, &mut pos) else { break };
                // Skeleton and split units carry an extra 8-byte signature.
                if ut == 0x04 || ut == 0x05 || ut == 0x02 {
                    pos += 8;
                }
                (ut, as_, u64::from(ab))
            } else {
                let Some(ab) = read_u32(info, &mut pos) else { break };
                let Some(as_) = read_u8(info, &mut pos) else { break };
                (0x01u8, as_, u64::from(ab))
            };
            if addr_size != 4 && addr_size != 8 {
                notes.push(format!("unit at {unit_start:#x}: address size {addr_size}"));
            }
            if is64 {
                notes.push("64-bit DWARF unit lengths are parsed but rare".into());
            }
            let Some(abbrevs) = parse_abbrevs(abbrev, abbrev_off as usize) else {
                notes.push(format!("unit at {unit_start:#x}: abbrev table missing"));
                pos = unit_end;
                continue;
            };

            let context = Context {
                version,
                addr_size,
                is64,
                str_sec,
                line_str,
                str_offsets,
                addr_sec,
                loc_sec,
                // DWARF 5's per-unit header is eight bytes. The unit's own
                // `DW_AT_str_offsets_base`/`DW_AT_addr_base` override this as
                // soon as the root DIE has been read.
                str_offsets_base: std::cell::Cell::new(8),
                addr_base: std::cell::Cell::new(8),
            };
            let (mut dies, _consumed) = read_dies(&context, info, pos, unit_start, &abbrevs, addr_size);
            // The root DIE's own string references were read before its
            // base was known. When the unit names a base other than the
            // usual 8, read it again so the unit's name comes from the right
            // string too.
            let sb = context.str_offsets_base.get();
            let ab = context.addr_base.get();
            if sb != 8 || ab != 8 {
                let again = context.clone();
                again.str_offsets_base.set(sb);
                again.addr_base.set(ab);
                let (d, _) = read_dies(&again, info, pos, unit_start, &abbrevs, addr_size);
                dies = d;
            }

            let mut unit = Unit {
                offset: unit_start as u64,
                producer: None,
                language: None,
                comp_dir: None,
                name: None,
                functions: Vec::new(),
                types: Vec::new(),
                lines: LineTable::default(),
            };
            build_unit(&context, &dies, &mut unit, line_sec, view);
            if unit_type == 0x04 || unit_type == 0x05 {
                notes.push(format!(
                    "unit at {unit_start:#x}: skeleton/split unit, likely incomplete"
                ));
            }
            units.push(unit);
            pos = unit_end;
        }

        if units.is_empty() {
            notes.push("no compilation units could be parsed".into());
        }
        Some(Dwarf { units, notes })
    }

    /// Source location for an address: `(file, line, column)`.
    pub fn line_for(&self, addr: Addr) -> Option<(String, u32, u32)> {
        let mut best: Option<(&LineRow, &Unit)> = None;
        for u in &self.units {
            if let Some(row) = u.lines.row_for(addr) {
                match best {
                    Some((b, _)) if b.addr <= row.addr => {}
                    _ => best = Some((row, u)),
                }
            }
        }
        best.map(|(r, _)| (r.file.clone(), r.line, r.column))
    }

    /// Inline frames covering `addr`, outermost first.
    pub fn inline_at(&self, addr: Addr) -> Vec<&InlineFrame> {
        let mut frames: Vec<&InlineFrame> = self
            .units
            .iter()
            .flat_map(|u| u.functions.iter())
            .flat_map(|f| f.inlined.iter())
            .filter(|fr| fr.low_pc.0 <= addr.0 && addr.0 < fr.high_pc.0)
            .collect();
        // Outermost first: the widest range starts highest.
        frames.sort_by_key(|f| std::cmp::Reverse(f.high_pc.0 - f.low_pc.0));
        frames
    }

    /// Debug function containing `addr`.
    pub fn function_for(&self, addr: Addr) -> Option<&DwarfFunction> {
        self.units
            .iter()
            .flat_map(|u| u.functions.iter())
            .filter(|f| f.size() > 0 && addr.0 >= f.low_pc.0 && addr.0 < f.high_pc.0)
            .min_by_key(|f| f.size())
    }

    /// Debug function by exact name or linkage name.
    pub fn function_by_name(&self, name: &str) -> Option<&DwarfFunction> {
        self.units
            .iter()
            .flat_map(|u| u.functions.iter())
            .find(|f| f.name == name || f.linkage_name.as_deref() == Some(name))
    }

    /// Total number of functions with a usable address range.
    pub fn function_count(&self) -> usize {
        self.units
            .iter()
            .map(|u| u.functions.iter().filter(|f| f.size() > 0).count())
            .sum()
    }

    /// Every type, flattened.
    pub fn types(&self) -> impl Iterator<Item = &DwarfType> {
        self.units.iter().flat_map(|u| u.types.iter())
    }

    /// Source files named by any unit.
    pub fn source_files(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .units
            .iter()
            .flat_map(|u| u.lines.files.iter().cloned())
            .collect();
        v.sort();
        v.dedup();
        v
    }
}

fn section_bytes<'a>(view: &'a View<'_>, name: &str) -> Option<&'a [u8]> {
    // Debug sections are not allocated, so they are read by file offset.
    view.section_data_named(name)
}

#[derive(Clone)]
struct Context<'a> {
    version: u16,
    addr_size: u8,
    /// 64-bit DWARF: unit lengths and section offsets are 8 bytes wide.
    is64: bool,
    /// `.debug_loc` for DWARF 2-4 or `.debug_loclists` for DWARF 5.
    loc_sec: &'a [u8],
    str_sec: &'a [u8],
    line_str: &'a [u8],
    str_offsets: &'a [u8],
    addr_sec: &'a [u8],
    /// Where this unit's offsets start in `.debug_str_offsets`.
    ///
    /// DWARF 5 puts an 8-byte header in front of each unit's offset array, and
    /// a unit names its own start with `DW_AT_str_offsets_base`. Reading from
    /// zero read the header's length as an offset, which produced names that
    /// belonged to a different string: `sum.c` where `total` should be.
    str_offsets_base: std::cell::Cell<u64>,
    /// Where this unit's addresses start in `.debug_addr`.
    addr_base: std::cell::Cell<u64>,
}

/// A parsed DIE, kept flat with a depth field.
struct Die {
    offset: usize,
    tag: u64,
    depth: u32,
    attrs: Vec<(u64, Value)>,
}

impl Die {
    fn get(&self, at: u64) -> Option<&Value> {
        self.attrs.iter().find(|(a, _)| *a == at).map(|(_, v)| v)
    }
}

#[derive(Clone, Debug)]
struct Abbrev {
    tag: u64,
    children: bool,
    attrs: Vec<(u64, u64, i64)>,
}

fn parse_abbrevs(data: &[u8], offset: usize) -> Option<HashMap<u64, Abbrev>> {
    let mut out = HashMap::new();
    let mut pos = offset;
    loop {
        let code = uleb(data, &mut pos)?;
        if code == 0 {
            break;
        }
        let tag = uleb(data, &mut pos)?;
        let children = *data.get(pos)? != 0;
        pos += 1;
        let mut attrs = Vec::new();
        loop {
            let at = uleb(data, &mut pos)?;
            let form = uleb(data, &mut pos)?;
            if at == 0 && form == 0 {
                break;
            }
            let implicit = if form == 0x21 {
                sleb(data, &mut pos)?
            } else {
                0
            };
            attrs.push((at, form, implicit));
        }
        out.insert(code, Abbrev {
            tag,
            children,
            attrs,
        });
    }
    Some(out)
}

fn read_dies(
    ctx: &Context<'_>,
    info: &[u8],
    start: usize,
    unit_start: usize,
    abbrevs: &HashMap<u64, Abbrev>,
    addr_size: u8,
) -> (Vec<Die>, usize) {
    let mut dies = Vec::new();
    let mut pos = start;
    let mut depth = 0u32;
    // The first DIE is the unit itself; its end is marked by a zero code at
    // depth zero.
    while pos < info.len() {
        // References point at the abbreviation code byte, so the offset has to
        // be taken before the code is consumed.
        let die_offset = pos;
        let Some(code) = uleb(info, &mut pos) else { break };
        if code == 0 {
            if depth == 0 {
                break;
            }
            depth -= 1;
            continue;
        }
        let Some(ab) = abbrevs.get(&code) else { break };
        let mut attrs = Vec::with_capacity(ab.attrs.len());
        let mut bad = false;
        for (at, form, implicit) in &ab.attrs {
            match read_form(ctx, info, &mut pos, *form, *implicit, addr_size, unit_start) {
                Some(v) => attrs.push((*at, v)),
                None => {
                    bad = true;
                    break;
                }
            }
        }
        if bad {
            break;
        }
        // The root DIE names where this unit's string and address tables
        // start. Every later DIE in the unit uses those bases, so the values
        // have to be picked up before the next DIE is read.
        if depth == 0 {
            for (at, v) in &attrs {
                let Value::Uint(n) = v else { continue };
                match *at {
                    DW_AT_STR_OFFSETS_BASE => ctx.str_offsets_base.set(*n),
                    DW_AT_ADDR_BASE => ctx.addr_base.set(*n),
                    _ => {}
                }
            }
        }
        dies.push(Die {
            offset: die_offset,
            tag: ab.tag,
            depth,
            attrs,
        });
        if ab.children {
            depth += 1;
        }
    }
    (dies, pos)
}

#[allow(clippy::too_many_arguments)]
fn read_form(
    ctx: &Context<'_>,
    data: &[u8],
    pos: &mut usize,
    form: u64,
    implicit: i64,
    addr_size: u8,
    unit_start: usize,
) -> Option<Value> {
    Some(match form {
        0x01 => Value::Addr(read_addr(data, pos, addr_size)?), // addr
        0x03 => {
            let n = read_u16(data, pos)? as usize;
            let b = data.get(*pos..*pos + n)?.to_vec();
            *pos += n;
            Value::Block(b)
        }
        0x04 => {
            let n = read_u32(data, pos)? as usize;
            let b = data.get(*pos..*pos + n)?.to_vec();
            *pos += n;
            Value::Block(b)
        }
        0x05 => Value::Uint(u64::from(read_u16(data, pos)?)),
        0x06 => Value::Uint(u64::from(read_u32(data, pos)?)),
        0x07 => Value::Uint(read_u64(data, pos)?),
        0x08 => {
            // Inline NUL-terminated string, right here in .debug_info.
            let rest = data.get(*pos..)?;
            let end = rest.iter().position(|b| *b == 0)?;
            let s = String::from_utf8_lossy(&rest[..end]).into_owned();
            *pos += end + 1;
            Value::Str(s)
        }
        0x09 => {
            let n = uleb(data, pos)? as usize;
            let b = data.get(*pos..*pos + n)?.to_vec();
            *pos += n;
            Value::Block(b)
        }
        0x0a => {
            let n = usize::from(*data.get(*pos)?);
            *pos += 1;
            let b = data.get(*pos..*pos + n)?.to_vec();
            *pos += n;
            Value::Block(b)
        }
        0x0b => {
            let v = *data.get(*pos)?;
            *pos += 1;
            Value::Uint(u64::from(v))
        }
        0x0c => {
            let v = *data.get(*pos)?;
            *pos += 1;
            Value::Flag(v != 0)
        }
        0x0d => Value::Sint(sleb(data, pos)?),
        0x0e => Value::Str(cstr(ctx.str_sec, read_u32(data, pos)? as usize)),
        0x0f => Value::Uint(uleb(data, pos)?),
        0x10 => {
            // Section-relative, unlike every other reference form. DWARF 2
            // sized it like an address; later versions follow the unit length
            // format instead.
            let v = if ctx.version <= 2 {
                read_addr(data, pos, addr_size)?
            } else if ctx.is64 {
                read_u64(data, pos)?
            } else {
                u64::from(read_u32(data, pos)?)
            };
            Value::Ref(v)
        }
        // CU-relative references: rebase them onto the section so a single
        // offset map serves every unit in the file.
        0x11 => Value::Ref(cu_ref(unit_start, u64::from(*data.get(*pos)?))?),
        0x12 => Value::Ref(cu_ref(unit_start, u64::from(read_u16(data, pos)?))?),
        0x13 => Value::Ref(cu_ref(unit_start, u64::from(read_u32(data, pos)?))?),
        0x14 => Value::Ref(cu_ref(unit_start, read_u64(data, pos)?)?),
        0x15 => Value::Ref(cu_ref(unit_start, uleb(data, pos)?)?),
        0x16 => {
            // DW_FORM_indirect: the real form follows as a ULEB.
            let real = uleb(data, pos)?;
            return read_form(ctx, data, pos, real, implicit, addr_size, unit_start);
        }
        0x17 => {
            let v = if ctx.is64 {
                read_u64(data, pos)?
            } else {
                u64::from(read_u32(data, pos)?)
            };
            Value::Uint(v)
        }
        0x18 => {
            let n = uleb(data, pos)? as usize;
            let b = data.get(*pos..*pos + n)?.to_vec();
            *pos += n;
            Value::Block(b)
        }
        0x19 => Value::Flag(true),
        0x1a => {
            let idx = uleb(data, pos)?;
            Value::Str(read_strx(ctx, idx))
        }
        0x1b => Value::Uint(read_addr_index(ctx, data, pos, 1)?),
        0x1c => {
            *pos += 4;
            Value::Unresolved("ref_sup4")
        }
        0x1d => {
            *pos += 4;
            Value::Unresolved("strp_sup")
        }
        0x1e => {
            let b = data.get(*pos..*pos + 16)?.to_vec();
            *pos += 16;
            Value::Block(b)
        }
        0x1f => Value::Str(cstr(ctx.line_str, read_u32(data, pos)? as usize)),
        0x20 => {
            let b = data.get(*pos..*pos + 8)?.to_vec();
            *pos += 8;
            Value::Block(b)
        }
        0x21 => Value::Sint(implicit),
        0x22 => Value::Uint(uleb(data, pos)?),
        0x23 => Value::Uint(uleb(data, pos)?),
        0x24 => {
            *pos += 8;
            Value::Unresolved("ref_sup8")
        }
        0x25 => {
            let idx = u64::from(*data.get(*pos)?);
            *pos += 1;
            Value::Str(read_strx(ctx, idx))
        }
        0x26 => {
            let idx = u64::from(read_u16(data, pos)?);
            Value::Str(read_strx(ctx, idx))
        }
        0x27 => {
            let idx = read_uint_bytes(data, pos, 3)?;
            Value::Str(read_strx(ctx, idx))
        }
        0x28 => {
            let idx = u64::from(read_u32(data, pos)?);
            Value::Str(read_strx(ctx, idx))
        }
        0x29 => Value::Uint(read_addr_index(ctx, data, pos, 1)?),
        0x2a => Value::Uint(read_addr_index(ctx, data, pos, 2)?),
        0x2b => Value::Uint(read_addr_index(ctx, data, pos, 3)?),
        0x2c => Value::Uint(read_addr_index(ctx, data, pos, 4)?),
        _ => return None,
    })
}

/// Rebase a compilation-unit-relative reference onto the section.
fn cu_ref(unit_start: usize, offset: u64) -> Option<u64> {
    u64::try_from(unit_start).ok()?.checked_add(offset)
}

fn read_uint_bytes(data: &[u8], pos: &mut usize, n: usize) -> Option<u64> {
    let b = data.get(*pos..*pos + n)?;
    let mut v = 0u64;
    for (i, byte) in b.iter().enumerate() {
        v |= u64::from(*byte) << (8 * i);
    }
    *pos += n;
    Some(v)
}

fn read_strx(ctx: &Context<'_>, idx: u64) -> String {
    let width = if ctx.is64 { 8 } else { 4 };
    let at = (ctx.str_offsets_base.get() as usize).saturating_add(
        (idx as usize).saturating_mul(width),
    );
    let Some(b) = ctx.str_offsets.get(at..at + width) else {
        return String::new();
    };
    let mut off = 0u64;
    for (i, byte) in b.iter().enumerate() {
        off |= u64::from(*byte) << (8 * i);
    }
    cstr(ctx.str_sec, off as usize)
}

fn read_addr_index(ctx: &Context<'_>, data: &[u8], pos: &mut usize, bytes: usize) -> Option<u64> {
    let mut v = 0u64;
    for i in 0..bytes {
        v |= u64::from(*data.get(*pos + i)?) << (8 * i);
    }
    *pos += bytes;
    // Prefer .debug_addr when the compiler put the real address there.
    let at = (ctx.addr_base.get() as usize)
        .saturating_add((v as usize).saturating_mul(usize::from(ctx.addr_size)));
    if let Some(b) = ctx.addr_sec.get(at..at + usize::from(ctx.addr_size)) {
        let mut addr = 0u64;
        for (i, byte) in b.iter().enumerate() {
            addr |= u64::from(*byte) << (8 * i);
        }
        if addr != 0 {
            return Some(addr);
        }
    }
    Some(v)
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

fn cstr(data: &[u8], off: usize) -> String {
    if off >= data.len() {
        return String::new();
    }
    let rest = &data[off..];
    let end = rest.iter().position(|b| *b == 0).unwrap_or(rest.len());
    String::from_utf8_lossy(&rest[..end]).into_owned()
}

fn uleb(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *data.get(*pos)?;
        *pos += 1;
        result |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

fn sleb(data: &[u8], pos: &mut usize) -> Option<i64> {
    let mut result = 0i64;
    let mut shift = 0u32;
    loop {
        let b = *data.get(*pos)?;
        *pos += 1;
        result |= i64::from(b & 0x7f) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            if shift < 64 && b & 0x40 != 0 {
                result |= -1i64 << shift;
            }
            return Some(result);
        }
        if shift > 63 {
            return None;
        }
    }
}

fn read_u8(data: &[u8], pos: &mut usize) -> Option<u8> {
    let v = *data.get(*pos)?;
    *pos += 1;
    Some(v)
}

fn read_u16(data: &[u8], pos: &mut usize) -> Option<u16> {
    let b = data.get(*pos..*pos + 2)?;
    *pos += 2;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn read_u32(data: &[u8], pos: &mut usize) -> Option<u32> {
    let b = data.get(*pos..*pos + 4)?;
    *pos += 4;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u64(data: &[u8], pos: &mut usize) -> Option<u64> {
    let b = data.get(*pos..*pos + 8)?;
    *pos += 8;
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    Some(u64::from_le_bytes(a))
}

// ---------------------------------------------------------------------------
// semantic model
// ---------------------------------------------------------------------------

const DW_TAG_COMPILE_UNIT: u64 = 0x11;
const DW_TAG_SUBPROGRAM: u64 = 0x2e;
const DW_TAG_INLINE_SUBROUTINE: u64 = 0x1d;
const DW_TAG_VARIABLE: u64 = 0x34;
const DW_TAG_PARAM: u64 = 0x05;
const DW_TAG_BASE_TYPE: u64 = 0x24;
const DW_TAG_POINTER: u64 = 0x0f;
const DW_TAG_CONST: u64 = 0x26;
const DW_TAG_TYPEDEF: u64 = 0x16;
const DW_TAG_STRUCT: u64 = 0x13;
const DW_TAG_CLASS: u64 = 0x02;
const DW_TAG_UNION: u64 = 0x17;
const DW_TAG_ARRAY: u64 = 0x01;
const DW_TAG_ENUM: u64 = 0x04;
const DW_TAG_SUBROUTINE_TYPE: u64 = 0x15;
const DW_TAG_MEMBER: u64 = 0x0d;

const DW_AT_NAME: u64 = 0x03;
const DW_AT_LOCATION: u64 = 0x02;
const DW_AT_FRAME_BASE: u64 = 0x40;
const DW_AT_STR_OFFSETS_BASE: u64 = 0x72;
const DW_AT_ADDR_BASE: u64 = 0x73;
const DW_AT_BYTE_SIZE: u64 = 0x0b;
const DW_AT_LOW_PC: u64 = 0x11;
const DW_AT_HIGH_PC: u64 = 0x12;
const DW_AT_LANGUAGE: u64 = 0x13;
const DW_AT_COMP_DIR: u64 = 0x1b;
const DW_AT_DECL_FILE: u64 = 0x3a;
const DW_AT_DECL_LINE: u64 = 0x3b;
const DW_AT_TYPE: u64 = 0x49;
const DW_AT_PRODUCER: u64 = 0x25;
const DW_AT_LINKAGE_NAME: u64 = 0x6e;
const DW_AT_SPECIFICATION: u64 = 0x47;
const DW_AT_ABSTRACT_ORIGIN: u64 = 0x31;
const DW_AT_STMT_LIST: u64 = 0x10;
const DW_AT_DATA_MEMBER_LOCATION: u64 = 0x38;
const DW_AT_UPPER_BOUND: u64 = 0x2f;
const DW_AT_COUNT: u64 = 0x37;
const DW_AT_CALL_FILE: u64 = 0x58;
const DW_AT_CALL_LINE: u64 = 0x59;

fn build_unit(
    ctx: &Context<'_>,
    dies: &[Die],
    unit: &mut Unit,
    line_sec: &[u8],
    view: &View<'_>,
) {
    if let Some(root) = dies.first() {
        if root.tag == DW_TAG_COMPILE_UNIT {
            unit.producer = root.get(DW_AT_PRODUCER).and_then(|v| v.as_str()).map(str::to_string);
            unit.language = root.get(DW_AT_LANGUAGE).and_then(|v| v.as_u64());
            unit.comp_dir = root.get(DW_AT_COMP_DIR).and_then(|v| v.as_str()).map(str::to_string);
            unit.name = root.get(DW_AT_NAME).and_then(|v| v.as_str()).map(str::to_string);
            if let Some(Value::Uint(off)) = root.get(DW_AT_STMT_LIST) {
                unit.lines = line::parse(
                    line_sec,
                    *off as usize,
                    ctx.version,
                    ctx.addr_size,
                    ctx.line_str,
                    unit.comp_dir.as_deref(),
                );
            }
        }
    }

    // Offset-based location lists are relative to the unit's base address.
    let unit_base = dies
        .first()
        .and_then(|d| d.get(DW_AT_LOW_PC))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // Type names by DIE index, used to label variables.
    let mut type_name_by_idx: HashMap<usize, String> = HashMap::new();
    for (i, d) in dies.iter().enumerate() {
        if is_type_tag(d.tag)
            && let Some(v) = d.get(DW_AT_NAME).and_then(|v| v.as_str())
        {
            type_name_by_idx.insert(i, v.to_string());
        }
    }
    // Fall back to formatting for base types with no name.
    for (i, d) in dies.iter().enumerate() {
        if d.tag == DW_TAG_BASE_TYPE && !type_name_by_idx.contains_key(&i) {
            if let Some(size) = d.get(DW_AT_BYTE_SIZE).and_then(|v| v.as_u64()) {
                type_name_by_idx.insert(i, format!("int{size}"));
                let _ = size;
            }
        }
    }

    // DIE offsets are how DW_AT_type refers to a type; index them so a
    // reference resolves in one lookup instead of a rescan.
    let mut by_offset: HashMap<usize, usize> = HashMap::with_capacity(dies.len());
    for (i, d) in dies.iter().enumerate() {
        by_offset.insert(d.offset, i);
    }

    // The flattened DIE list lost parent links; walk it again with depths to
    // rebuild them. `parent_of[i]` is the enclosing DIE index.
    let mut stack: Vec<usize> = Vec::new();
    let mut parent_of: Vec<Option<usize>> = vec![None; dies.len()];
    for (i, d) in dies.iter().enumerate() {
        while stack.last().is_some_and(|p| dies[*p].depth >= d.depth) {
            stack.pop();
        }
        parent_of[i] = stack.last().copied();
        stack.push(i);
    }

    // A local can sit inside a lexical block, which is a child of the
    // subprogram rather than the subprogram itself. Variables belong to the
    // nearest enclosing subprogram, whatever blocks lie between them, or
    // `for (int i = ...)` loses its induction variable.
    let mut owner_of: Vec<Option<usize>> = vec![None; dies.len()];
    for i in 0..dies.len() {
        let mut cur = Some(i);
        while let Some(c) = cur {
            if dies[c].tag == DW_TAG_SUBPROGRAM {
                owner_of[i] = Some(c);
                break;
            }
            cur = parent_of[c];
        }
    }

    for (i, d) in dies.iter().enumerate() {
        match d.tag {
            DW_TAG_SUBPROGRAM => {
                let Some(low) = d.get(DW_AT_LOW_PC).and_then(|v| v.as_u64()) else {
                    continue;
                };
                let high = match d.get(DW_AT_HIGH_PC) {
                    Some(Value::Addr(a)) => *a,
                    Some(v) => v.as_u64().map(|n| low + n).unwrap_or(low),
                    None => low,
                };
                let mut f = DwarfFunction {
                    name: attr_chain(dies, &by_offset, i, DW_AT_NAME)
                        .and_then(|v| v.as_str().map(str::to_string))
                        .filter(|s| !s.is_empty())
                        .or_else(|| {
                            attr_chain(dies, &by_offset, i, DW_AT_LINKAGE_NAME)
                                .and_then(|v| v.as_str().map(str::to_string))
                        })
                        .unwrap_or_else(|| "?".into()),
                    linkage_name: d
                        .get(DW_AT_LINKAGE_NAME)
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    low_pc: Addr(low),
                    high_pc: Addr(high),
                    decl_file: decl_file(d, unit, ctx.version),
                    decl_line: d.get(DW_AT_DECL_LINE).and_then(|v| v.as_u64()).map(|n| n as u32),
                    return_type: d
                        .get(DW_AT_TYPE)
                        .and_then(|v| v.as_u64())
                        .and_then(|r| type_name_at(dies, &by_offset, r, 0)),
                    // The frame base is usually a block holding a single
                    // `DW_OP_call_frame_cfa`; a location list here would need
                    // the address to pick an entry, and no consumer of this
                    // field has one yet.
                    frame_base: match attr_chain(dies, &by_offset, i, DW_AT_FRAME_BASE) {
                        Some(Value::Block(b)) => Some(describe_expr(&b)),
                        _ => None,
                    },
                    params: Vec::new(),
                    locals: Vec::new(),
                    inlined: Vec::new(),
                };
                // Children of this subprogram.
                let mut j = i + 1;
                while j < dies.len() && dies[j].depth > d.depth {
                    if owner_of[j] == Some(i) {
                        match dies[j].tag {
                            DW_TAG_PARAM | DW_TAG_VARIABLE => {
                                let name = attr_chain(dies, &by_offset, j, DW_AT_NAME)
                                    .and_then(|v| v.as_str().map(str::to_string))
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or_else(|| "?".to_string());
                                let type_name = attr_chain(dies, &by_offset, j, DW_AT_TYPE)
                                    .and_then(|v| v.as_u64())
                                    .and_then(|r| type_name_at(dies, &by_offset, r, 0));
                                let (location, ranges) =
                                    match attr_chain(dies, &by_offset, j, DW_AT_LOCATION) {
                                        Some(Value::Block(b)) => {
                                            (Some(describe_expr(&b)), Vec::new())
                                        }
                                        Some(v) => {
                                            let ranges = locate(ctx, v.as_u64(), unit_base);
                                            let first =
                                                ranges.first().map(|r| r.expr.clone());
                                            (first, ranges)
                                        }
                                        None => (None, Vec::new()),
                                    };
                                let var = Variable {
                                    name,
                                    type_name,
                                    location,
                                    ranges,
                                    decl_line: attr_chain(dies, &by_offset, j, DW_AT_DECL_LINE)
                                        .and_then(|v| v.as_u64())
                                        .map(|n| n as u32),
                                    is_param: dies[j].tag == DW_TAG_PARAM,
                                };
                                if var.is_param {
                                    f.params.push(var);
                                } else {
                                    f.locals.push(var);
                                }
                            }
                            DW_TAG_INLINE_SUBROUTINE => {
                                if let Some(frame) = inline_frame(dies, &by_offset, j, unit, ctx)
                                {
                                    f.inlined.push(frame);
                                }
                            }
                            _ => {}
                        }
                    }
                    j += 1;
                }
                unit.functions.push(f);
            }
            DW_TAG_BASE_TYPE | DW_TAG_TYPEDEF | DW_TAG_STRUCT | DW_TAG_CLASS | DW_TAG_UNION
            | DW_TAG_ARRAY | DW_TAG_ENUM | DW_TAG_POINTER | DW_TAG_CONST => {
                let kind = match d.tag {
                    DW_TAG_BASE_TYPE => "base",
                    DW_TAG_TYPEDEF => "typedef",
                    DW_TAG_STRUCT => "struct",
                    DW_TAG_CLASS => "class",
                    DW_TAG_UNION => "union",
                    DW_TAG_ARRAY => "array",
                    DW_TAG_ENUM => "enum",
                    DW_TAG_POINTER => "pointer",
                    _ => "const",
                };
                let target = d
                    .get(DW_AT_TYPE)
                    .and_then(|v| v.as_u64())
                    .and_then(|r| type_name_at(dies, &by_offset, r, 0));
                let mut members = Vec::new();
                let mut j = i + 1;
                while j < dies.len() && dies[j].depth > d.depth {
                    if parent_of[j] == Some(i) && dies[j].tag == DW_TAG_MEMBER {
                        members.push(Member {
                            name: dies[j]
                                .get(DW_AT_NAME)
                                .and_then(|v| v.as_str())
                                .unwrap_or("?")
                                .to_string(),
                            type_name: dies[j]
                                .get(DW_AT_TYPE)
                                .and_then(|v| v.as_u64())
                                .and_then(|r| type_name_at(dies, &by_offset, r, 0)),
                            offset: dies[j]
                                .get(DW_AT_DATA_MEMBER_LOCATION)
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            size: dies[j].get(DW_AT_BYTE_SIZE).and_then(|v| v.as_u64()),
                        });
                    }
                    j += 1;
                }
                unit.types.push(DwarfType {
                    name: d
                        .get(DW_AT_NAME)
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("<anon {kind} @{i}>")),
                    kind: kind.to_string(),
                    byte_size: d.get(DW_AT_BYTE_SIZE).and_then(|v| v.as_u64()),
                    members,
                    target,
                });
            }
            _ => {}
        }
    }
    let _ = view;
}

fn is_type_tag(tag: u64) -> bool {
    matches!(
        tag,
        DW_TAG_BASE_TYPE
            | DW_TAG_TYPEDEF
            | DW_TAG_STRUCT
            | DW_TAG_CLASS
            | DW_TAG_UNION
            | DW_TAG_ARRAY
            | DW_TAG_ENUM
            | DW_TAG_POINTER
            | DW_TAG_CONST
            | DW_TAG_SUBROUTINE_TYPE
    )
}

/// Resolve a `DW_AT_type` reference to a name, following one level of
/// typedef/const indirection.
fn type_name_at(
    dies: &[Die],
    by_offset: &HashMap<usize, usize>,
    offset: u64,
    depth: u32,
) -> Option<String> {
    if depth > 4 {
        return None;
    }
    let i = *by_offset.get(&(offset as usize))?;
    let d = &dies[i];
    if let Some(name) = d.get(DW_AT_NAME).and_then(|v| v.as_str())
        && !name.is_empty()
    {
        // A pointer reads better as `T *` than as the bare target name.
        if d.tag == DW_TAG_POINTER {
            return Some(ptr_to(name));
        }
        return Some(name.to_string());
    }
    match d.tag {
        DW_TAG_POINTER => d
            .get(DW_AT_TYPE)
            .and_then(|v| v.as_u64())
            .and_then(|r| type_name_at(dies, by_offset, r, depth + 1))
            .map(|t| ptr_to(&t)),
        DW_TAG_CONST => d
            .get(DW_AT_TYPE)
            .and_then(|v| v.as_u64())
            .and_then(|r| type_name_at(dies, by_offset, r, depth + 1))
            .map(|t| format!("const {t}")),
        DW_TAG_TYPEDEF => d
            .get(DW_AT_TYPE)
            .and_then(|v| v.as_u64())
            .and_then(|r| type_name_at(dies, by_offset, r, depth + 1)),
        DW_TAG_ARRAY => {
            let elem = d
                .get(DW_AT_TYPE)
                .and_then(|v| v.as_u64())
                .and_then(|r| type_name_at(dies, by_offset, r, depth + 1))?;
            // Element count is either explicit or derivable from the byte size.
            let count = d
                .get(DW_AT_COUNT)
                .and_then(|v| v.as_u64())
                // DWARF 2-4 put the last index in the subrange child; the
                // common case of one subrange is handled here as a fallback.
                .or_else(|| {
                    d.get(DW_AT_UPPER_BOUND)
                        .and_then(|v| v.as_u64())
                        .map(|n| n + 1)
                })
                .or_else(|| {
                    let total = d.get(DW_AT_BYTE_SIZE).and_then(|v| v.as_u64())?;
                    let elem = d.get(DW_AT_TYPE).and_then(|v| v.as_u64())?;
                    let esize = byte_size_at(dies, by_offset, elem, 0)?;
                    (esize > 0).then(|| total / esize)
                });
            Some(match count {
                Some(n) => format!("{elem}[{n}]"),
                None => format!("{elem}[]"),
            })
        }
        DW_TAG_BASE_TYPE => d
            .get(DW_AT_BYTE_SIZE)
            .and_then(|v| v.as_u64())
            .map(|n| format!("int{n}")),
        _ => None,
    }
}

/// Append a pointer star without doubling the space of `char * *`.
fn ptr_to(t: &str) -> String {
    if t.ends_with('*') {
        format!("{t}*")
    } else {
        format!("{t} *")
    }
}

/// Size in bytes of the type at `offset`, following references.
fn byte_size_at(
    dies: &[Die],
    by_offset: &HashMap<usize, usize>,
    offset: u64,
    depth: u32,
) -> Option<u64> {
    if depth > 6 {
        return None;
    }
    let i = *by_offset.get(&(offset as usize))?;
    let d = &dies[i];
    if let Some(n) = d.get(DW_AT_BYTE_SIZE).and_then(|v| v.as_u64()) {
        return Some(n);
    }
    let next = d.get(DW_AT_TYPE).and_then(|v| v.as_u64())?;
    byte_size_at(dies, by_offset, next, depth + 1)
}

/// Resolve `DW_AT_decl_file`, honouring the DWARF 5 change from 1-based to
/// 0-based file indices.
fn decl_file(d: &Die, unit: &Unit, version: u16) -> Option<String> {
    let idx = d.get(DW_AT_DECL_FILE).and_then(|v| v.as_u64())? as usize;
    let idx = if version >= 5 { idx } else { idx.checked_sub(1)? };
    unit.lines.files.get(idx).cloned()
}

/// Build one inline frame from a `DW_TAG_inlined_subroutine` DIE.
fn inline_frame(
    dies: &[Die],
    by_offset: &HashMap<usize, usize>,
    idx: usize,
    unit: &Unit,
    ctx: &Context<'_>,
) -> Option<InlineFrame> {
    let d = dies.get(idx)?;
    let low = d.get(DW_AT_LOW_PC).and_then(|v| v.as_u64())?;
    let high = match d.get(DW_AT_HIGH_PC) {
        Some(Value::Addr(a)) => *a,
        Some(v) => v.as_u64().map(|n| low + n).unwrap_or(low),
        None => low,
    };
    let name = attr_chain(dies, by_offset, idx, DW_AT_NAME)
        .and_then(|v| v.as_str().map(str::to_string))
        .or_else(|| {
            attr_chain(dies, by_offset, idx, DW_AT_LINKAGE_NAME)
                .and_then(|v| v.as_str().map(str::to_string))
        })
        .unwrap_or_else(|| "?".into());
    let call_file = d
        .get(DW_AT_CALL_FILE)
        .and_then(|v| v.as_u64())
        .and_then(|n| {
            let n = n as usize;
            let n = if ctx.version >= 5 { n } else { n.checked_sub(1)? };
            unit.lines.files.get(n).cloned()
        });
    Some(InlineFrame {
        name,
        low_pc: Addr(low),
        high_pc: Addr(high),
        call_file,
        call_line: d.get(DW_AT_CALL_LINE).and_then(|v| v.as_u64()).map(|n| n as u32),
    })
}

/// Read an attribute, following `DW_AT_abstract_origin` and
/// `DW_AT_specification` when the DIE itself does not carry it.
///
/// Optimised builds split a function into an abstract DIE that holds the name,
/// types and locations and a concrete DIE that holds only the addresses.
fn attr_chain(
    dies: &[Die],
    by_offset: &HashMap<usize, usize>,
    idx: usize,
    at: u64,
) -> Option<Value> {
    let mut cur = idx;
    for _ in 0..8 {
        let d = dies.get(cur)?;
        if let Some(v) = d.get(at) {
            return Some(v.clone());
        }
        let next = [DW_AT_ABSTRACT_ORIGIN, DW_AT_SPECIFICATION]
            .iter()
            .find_map(|a| match d.get(*a) {
                Some(Value::Ref(off)) => Some(*off),
                _ => None,
            })?;
        cur = *by_offset.get(&usize::try_from(next).ok()?)?;
    }
    None
}


/// Resolve a location-list reference into address ranges.
///
/// `DW_AT_location` is either an inline expression or an offset into the
/// location-list section; only the second case reaches here.
fn locate(ctx: &Context<'_>, offset: Option<u64>, cu_base: u64) -> Vec<LocRange> {
    let Some(off) = offset else {
        return Vec::new();
    };
    let Ok(off) = usize::try_from(off) else {
        return Vec::new();
    };
    if ctx.loc_sec.is_empty() || off >= ctx.loc_sec.len() {
        return Vec::new();
    }
    if ctx.version >= 5 {
        locs::parse_debug_loclists(ctx.loc_sec, off, ctx.addr_size, Some(ctx.addr_sec), cu_base)
    } else {
        locs::parse_debug_loc(ctx.loc_sec, off, ctx.addr_size, cu_base)
    }
}

fn describe_expr(bytes: &[u8]) -> String {
    let mut parts: Vec<String> = Vec::new();
    render_expr(bytes, &mut parts, 0);
    if parts.is_empty() {
        "?".into()
    } else {
        parts.join(" ")
    }
}

/// Stack operations with no operands, rendered as symbols where one exists.
const SIMPLE_OPS: &[(u8, &str)] = &[
    (0x06, "deref"),
    (0x12, "dup"),
    (0x13, "drop"),
    (0x14, "over"),
    (0x16, "swap"),
    (0x17, "rot"),
    (0x18, "xderef"),
    (0x19, "abs"),
    (0x1a, "and"),
    (0x1b, "div"),
    (0x1c, "minus"),
    (0x1d, "mod"),
    (0x1e, "mul"),
    (0x1f, "neg"),
    (0x20, "not"),
    (0x21, "or"),
    (0x22, "plus"),
    (0x24, "shl"),
    (0x25, "shr"),
    (0x26, "shra"),
    (0x27, "xor"),
    (0x29, "eq"),
    (0x2a, "ge"),
    (0x2b, "gt"),
    (0x2c, "le"),
    (0x2d, "lt"),
    (0x2e, "ne"),
    (0x96, "nop"),
    (0x97, "push_object_address"),
    (0x9b, "form_tls_address"),
    (0x9c, "call_frame_cfa"),
];

fn render_expr(bytes: &[u8], parts: &mut Vec<String>, depth: u32) {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let op = bytes[pos];
        pos += 1;
        // Literals: 0x30..0x4f push a small signed constant.
        if (0x30..=0x4f).contains(&op) {
            let v = i64::from(op) - 0x30;
            parts.push(format!("const {v}"));
            continue;
        }
        // reg0..reg31
        if (0x50..=0x6f).contains(&op) {
            parts.push(format!("reg{}", op - 0x50));
            continue;
        }
        // breg0..breg31, followed by a signed offset.
        if (0x70..=0x8f).contains(&op) {
            match sleb(bytes, &mut pos) {
                Some(off) if off < 0 => parts.push(format!("breg{}{off}", op - 0x70)),
                Some(off) => parts.push(format!("breg{}+{off}", op - 0x70)),
                None => parts.push(format!("breg{}?", op - 0x70)),
            }
            continue;
        }
        if let Some((_, name)) = SIMPLE_OPS.iter().find(|(code, _)| *code == op) {
            parts.push((*name).to_string());
            continue;
        }
        match op {
            0x03 => match read_u64(bytes, &mut pos) {
                Some(a) => parts.push(format!("addr {a:#x}")),
                None => break,
            },
            0x08 => match read_u8(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {n}")),
                None => break,
            },
            0x09 => match read_u8(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {}", n as i8)),
                None => break,
            },
            0x0a => match read_u16(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {n}")),
                None => break,
            },
            0x0b => match read_u16(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {}", n as i16)),
                None => break,
            },
            0x0c => match read_u32(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {n}")),
                None => break,
            },
            0x0d => match read_u32(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {}", n as i32)),
                None => break,
            },
            0x0e => match read_u64(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {n}")),
                None => break,
            },
            0x0f => match read_u64(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {}", n as i64)),
                None => break,
            },
            0x10 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {n}")),
                None => break,
            },
            0x11 => match sleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("const {n}")),
                None => break,
            },
            0x15 => match read_u8(bytes, &mut pos) {
                Some(n) => parts.push(format!("pick {n}")),
                None => break,
            },
            0x23 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("plus_uconst {n}")),
                None => break,
            },
            // Branching: relative offsets, kept as targets nobody will follow.
            0x28 | 0x2f => match read_u16(bytes, &mut pos) {
                Some(t) => parts.push(format!("{} {t:+}", if op == 0x28 { "bra" } else { "skip" })),
                None => break,
            },
            0x90 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("regx {n}")),
                None => break,
            },
            0x91 => match sleb(bytes, &mut pos) {
                Some(off) if off < 0 => parts.push(format!("fbreg{off}")),
                Some(off) => parts.push(format!("fbreg+{off}")),
                None => break,
            },
            0x92 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("bregx {n}")),
                None => break,
            },
            0x93 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("piece {n}")),
                None => break,
            },
            0x94 | 0x95 => match read_u8(bytes, &mut pos) {
                Some(n) => {
                    let name = if op == 0x94 { "deref_size" } else { "xderef_size" };
                    parts.push(format!("{name} {n}"));
                }
                None => break,
            },
            0x98 | 0x99 => {
                let width = if op == 0x98 { 2 } else { 4 };
                if read_uint_bytes(bytes, &mut pos, width).is_none() {
                    break;
                }
                parts.push("call".into());
            }
            0x9a => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("call_ref {n}")),
                None => break,
            },
            0x9d => {
                let (Some(size), Some(offset)) = (uleb(bytes, &mut pos), uleb(bytes, &mut pos))
                else {
                    break;
                };
                parts.push(format!("bit_piece {size} {offset}"));
            }
            0x9e => {
                let Some(len) = uleb(bytes, &mut pos) else { break };
                let Ok(len) = usize::try_from(len) else { break };
                let Some(block) = bytes.get(pos..pos + len) else { break };
                pos += len;
                parts.push(format!("implicit_value {}", hex_block(block)));
            }
            0x9f => parts.push("stack_value".into()),
            0xa0 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("implicit_pointer {n}")),
                None => break,
            },
            0xa1 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("addr_index {n}")),
                None => break,
            },
            0xa2 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("const_index {n}")),
                None => break,
            },
            0xa3 => {
                // A nested expression inside a ULEB-length block.
                let Some(len) = uleb(bytes, &mut pos) else { break };
                let Ok(len) = usize::try_from(len) else { break };
                let Some(block) = bytes.get(pos..pos + len) else { break };
                pos += len;
                let mut inner = Vec::new();
                if depth < 4 {
                    render_expr(block, &mut inner, depth + 1);
                }
                parts.push(format!("entry_value({})", inner.join(" ")));
            }
            0xa4 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("const_type {n}")),
                None => break,
            },
            0xa5 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("regval_type {n}")),
                None => break,
            },
            0xa6 | 0xa7 => match uleb(bytes, &mut pos) {
                Some(n) => {
                    let name = if op == 0xa6 { "deref_type" } else { "xderef_type" };
                    parts.push(format!("{name} {n}"));
                }
                None => break,
            },
            0xa8 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("convert {n}")),
                None => break,
            },
            0xa9 => match uleb(bytes, &mut pos) {
                Some(n) => parts.push(format!("reinterpret {n}")),
                None => break,
            },
            other => {
                parts.push(format!("op{other:#04x}"));
            }
        }
    }
}

/// Short hex rendering for embedded blocks.
fn hex_block(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2 + 2);
    s.push_str("0x");
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leb_readers_handle_two_byte_values() {
        let data = [0x80, 0x02];
        let mut pos = 0;
        assert_eq!(uleb(&data, &mut pos), Some(0x100));
        assert_eq!(pos, 2);
    }

    #[test]
    fn sleb_extends_sign() {
        let data = [0x7f];
        let mut pos = 0;
        assert_eq!(sleb(&data, &mut pos), Some(-1));
    }

    #[test]
    fn location_expressions_render() {
        // DW_OP_fbreg -16 (SLEB 0x70)
        assert_eq!(describe_expr(&[0x91, 0x70]), "fbreg-16");
        assert_eq!(describe_expr(&[0x9c]), "call_frame_cfa");
        assert_eq!(describe_expr(&[0x50]), "reg0");
        // DW_OP_breg5 +8 ; DW_OP_deref_size 4
        assert_eq!(describe_expr(&[0x75, 0x08, 0x94, 0x04]), "breg5+8 deref_size 4");
        // DW_OP_lit3 ; DW_OP_stack_value ; DW_OP_piece 4
        assert_eq!(describe_expr(&[0x33, 0x9f, 0x93, 0x04]), "const 3 stack_value piece 4");
        // DW_OP_addr 0x1000
        assert_eq!(
            describe_expr(&[0x03, 0x00, 0x10, 0, 0, 0, 0, 0, 0]),
            "addr 0x1000"
        );
        // DW_OP_entry_value with a nested DW_OP_reg5
        assert_eq!(describe_expr(&[0xa3, 0x01, 0x55]), "entry_value(reg5)");
    }

    #[test]
    fn cstr_is_bounded() {
        assert_eq!(cstr(b"abc\0def", 0), "abc");
        assert_eq!(cstr(b"abc", 0), "abc");
        assert_eq!(cstr(b"abc", 99), "");
    }
}
