//! WebAssembly bytecode.
//!
//! A WASM function body is a linear byte stream with structured control flow:
//! `block`/`loop`/`if` open a frame, `end` closes it, and a branch names a frame
//! by depth rather than an address. Decoding therefore happens in two passes —
//! one to read instructions and remember which frame each branch names, one to
//! turn the frame table into byte offsets. Offsets are the identity of
//! everything in a module (there is no address space), so a branch target here
//! is a file offset that a caller can hexdump.
//!
//! What is decoded: the MVP instruction set, sign extension, saturating
//! truncation, bulk memory, reference types, and the `block`/`loop`/`if`
//! structure with resolved branch targets. What is not: SIMD and threads. Both
//! are signalled at the exact offset they occur rather than guessed at, because
//! a wrong length would silently misalign every instruction after it.

use occ_core::{Addr, CallTarget, FlowKind, Instruction, MAX_INSN_LEN, WasmFuncType};

/// One decoded function body.
pub struct Body {
    /// Instructions in offset order.
    pub insns: Vec<Instruction>,
    /// Branch targets per instruction offset, including every `br_table` entry.
    pub targets: Vec<(Addr, Vec<Addr>)>,
    /// Where decoding stopped and why, when it did.
    pub truncated: Option<(Addr, String)>,
    /// The number of local variables the function declares.
    ///
    /// The declarations are not instructions, but they say how many slots
    /// the function has, and a local that is read before it is written is
    /// uninitialised only if there is a slot for it at all.
    pub locals: u32,
    /// The label each `br_table` names, one per case, with the default last.
    ///
    /// The order is the order of the cases, which is what makes a `br_table`
    /// a switch: the index the instruction pops selects one of them. The
    /// default comes last because the module writes it last.
    pub br_tables: Vec<(Addr, Vec<u32>)>,
}

/// How an instruction's immediates are encoded.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Imm {
    /// No immediates.
    None,
    /// `block`/`loop`/`if`: a block type.
    BlockType,
    /// A branch depth.
    Label,
    /// `br_table`: a vector of depths plus a default.
    BrTable,
    /// A function index.
    Func,
    /// `call_indirect`: a type index and a table index.
    TypeTable,
    /// A local, global or table index.
    Var,
    /// A load or store: alignment and offset.
    MemArg,
    /// `memory.size`/`memory.grow`: a memory index.
    Mem,
    /// A 32-bit signed constant.
    I32,
    /// A 64-bit signed constant.
    I64,
    /// A 32-bit float.
    F32,
    /// A 64-bit float.
    F64,
    /// `ref.null`: a heap type.
    Heap,
    /// `select t*`: a vector of value types.
    SelectTypes,
}

/// Instructions with a fixed encoding outside the arithmetic block.
const SIMPLE: &[(u8, &str, Imm)] = &[
    (0x00, "unreachable", Imm::None),
    (0x01, "nop", Imm::None),
    (0x02, "block", Imm::BlockType),
    (0x03, "loop", Imm::BlockType),
    (0x04, "if", Imm::BlockType),
    (0x05, "else", Imm::None),
    (0x0b, "end", Imm::None),
    (0x0c, "br", Imm::Label),
    (0x0d, "br_if", Imm::Label),
    (0x0e, "br_table", Imm::BrTable),
    (0x0f, "return", Imm::None),
    (0x10, "call", Imm::Func),
    (0x11, "call_indirect", Imm::TypeTable),
    (0x12, "return_call", Imm::Func),
    (0x13, "return_call_indirect", Imm::TypeTable),
    (0x1a, "drop", Imm::None),
    (0x1b, "select", Imm::None),
    (0x1c, "select_t", Imm::SelectTypes),
    (0x20, "local.get", Imm::Var),
    (0x21, "local.set", Imm::Var),
    (0x22, "local.tee", Imm::Var),
    (0x23, "global.get", Imm::Var),
    (0x24, "global.set", Imm::Var),
    (0x25, "table.get", Imm::Var),
    (0x26, "table.set", Imm::Var),
    (0x28, "i32.load", Imm::MemArg),
    (0x29, "i64.load", Imm::MemArg),
    (0x2a, "f32.load", Imm::MemArg),
    (0x2b, "f64.load", Imm::MemArg),
    (0x2c, "i32.load8_s", Imm::MemArg),
    (0x2d, "i32.load8_u", Imm::MemArg),
    (0x2e, "i32.load16_s", Imm::MemArg),
    (0x2f, "i32.load16_u", Imm::MemArg),
    (0x30, "i64.load8_s", Imm::MemArg),
    (0x31, "i64.load8_u", Imm::MemArg),
    (0x32, "i64.load16_s", Imm::MemArg),
    (0x33, "i64.load16_u", Imm::MemArg),
    (0x34, "i64.load32_s", Imm::MemArg),
    (0x35, "i64.load32_u", Imm::MemArg),
    (0x36, "i32.store", Imm::MemArg),
    (0x37, "i64.store", Imm::MemArg),
    (0x38, "f32.store", Imm::MemArg),
    (0x39, "f64.store", Imm::MemArg),
    (0x3a, "i32.store8", Imm::MemArg),
    (0x3b, "i32.store16", Imm::MemArg),
    (0x3c, "i64.store8", Imm::MemArg),
    (0x3d, "i64.store16", Imm::MemArg),
    (0x3e, "i64.store32", Imm::MemArg),
    (0x3f, "memory.size", Imm::Mem),
    (0x40, "memory.grow", Imm::Mem),
    (0x41, "i32.const", Imm::I32),
    (0x42, "i64.const", Imm::I64),
    (0x43, "f32.const", Imm::F32),
    (0x44, "f64.const", Imm::F64),
    (0xd0, "ref.null", Imm::Heap),
    (0xd1, "ref.is_null", Imm::None),
    (0xd2, "ref.func", Imm::Func),
];

/// 0x45..=0xc4, the arithmetic block, in encoding order.
const ARITH: [&str; 0x80] = [
    "i32.eqz", "i32.eq", "i32.ne", "i32.lt_s", "i32.lt_u", "i32.gt_s", "i32.gt_u", "i32.le_s",
    "i32.le_u", "i32.ge_s", "i32.ge_u", "i64.eqz", "i64.eq", "i64.ne", "i64.lt_s", "i64.lt_u",
    "i64.gt_s", "i64.gt_u", "i64.le_s", "i64.le_u", "i64.ge_s", "i64.ge_u", "f32.eq", "f32.ne",
    "f32.lt", "f32.gt", "f32.le", "f32.ge", "f64.eq", "f64.ne", "f64.lt", "f64.gt", "f64.le",
    "f64.ge", "i32.clz", "i32.ctz", "i32.popcnt", "i32.add", "i32.sub", "i32.mul", "i32.div_s",
    "i32.div_u", "i32.rem_s", "i32.rem_u", "i32.and", "i32.or", "i32.xor", "i32.shl", "i32.shr_s",
    "i32.shr_u", "i32.rotl", "i32.rotr", "i64.clz", "i64.ctz", "i64.popcnt", "i64.add", "i64.sub",
    "i64.mul", "i64.div_s", "i64.div_u", "i64.rem_s", "i64.rem_u", "i64.and", "i64.or", "i64.xor",
    "i64.shl", "i64.shr_s", "i64.shr_u", "i64.rotl", "i64.rotr", "f32.abs", "f32.neg", "f32.ceil",
    "f32.floor", "f32.trunc", "f32.nearest", "f32.sqrt", "f32.add", "f32.sub", "f32.mul",
    "f32.div", "f32.min", "f32.max", "f32.copysign", "f64.abs", "f64.neg", "f64.ceil", "f64.floor",
    "f64.trunc", "f64.nearest", "f64.sqrt", "f64.add", "f64.sub", "f64.mul", "f64.div", "f64.min",
    "f64.max", "f64.copysign", "i32.wrap_i64", "i32.trunc_f32_s", "i32.trunc_f32_u",
    "i32.trunc_f64_s", "i32.trunc_f64_u", "i64.extend_i32_s", "i64.extend_i32_u",
    "i64.trunc_f32_s", "i64.trunc_f32_u", "i64.trunc_f64_s", "i64.trunc_f64_u",
    "f32.convert_i32_s", "f32.convert_i32_u", "f32.convert_i64_s", "f32.convert_i64_u",
    "f32.demote_f64", "f64.convert_i32_s", "f64.convert_i32_u", "f64.convert_i64_s",
    "f64.convert_i64_u", "f64.promote_f32", "i32.reinterpret_f32", "i64.reinterpret_f64",
    "f32.reinterpret_i32", "f64.reinterpret_i64", "i32.extend8_s", "i32.extend16_s",
    "i64.extend8_s", "i64.extend16_s", "i64.extend32_s",
];

/// 0xfc sub-opcodes up to 7 take no immediates.
const SAT: [&str; 8] = [
    "i32.trunc_sat_f32_s",
    "i32.trunc_sat_f32_u",
    "i32.trunc_sat_f64_s",
    "i32.trunc_sat_f64_u",
    "i64.trunc_sat_f32_s",
    "i64.trunc_sat_f32_u",
    "i64.trunc_sat_f64_s",
    "i64.trunc_sat_f64_u",
];

/// A control frame, used to resolve a branch depth to an offset.
struct Frame {
    /// `block`, `loop` or `if`.
    kind: u8,
    /// Instruction index of the opening instruction.
    open: usize,
    /// Instruction index of `else`, for an `if` that has one.
    else_at: Option<usize>,
    /// Instruction index of the closing `end`.
    end: Option<usize>,
}

/// One instruction before branch targets are known.
struct Raw {
    offset: u64,
    len: u32,
    text: String,
    mnemonic: &'static str,
    flow: FlowKind,
    /// Frame a branch names, as an index into the frame table.
    branch_frame: Option<usize>,
    /// Call target offset, when the index resolved.
    call_target: Option<u64>,
    /// `br_table` frames, including the default.
    table_frames: Vec<usize>,
    /// `br_table` label depths, one per case, with the default last.
    table_labels: Vec<u32>,
    /// Offset of the `else`/`end` an `if` jumps to when false.
    if_false_frame: Option<usize>,
}

/// Decode the body of one function.
///
/// `body` must be exactly the function body — the leading local declarations
/// included — and `start` is its file offset, which is what every instruction
/// address is measured from. `types` is the module's type section, which is
/// what says what a block with a type index takes and leaves.
pub fn decode(
    body: &[u8],
    start: u64,
    call_target: &dyn Fn(u32) -> Option<Addr>,
    types: &[WasmFuncType],
) -> Body {
    let mut raws: Vec<Raw> = Vec::new();
    let mut frames: Vec<Frame> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut truncated = None;
    let mut pos = 0usize;

    // Local declarations: a count, then (count, type) pairs. They are not
    // instructions, but they are part of the body's byte range.
    //
    // The count of them is the number of slots the function has. The
    // parameters come before the locals, so the number of slots is the
    // number of parameters plus the number of local groups, each of
    // which may declare more than one slot.
    let mut locals = 0u32;
    match leb_u32(body, &mut pos) {
        Some(groups) => {
            for _ in 0..groups.min(1_000_000) {
                let Some(count) = leb_u32(body, &mut pos) else {
                    break;
                };
                if body.get(pos).is_none() {
                    break;
                }
                pos += 1;
                locals = locals.saturating_add(count);
            }
        }
        None => {
            return Body {
                insns: Vec::new(),
                targets: Vec::new(),
                truncated: Some((Addr(start), "local declarations are malformed".into())),
                locals: 0,
                br_tables: Vec::new(),
            };
        }
    }

    while pos < body.len() {
        let offset = start + pos as u64;
        let begin = pos;
        let op = body[pos];
        pos += 1;
        let (name, imm) = match SIMPLE.binary_search_by_key(&op, |e| e.0) {
            Ok(i) => (SIMPLE[i].1, SIMPLE[i].2),
            Err(_) if (0x45..=0xc4).contains(&op) => (ARITH[usize::from(op - 0x45)], Imm::None),
            Err(_) => match op {
                0xfc => {
                    let Some(sub) = leb_u32(body, &mut pos) else { break };
                    match decode_fc(sub, body, &mut pos) {
                        Some(name) => (name, Imm::None),
                        None => {
                            truncated = Some((
                                Addr(offset),
                                format!("0xfc sub-opcode {sub} is not decoded"),
                            ));
                            break;
                        }
                    }
                }
                0xfd => {
                    truncated = Some((
                        Addr(offset),
                        "SIMD (0xfd) is not decoded; the body stops here".into(),
                    ));
                    break;
                }
                0xfe => {
                    truncated = Some((
                        Addr(offset),
                        "threads (0xfe) are not decoded; the body stops here".into(),
                    ));
                    break;
                }
                _ => {
                    truncated = Some((
                        Addr(offset),
                        format!("opcode 0x{op:02x} is not decoded; the body stops here"),
                    ));
                    break;
                }
            },
        };

        let mut text = name.to_string();
        let mut branch_frame = None;
        let mut table_frames = Vec::new();
        let mut table_labels: Vec<u32> = Vec::new();
        let mut call_target_at = None;
        let mut if_false_frame = None;

        match imm {
            Imm::None => {}
            Imm::BlockType => {
                let Some(bt) = block_type(body, &mut pos) else { break };
                let suffix = bt.describe(types);
                if !suffix.is_empty() {
                    text.push_str(&suffix);
                }
            }
            Imm::Label => {
                let Some(depth) = leb_u32(body, &mut pos) else { break };
                text.push(' ');
                match frame_at(&stack, depth) {
                    Some(f) => {
                        text.push_str(&depth.to_string());
                        branch_frame = Some(f);
                    }
                    None => text.push_str(&format!("{depth} (out of range)")),
                }
            }
            Imm::BrTable => {
                let Some(count) = leb_u32(body, &mut pos) else { break };
                let mut depths = Vec::new();
                let mut ok = true;
                for _ in 0..count.min(1_000_000) {
                    match leb_u32(body, &mut pos) {
                        Some(d) => depths.push(d),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                let Some(default) = leb_u32(body, &mut pos) else { break };
                if !ok {
                    break;
                }
                // The cases are named in the order the module writes them,
                // and the default comes last: the index the instruction pops
                // picks one of the cases, and the default is what it picks
                // when the index is out of range. Printing them is what
                // turns the instruction from "a branch somewhere" into a
                // dispatch.
                text.push_str(&format!(
                    " {} entr{}",
                    depths.len(),
                    if depths.len() == 1 { "y" } else { "ies" }
                ));
                let mut labels = Vec::new();
                for d in &depths {
                    labels.push(d.to_string());
                    if let Some(f) = frame_at(&stack, *d) {
                        table_frames.push(f);
                    }
                }
                text.push_str(&format!(", labels {}", labels.join(" ")));
                text.push_str(&format!(", default {default}"));
                if let Some(f) = frame_at(&stack, default) {
                    table_frames.push(f);
                }
                table_labels = depths
                    .iter()
                    .chain(std::iter::once(&default))
                    .copied()
                    .collect();
            }
            Imm::Func => {
                let Some(index) = leb_u32(body, &mut pos) else { break };
                text.push_str(&format!(" {index}"));
                if let Some(to) = call_target(index) {
                    call_target_at = Some(to.get());
                    text.push_str(&format!(" -> {to}"));
                }
            }
            Imm::TypeTable => {
                let Some(ty) = leb_u32(body, &mut pos) else { break };
                let Some(table) = leb_u32(body, &mut pos) else { break };
                text.push_str(&format!(" type {ty}, table {table}"));
            }
            Imm::Var => {
                let Some(index) = leb_u32(body, &mut pos) else { break };
                text.push_str(&format!(" {index}"));
            }
            Imm::MemArg => {
                let Some(align) = leb_u32(body, &mut pos) else { break };
                let Some(off) = leb_u32(body, &mut pos) else { break };
                text.push_str(&format!(" align={align} offset={off}"));
            }
            Imm::Mem => {
                let Some(index) = body.get(pos).copied() else { break };
                pos += 1;
                text.push_str(&format!(" {index}"));
            }
            Imm::I32 => {
                let Some(v) = leb_i32(body, &mut pos) else { break };
                text.push_str(&format!(" {v}"));
            }
            Imm::I64 => {
                let Some(v) = leb_i64(body, &mut pos) else { break };
                text.push_str(&format!(" {v}"));
            }
            Imm::F32 => {
                let Some(chunk) = body.get(pos..pos + 4) else { break };
                pos += 4;
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                text.push_str(&format!(" {}", f32::from_bits(bits)));
            }
            Imm::F64 => {
                let Some(chunk) = body.get(pos..pos + 8) else { break };
                pos += 8;
                let mut raw = [0u8; 8];
                raw.copy_from_slice(chunk);
                text.push_str(&format!(" {}", f64::from_bits(u64::from_le_bytes(raw))));
            }
            Imm::Heap => {
                let Some(heap) = body.get(pos).copied() else { break };
                pos += 1;
                text.push_str(&format!(" {}", heap_name(heap)));
            }
            Imm::SelectTypes => {
                let Some(count) = leb_u32(body, &mut pos) else { break };
                for _ in 0..count.min(64) {
                    let Some(t) = body.get(pos).copied() else { break };
                    pos += 1;
                    text.push_str(&format!(" {}", heap_name(t)));
                }
            }
        }

        let flow = match op {
            0x00 => FlowKind::Exception,
            0x0c => FlowKind::UnconditionalBranch,
            0x0d | 0x04 => FlowKind::ConditionalBranch,
            0x0e => FlowKind::IndirectBranch,
            0x0f => FlowKind::Return,
            // `return_call` and its indirect form are calls for every purpose
            // this layer serves: the callee is reachable and has no return edge
            // back into this frame, which the caller's own `return` covers.
            0x10 | 0x12 => FlowKind::Call,
            0x11 | 0x13 => FlowKind::IndirectCall,
            _ => FlowKind::Next,
        };

        let index = raws.len();
        match op {
            0x02..=0x04 => {
                frames.push(Frame {
                    kind: op,
                    open: index,
                    else_at: None,
                    end: None,
                });
                stack.push(frames.len() - 1);
                if op == 0x04 {
                    // The false edge leaves the then-branch: after `else` when
                    // there is one, past `end` otherwise.
                    if_false_frame = Some(frames.len() - 1);
                }
            }
            0x05 => {
                if let Some(&f) = stack.last()
                    && frames[f].kind == 0x04
                {
                    frames[f].else_at = Some(index);
                }
            }
            0x0b => {
                if let Some(f) = stack.pop() {
                    frames[f].end = Some(index);
                }
            }
            _ => {}
        }

        raws.push(Raw {
            offset,
            len: (pos - begin) as u32,
            text,
            mnemonic: name,
            flow,
            branch_frame,
            call_target: call_target_at,
            table_frames,
            table_labels,
            if_false_frame,
        });
    }

    // Second pass: turn frames into offsets.
    let end_of = |f: usize| -> Option<u64> {
        let frame = &frames[f];
        match frame.kind {
            // A branch to a `loop` restarts it: the target is its first body
            // instruction, not the instruction after `end`.
            0x03 => raws.get(frame.open + 1).map(|r| r.offset),
            _ => frame
                .end
                .and_then(|e| raws.get(e + 1))
                .map(|r| r.offset)
                // An `end` at the very end of the body: the target is one past
                // the last byte, which is where the caller resumes.
                .or_else(|| {
                    frame
                        .end
                        .map(|e| raws[e].offset + u64::from(raws[e].len))
                }),
        }
    };

    let mut insns = Vec::with_capacity(raws.len());
    let mut targets: Vec<(Addr, Vec<Addr>)> = Vec::new();
    let mut br_tables: Vec<(Addr, Vec<u32>)> = Vec::new();
    for raw in &raws {
        let mut target = None;
        if let Some(f) = raw.branch_frame {
            // A `br` to an `if` frame means the same as to a block: past `end`.
            target = end_of(f).map(|t| CallTarget::Direct { to: Addr(t) });
        }
        if let Some(f) = raw.if_false_frame {
            // Falling out of the then-branch lands after `else`, or past `end`
            // when there is none.
            let to = frames[f]
                .else_at
                .and_then(|e| raws.get(e + 1))
                .map(|r| r.offset)
                .or_else(|| end_of(f));
            if let Some(to) = to {
                target = Some(CallTarget::Direct { to: Addr(to) });
            }
        }
        if let Some(to) = raw.call_target {
            target = Some(CallTarget::Direct { to: Addr(to) });
        }
        if raw.flow == FlowKind::IndirectBranch {
            target = Some(CallTarget::Indirect { via: None });
        }

        let mut listed: Vec<Addr> = Vec::new();
        for f in &raw.table_frames {
            if let Some(to) = end_of(*f) {
                listed.push(Addr(to));
            }
        }
        if !raw.table_labels.is_empty() {
            br_tables.push((Addr(raw.offset), raw.table_labels.clone()));
        }
        if let Some(CallTarget::Direct { to }) = &target
            && raw.flow == FlowKind::ConditionalBranch
        {
            listed.push(*to);
        }
        if !listed.is_empty() {
            listed.sort_by_key(|a| a.get());
            listed.dedup();
            targets.push((Addr(raw.offset), listed));
        }

        let mut bytes = [0u8; MAX_INSN_LEN];
        let from = (raw.offset - start) as usize;
        let take = (raw.len as usize).min(MAX_INSN_LEN);
        if let Some(chunk) = body.get(from..from + take) {
            bytes[..take].copy_from_slice(chunk);
        }
        insns.push(Instruction {
            addr: Addr(raw.offset),
            len: raw.len,
            bytes,
            text: raw.text.clone(),
            mnemonic: raw.mnemonic.into(),
            flow: raw.flow,
            target,
            mem_reads: Vec::new(),
            mem_writes: Vec::new(),
        });
    }

    Body {
        insns,
        targets,
        truncated,
        locals,
        br_tables,
    }
}

fn frame_at(stack: &[usize], depth: u32) -> Option<usize> {
    let depth = depth as usize;
    if depth >= stack.len() {
        return None;
    }
    stack.get(stack.len() - 1 - depth).copied()
}

/// 0xfc sub-opcodes. Only the ones with a known length are returned.
fn decode_fc(sub: u32, body: &[u8], pos: &mut usize) -> Option<&'static str> {
    if let Some(name) = SAT.get(sub as usize) {
        return Some(name);
    }
    let name: &'static str = match sub {
        8 => "memory.init",
        9 => "data.drop",
        10 => "memory.copy",
        11 => "memory.fill",
        12 => "table.init",
        13 => "elem.drop",
        14 => "table.copy",
        15 => "table.grow",
        16 => "table.size",
        17 => "table.fill",
        _ => return None,
    };
    // Lengths: memory.init/data.drop take one index, memory.copy two,
    // table.copy two, table.init two, the rest one.
    let count = match sub {
        10 | 12 | 14 => 2,
        _ => 1,
    };
    for _ in 0..count {
        leb_u32(body, pos)?;
    }
    Some(name)
}

/// The type of a block: what it takes and what it leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockType {
    /// Nothing in, nothing out.
    Void,
    /// One value of a named type.
    One(&'static str),
    /// A type index: a block with parameters and results.
    Index(u32),
}

impl BlockType {
    /// The type the instruction is printed with.
    ///
    /// The parameters and the results are named, because a block that takes
    /// a value and gives one back says what that value is: a reader who
    /// knows the block carries an `i32` across it knows that a `br` out of
    /// it carries an `i32` with it.
    fn describe(self, types: &[WasmFuncType]) -> String {
        match self {
            BlockType::Void => String::new(),
            BlockType::One(name) => format!(" (result {name})"),
            BlockType::Index(index) => match types.get(index as usize) {
                Some(ty) => {
                    let mut out = String::new();
                    if !ty.params.is_empty() {
                        out.push_str(" (param");
                        for p in &ty.params {
                            out.push(' ');
                            out.push_str(p.name());
                        }
                    }
                    if !ty.results.is_empty() {
                        if out.is_empty() {
                            out.push_str(" (result");
                        } else {
                            out.push_str(", result");
                        }
                        for r in &ty.results {
                            out.push(' ');
                            out.push_str(r.name());
                        }
                    }
                    if !out.is_empty() {
                        out.push(')');
                    }
                    out
                }
                // A type index the module does not declare is a module this
                // parser could not read; saying so is better than guessing.
                None => format!(" (type {index})"),
            },
        }
    }
}

/// Read a block type.
fn block_type(body: &[u8], pos: &mut usize) -> Option<BlockType> {
    let b = *body.get(*pos)?;
    // A negative byte is a value type or the empty 0x40; anything else starts a
    // signed 33-bit type index.
    if b == 0x40 {
        *pos += 1;
        return Some(BlockType::Void);
    }
    if b & 0x80 == 0 {
        // A single byte below 0x80 is a value type, except for the ones that
        // are not: 0x00 is the first type index and means a block with no
        // parameters and no results.
        let name = match b {
            0x7f => "i32",
            0x7e => "i64",
            0x7d => "f32",
            0x7c => "f64",
            0x7b => "v128",
            0x70 => "funcref",
            0x6f => "externref",
            _ => {
                // A type index, read as a signed 33-bit number.
                let index = leb_i64(body, pos)?;
                if index < 0 {
                    return None;
                }
                return Some(BlockType::Index(index as u32));
            }
        };
        *pos += 1;
        return Some(BlockType::One(name));
    }
    // A multi-byte signed number is a type index.
    let index = leb_i64(body, pos)?;
    if index < 0 {
        return None;
    }
    Some(BlockType::Index(index as u32))
}

fn heap_name(byte: u8) -> &'static str {
    match byte {
        0x70 => "funcref",
        0x6f => "externref",
        0x7f => "i32",
        0x7e => "i64",
        0x7d => "f32",
        0x7c => "f64",
        0x7b => "v128",
        _ => "?",
    }
}

fn leb_u32(data: &[u8], pos: &mut usize) -> Option<u32> {
    let mut result = 0u32;
    let mut shift = 0u32;
    loop {
        let b = *data.get(*pos)?;
        *pos += 1;
        result |= u32::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 35 {
            return None;
        }
    }
}

fn leb_i64(data: &[u8], pos: &mut usize) -> Option<i64> {
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
        if shift >= 70 {
            return None;
        }
    }
}

fn leb_i32(data: &[u8], pos: &mut usize) -> Option<i32> {
    leb_i64(data, pos).map(|v| v as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_calls(_index: u32) -> Option<Addr> {
        None
    }

    /// `(func (result i32) i32.const 42)`, local declarations included.
    #[test]
    fn decodes_a_constant_body() {
        let body = [0x00, 0x41, 0x2a, 0x0b];
        let out = decode(&body, 0x100, &no_calls, &[]);
        assert_eq!(out.insns.len(), 2);
        assert_eq!(out.insns[0].text, "i32.const 42");
        assert_eq!(out.insns[0].addr, Addr(0x101));
        assert_eq!(out.insns[0].len, 2);
        assert_eq!(out.insns[1].text, "end");
        assert!(out.truncated.is_none());
    }

    /// A `block` whose `br 0` targets the instruction after `end`.
    #[test]
    fn a_branch_resolves_to_the_instruction_after_the_end() {
        // block ; br 0 ; nop ; end ; nop ; end
        let body = [0x00, 0x02, 0x40, 0x0c, 0x00, 0x01, 0x0b, 0x01, 0x0b];
        let out = decode(&body, 0, &no_calls, &[]);
        let br = out.insns.iter().find(|i| i.mnemonic.as_ref() == "br").unwrap();
        assert_eq!(br.addr, Addr(3));
        match br.target.clone() {
            Some(CallTarget::Direct { to }) => assert_eq!(to, Addr(7)),
            other => panic!("expected a direct target, got {other:?}"),
        }
    }

    /// A branch to a `loop` goes back to its first body instruction.
    #[test]
    fn a_branch_to_a_loop_restarts_it() {
        // loop ; nop ; br 0 ; end ; end
        let body = [0x00, 0x03, 0x40, 0x01, 0x0c, 0x00, 0x0b, 0x0b];
        let out = decode(&body, 0, &no_calls, &[]);
        let br = out.insns.iter().find(|i| i.mnemonic.as_ref() == "br").unwrap();
        match br.target.clone() {
            Some(CallTarget::Direct { to }) => assert_eq!(to, Addr(3)),
            other => panic!("expected a direct target, got {other:?}"),
        }
    }

    #[test]
    fn an_if_targets_the_instruction_after_else() {
        // if ; else ; nop ; end ; end
        let body = [0x00, 0x04, 0x40, 0x05, 0x01, 0x0b, 0x0b];
        let out = decode(&body, 0, &no_calls, &[]);
        let cond = &out.insns[0];
        assert_eq!(cond.mnemonic.as_ref(), "if");
        assert_eq!(cond.flow, FlowKind::ConditionalBranch);
        match cond.target.clone() {
            Some(CallTarget::Direct { to }) => assert_eq!(to, Addr(4)),
            other => panic!("expected a direct target, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_opcode_stops_the_body_and_says_where() {
        let body = [0x00, 0x41, 0x01, 0xfd, 0x00, 0x0b];
        let out = decode(&body, 0x40, &no_calls, &[]);
        assert_eq!(out.insns.len(), 1);
        let (at, why) = out.truncated.expect("truncation is reported");
        assert_eq!(at, Addr(0x43));
        assert!(why.contains("SIMD"), "{why}");
    }

    #[test]
    fn a_truncated_body_never_panics() {
        let body = [0x00, 0x41, 0x2a, 0x0b];
        for cut in 0..body.len() {
            let _ = decode(&body[..cut], 0, &no_calls, &[]);
        }
    }
}
