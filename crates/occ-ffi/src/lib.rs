//! C ABI for `occ`.
//!
//! The point of this crate is that a caller does not need Rust, cargo, or a
//! vendored dependency tree: `cc t.c -L. -locc -lpthread -ldl -lm` is the whole
//! build line. Everything the caller can ask for is spelled out in `occ.h`,
//! which is written by hand rather than generated, because a generator is one
//! more thing to install.
//!
//! Three decisions shape the interface.
//!
//! **Results cross the boundary as JSON strings, never as structs.** A C
//! struct shared with Rust has to agree on field order, padding, enum width and
//! `repr` on both sides, and every one of those can drift silently. A JSON
//! document has one representation and `occ-json` already writes it for the MCP
//! server, so the C caller and the MCP client see the same bytes for the same
//! question.
//!
//! **There are two error channels, and they mean different things.** A `NULL`
//! return with text in the `err` buffer means the call did not happen: a null
//! pointer, a handle that is not open, a panic caught at the boundary. A parsed
//! JSON document with `"error": true` means the call happened and the analysis
//! declined — an address outside any function, a target that matched no symbol.
//! A caller that treats the two alike will report a bad argument as a fact about
//! the binary.
//!
//! **Every returned string is owned by the caller and released with
//! [`occ_free`].** The session owns its images; the strings are copies and
//! outlive the session if the caller keeps them, which is deliberate — a
//! summary should not become invalid because another thread closed a handle.
//!
//! A session serialises its own mutations behind a lock, so several threads may
//! call into one `OccSession`. Two rules are the caller's to keep: do not call
//! [`occ_close`] while another thread is inside any other function on the same
//! session, and do not free a string twice.

#![allow(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
// The library boundary may not panic; a test may.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use occ_analyze::view::View;
use occ_core::{Addr, Function, Image};
use occ_decode::AnyDecoder;
use occ_json::{Map, Value};
use occ_session::Session;
use occ_session::render::Envelope;

/// An open session, opaque to C.
///
/// One session may hold several images: opening an `ar` archive yields one per
/// member, and opening a directory is not supported. Use [`occ_image_at`] to
/// walk them.
pub struct OccSession {
    session: Session,
}

/// The most instructions one [`occ_disasm`] call will decode.
///
/// A caller that asks for a billion is not asking a question, and decoding one
/// would hold the session lock for minutes. The reply says when this clamped.
const MAX_INSNS: u32 = 100_000;

/// The most strings one [`occ_strings`] call will return.
const MAX_STRINGS: usize = 20_000;

/// The most cross references one [`occ_xrefs`] call will return.
const MAX_XREFS: usize = 10_000;

/// Fill the caller's buffer with `msg`, truncating on a character boundary.
///
/// Truncation rather than refusal, because a diagnostic that does not fit is
/// still worth more than silence. The buffer is always NUL terminated when
/// `len > 0`.
///
/// # Safety
///
/// `dst` must point to `len` writable bytes, or be null.
unsafe fn report(dst: *mut c_char, len: usize, msg: &str) {
    if dst.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let mut n = bytes.len().min(len - 1);
    while n > 0 && !msg.is_char_boundary(n) {
        n -= 1;
    }
    // SAFETY: the caller guaranteed `len` writable bytes, and `n < len`.
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), dst.cast::<u8>(), n);
        *dst.add(n) = 0;
    }
}

/// Move a JSON document to the C heap.
///
/// `CString` refuses interior NULs, and a JSON writer that emitted one would
/// break every consumer downstream, so a stray NUL is escaped rather than
/// silently truncating the document.
fn to_c(json: String) -> *mut c_char {
    let clean = if json.as_bytes().contains(&0) {
        json.replace('\0', "\\u0000")
    } else {
        json
    };
    match CString::new(clean) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Read a C string as `&str`.
///
/// # Safety
///
/// `p` must be null or point to a NUL-terminated string valid for this call.
unsafe fn cstr<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Err("occ: argument is a null pointer".into());
    }
    // SAFETY: the caller promised a NUL-terminated string.
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| "occ: argument is not valid UTF-8".to_string())
}

/// Run `f`, converting a panic and an error into the `NULL` channel.
///
/// The release profile builds with unwinding so that this boundary is real. A
/// panic that crossed into C would be undefined behaviour, and one that aborted
/// the process would take a server down over one malformed input.
fn call_json<F>(err: *mut c_char, err_len: usize, f: F) -> *mut c_char
where
    F: FnOnce() -> Result<String, String>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(json)) => to_c(json),
        Ok(Err(msg)) => {
            // SAFETY: the caller supplied the buffer and its length.
            unsafe { report(err, err_len, &msg) };
            ptr::null_mut()
        }
        Err(payload) => {
            let msg = panic_text(&payload);
            // SAFETY: as above.
            unsafe { report(err, err_len, &msg) };
            ptr::null_mut()
        }
    }
}

/// Extract the message from a panic payload, or name it if it is not a string.
fn panic_text(payload: &Box<dyn std::any::Any + Send>) -> String {
    let detail = if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string payload".to_string()
    };
    format!("occ: internal panic: {detail}")
}

/// Run `f`, converting a panic into a sentinel.
fn call_raw<T, F>(err: *mut c_char, err_len: usize, fail: T, f: F) -> T
where
    F: FnOnce() -> Result<T, String>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(msg)) => {
            // SAFETY: the caller supplied the buffer and its length.
            unsafe { report(err, err_len, &msg) };
            fail
        }
        Err(payload) => {
            let msg = panic_text(&payload);
            // SAFETY: as above.
            unsafe { report(err, err_len, &msg) };
            fail
        }
    }
}

/// Hex encode bytes, lowercase without separators.
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[usize::from(b >> 4)] as char);
        out.push(DIGITS[usize::from(b & 0x0f)] as char);
    }
    out
}

/// Build a JSON object from pairs without importing the `obj!` macro.
fn object(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k, v);
    }
    Value::Object(m)
}

/// A string array from anything string-like.
fn strings<I, S>(items: I) -> Value
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    Value::Array(items.into_iter().map(|s| Value::from(s.into())).collect())
}

/// Borrow the session behind a raw pointer.
///
/// # Safety
///
/// `p` must be a pointer returned by [`occ_open`] and not yet closed.
unsafe fn session<'a>(p: *const OccSession) -> Result<&'a OccSession, String> {
    // SAFETY: the caller promised a live session pointer.
    unsafe { p.as_ref() }.ok_or_else(|| "occ: session is a null pointer".to_string())
}

/// Open one image out of a session.
fn file_of(s: &OccSession, image: i64) -> Result<std::sync::Arc<occ_session::ImageFile>, String> {
    if image < 0 {
        return Err(format!("occ: image id {image} is not valid"));
    }
    s.session
        .image(image as u32)
        .ok_or_else(|| format!("occ: no image with id {image} in this session"))
}

/// Resolve a name or an address to a recovered function.
fn find<'a>(image: &'a Image, analysis: &'a occ_analyze::Analysis, target: &str) -> Option<&'a Function> {
    if let Some(addr) = occ_core::addr::parse_addr(target)
        && let Some(f) = analysis.function_covering(addr)
    {
        return Some(f);
    }
    if let Some(f) = analysis.function_by_name(target) {
        return Some(f);
    }
    // A symbol the analysis did not turn into a function can still name an
    // address inside one. Searching the table is linear, which is fine: this
    // runs once per call, not once per instruction.
    let sym = image.symbols.iter().find(|s| s.name == target)?;
    analysis
        .function_at(sym.addr)
        .or_else(|| analysis.function_covering(sym.addr))
}

/// The library version as a static NUL-terminated string.
///
/// No allocation and no failure: this is safe to call before anything else and
/// the pointer is valid for the life of the process.
#[unsafe(no_mangle)]
pub extern "C" fn occ_version() -> *const c_char {
    const V: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    V.as_ptr().cast()
}

/// Open an image and return a session, or `NULL` with `err` filled.
///
/// Opening an archive yields one image per member; call [`occ_image_count`] to
/// see how many, then [`occ_image_at`] for each id.
///
/// # Safety
///
/// `path` must be a NUL-terminated string. `err` must be null or point to
/// `err_len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_open(
    path: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> *mut OccSession {
    call_raw(err, err_len, ptr::null_mut(), || {
        // SAFETY: the caller supplied a NUL-terminated path.
        let p = unsafe { cstr(path) }?;
        let session = Session::new();
        session.open_path(p).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(OccSession { session })))
    })
}

/// Close a session and release every image it holds.
///
/// Strings handed out earlier stay valid until [`occ_free`]; they are copies,
/// not views into the session.
///
/// # Safety
///
/// `s` must come from [`occ_open`] and must not be used afterwards. Passing
/// null is allowed and does nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_close(s: *mut OccSession) {
    if s.is_null() {
        return;
    }
    // SAFETY: the pointer came from `Box::into_raw` in `occ_open` and the
    // caller promised not to use it again.
    drop(unsafe { Box::from_raw(s) });
}

/// Number of images in the session.
///
/// # Safety
///
/// `s` must be a live session, or null (which answers zero).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_image_count(s: *const OccSession) -> u32 {
    // SAFETY: the caller promised a live session.
    match unsafe { s.as_ref() } {
        Some(s) => s.session.image_ids().len() as u32,
        None => 0,
    }
}

/// The image id at `index`, or `-1` when the index is past the end.
///
/// # Safety
///
/// `s` must be a live session, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_image_at(s: *const OccSession, index: u32) -> i64 {
    // SAFETY: the caller promised a live session.
    match unsafe { s.as_ref() } {
        Some(s) => s
            .session
            .image_ids()
            .get(index as usize)
            .map(|id| i64::from(*id))
            .unwrap_or(-1),
        None => -1,
    }
}

/// Structure of one image: format, architecture, sections, symbols, coverage.
///
/// # Safety
///
/// `s` must be a live session; `err` must be null or hold `err_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_info(
    s: *const OccSession,
    image: i64,
    err: *mut c_char,
    err_len: usize,
) -> *mut c_char {
    call_json(err, err_len, || {
        // SAFETY: the caller promised a live session.
        let s = unsafe { session(s) }?;
        let f = file_of(s, image)?;
        let img = &f.image;
        let a = &f.analysis;
        let sections: Vec<Value> = img
            .sections
            .iter()
            .take(512)
            .map(|sec| {
                object(vec![
                    ("name", Value::from(sec.name.as_str())),
                    ("addr", Value::from(sec.addr.to_string())),
                    ("size", Value::from(sec.size)),
                    ("file_size", Value::from(sec.file_size)),
                    ("kind", Value::from(format!("{:?}", sec.kind))),
                ])
            })
            .collect();
        let imports: Vec<Value> = img
            .imports
            .iter()
            .take(256)
            .map(|i| Value::from(i.name.as_str()))
            .collect();
        let exports: Vec<Value> = img
            .exports
            .iter()
            .take(256)
            .map(|e| Value::from(e.name.as_str()))
            .collect();
        let env = Envelope::ok(format!(
            "{} {:?} {:?} entry {} · {} functions · {:.1}% of code recovered",
            img.path,
            img.format,
            img.arch,
            img.entry,
            a.functions.len(),
            a.coverage.fraction() * 100.0
        ))
        .field(
            "image",
            object(vec![
                ("path", Value::from(img.path.as_str())),
                ("format", Value::from(format!("{:?}", img.format))),
                ("arch", Value::from(format!("{:?}", img.arch))),
                ("bits", Value::from(u64::from(img.bits))),
                ("endian", Value::from(format!("{:?}", img.endian))),
                ("entry", Value::from(img.entry.to_string())),
                ("image_base", Value::from(img.image_base.to_string())),
                ("file_size", Value::from(img.file_size)),
                ("sha256", Value::from(img.sha256.as_str())),
                ("segment_count", Value::from(img.segments.len())),
                ("section_count", Value::from(img.sections.len())),
                ("symbol_count", Value::from(img.symbols.len())),
                ("import_count", Value::from(img.imports.len())),
                ("export_count", Value::from(img.exports.len())),
            ]),
        )
        .field(
            "coverage",
            object(vec![
                ("code_bytes", Value::from(a.coverage.code_bytes)),
                ("decoded_bytes", Value::from(a.coverage.decoded_bytes)),
                ("covered_bytes", Value::from(a.coverage.covered_bytes)),
                ("functions", Value::from(a.coverage.functions as u64)),
                ("instructions", Value::from(a.coverage.insns as u64)),
                ("blocks", Value::from(a.coverage.blocks as u64)),
                ("fraction", Value::from(a.coverage.fraction() as f64)),
            ]),
        )
        .field("sections", Value::Array(sections))
        .field("imports", Value::Array(imports))
        .field("exports", Value::Array(exports))
        .field("warnings", strings(img.warnings.iter().cloned()))
        .build();
        Ok(occ_json::to_string(&env))
    })
}

/// Decode `count` instructions starting at `addr`.
///
/// Counts above [`MAX_INSNS`] are clamped and the reply says so. Bytes that no
/// architecture in the table recognises decode as `.byte` one byte at a time,
/// which is what keeps the listing aligned after a gap.
///
/// # Safety
///
/// `s` must be a live session; `err` must be null or hold `err_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_disasm(
    s: *const OccSession,
    image: i64,
    addr: u64,
    count: u32,
    err: *mut c_char,
    err_len: usize,
) -> *mut c_char {
    call_json(err, err_len, || {
        // SAFETY: the caller promised a live session.
        let s = unsafe { session(s) }?;
        let f = file_of(s, image)?;
        let clamped = count.min(MAX_INSNS);
        let view = View::new(&f.image, &f.data);
        let dec = AnyDecoder::new(f.image.arch, f.image.bits)
            .ok_or_else(|| format!("occ: {:?} has no decoder", f.image.arch))?;
        let mut at = Addr(addr);
        let mut rows: Vec<Value> = Vec::new();
        for _ in 0..clamped {
            let Some(bytes) = view.bytes_to_end(at) else {
                break;
            };
            let Some(d) = dec.decode(at, bytes) else {
                break;
            };
            let insn = &d.insn;
            let used = (insn.len as usize).min(insn.bytes.len());
            rows.push(object(vec![
                ("addr", Value::from(at.to_string())),
                ("bytes", Value::from(hex(&insn.bytes[..used]))),
                ("len", Value::from(u64::from(insn.len))),
                ("text", Value::from(insn.text.as_str())),
                ("mnemonic", Value::from(&*insn.mnemonic)),
                ("flow", Value::from(format!("{:?}", insn.flow))),
            ]));
            let next = at.0.saturating_add(u64::from(insn.len));
            if next <= at.0 {
                break;
            }
            at = Addr(next);
        }
        let mut env = Envelope::ok(format!("{} instruction(s) at {}", rows.len(), Addr(addr)))
            .field("instructions", Value::Array(rows))
            .field("next_addr", Value::from(at.to_string()));
        let mut caveats = Vec::new();
        if u64::from(clamped) < u64::from(count) {
            env = env
                .field("requested", Value::from(u64::from(count)))
                .field("clamped_to", Value::from(u64::from(clamped)));
            caveats.push(format!(
                "the request asked for {count} instructions; the ceiling is {MAX_INSNS}"
            ));
        }
        if view.bytes_to_end(at).is_none() {
            caveats.push("the address is outside every mapped section".to_string());
        }
        if !caveats.is_empty() {
            env = env.caveats(caveats);
        }
        Ok(occ_json::to_string(&env.build()))
    })
}

/// One recovered function: extent, blocks, callers, callees, jump tables.
///
/// `target` is a symbol name, a `fn:0x401000` handle, or a bare address.
///
/// # Safety
///
/// `s` must be a live session; `target` must be a NUL-terminated string; `err`
/// must be null or hold `err_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_function(
    s: *const OccSession,
    image: i64,
    target: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> *mut c_char {
    call_json(err, err_len, || {
        // SAFETY: the caller promised a live session.
        let s = unsafe { session(s) }?;
        // SAFETY: the caller promised a NUL-terminated target.
        let target = unsafe { cstr(target) }?;
        let f = file_of(s, image)?;
        let func = find(&f.image, &f.analysis, target)
            .ok_or_else(|| format!("occ: no function matches {target}"))?;
        let blocks: Vec<Value> = func
            .blocks
            .iter()
            .map(|b| {
                let edges: Vec<Value> = b
                    .edges
                    .iter()
                    .map(|e| {
                        object(vec![
                            ("to", Value::from(e.to.to_string())),
                            ("kind", Value::from(format!("{:?}", e.kind))),
                        ])
                    })
                    .collect();
                object(vec![
                    ("start", Value::from(b.start.to_string())),
                    ("end", Value::from(b.end.to_string())),
                    ("insns", Value::from(b.len() as u64)),
                    ("edges", Value::Array(edges)),
                ])
            })
            .collect();
        let callers: Vec<Value> = func.callers.iter().map(|a| Value::from(a.to_string())).collect();
        let callees: Vec<Value> = func.callees.iter().map(|a| Value::from(a.to_string())).collect();
        let tables: Vec<Value> = func
            .jump_tables
            .iter()
            .map(|t| {
                object(vec![
                    ("at", Value::from(t.at.to_string())),
                    ("base", Value::from(t.base.to_string())),
                    ("entry_bytes", Value::from(u64::from(t.entry_bytes))),
                    ("count", Value::from(u64::from(t.count))),
                    ("relative", Value::from(t.relative)),
                ])
            })
            .collect();
        let env = Envelope::ok(format!(
            "{} at {} · {} bytes · {} instructions · {} blocks",
            func.display_name(),
            func.entry,
            func.size,
            func.insns.len(),
            func.blocks.len()
        ))
        .handle(format!("fn:{}", func.entry))
        .field(
            "function",
            object(vec![
                ("entry", Value::from(func.entry.to_string())),
                ("name", Value::from(func.display_name())),
                ("size", Value::from(func.size)),
                ("insn_count", Value::from(func.insns.len() as u64)),
                ("block_count", Value::from(func.blocks.len() as u64)),
                ("is_thunk", Value::from(func.is_thunk)),
                ("callers", Value::Array(callers)),
                ("callees", Value::Array(callees)),
                ("jump_tables", Value::Array(tables)),
            ]),
        )
        .field("blocks", Value::Array(blocks))
        .evidence(func.boundary_evidence.clone())
        .confidence(func.boundary_confidence)
        .build();
        Ok(occ_json::to_string(&env))
    })
}

/// Cross references to an address: who reads, writes, calls or jumps here.
///
/// # Safety
///
/// `s` must be a live session; `target` must be a NUL-terminated string; `err`
/// must be null or hold `err_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_xrefs(
    s: *const OccSession,
    image: i64,
    target: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> *mut c_char {
    call_json(err, err_len, || {
        // SAFETY: the caller promised a live session.
        let s = unsafe { session(s) }?;
        // SAFETY: the caller promised a NUL-terminated target.
        let target = unsafe { cstr(target) }?;
        let f = file_of(s, image)?;
        let at = if let Some(a) = occ_core::addr::parse_addr(target) {
            a
        } else if let Some(sym) = f.image.symbols.iter().find(|s| s.name == target) {
            sym.addr
        } else {
            return Err(format!("occ: no symbol or address matches {target}"));
        };
        let mut rows: Vec<Value> = Vec::new();
        let mut total = 0usize;
        for x in f.analysis.xrefs.to(at) {
            total += 1;
            if rows.len() >= MAX_XREFS {
                continue;
            }
            rows.push(object(vec![
                ("from", Value::from(x.from.to_string())),
                ("to", Value::from(x.to.to_string())),
                ("kind", Value::from(format!("{:?}", x.kind))),
                (
                    "from_function",
                    match x.from_function {
                        Some(a) => Value::from(a.to_string()),
                        None => Value::Null,
                    },
                ),
            ]));
        }
        let mut env = Envelope::ok(format!("{total} reference(s) to {at}"))
            .field("to", Value::from(at.to_string()))
            .field("count", Value::from(total as u64))
            .field("refs", Value::Array(rows));
        if total > MAX_XREFS {
            env = env.truncated(true).caveats(vec![format!(
                "only the first {MAX_XREFS} of {total} references are listed"
            )]);
        }
        Ok(occ_json::to_string(&env.build()))
    })
}

/// Decompile one function to pseudo-C.
///
/// The body carries a `unstructured` count: every construct that needed a
/// `goto`. A caller reading the code should treat that number as part of the
/// answer rather than as metadata.
///
/// # Safety
///
/// `s` must be a live session; `target` must be a NUL-terminated string; `err`
/// must be null or hold `err_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_decompile(
    s: *const OccSession,
    image: i64,
    target: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> *mut c_char {
    call_json(err, err_len, || {
        // SAFETY: the caller promised a live session.
        let s = unsafe { session(s) }?;
        // SAFETY: the caller promised a NUL-terminated target.
        let target = unsafe { cstr(target) }?;
        let f = file_of(s, image)?;
        let func = find(&f.image, &f.analysis, target)
            .ok_or_else(|| format!("occ: no function matches {target}"))?;
        let analysis = &f.analysis;
        let name_of = |a: Addr| -> String {
            analysis
                .function_at(a)
                .map(|g| g.display_name())
                .unwrap_or_else(|| format!("sub_{:x}", a.get()))
        };
        let opts = occ_decompile::Options::default();
        let d = occ_decompile::decompile(&f.image, func, &opts, &name_of);
        let code = occ_decompile::render::render(&d);
        let mut env = Envelope::ok(format!(
            "{}: {} line(s), {} block(s), {} unstructured",
            func.display_name(),
            code.lines().count(),
            func.blocks.len(),
            d.unstructured
        ))
        .handle(format!("fn:{}", func.entry))
        .field(
            "function",
            object(vec![
                ("entry", Value::from(func.entry.to_string())),
                ("name", Value::from(d.name.as_str())),
                ("params", strings(d.params.iter().cloned())),
                ("unstructured", Value::from(d.unstructured as u64)),
                ("lines", Value::from(code.lines().count() as u64)),
            ]),
        )
        .field("code", Value::from(code))
        .field("notes", strings(d.notes.iter().cloned()))
        .evidence(func.boundary_evidence.clone())
        .confidence((func.boundary_confidence * 0.9).clamp(0.0, 1.0));
        if !occ_decompile::is_liftable(f.image.arch) {
            env = env.caveats(vec![format!(
                "{:?} has no lifter; the body is an outline, not decompiled code",
                f.image.arch
            )]);
        }
        Ok(occ_json::to_string(&env.build()))
    })
}

/// Strings found in the image, in address order.
///
/// # Safety
///
/// `s` must be a live session; `err` must be null or hold `err_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_strings(
    s: *const OccSession,
    image: i64,
    err: *mut c_char,
    err_len: usize,
) -> *mut c_char {
    call_json(err, err_len, || {
        // SAFETY: the caller promised a live session.
        let s = unsafe { session(s) }?;
        let f = file_of(s, image)?;
        let all = &f.analysis.strings;
        let rows: Vec<Value> = all
            .iter()
            .take(MAX_STRINGS)
            .map(|x| {
                object(vec![
                    ("addr", Value::from(x.addr.to_string())),
                    ("text", Value::from(x.text.as_str())),
                    ("encoding", Value::from(format!("{:?}", x.encoding))),
                ])
            })
            .collect();
        let mut env = Envelope::ok(format!("{} string(s)", all.len()))
            .field("count", Value::from(all.len() as u64))
            .field("strings", Value::Array(rows));
        if all.len() > MAX_STRINGS {
            env = env.truncated(true).caveats(vec![format!(
                "only the first {MAX_STRINGS} of {} strings are listed",
                all.len()
            )]);
        }
        Ok(occ_json::to_string(&env.build()))
    })
}

/// Raw bytes of one section, hex encoded, for a caller that wants its own parser.
///
/// # Safety
///
/// `s` must be a live session; `name` must be a NUL-terminated string; `err`
/// must be null or hold `err_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_section_bytes(
    s: *const OccSession,
    image: i64,
    name: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> *mut c_char {
    call_json(err, err_len, || {
        // SAFETY: the caller promised a live session.
        let s = unsafe { session(s) }?;
        // SAFETY: the caller promised a NUL-terminated name.
        let name = unsafe { cstr(name) }?;
        let f = file_of(s, image)?;
        let view = View::new(&f.image, &f.data);
        let sec = f
            .image
            .section_by_name(name)
            .ok_or_else(|| format!("occ: no section named {name}"))?;
        let data = view
            .section_data(sec)
            .ok_or_else(|| format!("occ: section {name} has no file bytes"))?;
        let env = Envelope::ok(format!("{}: {} byte(s)", name, data.len()))
            .field("section", Value::from(name))
            .field("addr", Value::from(sec.addr.to_string()))
            .field("size", Value::from(data.len() as u64))
            .field("hex", Value::from(hex(data)))
            .build();
        Ok(occ_json::to_string(&env))
    })
}

/// Release a string returned by any other function in this header.
///
/// Passing null is allowed and does nothing. Freeing the same pointer twice is
/// undefined, as it is for `free`.
///
/// # Safety
///
/// `p` must be null or a pointer returned by this library and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn occ_free(p: *mut c_char) {
    if p.is_null() {
        return;
    }
    // SAFETY: the pointer came from `CString::into_raw` in `to_c`, which is the
    // only producer of strings this header hands out.
    drop(unsafe { CString::from_raw(p) });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn open(path: &str) -> (CString, *mut OccSession) {
        let p = CString::new(path).unwrap();
        let mut err = [0 as c_char; 256];
        let s = unsafe { occ_open(p.as_ptr(), err.as_mut_ptr(), err.len()) };
        assert!(!s.is_null(), "open failed: {:?}", unsafe {
            CStr::from_ptr(err.as_ptr())
        });
        (p, s)
    }

    #[test]
    fn version_is_static_and_nul_terminated() {
        let v = unsafe { CStr::from_ptr(occ_version()) };
        assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn opening_a_binary_and_reading_its_info_works() {
        let (_p, s) = open("/bin/true");
        let mut err = [0 as c_char; 256];
        assert_eq!(unsafe { occ_image_count(s) }, 1);
        let id = unsafe { occ_image_at(s, 0) };
        assert_eq!(id, 0);
        let j = unsafe { occ_info(s, id, err.as_mut_ptr(), err.len()) };
        assert!(!j.is_null());
        let text = unsafe { CStr::from_ptr(j) }.to_str().unwrap().to_string();
        unsafe { occ_free(j) };
        assert!(text.contains("\"format\""), "{text}");
        assert!(text.contains("\"coverage\""), "{text}");
        unsafe { occ_close(s) };
    }

    #[test]
    fn disassembly_stays_aligned_and_reports_the_next_address() {
        let (_p, s) = open("/bin/true");
        let mut err = [0 as c_char; 256];
        let id = unsafe { occ_image_at(s, 0) };
        let f = file_of(unsafe { &*s }, id).unwrap();
        let entry = f.image.entry.0;
        let j = unsafe { occ_disasm(s, id, entry, 8, err.as_mut_ptr(), err.len()) };
        assert!(!j.is_null());
        let text = unsafe { CStr::from_ptr(j) }.to_str().unwrap().to_string();
        unsafe { occ_free(j) };
        assert!(text.contains("\"instructions\""), "{text}");
        assert!(text.contains("\"mnemonic\""), "{text}");
        unsafe { occ_close(s) };
    }

    #[test]
    fn decompile_returns_code_and_an_unstructured_count() {
        let (_p, s) = open("/bin/true");
        let mut err = [0 as c_char; 256];
        let id = unsafe { occ_image_at(s, 0) };
        let f = file_of(unsafe { &*s }, id).unwrap();
        let name = f
            .analysis
            .functions
            .iter()
            .find(|g| g.insns.len() > 4)
            .map(|g| g.display_name())
            .unwrap();
        let t = CString::new(name).unwrap();
        let j = unsafe { occ_decompile(s, id, t.as_ptr(), err.as_mut_ptr(), err.len()) };
        assert!(!j.is_null(), "{}", unsafe {
            CStr::from_ptr(err.as_ptr())
        }.to_str().unwrap());
        let text = unsafe { CStr::from_ptr(j) }.to_str().unwrap().to_string();
        unsafe { occ_free(j) };
        assert!(text.contains("\"unstructured\""), "{text}");
        unsafe { occ_close(s) };
    }

    #[test]
    fn a_null_argument_is_refused_with_a_message_not_a_crash() {
        let mut err = [0 as c_char; 128];
        let s = unsafe { occ_open(ptr::null(), err.as_mut_ptr(), err.len()) };
        assert!(s.is_null());
        let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_str().unwrap();
        assert!(msg.contains("null"), "{msg}");
    }

    #[test]
    fn an_unknown_image_id_is_an_error_not_a_panic() {
        let (_p, s) = open("/bin/true");
        let mut err = [0 as c_char; 128];
        let j = unsafe { occ_info(s, 99, err.as_mut_ptr(), err.len()) };
        assert!(j.is_null());
        let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_str().unwrap();
        assert!(msg.contains("99"), "{msg}");
        unsafe { occ_close(s) };
    }

    #[test]
    fn a_target_that_matches_nothing_says_so() {
        let (_p, s) = open("/bin/true");
        let mut err = [0 as c_char; 128];
        let id = unsafe { occ_image_at(s, 0) };
        let t = CString::new("definitely_not_a_symbol_xyz").unwrap();
        let j = unsafe { occ_function(s, id, t.as_ptr(), err.as_mut_ptr(), err.len()) };
        assert!(j.is_null());
        unsafe { occ_close(s) };
    }

    #[test]
    fn strings_are_bounded_and_counted() {
        let (_p, s) = open("/bin/true");
        let mut err = [0 as c_char; 128];
        let id = unsafe { occ_image_at(s, 0) };
        let j = unsafe { occ_strings(s, id, err.as_mut_ptr(), err.len()) };
        assert!(!j.is_null());
        let text = unsafe { CStr::from_ptr(j) }.to_str().unwrap().to_string();
        unsafe { occ_free(j) };
        assert!(text.contains("\"count\""), "{text}");
        unsafe { occ_close(s) };
    }

    #[test]
    fn hex_is_lowercase_and_unpadded() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }
}
