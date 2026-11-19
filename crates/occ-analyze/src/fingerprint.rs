//! Toolchain and language fingerprinting.
//!
//! The container parsers answer "what is in this file". This module answers
//! "what produced it", which changes what an analyst should expect: a Rust
//! binary has different panic paths and monomorphised names than a C one, an
//! MSVC binary inlines its CRT, and a Go binary carries its own scheduler.
//!
//! Every conclusion here is a [`Claim`] with the evidence that produced it.
//! The confidence is computed from that evidence, not asserted, so a caller can
//! see that "clang" rests on one `.comment` line while "gcc" had three
//! agreeing signals.

use occ_core::{
    Claim, CompilerFamily, Evidence, EvidenceKind, EvidenceSource, Fingerprint, Format, Handle,
    Image, LangGuess, SectionKind,
};

use crate::entropy::{HIGH_ENTROPY, per_section};
use crate::view::View;

/// Compute a fingerprint, replacing the placeholder the parser installed.
pub fn fingerprint(image: &Image, view: &View) -> Fingerprint {
    // Start from what the container parser established (linkage, and for
    // Mach-O whether symbols were stripped) and refine it. Starting from
    // `unknown` discarded those answers and made the same file report `pie`
    // from its header and `linkage?` from its fingerprint.
    let mut fp = image.fingerprint.clone();
    let handle = Handle::image(image.id.0);

    let comment = read_comment(view);
    let symbols = symbol_blob(image);
    let imports = import_blob(image);

    // Runtime toolchains that own the entry point and the calling convention.
    // These are decided before the generic compiler heuristics because a
    // VB6 or CLR image contains MSVC-compiled runtime code inside it: the
    // question "which compiler built this file" would answer `msvc` for the
    // runtime and miss what actually produced the executable around it.
    let msvb = runtime_import(image, &["msvbvm50", "msvbvm60"])
        || image.imports.iter().any(|i| i.name.starts_with("__vba"));
    let dotnet = runtime_import(image, &["mscoree"])
        || image
            .imports
            .iter()
            .any(|i| matches!(i.name.as_str(), "_CorExeMain" | "_CorDllMain" | "_CorMain"));
    // Borland's RTL emits a section set no Microsoft link produces, and the
    // compiler's own runtime strings name the product directly. Both mean
    // "Borland toolchain"; only Pascal symbol mangling means "Delphi" rather
    // than C++ Builder.
    let borland_layout = borland_sections(image);
    let borland_marker = borland_product(view);
    // A stripped Delphi build loses the Pascal-mangled symbols, but it keeps
    // the VCL unit names the RTL registers at startup ("Classes", "Forms",
    // "Controls", ...). C++ Builder carries the same units *and* its own
    // product string, so the string is the tie-breaker: without it, a
    // Borland layout plus several VCL unit names is Delphi, not C++.
    let vcl_units = borland_layout && borland_marker.is_none() && vcl_unit_names(view.file());
    let delphi = delphi_symbols(&symbols)
        || matches!(borland_marker, Some((CompilerFamily::Delphi, _)))
        || vcl_units;
    // CODE/DATA/BSS is not Borland's alone: TASM and hand-written Win32
    // assembly produce the same section names, and so do a few linkers for
    // other toolchains. The layout is a hint that needs one corroborating
    // Borland fact — its product string, its RTL symbols, or Pascal/Delphi
    // mangling. Claiming `borland` from sections alone labelled a TASM
    // crackme as C++ Builder.
    // A full Borland RTL link also emits BSS and a TLS directory; TASM does
    // not, which is the difference between a C++ Builder binary and the
    // SECTION names happening to match.
    let borland_full = borland_layout
        && image.section_by_name("BSS").is_some()
        && (image.section_by_name("TLS").is_some() || image.section_by_name(".tls").is_some());
    let borland_rt = borland_full
        || symbols.contains("@System@@")
        || symbols.contains("Borland")
        || symbols.contains("Delphi")
        || delphi;
    let borland = (borland_layout && borland_rt) || borland_marker.is_some() || delphi;
    let msvc_rt = msvc_runtime(image);
    let msvc_link = rich_header(view);

    // Compiler family: strongest signal first.
    let mut claim = Claim::new(handle.clone(), "compiler family has not been determined");
    if let Some(c) = &comment {
        if c.contains("GCC:") {
            claim = claim.with(Evidence::support(
                EvidenceKind::Bytes,
                EvidenceSource::Loader,
                format!("comment section records '{}'", first_line(c)),
                0.7,
            ));
        }
        if c.contains("clang version") {
            claim = claim.with(Evidence::support(
                EvidenceKind::Bytes,
                EvidenceSource::Loader,
                format!("comment section records '{}'", first_line(c)),
                0.75,
            ));
        }
        if c.contains("rustc version") || c.contains("rustc ") {
            claim = claim.with(Evidence::support(
                EvidenceKind::Bytes,
                EvidenceSource::Loader,
                format!("comment section records '{}'", first_line(c)),
                0.8,
            ));
        }
    }
    if symbols.contains("rust_eh_personality") || symbols.contains("_ZN") && symbols.contains("17h")
    {
        claim = claim.with(Evidence::support(
            EvidenceKind::Symbol,
            EvidenceSource::Loader,
            "symbol names carry Rust's legacy hash suffix or Rust personality routine",
            0.6,
        ));
    }
    if symbols.contains("__C_specific_handler") || imports.contains("__C_specific_handler") {
        claim = claim.with(Evidence::support(
            EvidenceKind::ImportCall,
            EvidenceSource::Loader,
            "the import table names `__C_specific_handler`; MSVC and MinGW SEH both do",
            0.5,
        ));
    }
    if symbols.contains("$s") || symbols.contains("__swift") {
        claim = claim.with(Evidence::support(
            EvidenceKind::Symbol,
            EvidenceSource::Loader,
            "symbol names use the Swift mangling prefix",
            0.6,
        ));
    }
    if image.section_by_name(".gopclntab").is_some()
        || image.section_by_name(".go.buildinfo").is_some()
    {
        claim = claim.with(Evidence::support(
            EvidenceKind::FormatMeta,
            EvidenceSource::Loader,
            "Go runtime tables are present",
            0.9,
        ));
    }
    if msvb {
        claim = claim.with(Evidence::support(
            EvidenceKind::ImportCall,
            EvidenceSource::Loader,
            "the import table names the Visual Basic 6 runtime (MSVBVM)",
            0.95,
        ));
    }
    if dotnet {
        claim = claim.with(Evidence::support(
            EvidenceKind::ImportCall,
            EvidenceSource::Loader,
            "the entry point is the CLR stub (_CorExeMain/mscoree)",
            0.95,
        ));
    }
    if borland && borland_layout {
        claim = claim.with(Evidence::support(
            EvidenceKind::FormatMeta,
            EvidenceSource::Loader,
            "Borland RTL section layout (CODE/DATA/BSS, no .text)",
            0.7,
        ));
    }
    // MinGW is GCC: it links libgcc and registers its frame information, and
    // it also links the UCRT and imports `__C_specific_handler` for SEH. Those
    // two facts are shared with MSVC, so a runtime import alone must not
    // outrank a libgcc reference.
    let libs = import_libraries(image);
    let mingw = symbols.contains("__register_frame_info")
        || symbols.contains("__deregister_frame_info")
        || libs
            .iter()
            .any(|l| l.contains("libgcc") || l.contains("libstdc++"))
        || contains_ascii_ci(view.file(), b"MinGW-w64")
        || contains_ascii_ci(view.file(), b"GCC: (MinGW")
        || comment.as_deref().is_some_and(|c| c.contains("MinGW"));
    if mingw {
        claim = claim.with(Evidence::support(
            EvidenceKind::Symbol,
            EvidenceSource::Loader,
            "the file names libgcc/MinGW-w64: the toolchain is GCC, which also \
             links the UCRT and imports the SEH handler",
            0.6,
        ));
    }
    let mut marker_language: Option<LangGuess> = None;
    if let Some((family, marker)) = borland_marker {
        claim = claim.with(Evidence::support(
            EvidenceKind::Bytes,
            EvidenceSource::Loader,
            format!("the file carries the compiler's own product string '{marker}'"),
            0.7,
        ));
        // A product string that names the C++ product settles the language;
        // Delphi's own string does the same for Pascal.
        marker_language = match family {
            CompilerFamily::Borland => Some(LangGuess::Cpp),
            _ => Some(LangGuess::Pascal),
        };
    }
    if msvc_link {
        claim = claim.with(Evidence::support(
            EvidenceKind::Bytes,
            EvidenceSource::Loader,
            "the DOS stub carries an MSVC linker Rich header (DanS/Rich)",
            0.6,
        ));
    }
    if let Some(rt) = msvc_rt {
        claim = claim.with(Evidence::support(
            EvidenceKind::ImportCall,
            EvidenceSource::Loader,
            format!("the import table links the MSVC runtime ({rt})"),
            0.6,
        ));
    }
    if vcl_units {
        claim = claim.with(Evidence::support(
            EvidenceKind::FormatMeta,
            EvidenceSource::Loader,
            "the file registers the VCL unit names Classes, Forms and Controls; \
             with no C++ Builder product string this is Delphi",
            0.6,
        ));
    }
    if delphi {
        claim = claim.with(Evidence::support(
            EvidenceKind::FormatMeta,
            EvidenceSource::Loader,
            "CODE/DATA/BSS/TLS section set and VCL symbol names are Delphi's layout",
            0.8,
        ));
    }
    claim.recompute();
    fp.compiler = if msvb {
        CompilerFamily::Msvb
    } else if dotnet {
        CompilerFamily::DotNet
    } else if delphi {
        CompilerFamily::Delphi
    } else if borland {
        CompilerFamily::Borland
    } else if claim.confidence >= 0.5 {
        if comment
            .as_deref()
            .is_some_and(|c| c.contains("clang version"))
        {
            CompilerFamily::Clang
        } else if comment.as_deref().is_some_and(|c| c.contains("rustc")) {
            CompilerFamily::Rustc
        } else if image.section_by_name(".gopclntab").is_some() {
            CompilerFamily::Go
        } else if symbols.contains("$s") {
            CompilerFamily::Swift
        } else if mingw && !msvc_link {
            CompilerFamily::Gcc
        } else if comment.as_deref().is_some_and(|c| c.contains("GCC:")) {
            CompilerFamily::Gcc
        } else if msvc_link || msvc_rt.is_some() {
            CompilerFamily::Msvc
        } else {
            CompilerFamily::Unknown
        }
    } else {
        CompilerFamily::Unknown
    };
    // The statement must say what was found, not what was asked. Leaving the
    // placeholder in place made a confident VB6 reading print "compiler family
    // has not been determined" next to the evidence that had settled it.
    claim.statement = match fp.compiler {
        CompilerFamily::Unknown => "compiler family has not been determined".to_string(),
        other => format!("compiler family is {}", other.label()),
    };
    fp.compiler_claim = claim;

    // Language.
    fp.language = match fp.compiler {
        CompilerFamily::Rustc => LangGuess::Rust,
        CompilerFamily::Go => LangGuess::Go,
        CompilerFamily::Swift => LangGuess::Swift,
        CompilerFamily::Zig => LangGuess::Zig,
        CompilerFamily::Msvb => LangGuess::VisualBasic,
        CompilerFamily::Delphi => LangGuess::Pascal,
        // Delphi vs C++ Builder is not decided by the section layout, so the
        // language stays unknown unless a product string or Pascal mangling
        // settled it above.
        CompilerFamily::Borland => marker_language.unwrap_or(LangGuess::Unknown),
        // A CLR image can be C#, VB.NET, F# or PowerShell; the container does
        // not say which. Saying `csharp` here would be a guess dressed as a
        // reading, so the language stays unknown and the compiler family
        // carries the finding.
        CompilerFamily::DotNet => LangGuess::Unknown,
        _ => {
            if symbols.contains("_Z") && !symbols.contains("rust_eh_personality") {
                LangGuess::Cpp
            } else if symbols.is_empty() {
                LangGuess::Unknown
            } else {
                LangGuess::C
            }
        }
    };

    // Linkage is the container's answer, set by the parser from `e_type`,
    // `IMAGE_FILE_DLL`, or the Mach-O file type. Re-deriving it here from
    // section names gave two answers for the same file — the header said `pie`
    // while this function said `exec` — so it is left alone.

    // Stripped: no named symbols at all.
    fp.stripped = !image.symbols.iter().any(|s| !s.name.is_empty());

    fp.has_go_buildinfo = image.section_by_name(".go.buildinfo").is_some()
        || image.section_by_name(".gopclntab").is_some();
    fp.has_dwarf = image
        .sections
        .iter()
        .any(|s| s.name.starts_with(".debug_") || s.name.starts_with(".zdebug_"));
    fp.section_entropy = per_section(view)
        .into_iter()
        .map(|e| (e.name, e.entropy))
        .collect();
    fp.runtime = runtime(image);

    // Packer suspicion: high entropy in a code section, or the section names
    // packers leave behind.
    let mut pack = Claim::new(handle, "the image may be packed");
    for (name, value) in &fp.section_entropy {
        let code = image
            .section_by_name(name)
            .map(|s| s.kind == SectionKind::Code)
            .unwrap_or(false);
        if code && *value >= HIGH_ENTROPY {
            pack = pack.with(Evidence::support(
                EvidenceKind::Entropy,
                EvidenceSource::Heuristic,
                format!("executable section {name} has entropy {value:.2}"),
                0.5,
            ));
        }
    }
    let packed_names = [
        "UPX0",
        "UPX1",
        "UPX2",
        ".aspack",
        ".adata",
        ".packed",
        "MPRESS1",
        "MPRESS2",
        ".nsp0",
        ".nsp1",
        ".petite",
        ".themida",
        ".vmp0",
        ".vmp1",
        ".winlice",
        ".perplex",
        ".kkrunchy",
        "PEC2",
        "ASPack",
    ];
    for n in packed_names {
        if image.section_by_name(n).is_some() {
            pack = pack.with(Evidence::support(
                EvidenceKind::FormatMeta,
                EvidenceSource::Loader,
                format!("section {n} is a known packer signature"),
                0.8,
            ));
        }
    }

    // A writable *and* executable section is the decompression stub's own
    // output buffer. Optimised compilers do not emit one: W^X is the norm on
    // every platform this tool reads. Two or more is a wrapper's fingerprint
    // regardless of which packer it is, which is what catches the builds that
    // rename their sections to values no packer name list contains.
    let rwx: Vec<&str> = image
        .sections
        .iter()
        .filter(|s| s.perm.write() && s.perm.exec())
        .map(|s| s.name.as_str())
        .collect();
    if !rwx.is_empty() {
        pack = pack.with(Evidence::support(
            EvidenceKind::FormatMeta,
            EvidenceSource::Heuristic,
            format!(
                "{} writable+executable section(s) ({}): compiled code is not W^X",
                rwx.len(),
                rwx.join(", ")
            ),
            if rwx.len() >= 2 { 0.7 } else { 0.45 },
        ));
    }

    // A stub that only needs GetModuleHandleA/LoadLibraryA and has almost no
    // imports is unpacking the rest of itself at run time. This inference
    // needs a container that *can* express imports: a COM image has no import
    // table at all, so "zero imports" there is a property of the format and
    // not evidence about the code.
    let has_import_table = matches!(image.format, Format::Elf | Format::Pe | Format::MachO);
    if has_import_table
        && image.imports.len() <= 2
        && image.sections.iter().any(|s| s.kind == SectionKind::Code)
    {
        pack = pack.with(Evidence::support(
            EvidenceKind::ImportCall,
            EvidenceSource::Heuristic,
            format!(
                "only {} import(s) for a native executable",
                image.imports.len()
            ),
            0.3,
        ));
    }
    if has_import_table && image.imports.len() <= 3 && image.section_by_name(".text").is_some() {
        pack = pack.with(Evidence::support(
            EvidenceKind::ImportCall,
            EvidenceSource::Heuristic,
            "very few imports for an executable",
            0.2,
        ));
    }
    pack.recompute();
    fp.packer_suspicion = (pack.confidence >= 0.5).then_some(pack);

    fp
}

fn runtime(image: &Image) -> Option<String> {
    if let Some(i) = &image.interpreter {
        if i.contains("ld-musl") {
            return Some("musl".into());
        }
        if i.contains("ld-linux") || i.contains("ld64") {
            return Some("glibc".into());
        }
        return Some(i.clone());
    }
    // The libraries are the answer for a PE. The symbol-name fallback below
    // must not run for one: `malloc` and `free` are imported by the UCRT
    // forwarders too, and answering `glibc` for a Windows binary because
    // those two names appear is a wrong answer, not a weak one.
    let libs = import_libraries(image);
    if libs
        .iter()
        .any(|l| l.starts_with("api-ms-win-crt-") || l == "ucrtbase")
    {
        return Some("ucrt".into());
    }
    // Name the C runtime rather than whichever of the pair sorts first:
    // `msvcp90` is the C++ library that sits beside `msvcr90`, and reporting
    // the C++ one as "the runtime" hides which ABI the binary links.
    if let Some(v) = libs.iter().find(|l| l.starts_with("msvcr")) {
        return Some(v.clone());
    }
    if let Some(v) = libs.iter().find(|l| l.starts_with("msvcp")) {
        return Some(v.clone());
    }
    if image.format == Format::Elf {
        let names: Vec<&str> = image.imports.iter().map(|i| i.name.as_str()).collect();
        if names.iter().any(|n| n.contains("ucrtbase")) {
            return Some("ucrt".into());
        }
        if names.iter().any(|n| n.contains("msvcrt")) {
            return Some("msvcrt".into());
        }
        if names.iter().any(|n| {
            matches!(
                *n,
                "__libc_start_main" | "__errno_location" | "malloc" | "free" | "memcpy"
            )
        }) {
            return Some("glibc".into());
        }
    }
    if image.section_by_name(".gopclntab").is_some() {
        return Some("go".into());
    }
    None
}

/// Importing libraries, lower-cased and without the extension.
fn import_libraries(image: &Image) -> Vec<String> {
    let mut v: Vec<String> = image
        .imports
        .iter()
        .filter_map(|i| i.library.as_deref())
        .map(|l| {
            let l = l.to_ascii_lowercase();
            l.split('.').next().unwrap_or(&l).to_string()
        })
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// The MSVC runtime family, if the import table links one.
///
/// `msvcr90`/`msvcp90` and the UCRT forwarders are only ever produced by MSVC.
/// A plain `msvcrt.dll` is not evidence: MinGW links it too, which is why it
/// is not in this match.
fn msvc_runtime(image: &Image) -> Option<&'static str> {
    let libs = import_libraries(image);
    if libs
        .iter()
        .any(|l| l.starts_with("api-ms-win-crt-") || l == "ucrtbase")
    {
        return Some("ucrt");
    }
    // `msvcr90`/`msvcr100` are versioned MSVC runtimes; `msvcrt` is the
    // system DLL that MinGW also links, so it must not be matched here.
    if libs
        .iter()
        .any(|l| (l.starts_with("msvcr") || l.starts_with("msvcp")) && !l.starts_with("msvcrt"))
    {
        return Some("msvcr");
    }
    None
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

/// True when something is imported from one of `libs` (case-insensitive,
/// extension-insensitive: PE records `MSVBVM60.dll`, some writers use
/// `MSVBVM60.DLL` or no extension at all).
fn runtime_import(image: &Image, libs: &[&str]) -> bool {
    image.imports.iter().any(|i| {
        i.library.as_deref().is_some_and(|l| {
            let l = l.to_ascii_lowercase();
            let stem = l.split('.').next().unwrap_or(&l);
            libs.contains(&stem)
        })
    })
}

/// Delphi and C++ Builder emit a fixed four-section layout: `CODE`, `DATA`,
/// `BSS`, `TLS` (with the dot missing), plus `.idata`/`.didata`. MSVC's
/// equivalent is `.text`/`.data`/`.bss`/`.tls`, so the combination is a
/// reliable discriminator rather than a loose "a section called DATA".
fn borland_sections(image: &Image) -> bool {
    let has = |n: &str| {
        image.section_by_name(n).is_some() || image.section_by_name(&format!(".{n}")).is_some()
    };
    // `CODE` and `DATA` alone are already something no MSVC link emits;
    // `BSS` is normally there but a small binary can merge it away, and
    // requiring it missed a Borland-built sample whose sections were
    // `CODE, DATA, .idata, .reloc, .rsrc`. `.text` still has to be absent
    // for the relaxed form, so an ordinary MSVC image cannot match.
    let no_text = image.section_by_name(".text").is_none();
    has("CODE") && has("DATA") && (has("BSS") || no_text)
}

/// Unstripped Delphi binaries keep VCL/RTL symbols, which are published as
/// `@System@@...`, `@Classes@...` or the older `System@@` spelling.
fn delphi_symbols(symbols: &str) -> bool {
    symbols.lines().any(|l| {
        l.starts_with("@System@") || l.starts_with("@Classes@") || l.starts_with("System@@")
    })
}

/// Unit names a VCL application registers at startup. A stripped Delphi
/// binary has no `@System@` symbols left, but these strings remain because
/// the RTL looks them up by name.
fn vcl_unit_names(data: &[u8]) -> bool {
    const UNITS: &[&str] = &[
        "Classes",
        "Forms",
        "Controls",
        "StdCtrls",
        "SysUtils",
        "Graphics",
    ];
    let mut found = 0;
    for unit in UNITS {
        if data
            .windows(unit.len())
            .any(|window| window == unit.as_bytes())
        {
            found += 1;
        }
    }
    found >= 2
}

/// The compiler's own product string, when the file carries one.
///
/// These are copyright notices baked into the runtime, so they name the
/// toolchain rather than describe the input. A file can contain the text by
/// accident (an embedded resource, a decoy in a crackme), which is why the
/// match is on the exact product wording and why the evidence carries a
/// weight below the structural signals.
fn borland_product(view: &View) -> Option<(CompilerFamily, &'static str)> {
    let file = view.file();
    if contains_ascii_ci(file, b"Borland C++") {
        return Some((CompilerFamily::Borland, "Borland C++"));
    }
    if contains_ascii_ci(file, b"Borland Delphi") {
        return Some((CompilerFamily::Delphi, "Borland Delphi"));
    }
    None
}

/// Case-insensitive substring search over raw bytes.
fn contains_ascii_ci(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack.len() >= needle.len()
        && haystack
            .windows(needle.len())
            .any(|w| w.eq_ignore_ascii_case(needle))
}

/// True when the DOS stub carries an MSVC linker `Rich` header.
///
/// `link.exe` writes a `DanS` marker followed by product ids and a `Rich`
/// checksum into the stub. Compilers that do not use Microsoft's linker
/// (Borland, MinGW's ld, Delphi) do not emit it, so its presence dates the
/// link step to MSVC — not necessarily the language, which is why the weight
/// is 0.6 and the runtime/import evidence still outranks it.
fn rich_header(view: &View) -> bool {
    let file = view.file();
    if view.image().format != Format::Pe {
        return false;
    }
    let stub = &file[..file.len().min(0x400)];
    let Some(pos) = stub.windows(4).position(|w| w == b"Rich") else {
        return false;
    };
    stub[..pos].windows(4).any(|w| w == b"DanS")
}

fn read_comment(view: &View) -> Option<String> {
    let s = view.image().section_by_name(".comment")?;
    if s.file_size == 0 {
        return None;
    }
    let bytes = view.bytes(s.addr, s.file_size as usize)?;
    Some(String::from_utf8_lossy(bytes).into_owned())
}

fn symbol_blob(image: &Image) -> String {
    let mut s = String::with_capacity(image.symbols.len() * 24);
    for sym in &image.symbols {
        s.push_str(&sym.name);
        s.push('\n');
    }
    s
}

fn import_blob(image: &Image) -> String {
    let mut s = String::with_capacity(image.imports.len() * 16);
    for imp in &image.imports {
        s.push_str(&imp.name);
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::image_with_sections;
    use occ_core::{Addr, Perm, Section, SectionKind};

    fn section(name: &str, off: u64, size: u64) -> Section {
        Section {
            index: 1,
            name: name.into(),
            addr: Addr(0x1000 + off),
            file_offset: off,
            size,
            file_size: size,
            kind: SectionKind::Other,
            perm: Perm::R,
        }
    }

    #[test]
    fn gcc_comment_is_recognised() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GCC: (Ubuntu 13.2.0) 13.2.0\0");
        let img = image_with_sections(vec![section(".comment", 0, data.len() as u64)]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.compiler, CompilerFamily::Gcc);
        assert!(fp.compiler_claim.confidence > 0.6);
    }

    #[test]
    fn rust_comment_wins_over_symbol_guesses() {
        let mut data = Vec::new();
        data.extend_from_slice(b"rustc version 1.80.0\0");
        let img = image_with_sections(vec![section(".comment", 0, data.len() as u64)]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.compiler, CompilerFamily::Rustc);
        assert_eq!(fp.language, LangGuess::Rust);
    }

    #[test]
    fn missing_comment_leaves_compiler_unknown() {
        let data = vec![0u8; 16];
        let img = image_with_sections(vec![section(".rodata", 0, 16)]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.compiler, CompilerFamily::Unknown);
        assert_eq!(fp.compiler_claim.confidence, 0.0);
    }

    fn with_imports(mut img: Image, imports: &[(&str, &str)]) -> Image {
        let mut slot = 0x3000;
        for (lib, name) in imports {
            img.imports.push(occ_core::Import {
                name: (*name).into(),
                library: Some((*lib).into()),
                ordinal: None,
                slot: Some(Addr(slot)),
                kind: occ_core::ImportKind::Normal,
            });
            slot += 8;
        }
        img
    }

    #[test]
    fn vb6_runtime_import_beats_the_msvc_code_inside_it() {
        let data = vec![0u8; 16];
        let base = image_with_sections(vec![section(".text", 0, 16)]);
        let img = with_imports(base, &[("MSVBVM60.dll", "ThunRTMain")]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.compiler, CompilerFamily::Msvb);
        assert_eq!(fp.language, LangGuess::VisualBasic);
        assert!(fp.compiler_claim.confidence >= 0.9);
    }

    #[test]
    fn clr_stub_is_reported_as_dotnet() {
        let data = vec![0u8; 16];
        let base = image_with_sections(vec![section(".text", 0, 16)]);
        let img = with_imports(base, &[("mscoree.dll", "_CorExeMain")]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.compiler, CompilerFamily::DotNet);
        // C#, VB.NET, F# all produce this image; the container does not say
        // which, so the language must stay unknown rather than guess C#.
        assert_eq!(fp.language, LangGuess::Unknown);
    }

    #[test]
    fn the_borland_layout_alone_is_not_enough() {
        // Delphi and C++ Builder share this layout, and so do TASM and some
        // hand-written assembly linkers. Without a Borland fact the layout
        // says nothing about the compiler.
        let data = vec![0u8; 16];
        let img = image_with_sections(vec![
            section("CODE", 0, 8),
            section("DATA", 8, 4),
            section("BSS", 12, 4),
        ]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.compiler, CompilerFamily::Unknown);
        assert_eq!(fp.language, LangGuess::Unknown);

        // Pascal symbol mangling is the evidence that separates Delphi from
        // C++ Builder, and it is what the language answer must rest on.
        let mut img2 = image_with_sections(vec![section("CODE", 0, 8)]);
        img2.symbols.push(occ_core::Symbol {
            name: "@System@@LoadString$".into(),
            addr: Addr(0x1000),
            size: 0,
            kind: occ_core::SymbolKind::Function,
            binding: occ_core::SymbolBinding::Global,
            section: None,
            defined: true,
            demangled: false,
        });
        let v2 = View::new(&img2, &data);
        let fp2 = fingerprint(&img2, &v2);
        assert_eq!(fp2.compiler, CompilerFamily::Delphi);
        assert_eq!(fp2.language, LangGuess::Pascal);
    }

    #[test]
    fn vcl_unit_names_identify_delphi_without_symbols() {
        // A stripped Delphi build keeps no Pascal-mangled symbols, but the
        // RTL still registers its unit names. With a Borland section layout
        // and no C++ Builder product string, that is Delphi.
        let data = b"Classes\0Forms\0Controls\0TForm1\0".to_vec();
        let len = data.len() as u64;
        let img = image_with_sections(vec![
            section("CODE", 0, len),
            section("DATA", len, 0),
            section("BSS", len, 0),
            section(".tls", len, 0),
        ]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.compiler, CompilerFamily::Delphi);
        assert_eq!(fp.language, LangGuess::Pascal);
    }

    #[test]
    fn ucrt_forwarders_are_not_mistaken_for_glibc() {
        let data = vec![0u8; 16];
        let base = image_with_sections(vec![section(".text", 0, 16)]);
        let img = with_imports(
            base,
            &[
                ("api-ms-win-crt-heap-l1-1-0.dll", "malloc"),
                ("api-ms-win-crt-runtime-l1-1-0.dll", "exit"),
            ],
        );
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.runtime.as_deref(), Some("ucrt"));
        assert_eq!(fp.compiler, CompilerFamily::Msvc);
    }

    #[test]
    fn versioned_msvcr_identifies_msvc_but_plain_msvcrt_does_not() {
        let data = vec![0u8; 16];
        let base = image_with_sections(vec![section(".text", 0, 16)]);
        let img = with_imports(
            base,
            &[
                ("MSVCR90.dll", "printf"),
                ("MSVCP90.dll", "??0exception@@QAE@XZ"),
            ],
        );
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        assert_eq!(fp.compiler, CompilerFamily::Msvc);
        assert_eq!(fp.runtime.as_deref(), Some("msvcr90"));

        // MinGW links plain msvcrt.dll: a compiler family is not derivable
        // from that alone, and reporting `msvc` would be a guess.
        let img2 = with_imports(
            image_with_sections(vec![section(".text", 0, 16)]),
            &[("msvcrt.dll", "printf")],
        );
        let v2 = View::new(&img2, &data);
        let fp2 = fingerprint(&img2, &v2);
        assert_eq!(fp2.compiler, CompilerFamily::Unknown);
    }

    #[test]
    fn borland_layout_with_delphi_mangling_names_delphi() {
        let data = vec![0u8; 16];
        let mut img = image_with_sections(vec![section("CODE", 0, 8), section("DATA", 8, 8)]);
        img.symbols.push(occ_core::Symbol {
            name: "@System@@HandleFinally$qqrv".into(),
            addr: Addr(0x1000),
            size: 0,
            kind: occ_core::SymbolKind::Function,
            binding: occ_core::SymbolBinding::Global,
            section: None,
            defined: true,
            demangled: false,
        });
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        // Pascal RTL mangling is the fact that separates Delphi from C++
        // Builder; the layout plus that mangle is a Delphi reading.
        assert_eq!(fp.compiler, CompilerFamily::Delphi);
        assert_eq!(fp.language, LangGuess::Pascal);
    }

    #[test]
    fn writable_executable_sections_raise_packer_suspicion() {
        // Two RWX code sections: no compiler emits this, a decompression stub
        // does. The section names are deliberately not on any known-packer
        // list, which is the case that used to pass unnoticed.
        let mut data = vec![0u8; 64];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        let img = image_with_sections(vec![
            Section {
                index: 1,
                name: ".rwx0".into(),
                addr: Addr(0x1000),
                file_offset: 0,
                size: 32,
                file_size: 32,
                kind: SectionKind::Code,
                perm: Perm::R.union(Perm::W).union(Perm::X),
            },
            Section {
                index: 2,
                name: ".rwx1".into(),
                addr: Addr(0x1020),
                file_offset: 32,
                size: 32,
                file_size: 32,
                kind: SectionKind::Code,
                perm: Perm::R.union(Perm::W).union(Perm::X),
            },
        ]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
        let pack = fp.packer_suspicion.expect("packer should be suspected");
        assert!(pack.confidence >= 0.5, "confidence {}", pack.confidence);
    }
}
