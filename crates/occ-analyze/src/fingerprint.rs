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
    Claim, CompilerFamily, Evidence, EvidenceKind, EvidenceSource, Fingerprint, Handle, Image,
    LangGuess, SectionKind,
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
    let delphi = delphi_sections(image) || delphi_symbols(&symbols);

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
            "MSVC exception handler is imported",
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
    if image.section_by_name(".gopclntab").is_some() || image.section_by_name(".go.buildinfo").is_some()
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
    } else if claim.confidence >= 0.5 {
        if comment.as_deref().is_some_and(|c| c.contains("clang version")) {
            CompilerFamily::Clang
        } else if comment.as_deref().is_some_and(|c| c.contains("rustc")) {
            CompilerFamily::Rustc
        } else if image.section_by_name(".gopclntab").is_some() {
            CompilerFamily::Go
        } else if symbols.contains("$s") {
            CompilerFamily::Swift
        } else if symbols.contains("__C_specific_handler") {
            CompilerFamily::Msvc
        } else if comment.as_deref().is_some_and(|c| c.contains("GCC:")) {
            CompilerFamily::Gcc
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

    fp.has_go_buildinfo =
        image.section_by_name(".go.buildinfo").is_some() || image.section_by_name(".gopclntab").is_some();
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
        "UPX0", "UPX1", "UPX2", ".aspack", ".adata", ".packed", "MPRESS1", "MPRESS2", ".nsp0",
        ".nsp1", ".petite", ".themida", ".vmp0", ".vmp1", ".winlice", ".perplex", ".kkrunchy",
        "PEC2", "ASPack",
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
    // rename their sections (`.kaine0`, `.kaine1`) to defeat the name list.
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
    // imports is unpacking the rest of itself at run time.
    if image.imports.len() <= 2 && image.sections.iter().any(|s| s.kind == SectionKind::Code) {
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
    if image.imports.len() <= 3 && image.section_by_name(".text").is_some() {
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
    if image.section_by_name(".gopclntab").is_some() {
        return Some("go".into());
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
fn delphi_sections(image: &Image) -> bool {
    let has = |n: &str| image.section_by_name(n).is_some() || image.section_by_name(&format!(".{n}")).is_some();
    has("CODE") && has("DATA") && has("BSS")
}

/// Unstripped Delphi binaries keep VCL/RTL symbols, which are published as
/// `@System@@...`, `@Classes@...` or the older `System@@` spelling.
fn delphi_symbols(symbols: &str) -> bool {
    symbols.lines().any(|l| {
        l.starts_with("@System@") || l.starts_with("@Classes@") || l.starts_with("System@@")
    })
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
    fn delphi_section_layout_is_detected() {
        let data = vec![0u8; 16];
        let img = image_with_sections(vec![
            section("CODE", 0, 8),
            section("DATA", 8, 4),
            section("BSS", 12, 4),
        ]);
        let v = View::new(&img, &data);
        let fp = fingerprint(&img, &v);
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
                name: ".kaine0".into(),
                addr: Addr(0x1000),
                file_offset: 0,
                size: 32,
                file_size: 32,
                kind: SectionKind::Code,
                perm: Perm::R.union(Perm::W).union(Perm::X),
            },
            Section {
                index: 2,
                name: ".pack".into(),
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
