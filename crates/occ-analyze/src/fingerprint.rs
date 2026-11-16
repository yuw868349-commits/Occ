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
    claim.recompute();
    fp.compiler = if claim.confidence >= 0.5 {
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
    fp.compiler_claim = claim;

    // Language.
    fp.language = match fp.compiler {
        CompilerFamily::Rustc => LangGuess::Rust,
        CompilerFamily::Go => LangGuess::Go,
        CompilerFamily::Swift => LangGuess::Swift,
        CompilerFamily::Zig => LangGuess::Zig,
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
    let packed_names = ["UPX0", "UPX1", ".aspack", ".adata", ".packed", "MPRESS1"];
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
}
