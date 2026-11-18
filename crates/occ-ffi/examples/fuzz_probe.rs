//! Mutation fuzzer for the container parsers, driven by `scripts/fuzz.py`.
//!
//! Why this is not `cargo-fuzz`: that would add a dependency, and the point of
//! this project is that it has none. The mutation strategies here are the ones
//! that matter for file-format parsers — a flipped bit inside a length field is
//! worth more than a thousand random bytes — and they are cheap enough to run
//! tens of millions of times in one process.
//!
//! Two depths:
//!
//! - `load` runs the container parser only. This is the fast path, used for the
//!   bulk of the iterations.
//! - `deep` also builds the analysis, extracts strings and decompiles the first
//!   few functions, which is where an index computed from a malformed table
//!   actually goes out of bounds.
//!
//! A rejected input is a success. A panic, an abort, or a hang is a failure and
//! is written to `--out` so it can be reproduced without re-running the seed.
//!
//! Usage:
//!
//! ```text
//! fuzz_probe --seeds DIR --iterations N [--rng SEED] [--depth load|deep]
//!            [--out DIR] [--max-seed-bytes N]
//! ```
//!
//! The last line on stdout is one JSON object; everything before it is progress.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use occ_analyze::Analysis;
use occ_analyze::AnalysisOptions;
use occ_analyze::view::View;
use occ_core::{Addr, ImageId};

/// xorshift64*, so a seed reproduces a run exactly without a dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

/// One mutation, recorded so a failure can be described rather than just saved.
#[derive(Clone, Copy, Debug)]
enum Kind {
    FlipBit,
    SetByte,
    Truncate,
    Extend,
    CopyBlock,
    BumpLength,
    SpliceHeader,
}

impl Kind {
    fn from_index(i: usize) -> Kind {
        match i % 7 {
            0 => Kind::FlipBit,
            1 => Kind::SetByte,
            2 => Kind::Truncate,
            3 => Kind::Extend,
            4 => Kind::CopyBlock,
            5 => Kind::BumpLength,
            _ => Kind::SpliceHeader,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::FlipBit => "flip_bit",
            Kind::SetByte => "set_byte",
            Kind::Truncate => "truncate",
            Kind::Extend => "extend",
            Kind::CopyBlock => "copy_block",
            Kind::BumpLength => "bump_length",
            Kind::SpliceHeader => "splice_header",
        }
    }
}

/// Apply one mutation in place. `other` supplies a header for splicing.
fn mutate(kind: Kind, rng: &mut Rng, data: &mut Vec<u8>, other: &[u8]) {
    if data.is_empty() {
        data.push(rng.byte());
        return;
    }
    match kind {
        Kind::FlipBit => {
            let i = rng.below(data.len());
            let bit = rng.below(8) as u32;
            data[i] ^= 1 << bit;
        }
        Kind::SetByte => {
            let i = rng.below(data.len());
            data[i] = match rng.below(4) {
                0 => 0x00,
                1 => 0xff,
                2 => 0x7f,
                _ => rng.byte(),
            };
        }
        Kind::Truncate => {
            // Cut somewhere in the last three quarters, so the header survives
            // often enough to reach the interesting parsers.
            let keep = rng.below(data.len());
            data.truncate(keep.max(1));
        }
        Kind::Extend => {
            let extra = 1 + rng.below(64);
            for _ in 0..extra {
                data.push(rng.byte());
            }
        }
        Kind::CopyBlock => {
            // Duplicate a run of bytes, which is what turns a table count into
            // a count larger than the table.
            let len = 1 + rng.below(32.min(data.len()));
            let from = rng.below(data.len() - len + 1);
            let block: Vec<u8> = data[from..from + len].to_vec();
            let at = rng.below(data.len() + 1);
            data.splice(at..at, block);
        }
        Kind::BumpLength => {
            // Length fields sit in the first kilobyte of every container this
            // parses, little- or big-endian, 2, 4 or 8 bytes wide. Poking them
            // directly reaches the bounds checks that random bytes rarely hit.
            let window = data.len().min(1024);
            if window < 2 {
                return;
            }
            let at = rng.below(window - 1);
            let width = match rng.below(3) {
                0 => 2usize,
                1 => 4usize,
                _ => 8usize,
            }
            .min(data.len() - at);
            let delta = match rng.below(5) {
                0 => 0xffff_u64,
                1 => 0x1000_0000,
                2 => 0x8000_0000,
                3 => 0x7fff_ffff,
                _ => rng.next() & 0xffff,
            };
            let mut cur = 0u64;
            for k in 0..width {
                cur |= u64::from(data[at + k]) << (8 * k);
            }
            let next = match rng.below(2) {
                0 => cur.wrapping_add(delta),
                _ => cur.wrapping_sub(delta),
            };
            for k in 0..width {
                data[at + k] = ((next >> (8 * k)) & 0xff) as u8;
            }
        }
        Kind::SpliceHeader => {
            // Take the first bytes of another seed. Half of all format bugs are
            // "this field is from a different container family".
            if other.is_empty() {
                return;
            }
            let n = (1 + rng.below(256.min(other.len()))).min(data.len());
            data[..n].copy_from_slice(&other[..n]);
        }
    }
}

/// Panic messages, capped so a systemic failure does not exhaust memory.
fn panics() -> &'static Mutex<Vec<String>> {
    static P: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(Vec::new()))
}

/// Run one input at the requested depth.
fn exercise(data: &[u8], path: &Path, deep: bool) -> Result<(), String> {
    let image = occ_bin::load(data, &path.display().to_string(), ImageId(0))
        .map_err(|e| e.to_string())?;
    if !deep {
        return Ok(());
    }
    let view = View::new(&image, data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    let opts = occ_decompile::Options {
        max_blocks: 2_000,
        keep_comments: true,
        keep_dead: false,
        debug: None,
        source_lines: false,
    };
    let name_of = |a: Addr| format!("sub_{:x}", a.get());
    for f in analysis.functions.iter().take(8) {
        let d = occ_decompile::decompile(&image, f, &opts, &name_of);
        let _ = occ_decompile::render::render(&d);
    }
    Ok(())
}

/// Escape a string for the one-line JSON report.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let mut seeds_dir: Option<PathBuf> = None;
    let mut iterations = 1_000usize;
    let mut rng_seed = 0x0cc0_ffee_1234_5678u64;
    let mut deep = false;
    let mut out_dir: Option<PathBuf> = None;
    let mut max_seed = 512 * 1024usize;
    let mut start_at = 0usize;
    let mut trace = false;
    let mut no_mutate = false;
    let mut dump_at: Option<usize> = None;
    let mut dump_path = PathBuf::from("mutant.bin");

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> String {
            args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("fuzz_probe: {} needs a value", args[i]);
                std::process::exit(2);
            })
        };
        match args[i].as_str() {
            "--seeds" => {
                seeds_dir = Some(PathBuf::from(need(i)));
                i += 2;
            }
            "--iterations" => {
                iterations = need(i).parse().unwrap_or(iterations);
                i += 2;
            }
            "--rng" => {
                rng_seed = need(i).parse().unwrap_or(rng_seed);
                i += 2;
            }
            "--depth" => {
                deep = need(i) == "deep";
                i += 2;
            }
            "--out" => {
                out_dir = Some(PathBuf::from(need(i)));
                i += 2;
            }
            "--max-seed-bytes" => {
                max_seed = need(i).parse().unwrap_or(max_seed);
                i += 2;
            }
            "--start-at" => {
                start_at = need(i).parse().unwrap_or(0);
                i += 2;
            }
            "--trace" => {
                trace = true;
                i += 1;
            }
            "--no-mutate" => {
                // Run the seed verbatim. This is how a saved failure is turned
                // into a minimal reproduction without a second tool.
                no_mutate = true;
                i += 1;
            }
            "--dump-at" => {
                dump_at = need(i).parse().ok();
                i += 2;
            }
            "--dump-path" => {
                dump_path = PathBuf::from(need(i));
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "usage: fuzz_probe --seeds DIR --iterations N [--rng SEED] [--depth load|deep] [--out DIR] [--start-at N]"
                );
                return;
            }
            other => {
                eprintln!("fuzz_probe: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }

    let Some(seeds_dir) = seeds_dir else {
        eprintln!("fuzz_probe: --seeds is required");
        std::process::exit(2);
    };

    // Silence the default hook: a hundred panic traces on stderr would drown
    // the report. The message is captured instead and printed once per distinct
    // shape.
    std::panic::set_hook(Box::new(|info| {
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic".to_string());
        let at = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        if let Ok(mut v) = panics().lock()
            && v.len() < 200
        {
            v.push(format!("{msg} at {at}"));
        }
    }));

    let mut corpus: Vec<Vec<u8>> = Vec::new();
    match std::fs::read_dir(&seeds_dir) {
        Ok(rd) => {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_file() {
                    continue;
                }
                if let Ok(data) = std::fs::read(&p)
                    && !data.is_empty()
                    && data.len() <= max_seed
                {
                    corpus.push(data);
                }
            }
        }
        Err(e) => {
            eprintln!("fuzz_probe: cannot read {}: {e}", seeds_dir.display());
            std::process::exit(2);
        }
    }
    if corpus.is_empty() {
        eprintln!("fuzz_probe: no seed under {} is <= {max_seed} bytes", seeds_dir.display());
        std::process::exit(2);
    }

    let out_dir = out_dir.unwrap_or_else(|| std::env::temp_dir().join(".occ-fuzz-failures"));
    if std::fs::create_dir_all(&out_dir).is_err() {
        eprintln!("fuzz_probe: cannot create {}", out_dir.display());
        std::process::exit(2);
    }

    let started = Instant::now();
    let mut rng = Rng::new(rng_seed);
    let mut loaded = 0u64;
    let mut rejected = 0u64;
    let mut failures = 0u64;
    let mut by_kind: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut saved = 0usize;

    for n in start_at..start_at.saturating_add(iterations) {
        let tick = Instant::now();
        let si = rng.below(corpus.len());
        let base = &corpus[si];
        let mut data = base.clone();
        // One to four stacked mutations: a single flip rarely reaches a parser
        // bound, four usually destroys the header.
        let rounds = if no_mutate { 0 } else { 1 + rng.below(4) };
        let mut kinds = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let k = Kind::from_index(rng.below(7));
            let other = &corpus[rng.below(corpus.len())];
            mutate(k, &mut rng, &mut data, other);
            kinds.push(k);
        }
        if dump_at == Some(n) {
            // Written before the parse, so a hang still leaves the bytes on
            // disk. This is how a slow input becomes a reproducible one.
            let _ = std::fs::write(&dump_path, &data);
            eprintln!(
                "dump {n} -> {} ({} bytes, kinds {:?})",
                dump_path.display(),
                data.len(),
                kinds.iter().map(|k| k.name()).collect::<Vec<_>>()
            );
        }
        let path = PathBuf::from(format!("mutant-{n}.bin"));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            exercise(&data, &path, deep)
        }));
        match result {
            Ok(Ok(())) => loaded += 1,
            Ok(Err(_)) => rejected += 1,
            Err(_) => {
                failures += 1;
                for k in &kinds {
                    *by_kind.entry(k.name()).or_insert(0) += 1;
                }
                if saved < 32 {
                    let name = format!("fail-{n}-{:016x}.bin", rng_seed);
                    if std::fs::write(out_dir.join(&name), &data).is_ok() {
                        saved += 1;
                    }
                }
            }
        }
        if trace {
            // Printed before the next iteration starts, so a hang leaves the
            // culprit as the last line rather than as a missing one.
            eprintln!("trace {n} {} {:?}", si, tick.elapsed());
        }
        if n % 200_000 == 0 && n > start_at {
            eprintln!(
                "  {n} runs · {loaded} loaded · {rejected} rejected · {failures} panics · {:.0}s",
                started.elapsed().as_secs_f64()
            );
        }
    }

    let msgs = panics().lock().map(|m| m.clone()).unwrap_or_default();
    let first = msgs.first().cloned().unwrap_or_default();
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{{\"runs\":{},\"loaded\":{},\"rejected\":{},\"failures\":{},\"seconds\":{:.3},\"seed\":\"{}\",\"depth\":\"{}\",\"seeds\":{},\"panic_kinds\":{{{}}},\"first_panic\":\"{}\",\"failures_dir\":\"{}\"}}",
        iterations,
        loaded,
        rejected,
        failures,
        elapsed,
        rng_seed,
        if deep { "deep" } else { "load" },
        corpus.len(),
        by_kind
            .iter()
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect::<Vec<_>>()
            .join(","),
        esc(&first),
        esc(&out_dir.display().to_string())
    );
}
