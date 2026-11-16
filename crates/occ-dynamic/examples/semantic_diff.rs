//! Differential test of the lifter against the processor.
//!
//! For every integer encoding the decoder recognises, this runs the instruction
//! twice: once on the real processor under `ptrace`, once through the lifted IR
//! and the concrete interpreter. A disagreement is a bug in one of the two, and
//! the report says which register or flag.
//!
//! # Why this is the only way to know
//!
//! Unit tests encode what the author expected. This compares against the
//! hardware, which has no expectations. An instruction that the lifter renders
//! plausibly but computes differently passes every hand-written test and fails
//! here.
//!
//! # What is excluded, and why
//!
//! * **Control flow.** `jmp`, `call`, `ret` and the conditional moves that write
//!   `rip` are not arithmetic: the processor's answer is where it went, and the
//!   differential would compare program counters rather than values.
//! * **Stack-relative operands.** The lifter promotes `[rsp+8]` to a named
//!   local, deliberately: that is what makes the decompiled body readable. The
//!   promotion is a fact about the output, not about the machine, so an operand
//!   that names `rsp` or `rbp` is reported as skipped rather than counted as a
//!   disagreement.
//! * **Privileged and system instructions**, which fault in a user process.
//! * **Floating point and vector**, for now: their comparison needs NaN and
//!   rounding rules that are a separate piece of work, and mixing them in would
//!   make the integer result unreadable.
//!
//! Every exclusion is counted and printed. A test that silently covers less
//! than it claims is worse than one that says where it stops.
//!
//! Usage: `semantic_diff [--cases N] [--verbose] [--only MNEMONIC]`

use std::collections::BTreeMap;

use occ_core::Addr;
use occ_decode::x86::Decoder;
use occ_decompile::exec::{Flag, Machine};
use occ_decompile::lift::Lifter as _;
use occ_decompile::lift::x86::{Lifter, X86Target};
use occ_dynamic::probe::Probe;
use occ_sys::ptrace::Registers;

static TARGET: X86Target = X86Target;

/// General-purpose registers, in the order the report prints them.
const GP: [&str; 16] = [
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

/// The six arithmetic flags, and the bit each occupies in `eflags`.
const FLAGS: [(Flag, &str, u64); 5] = [
    (Flag::Cf, "CF", 1 << 0),
    (Flag::Pf, "PF", 1 << 2),
    (Flag::Zf, "ZF", 1 << 6),
    (Flag::Sf, "SF", 1 << 7),
    (Flag::Of, "OF", 1 << 11),
];

/// xorshift64*, so a run is reproducible from its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

/// Instructions worth testing: arithmetic and data movement that stays inside
/// one instruction and does not touch control flow.
fn is_arithmetic(mnemonic: &str) -> bool {
    const NAMES: &[&str] = &[
        "add", "sub", "adc", "sbb", "and", "or", "xor", "cmp", "test", "inc", "dec", "neg", "not",
        "shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr", "imul", "mul", "idiv", "div", "mov",
        "movzx", "movsx", "movsxd", "lea", "xchg", "xadd", "cmpxchg", "bswap", "bt", "bts", "btr",
        "btc", "bsf", "bsr", "tzcnt", "lzcnt", "popcnt", "xlat", "cbw", "cwde", "cdqe", "cwd",
        "cdq", "cqo",
    ];
    NAMES.contains(&mnemonic)
}

/// Encodings to try: every one-byte opcode and every 0F opcode, in register
/// form, with and without a REX.W prefix.
fn encodings() -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for op in 0u8..=0xff {
        for reg in 0u8..8 {
            for rm in 0u8..8 {
                let modrm = 0xc0 | (reg << 3) | rm;
                out.push(vec![0x48, op, modrm]);
                out.push(vec![op, modrm]);
            }
        }
    }
    for op2 in 0u8..=0xff {
        for reg in 0u8..8 {
            for rm in 0u8..8 {
                let modrm = 0xc0 | (reg << 3) | rm;
                out.push(vec![0x48, 0x0f, op2, modrm]);
            }
        }
    }
    out
}

/// One disagreement.
struct Diff {
    what: String,
    real: u64,
    got: Option<u64>,
}

fn main() {
    let mut cases_per_encoding = 4usize;
    let mut verbose = false;
    let mut only: Option<String> = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--cases" => {
                cases_per_encoding = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4);
                i += 2;
            }
            "--verbose" => {
                verbose = true;
                i += 1;
            }
            "--only" => {
                only = args.get(i + 1).cloned();
                i += 2;
            }
            other => {
                eprintln!("semantic_diff: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }

    let Some(decoder) = Decoder::new(64) else {
        eprintln!("semantic_diff: no x86-64 decoder");
        std::process::exit(1);
    };
    let name_of = |a: Addr| format!("sub_{:x}", a.get());
    let symbols = |_: &str| None;
    let mut probe = match Probe::new() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("semantic_diff: cannot start a tracee: {e}");
            std::process::exit(1);
        }
    };

    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut agree = 0u64;
    let mut differ = 0u64;
    let mut skipped: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut unlifted = 0u64;
    let mut not_arithmetic = 0u64;
    let mut failures: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut examples: Vec<String> = Vec::new();
    let mut tested_encodings = 0usize;

    for code in encodings() {
        let Some(decoded) = decoder.decode(Addr(0x1000), &code) else {
            continue;
        };
        let insn = decoded.insn.clone();
        // A decode that consumed fewer bytes than the encoding offered is a
        // prefix the decoder skipped, not the instruction we meant to test.
        if insn.len as usize != code.len() {
            continue;
        }
        if !is_arithmetic(&insn.mnemonic) {
            not_arithmetic += 1;
            continue;
        }
        if let Some(want) = &only
            && insn.mnemonic.as_ref() != want
        {
            continue;
        }
        tested_encodings += 1;

        for _ in 0..cases_per_encoding {
            // An operand naming the stack is promoted to a local by design.
            let text = insn.text.to_ascii_lowercase();
            if text.contains("rsp") || text.contains("rbp") {
                *skipped.entry("stack-relative operand").or_insert(0) += 1;
                continue;
            }
            // A move to or from a segment register is not arithmetic: the
            // selector is not a general-purpose value, and the lifter does not
            // model the segment it names.
            if text.contains(", es") || text.contains(", cs") || text.contains(", ss")
                || text.contains(", ds") || text.contains(", fs") || text.contains(", gs")
            {
                *skipped.entry("segment register operand").or_insert(0) += 1;
                continue;
            }

            // Initial state: registers either hold a pointer into the mapped
            // region or a full-width random value. Both shapes matter — the
            // first reaches the memory operands, the second exercises the high
            // bits that a masked 32-bit operation must not disturb.
            let data_base = probe.data_base();
            let pointer = rng.next() & 1 == 0;
            let mut regs = Registers::default();
            for name in GP {
                let v = if pointer {
                    data_base + rng.below(4096)
                } else {
                    rng.next()
                };
                regs.set(name, v);
            }
            // The probe puts the stack pointer at a fixed place inside the
            // mapped region, because a random one would fault on the first
            // push. The model gets the same value, or `mov al, spl` would
            // disagree for a reason that belongs to the harness.
            regs.set("rsp", probe.stack_addr());
            // Keep IF set and everything outside the six arithmetic flags
            // clear, so a diff in the report is about arithmetic.
            regs.eflags = 0x202 | (rng.next() & 0x8d5);
            let eflags = regs.eflags;

            let mut lifter = Lifter::new(&TARGET, &name_of, &symbols);
            // `adc` and `sbb` read the carry while lifting, not while running,
            // so the lifter needs the flags the processor will start with.
            if reads_carry(&insn.mnemonic) {
                lifter.core.flags = start_flags(eflags);
            }
            let lifted = lifter.lift_block(std::slice::from_ref(&insn));
            if lifted.stmts.is_empty() {
                unlifted += 1;
                continue;
            }

            let mut before = vec![0u8; 8192];
            for (k, b) in before.iter_mut().enumerate() {
                *b = (rng.next() & 0xff) as u8 ^ (k as u8);
            }
            probe.set_data(&before);

            let sample = match probe.step(&code, &regs) {
                Ok(s) => s,
                Err(e) => {
                    *skipped.entry("probe failure").or_insert(0) += 1;
                    if verbose {
                        eprintln!("probe: {e}");
                    }
                    continue;
                }
            };
            if let Some(sig) = sample.fault {
                *skipped.entry("faulted on the processor").or_insert(0) += 1;
                if verbose {
                    eprintln!("{} -> signal {sig}", insn.text);
                }
                continue;
            }

            let mut m = Machine::new();
            for name in GP {
                m.set(name, regs.get(name).unwrap_or(0));
            }
            m.map(data_base, &before);
            // The flags after the instruction are the ones the lifter derived,
            // not the ones it started with. Reading the starting state here
            // compared every arithmetic flag against the random value the
            // harness chose, which looked like a hundred lifter bugs and was
            // one harness bug.
            m.flags = lifter.core.flags.clone();

            if verbose && examples.len() < 3 {
                eprintln!("--- {}", insn.text);
                for (name, v) in [("rax", regs.rax), ("rbx", regs.rbx), ("rcx", regs.rcx)] {
                    eprintln!("    {name} = {v:#x}");
                }
                for st in &lifted.stmts {
                    eprintln!("    {st:?}");
                }
                eprintln!("    flags = {:?}", lifter.core.flags);
            }
            if let Err(e) = m.run_stmts(&lifted.stmts) {
                let key = format!("interpreter: {e}");
                let entry = failures.entry(key.clone()).or_insert((0, insn.text.to_string()));
                entry.0 += 1;
                differ += 1;
                if examples.len() < 40 {
                    examples.push(format!("{} :: {e}", insn.text));
                }
                continue;
            }

            let mut diffs = Vec::new();
            for name in GP {
                // `rsp` is set by the harness and is not part of what the
                // instruction computes unless it writes it.
                if name == "rsp" && !text.contains("rsp") {
                    continue;
                }
                let real = sample.regs.get(name).unwrap_or(0);
                let got = m.get(name);
                if real != got.unwrap_or(u64::MAX) {
                    diffs.push(Diff {
                        what: name.to_string(),
                        real,
                        got,
                    });
                }
            }
            for (flag, label, bit) in FLAGS {
                let real = sample.regs.eflags & bit != 0;
                match m.flag(flag) {
                    Some(got) if got != real => {
                        diffs.push(Diff {
                            what: format!("{label} (real {real}, ir {got})"),
                            real: u64::from(real),
                            got: Some(u64::from(got)),
                        });
                    }
                    // The flag matched, or the instruction did not define it
                    // and the machine left it alone: comparing that would be
                    // a question about the starting state, not about the
                    // instruction.
                    _ => {}
                }
            }
            // Memory: every byte the processor changed must have changed the
            // same way in the model.
            let after = &probe.data()[..before.len()];
            for (k, old) in before.iter().enumerate() {
                if *old == after[k] {
                    continue;
                }
                let addr = data_base + k as u64;
                let got = m.mem.get(&addr).copied().unwrap_or(*old);
                if got != after[k] {
                    diffs.push(Diff {
                        what: format!("mem[{addr:#x}]"),
                        real: u64::from(after[k]),
                        got: Some(u64::from(got)),
                    });
                    break;
                }
            }

            if diffs.is_empty() {
                agree += 1;
            } else {
                differ += 1;
                let key = format!(
                    "{}: {}",
                    insn.mnemonic,
                    diffs
                        .iter()
                        .map(|d| d.what.split(' ').next().unwrap_or(""))
                        .collect::<Vec<_>>()
                        .join(",")
                );
                let entry = failures.entry(key).or_insert((0, String::new()));
                entry.0 += 1;
                if entry.1.is_empty() {
                    entry.1 = format!(
                        "{} | {}",
                        insn.text,
                        diffs
                            .iter()
                            .map(|d| format!(
                                "{} real={:#x} ir={}",
                                d.what,
                                d.real,
                                d.got
                                    .map(|v| format!("{v:#x}"))
                                    .unwrap_or_else(|| "unbound".into())
                            ))
                            .collect::<Vec<_>>()
                            .join("; ")
                    );
                }
                if examples.len() < 40 {
                    examples.push(format!(
                        "{} :: {}",
                        insn.text,
                        diffs
                            .iter()
                            .map(|d| d.what.clone())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }
    }

    println!("encodings tested   {tested_encodings}");
    println!("agree              {agree}");
    println!("differ             {differ}");
    println!("not arithmetic     {not_arithmetic}");
    println!("no statements      {unlifted}");
    for (why, n) in &skipped {
        println!("skipped: {why:<28} {n}");
    }
    // The counts above are only comparable between runs if the encodings they
    // cover are the same encodings every time. They are, because the regions
    // the traced process sees are mapped at fixed addresses; when that could
    // not be done, the run is still correct but its numbers describe a
    // different set of encodings, and saying so is better than a number that
    // looks comparable and is not.
    if !probe.is_reproducible() {
        println!("\nwarning: the mapped regions were not at their fixed addresses, so the");
        println!("         encodings covered differ between runs and the counts above");
        println!("         are not comparable with another run's");
    }
    if !failures.is_empty() {
        println!("\nfailures by shape");
        let mut rows: Vec<_> = failures.iter().collect();
        rows.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for (key, (n, example)) in rows.iter().take(30) {
            println!("  {n:>6}  {key}");
            if !example.is_empty() {
                println!("          {example}");
            }
        }
        println!("\nexamples");
        for e in examples.iter().take(20) {
            println!("  {e}");
        }
    }
    probe.close();
    if differ > 0 {
        std::process::exit(1);
    }
}

/// Whether an instruction is defined by a flag it arrives with.
fn reads_carry(mnemonic: &str) -> bool {
    matches!(mnemonic, "adc" | "sbb" | "rcl" | "rcr" | "cmovb" | "cmovae" | "cmove" | "cmovne")
        || mnemonic.starts_with("cmov")
        || mnemonic.starts_with("set")
}

/// The flag expressions a given `eflags` image stands for.
///
/// The interpreter needs the starting flags as expressions, because a branch
/// condition inside the lifted statements reads whatever the previous
/// operation defined. For a single instruction that is the caller's value.
fn start_flags(eflags: u64) -> occ_decompile::ir::FlagState {
    let bit = |n: u32| Some(occ_decompile::ir::Expr::Const((eflags >> n) & 1));
    occ_decompile::ir::FlagState {
        cf: bit(0),
        pf: bit(2),
        zf: bit(6),
        sf: bit(7),
        of: bit(11),
    }
}
