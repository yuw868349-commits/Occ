//! Run one instruction through the lifter and the interpreter, and print what
//! happened.
//!
//! `debug_one <hex bytes> [reg=value ...]` lifts the instruction, prints every
//! statement, then executes it and prints the registers and flags. With
//! `--probe` the same instruction is run on the processor under `ptrace` and
//! both results are printed side by side. This is the tool that turns "the
//! differential says `imul al` disagrees" into "here is the expression".

use occ_core::Addr;
use occ_decode::x86::Decoder;
use occ_decompile::exec::{Flag, Machine};
use occ_decompile::lift::Lifter as _;
use occ_decompile::lift::x86::{Lifter, X86Target};
use occ_dynamic::probe::Probe;
use occ_sys::ptrace::Registers;

static TARGET: X86Target = X86Target::sysv();

const GP: [&str; 16] = [
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

fn main() {
    let mut bytes: Vec<u8> = Vec::new();
    let mut named: Vec<(String, u64)> = Vec::new();
    let mut probe = false;
    let mut eflags: u64 = 0x202;
    for arg in std::env::args().skip(1) {
        if arg == "--probe" {
            probe = true;
        } else if let Some(rest) = arg.strip_prefix("0x") {
            let clean: String = rest.chars().filter(|c| c.is_ascii_hexdigit()).collect();
            for i in (0..clean.len()).step_by(2) {
                bytes.push(u8::from_str_radix(&clean[i..i + 2], 16).unwrap_or(0));
            }
        } else if let Some((name, value)) = arg.split_once('=') {
            let value = value.trim_start_matches("0x");
            let value = u64::from_str_radix(value, 16).unwrap_or(0);
            named.push((name.to_string(), value));
        }
    }
    if bytes.is_empty() {
        eprintln!("usage: debug_one 0x<hex> [rax=0x..] [--probe]");
        return;
    }
    for (name, value) in &named {
        if *name == "eflags" {
            eflags = *value;
        }
    }
    let Some(decoder) = Decoder::new(64) else {
        eprintln!("no decoder");
        return;
    };
    let Some(decoded) = decoder.decode(Addr(0x1000), &bytes) else {
        eprintln!("decode failed");
        return;
    };
    let insn = decoded.insn.clone();
    println!("text  {}  ({} bytes)", insn.text, insn.len);
    let name_of = |a: Addr| format!("sub_{:x}", a.get());
    let symbols = |_: &str| None;
    let mut lifter = Lifter::new(&TARGET, &name_of, &symbols);
    let lifted = lifter.lift_block(std::slice::from_ref(&insn));
    for stmt in &lifted.stmts {
        println!("  {stmt:?}");
    }
    println!("flags {:?}", lifter.core.flags);

    let mut m = Machine::new();
    for name in GP {
        let value = named
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
            .unwrap_or(0);
        m.set(name, value);
    }
    m.flags = lifter.core.flags.clone();
    match m.run_stmts(&lifted.stmts) {
        Ok(_) => {}
        Err(e) => println!("fault {e:?}"),
    }
    let dump = |m: &Machine, label: &str| {
        for name in GP {
            let v = m.get(name).unwrap_or(0);
            if named.iter().any(|(n, _)| n == name) || v != 0 {
                println!("{label} {name:>4} = {v:#x}");
            }
        }
        let f = |which: Flag, bit: u64| match m.flag(which) {
            Some(v) => u64::from(v) * bit,
            None => 0,
        };
        println!(
            "{label} eflags = {:#x}",
            f(Flag::Cf, 1)
                | f(Flag::Pf, 4)
                | f(Flag::Zf, 0x40)
                | f(Flag::Sf, 0x80)
                | f(Flag::Of, 0x800)
        );
    };
    dump(&m, "ir  ");

    if probe {
        let mut regs = Registers::default();
        for name in GP {
            let value = named
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| *v)
                .unwrap_or(0);
            regs.set(name, value);
        }
        regs.eflags = eflags;
        let mut p = match Probe::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("probe failed: {e}");
                return;
            }
        };
        regs.rsp = p.stack_addr();
        match p.step(&bytes, &regs) {
            Ok(sample) => {
                println!("cpu  fault {:?}", sample.fault);
                for name in GP {
                    let v = sample.regs.get(name).unwrap_or(0);
                    if named.iter().any(|(n, _)| n == name) || v != 0 {
                        println!("cpu  {name:>4} = {v:#x}");
                    }
                }
                println!("cpu  eflags = {:#x}", sample.regs.eflags & 0x8d5);
            }
            Err(e) => eprintln!("probe step failed: {e}"),
        }
    }
}
