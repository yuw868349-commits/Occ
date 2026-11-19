//! The probe's own behaviour, checked against the processor.
//!
//! These tests exist because the probe is what makes the differential run
//! meaningful. If the probe lies about a register, or the regions it maps
//! drift between runs, then a run that reports "no disagreements" says
//! nothing — and the failure is invisible in the results, because a probe
//! that lies consistently lies the same way twice.

use occ_dynamic::probe::Probe;
use occ_sys::ptrace::Registers;

fn probe() -> Option<Probe> {
    Probe::new().ok()
}

/// Run one instruction and read a register back.
fn step(p: &mut Probe, code: &[u8], set: &[(&str, u64)]) -> Option<Registers> {
    let mut regs = Registers::default();
    regs.set("rax", 0x11223344556677aa);
    regs.set("rcx", 0x88776655443322bb);
    for (name, value) in set {
        regs.set(name, *value);
    }
    let sample = p.step(code, &regs).ok()?;
    if sample.fault.is_some() {
        return None;
    }
    Some(sample.regs)
}

#[test]
fn the_regions_are_at_fixed_addresses_so_a_run_can_be_repeated() {
    let Some(p) = probe() else { return };
    assert!(
        p.is_reproducible(),
        "the mapped regions did not land at their fixed addresses, so two runs \
         would cover different encodings"
    );
}

#[test]
fn two_probes_do_not_share_the_same_regions() {
    let (Some(a), Some(b)) = (probe(), probe()) else {
        return;
    };
    // Each probe takes the next slot, so the second one cannot overwrite the
    // first one's data. Two probes sharing an address would mean the second
    // one read the first one's bytes.
    assert_ne!(
        a.data_base(),
        b.data_base(),
        "two probes must not map their data over each other"
    );
    assert_ne!(a.stack_addr(), b.stack_addr());
}

#[test]
fn the_processor_computes_what_the_probe_reports() {
    let Some(mut p) = probe() else { return };
    // `add rax, rcx` with the two values the harness sets.
    let regs = step(&mut p, &[0x48, 0x01, 0xc8], &[]).expect("the instruction runs");
    assert_eq!(
        regs.rax,
        0x11223344556677aa_u64.wrapping_add(0x88776655443322bb),
        "the probe must report what the processor did, not what the test expected"
    );
    // The other register is untouched.
    assert_eq!(regs.rcx, 0x88776655443322bb);
}

#[test]
fn a_subtraction_borrows_and_the_carry_flag_says_so() {
    let Some(mut p) = probe() else { return };
    // `sub rax, rcx` where the minuend is smaller than the subtrahend: the
    // borrow must come out of the carry flag.
    let regs =
        step(&mut p, &[0x48, 0x29, 0xc8], &[("rax", 1), ("rcx", 2)]).expect("the instruction runs");
    assert_eq!(regs.rax, u64::MAX, "1 - 2 wraps to all ones");
    assert_ne!(regs.eflags & 1, 0, "the borrow must set the carry flag");
}

#[test]
fn a_divide_by_zero_faults_instead_of_returning_a_number() {
    let Some(mut p) = probe() else { return };
    let mut regs = Registers::default();
    regs.set("rax", 100);
    regs.set("rcx", 0);
    let sample = p.step(&[0x48, 0xf7, 0xf1], &regs).expect("the step runs");
    assert!(
        sample.fault.is_some(),
        "dividing by zero must fault, not produce a quotient"
    );
}

#[test]
fn an_instruction_that_faults_is_reported_as_a_fault_and_the_probe_recovers() {
    let Some(mut p) = probe() else { return };
    let mut regs = Registers::default();
    regs.set("rax", 100);
    regs.set("rcx", 0);
    let first = p.step(&[0x48, 0xf7, 0xf1], &regs).expect("the step runs");
    assert!(first.fault.is_some());
    // The child died with the fault, so the probe must have replaced it.
    // The next instruction has to run on a fresh child, not on a corpse.
    let regs = step(&mut p, &[0x48, 0x01, 0xc8], &[]).expect("the next step runs");
    assert_eq!(
        regs.rax,
        0x11223344556677aa_u64.wrapping_add(0x88776655443322bb)
    );
    assert!(p.forks >= 2, "a fault must cost a child");
}

#[test]
fn a_memory_operand_reads_the_region_the_probe_wrote() {
    let Some(mut p) = probe() else { return };
    let mut bytes = vec![0u8; 64];
    bytes[8] = 0xab;
    bytes[9] = 0xcd;
    p.set_data(&bytes);
    let base = p.data_base();
    // `mov al, [rbx + 8]` — read the byte the probe wrote.
    let regs = step(&mut p, &[0x8a, 0x43, 0x08], &[("rbx", base)]).expect("the instruction runs");
    assert_eq!(
        regs.rax & 0xff,
        0xab,
        "the probe must read back what it wrote"
    );
}

#[test]
fn a_write_to_the_region_is_visible_to_the_parent() {
    let Some(mut p) = probe() else { return };
    let base = p.data_base();
    // `mov byte [rbx + 16], 0x5a` — the instruction writes memory, not a
    // register, so the register is checked for being untouched instead.
    let regs =
        step(&mut p, &[0xc6, 0x43, 0x10, 0x5a], &[("rbx", base)]).expect("the instruction runs");
    assert_eq!(
        regs.rax & 0xff,
        0xaa,
        "the instruction must not have touched the register"
    );
    assert_eq!(
        p.data()[16],
        0x5a,
        "the parent must see the byte the child wrote"
    );
}
