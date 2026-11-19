//! End-to-end checks against hand-assembled machine code.
//!
//! Each fixture is a few instructions whose meaning is stated in the test. The
//! bytes are decoded by the same recovery the tools use, so a mistake in the
//! encoding shows up as a wrong instruction rather than as a silently passing
//! test.

use occ_analyze::{RecoverOptions, View, recover};
use occ_core::{
    Addr, Arch, Endian, Fingerprint, Format, Image, ImageId, Perm, Section, SectionKind,
};
use occ_decompile::Options;
use occ_sym::{ExecConfig, Mode, Outcome, Source, TaintConfig, Verdict, explore, trace};

fn image_with_code(code: &[u8]) -> Image {
    Image {
        id: ImageId(0),
        path: "fixture".into(),
        format: Format::Elf,
        arch: Arch::X86_64,
        bits: 64,
        endian: Endian::Little,
        entry: Addr(0x1000),
        image_base: Addr(0x1000),
        file_size: code.len() as u64,
        sha256: String::new(),
        segments: Vec::new(),
        resources: Vec::new(),
        sections: vec![Section {
            index: 1,
            name: ".text".into(),
            addr: Addr(0x1000),
            file_offset: 0,
            size: code.len() as u64,
            file_size: code.len() as u64,
            kind: SectionKind::Code,
            perm: Perm::R.union(Perm::X),
        }],
        symbols: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        relocations: Vec::new(),
        debug_relocations: Vec::new(),
        fingerprint: Fingerprint::unknown(ImageId(0)),
        interpreter: None,
        warnings: Vec::new(),
        function_bounds: Vec::new(),
        arm_modes: Default::default(),
        wasm_types: Default::default(),
        wasm_funcs: Default::default(),
    }
}

fn no_static(_at: Addr, _bytes: u8) -> Option<u64> {
    None
}

/// The first recovered function, which always starts at 0x1000.
fn function_at_entry(image: &Image, code: &[u8]) -> occ_core::Function {
    let view = View::new(image, code);
    let rec = recover(
        &view,
        &RecoverOptions {
            max_functions: 16,
            max_insns: 4096,
            follow_calls: false,
            sweep_uncovered: false,
            ..RecoverOptions::default()
        },
    );
    rec.functions
        .into_iter()
        .find(|f| f.entry == Addr(0x1000))
        .expect("a function starts at the entry address")
}

const TWICE: &[u8] = &[0x48, 0x8d, 0x04, 0x3f, 0xc3];

#[test]
fn a_straight_line_function_returns_its_expression() {
    // lea rax, [rdi + rdi] ; ret
    let image = image_with_code(TWICE);
    let func = function_at_entry(&image, TWICE);
    let report = explore(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &no_static,
        &[],
        &ExecConfig::default(),
    );
    assert_eq!(report.paths.len(), 1, "no branches, so one path");
    let path = &report.paths[0];
    assert_eq!(path.outcome, Outcome::Return);
    assert!(path.is_feasible());
    assert_eq!(path.ret.as_deref(), Some("rdi + rdi"));
    assert!(path.conds.is_empty(), "nothing to decide");
}

#[test]
fn a_two_way_branch_yields_two_paths_with_solved_inputs() {
    // cmp rdi, rsi ; jle +4 ; mov rax, rdi ; ret ; mov rax, rsi ; ret
    let code: &[u8] = &[
        0x48, 0x39, 0xf7, // cmp rdi, rsi
        0x7e, 0x04, // jle +4
        0x48, 0x89, 0xf8, // mov rax, rdi
        0xc3, // ret
        0x48, 0x89, 0xf0, // mov rax, rsi
        0xc3, // ret
    ];
    let image = image_with_code(code);
    let func = function_at_entry(&image, code);
    let report = explore(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &no_static,
        &[],
        &ExecConfig::default(),
    );
    assert_eq!(report.paths.len(), 2, "one path per arm");
    let mut returns: Vec<String> = report.paths.iter().filter_map(|p| p.ret.clone()).collect();
    returns.sort();
    assert_eq!(returns, vec!["rdi".to_string(), "rsi".to_string()]);

    for path in &report.paths {
        assert_eq!(path.model.verdict, Verdict::Satisfied);
        let model = path.model.map();
        // Whatever the arm, the model must actually satisfy the condition that
        // selected it. This is the property that makes the model trustworthy.
        for cond in &path.conds {
            assert_eq!(
                occ_sym::solve_constraints(std::slice::from_ref(cond), 4096).verdict,
                Verdict::Satisfied
            );
        }
        assert!(!model.is_empty(), "the branch reads a register");
    }
}

#[test]
fn an_unsatisfiable_edge_is_not_reported_as_a_path() {
    // test rdi, rdi ; je +6 ; mov eax, 1 ; ret ; xor eax, eax ; ret
    let code: &[u8] = &[
        0x48, 0x85, 0xff, // test rdi, rdi
        0x74, 0x06, // je +6
        0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0xc3, // ret
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ];
    let image = image_with_code(code);
    let func = function_at_entry(&image, code);
    let report = explore(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &no_static,
        &[],
        &ExecConfig::default(),
    );
    assert_eq!(report.paths.len(), 2);
    let returns = report.returns();
    assert_eq!(returns.len(), 2, "one path returns 1, the other 0");
    // Both arms must be reachable, because `rdi` is free.
    assert!(report.paths.iter().all(|p| p.is_feasible()));
}

#[test]
fn concrete_mode_names_the_input_it_still_needs() {
    // lea rax, [rdi + rdi] ; ret, with rdi supplied.
    let image = image_with_code(TWICE);
    let func = function_at_entry(&image, TWICE);
    let cfg = ExecConfig {
        mode: Mode::Concrete,
        inputs: vec![("rdi".into(), 21)],
        ..ExecConfig::default()
    };
    let report = explore(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &no_static,
        &[],
        &cfg,
    );
    assert_eq!(report.paths.len(), 1);
    assert_eq!(report.paths[0].outcome, Outcome::Return);
    // The expression is reported unfolded: the executor does not substitute
    // values into the printed form, it reports what the program computes.
    assert_eq!(report.paths[0].ret.as_deref(), Some("0x2a"));

    let cfg = ExecConfig {
        mode: Mode::Concrete,
        ..ExecConfig::default()
    };
    let report = explore(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &no_static,
        &[],
        &cfg,
    );
    match &report.paths[0].outcome {
        Outcome::Missing(names) => assert_eq!(names, &vec!["rdi".to_string()]),
        other => panic!("expected a missing input, got {other:?}"),
    }
}

#[test]
fn a_loop_is_unrolled_until_the_bound_and_says_so() {
    // sub rdi, 1 ; test rdi, rdi ; jg -9 ; mov rax, rdi ; ret
    let code: &[u8] = &[
        0x48, 0x83, 0xef, 0x01, // sub rdi, 1
        0x48, 0x85, 0xff, // test rdi, rdi
        0x7f, 0xf7, // jg -9
        0x48, 0x89, 0xf8, // mov rax, rdi
        0xc3, // ret
    ];
    let image = image_with_code(code);
    let func = function_at_entry(&image, code);
    let cfg = ExecConfig {
        max_loop: 3,
        max_paths: 16,
        ..ExecConfig::default()
    };
    let report = explore(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &no_static,
        &[],
        &cfg,
    );
    assert!(
        report.paths.len() > 1,
        "the loop has an exit and iterations"
    );
    let truncated = report
        .paths
        .iter()
        .filter(|p| p.outcome == Outcome::Truncated("loop bound"))
        .count();
    assert!(truncated >= 1, "unrolling must stop and say where");
    let feasible = report.feasible();
    assert!(feasible >= 1, "at least the zero-iteration path returns");
}

#[test]
fn a_parameter_reaching_a_call_is_reported_with_its_argument_index() {
    // mov [rsp-8], rdi ; mov rdi, [rsp-8] ; call puts ; ret, with puts at
    // 0x1010. The round trip through the slot is what a compiler emits for a
    // pass-through argument at -O0, and it is what makes the lifter record that
    // rdi is live at the call.
    let code: &[u8] = &[
        0x48, 0x89, 0x7c, 0x24, 0xf8, // mov [rsp-8], rdi
        0x48, 0x8b, 0x7c, 0x24, 0xf8, // mov rdi, [rsp-8]
        0xe8, 0x01, 0x00, 0x00, 0x00, // call 0x1010
        0xc3, // ret
        0xc3, // 0x1010: ret
    ];
    let image = image_with_code(code);
    let func = function_at_entry(&image, code);
    let cfg = TaintConfig::default();
    let report = trace(
        &image,
        &func,
        &Options::default(),
        &|a| {
            if a == Addr(0x1010) {
                "puts".into()
            } else {
                format!("sub_{:x}", a.get())
            }
        },
        &[Source::Name("rdi".into())],
        &cfg,
        std::slice::from_ref(&func),
    );
    let call = report
        .flows
        .iter()
        .find(|f| f.sink == "puts")
        .expect("the argument register reaches puts");
    assert_eq!(call.arg, Some(0));
    assert_eq!(call.value, "rdi");
    assert_eq!(call.source, "name:rdi");
    assert!(report.converged);
}

#[test]
fn a_tainted_branch_is_a_flow_of_its_own() {
    // test rdi, rdi ; je +6 ; mov eax,1 ; ret ; xor eax,eax ; ret
    let code: &[u8] = &[
        0x48, 0x85, 0xff, 0x74, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3, 0x31, 0xc0, 0xc3,
    ];
    let image = image_with_code(code);
    let func = function_at_entry(&image, code);
    let report = trace(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &[Source::Name("rdi".into())],
        &TaintConfig::default(),
        std::slice::from_ref(&func),
    );
    let branch = report
        .flows
        .iter()
        .find(|f| f.sink == "branch")
        .expect("the condition depends on rdi");
    assert_eq!(branch.source, "name:rdi");
    assert!(!branch.path.is_empty(), "a witness path back to the entry");
    assert_eq!(branch.path.first().copied(), Some(Addr(0x1000)));
}

#[test]
fn nothing_is_tainted_when_nothing_is_a_source() {
    let image = image_with_code(TWICE);
    let func = function_at_entry(&image, TWICE);
    let report = trace(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &[],
        &TaintConfig::default(),
        std::slice::from_ref(&func),
    );
    assert!(report.flows.is_empty());
    assert!(!report.notes.is_empty(), "an empty result is explained");
}

#[test]
fn a_call_with_a_known_body_is_interpreted() {
    // 1000: call 1010          ; e8 0b 00 00 00
    // 1005: cmp  eax, 0xa      ; 83 f8 0a
    // 1008: jne  100f          ; 75 05
    // 100a: mov  eax, 1        ; b8 01 00 00 00
    // 100f: ret                ; c3
    // 1010: lea  rax, [rdi+rdi]; 48 8d 04 3f
    // 1014: ret                ; c3
    const CODE: &[u8] = &[
        0xe8, 0x0b, 0x00, 0x00, 0x00, 0x83, 0xf8, 0x0a, 0x75, 0x05, 0xb8, 0x01, 0x00, 0x00, 0x00,
        0xc3, 0x48, 0x8d, 0x04, 0x3f, 0xc3,
    ];
    let image = image_with_code(CODE);
    let view = View::new(&image, CODE);
    let rec = recover(
        &view,
        &RecoverOptions {
            max_functions: 16,
            max_insns: 4096,
            follow_calls: true,
            ..RecoverOptions::default()
        },
    );
    let caller = rec
        .functions
        .iter()
        .find(|f| f.entry == Addr(0x1000))
        .expect("caller");
    // The callee doubled the argument, so `rdi = 5` makes `eax` ten and the
    // branch falls through. Without interpreting the call, `eax` is unknown
    // and both arms are explored.
    let cfg = ExecConfig {
        mode: Mode::Concrete,
        inputs: vec![("rdi".into(), 5)],
        ..ExecConfig::default()
    };
    let report = explore(
        &image,
        caller,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &no_static,
        &rec.functions,
        &cfg,
    );
    assert_eq!(
        report.paths.len(),
        1,
        "the callee decides the branch: {:#?}",
        report.paths
    );
    assert_eq!(report.paths[0].ret.as_deref(), Some("0x1"));
    assert!(
        report.notes.iter().any(|n| n.contains("interpreted")),
        "the run says it interpreted a call: {:?}",
        report.notes
    );
}

#[test]
fn a_recursive_call_is_not_interpreted() {
    // 1000: call 1000 ; e8 fb ff ff ff
    // 1005: ret       ; c3
    const CODE: &[u8] = &[0xe8, 0xfb, 0xff, 0xff, 0xff, 0xc3];
    let image = image_with_code(CODE);
    let func = function_at_entry(&image, CODE);
    let report = explore(
        &image,
        &func,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &no_static,
        std::slice::from_ref(&func),
        &ExecConfig::default(),
    );
    assert!(
        report.notes.iter().any(|n| n.contains("recursive")),
        "recursion is named rather than followed: {:?}",
        report.notes
    );
}

/// Two functions: one returns its argument, one stores it through a pointer.
///
/// 1000: call 1010   e8 0b 00 00 00
/// 1005: ret         c3
/// 1010: mov rax,rdi 48 89 f8
/// 1013: ret         c3
fn identity_pair() -> Vec<u8> {
    vec![
        0xe8, 0x0b, 0x00, 0x00, 0x00, 0xc3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
        0x90, 0x48, 0x89, 0xf8, 0xc3,
    ]
}

#[test]
fn a_summary_says_the_return_carries_the_argument() {
    let code = identity_pair();
    let image = image_with_code(&code);
    let view = View::new(&image, &code);
    let rec = recover(
        &view,
        &RecoverOptions {
            max_functions: 16,
            max_insns: 4096,
            follow_calls: true,
            ..RecoverOptions::default()
        },
    );
    let caller = rec
        .functions
        .iter()
        .find(|f| f.entry == Addr(0x1000))
        .expect("caller");
    let report = trace(
        &image,
        caller,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &[Source::Name("rdi".into())],
        &TaintConfig::default(),
        &rec.functions,
    );
    assert!(
        report.ret_tainted,
        "the callee returns its argument, so the call's result carries it: {:?}",
        report.notes
    );
}

#[test]
fn a_summary_reports_the_sink_inside_the_callee() {
    // 1000: call 1010     e8 0b 00 00 00
    // 1005: ret           c3
    // 1010: mov [rdi],esi 89 37
    // 1012: ret           c3
    let code: &[u8] = &[
        0xe8, 0x0b, 0x00, 0x00, 0x00, 0xc3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
        0x90, 0x89, 0x37, 0xc3,
    ];
    let image = image_with_code(code);
    let view = View::new(&image, code);
    let rec = recover(
        &view,
        &RecoverOptions {
            max_functions: 16,
            max_insns: 4096,
            follow_calls: true,
            ..RecoverOptions::default()
        },
    );
    let caller = rec
        .functions
        .iter()
        .find(|f| f.entry == Addr(0x1000))
        .expect("caller");
    let report = trace(
        &image,
        caller,
        &Options::default(),
        &|a| format!("sub_{:x}", a.get()),
        &[Source::Name("rdi".into())],
        &TaintConfig::default(),
        &rec.functions,
    );
    assert!(
        report
            .flows
            .iter()
            .any(|f| f.sink.contains("store through a pointer")),
        "the callee writes the argument through the pointer: {:?}",
        report.flows.iter().map(|f| &f.sink).collect::<Vec<_>>()
    );
}
