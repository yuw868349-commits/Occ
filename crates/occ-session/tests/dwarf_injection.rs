//! End-to-end check that debug information reaches the decompiler.
//!
//! The fixture is compiled here rather than checked in, so the test reads the
//! debug info a real `-g` build produces on this host. A host without a C
//! compiler skips the test: what is under test is the wiring, not the
//! compiler.
//!
//! The assertions are about facts the source states — the declared names, the
//! declared types, the source file — and never about the shape of the body,
//! which is the lifter's business.

use std::path::PathBuf;
use std::process::Command;

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::{Addr, ImageId};
use occ_session::ImageFile;

const SOURCE: &str = r#"
struct point { int x; int y; };

int add(int a, int b) {
    int sum = a + b;
    return sum;
}

int scaled(struct point *p, int factor) {
    return p->x * factor;
}
"#;

/// Compile the fixture into its own directory, returning the object path.
///
/// Each test gets a directory of its own: the two run in parallel, and one
/// writing the object while the other reads it is a race the test would
/// report as a decompiler defect. `None` when no C compiler runs.
fn build(name: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir().join(format!("occ_dwarf_injection_{name}"));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("dwarf_fixture.c");
    std::fs::write(&src, SOURCE).ok()?;
    let obj = dir.join("dwarf_fixture.o");
    let status = Command::new("gcc")
        .args(["-g", "-O0", "-c", "-o"])
        .arg(&obj)
        .arg(&src)
        .status()
        .ok()?;
    status.success().then_some(obj)
}

fn load(path: &PathBuf) -> ImageFile {
    let mut data = std::fs::read(path).expect("read object");
    let mut image = occ_bin::load(&data, path.to_str().unwrap(), ImageId(0)).expect("parse object");
    // A relocatable object's DWARF is a set of placeholders until a linker
    // fills it in; the session applies them, so the test does too.
    if let Some(applied) = occ_bin::reloc::apply(&mut data, &image) {
        image.warnings.extend(applied.notes);
    }
    let (analysis, dwarf) = {
        let view = View::new(&image, &data);
        (
            Analysis::build(&view, &AnalysisOptions::default()),
            occ_dwarf::Dwarf::load(&view),
        )
    };
    ImageFile {
        image,
        data,
        analysis,
        dwarf,
        opened_seq: 0,
    }
}

#[test]
fn names_types_and_lines_reach_the_output() {
    let Some(obj) = build("add") else {
        eprintln!("no C compiler on this host; skipping");
        return;
    };
    let file = load(&obj);

    let add = file
        .analysis
        .function_by_name("add")
        .expect("the object defines add");
    let info = occ_session::dbginfo::debug_info_for(&file, add).expect("add has debug information");
    assert_eq!(info.name.as_deref(), Some("add"));
    assert_eq!(info.return_type.as_deref(), Some("int"));
    let names: Vec<&str> = info.params.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["a", "b"], "parameter names come from the source");

    let opts = occ_decompile::Options {
        debug: Some(info.clone()),
        source_lines: true,
        ..Default::default()
    };
    let name_of = |a: Addr| format!("sub_{:x}", a.0);
    let d = occ_decompile::decompile(&file.image, add, &opts, &name_of);
    let text = occ_decompile::render::render(&d);

    assert!(
        text.starts_with("int add(int a, int b) {"),
        "the signature is the declared one, got:\n{text}"
    );
    assert!(
        text.contains("dwarf_fixture.c:"),
        "the body names the source file, got:\n{text}"
    );
    assert!(
        text.contains("// line "),
        "the body carries line marks, got:\n{text}"
    );
    // The lines must climb: a listing whose marks go backwards is worse than
    // none, because it claims a source order the code does not have.
    let mut last = 0u32;
    for line in text
        .lines()
        .filter_map(|l| l.trim().strip_prefix("// line "))
    {
        let n: u32 = line.parse().expect("line mark is a number");
        assert!(n >= last, "line marks go backwards: {last} then {n}");
        last = n;
    }
}

#[test]
fn a_pointer_parameter_keeps_its_struct_type() {
    let Some(obj) = build("scaled") else {
        eprintln!("no C compiler on this host; skipping");
        return;
    };
    let file = load(&obj);

    let scaled = file
        .analysis
        .function_by_name("scaled")
        .expect("the object defines scaled");
    let info =
        occ_session::dbginfo::debug_info_for(&file, scaled).expect("scaled has debug information");
    let types: Vec<&str> = info
        .params
        .iter()
        .filter_map(|p| p.type_name.as_deref())
        .collect();
    assert!(
        types.iter().any(|t| t.contains("struct point")),
        "the pointer parameter keeps its struct type, got {types:?}"
    );

    // The type table has to carry the members, or `p->x` cannot be printed as
    // a member access.
    let point = info
        .types
        .iter()
        .find(|t| t.name == "point")
        .expect("the struct is in the type table");
    let members: Vec<&str> = point.members.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(members, ["x", "y"], "members in declaration order");
}
