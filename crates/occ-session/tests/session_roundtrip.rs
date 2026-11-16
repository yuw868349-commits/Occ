//! What a saved session must survive.
//!
//! A session file is the only thing that carries a long investigation across
//! processes. If it drops the standing a hypothesis had reached, the next
//! process starts from "no conclusion" and cannot tell that from "we never
//! looked" — the two states the model is supposed to distinguish. And because
//! the file is an input, every field in it is checked: a version this build
//! does not know, a state name it cannot represent, a number outside its range
//! and a path that climbs out of its directory all have to fail loudly.

use std::path::{Path, PathBuf};

use occ_core::{
    Addr, Evidence, EvidenceKind, EvidenceSource, Handle, HypothesisState,
};
use occ_session::Session;

/// A scratch directory that removes itself, so a passing run leaves `/tmp` as
/// it found it.
struct Scratch(PathBuf);

impl Scratch {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(name: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("occ-session-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    Scratch(dir)
}

fn sample_evidence() -> Evidence {
    Evidence {
        kind: EvidenceKind::ControlFlow,
        source: EvidenceSource::Disassembler,
        at: Some(Addr(0x401000)),
        span: Some(16),
        detail: "the loop body matches a byte-wise copy".to_string(),
        weight: 0.9,
    }
}

#[test]
fn save_restore_keeps_the_standing_of_a_hypothesis() {
    let dir = scratch("roundtrip");
    let path = dir.path().join("session.json");
    let path_str = path.to_str().unwrap();

    let first = Session::new();
    first.note("opened /bin/ls");
    let id = first
        .hypothesise(Handle::image(0), Some(0), "the loop is memcpy")
        .expect("file hypothesis");
    let (verdict, confidence) = first.add_evidence(id, sample_evidence()).expect("add evidence");
    assert_eq!(verdict.label(), "confirmed");

    first.save(path_str).expect("save");

    let saved = occ_session::persist::load(path_str).expect("load");
    assert_eq!(saved.version, occ_session::persist::FORMAT_VERSION);
    assert_eq!(saved.hypotheses.len(), 1);
    let h = &saved.hypotheses[0];
    assert_eq!(h.state, HypothesisState::Supported);
    assert!((h.confidence - confidence).abs() < 1e-6);
    assert_eq!(h.evidence.len(), 1);
    assert_eq!(h.evidence[0].detail, "the loop body matches a byte-wise copy");
    assert_eq!(h.evidence[0].at, Some(Addr(0x401000)));
    assert_eq!(saved.notes, vec!["opened /bin/ls".to_string()]);

    let second = Session::new();
    second.restore(&saved).expect("restore");
    let restored = second.hypotheses();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].id, id);
    assert_eq!(restored[0].state, HypothesisState::Supported);
    assert!((restored[0].confidence - confidence).abs() < 1e-6);
    assert_eq!(restored[0].evidence.len(), 1);
    assert_eq!(restored[0].statement, "the loop is memcpy");
    assert_eq!(restored[0].handle, Handle::image(0));
    assert_eq!(restored[0].image, Some(0), "the claim keeps the image it was about");
}

#[test]
fn a_new_hypothesis_cannot_reuse_a_restored_id() {
    let dir = scratch("ids");
    let path = dir.path().join("session.json");
    let path_str = path.to_str().unwrap();

    let first = Session::new();
    let id = first.hypothesise(Handle::image(0), Some(0), "first").unwrap();
    first.save(path_str).unwrap();

    let second = Session::new();
    second.restore(&occ_session::persist::load(path_str).unwrap()).unwrap();
    let fresh = second.hypothesise(Handle::image(0), Some(0), "second").unwrap();
    assert!(fresh > id, "restored ids must reserve their slot");
    assert_eq!(second.hypotheses().len(), 2);
}

fn write(dir: &Scratch, name: &str, body: &str) -> String {
    let p = dir.path().join(name);
    std::fs::write(&p, body).unwrap();
    p.to_str().unwrap().to_string()
}

#[test]
fn an_unknown_version_is_refused() {
    let dir = scratch("version");
    let p = write(
        &dir,
        "v99.json",
        r#"{"version":99,"images":[],"hypotheses":[],"notes":[]}"#,
    );
    let err = occ_session::persist::load(&p).expect_err("version 99 must not load");
    assert!(err.to_string().contains("unsupported session version"), "{err}");
}

#[test]
fn a_version_1_file_still_loads_as_proposed() {
    let dir = scratch("v1");
    let p = write(
        &dir,
        "v1.json",
        r#"{"version":1,"images":[],"hypotheses":[{"id":1,"handle":"img:0","statement":"old"}],"notes":[]}"#,
    );
    let saved = occ_session::persist::load(&p).expect("version 1 is readable");
    assert_eq!(saved.hypotheses[0].state, HypothesisState::Proposed);
    assert_eq!(saved.hypotheses[0].confidence, 0.0);
    assert!(saved.hypotheses[0].evidence.is_empty());
}

#[test]
fn a_state_name_the_build_does_not_know_is_refused() {
    let dir = scratch("state");
    let p = write(
        &dir,
        "bad-state.json",
        r#"{"version":2,"images":[],"hypotheses":[
            {"id":1,"handle":"img:0","statement":"x","state":"probably","confidence":0.5,"evidence":[]}
        ],"notes":[]}"#,
    );
    let err = occ_session::persist::load(&p).expect_err("unknown state must not load");
    assert!(err.to_string().contains("`state` is not one of"), "{err}");
}

#[test]
fn a_confidence_outside_the_unit_interval_is_refused() {
    let dir = scratch("confidence");
    for (name, value) in [("high.json", "1.5"), ("negative.json", "-0.25")] {
        let body = format!(
            r#"{{"version":2,"images":[],"hypotheses":[
                {{"id":1,"handle":"img:0","statement":"x","state":"proposed","confidence":{value},"evidence":[]}}
            ],"notes":[]}}"#
        );
        let p = write(&dir, name, &body);
        let err = occ_session::persist::load(&p).expect_err(name);
        assert!(err.to_string().contains("confidence"), "{name}: {err}");
    }
}

#[test]
fn a_confidence_that_is_not_finite_is_refused() {
    let dir = scratch("non-finite");
    // JSON has no NaN literal, so the way a non-finite value reaches the loader
    // is an exponent that overflows to infinity.
    let p = write(
        &dir,
        "huge.json",
        r#"{"version":2,"images":[],"hypotheses":[
            {"id":1,"handle":"img:0","statement":"x","state":"proposed","confidence":1e400,"evidence":[]}
        ],"notes":[]}"#,
    );
    let err = occ_session::persist::load(&p).expect_err("1e400 must not load as a confidence");
    assert!(
        err.to_string().contains("confidence") || err.to_string().contains("number"),
        "{err}"
    );
}

#[test]
fn an_image_path_that_climbs_out_of_its_directory_is_refused() {
    let dir = scratch("traversal");
    let p = write(
        &dir,
        "traverse.json",
        r#"{"version":2,"images":[{"id":0,"path":"../../etc/shadow","sha256":""}],"hypotheses":[],"notes":[]}"#,
    );
    let err = occ_session::persist::load(&p).expect_err("traversal must not load");
    assert!(err.to_string().contains("climbs out"), "{err}");
}

#[test]
fn an_image_id_that_does_not_fit_is_refused() {
    let dir = scratch("image-id");
    let p = write(
        &dir,
        "big-id.json",
        r#"{"version":2,"images":[{"id":4294967296,"path":"/bin/ls","sha256":""}],"hypotheses":[],"notes":[]}"#,
    );
    let err = occ_session::persist::load(&p).expect_err("id overflow must not load");
    assert!(err.to_string().contains("does not fit"), "{err}");
}

#[test]
fn evidence_with_an_unknown_kind_is_refused() {
    let dir = scratch("evidence");
    let p = write(
        &dir,
        "bad-evidence.json",
        r#"{"version":2,"images":[],"hypotheses":[
            {"id":1,"handle":"img:0","statement":"x","state":"proposed","confidence":0.0,
             "evidence":[{"kind":"vibes","source":"disasm","at":null,"span":null,"detail":"d","weight":0.1}]}
        ],"notes":[]}"#,
    );
    let err = occ_session::persist::load(&p).expect_err("unknown kind must not load");
    assert!(err.to_string().contains("`kind`"), "{err}");
}
