//! Session persistence.
//!
//! What is saved is the *question*, not the answer: image paths with their
//! hashes, hypotheses with the standing they had reached, and notes. Analyses
//! are recomputed on load, because a cached analysis that silently disagrees
//! with the file on disk is worse than a two-second delay, and the hash check
//! makes the disagreement visible.
//!
//! A saved session is an input from outside this process: it can be edited,
//! truncated, or written by a different build. So loading validates every
//! field it reads and refuses the file when a value is not one this build can
//! represent, instead of substituting a default and letting the caller believe
//! the session is intact.

use std::path::{Component, Path};

use occ_core::{Addr, CoreError, Evidence, EvidenceKind, EvidenceSource, HypothesisState, Result};
use occ_json::{Map, Value};

use crate::session::Session;

/// Format this build writes.
pub const FORMAT_VERSION: u32 = 2;

/// Oldest format this build can read.
///
/// Version 1 saved the hypothesis table without its standing: no state, no
/// confidence, no evidence. Those files restore as hypotheses that were filed
/// and never examined, which is what version 1 actually recorded.
const OLDEST_VERSION: u32 = 1;

/// One image reference.
#[derive(Clone, Debug)]
pub struct SavedImage {
    /// Session-local id at save time.
    pub id: u32,
    /// Path as opened.
    pub path: String,
    /// SHA-256 at save time.
    pub sha256: String,
}

/// One hypothesis, with the standing it had reached.
#[derive(Clone, Debug)]
pub struct SavedHypothesis {
    /// Id.
    pub id: u64,
    /// Handle string.
    pub handle: String,
    /// Claim under test.
    pub statement: String,
    /// Image the claim is about, when the session knew.
    pub image: Option<u32>,
    /// Where the hypothesis stood.
    pub state: HypothesisState,
    /// Confidence derived from `evidence`.
    pub confidence: f32,
    /// Observations gathered so far.
    pub evidence: Vec<Evidence>,
    /// Sequence number of creation.
    pub created_seq: u64,
    /// Sequence number of the last update.
    pub updated_seq: u64,
}

/// A saved session.
#[derive(Clone, Debug, Default)]
pub struct SavedSession {
    /// Format version.
    pub version: u32,
    /// Images.
    pub images: Vec<SavedImage>,
    /// Hypotheses.
    pub hypotheses: Vec<SavedHypothesis>,
    /// Notes.
    pub notes: Vec<String>,
}

impl Session {
    /// Serialise the session to `path`.
    pub fn save(&self, path: &str) -> Result<()> {
        let mut images = Vec::new();
        for id in self.image_ids() {
            if let Some(f) = self.image(id) {
                images.push(SavedImage {
                    id,
                    path: f.image.path.clone(),
                    sha256: f.image.sha256.clone(),
                });
            }
        }
        let hypotheses: Vec<SavedHypothesis> = self
            .hypotheses()
            .into_iter()
            .map(|h| SavedHypothesis {
                id: h.id,
                handle: h.handle.to_string(),
                statement: h.statement,
                image: h.image,
                state: h.state,
                confidence: h.confidence,
                evidence: h.evidence,
                created_seq: h.created_seq,
                updated_seq: h.updated_seq,
            })
            .collect();
        let saved = SavedSession {
            version: FORMAT_VERSION,
            images,
            hypotheses,
            notes: self.notes(),
        };
        let text = occ_json::to_string_pretty(&to_json(&saved));
        std::fs::write(Path::new(path), text).map_err(|e| CoreError::io(path.to_string(), e))
    }
}

fn to_json(s: &SavedSession) -> Value {
    let mut m = Map::new();
    m.insert("version", Value::from(FORMAT_VERSION));
    m.insert(
        "images",
        Value::Array(
            s.images
                .iter()
                .map(|i| {
                    let mut o = Map::new();
                    o.insert("id", Value::from(i.id));
                    o.insert("path", Value::String(i.path.clone()));
                    o.insert("sha256", Value::String(i.sha256.clone()));
                    Value::Object(o)
                })
                .collect(),
        ),
    );
    m.insert(
        "hypotheses",
        Value::Array(
            s.hypotheses
                .iter()
                .map(|h| {
                    let mut o = Map::new();
                    o.insert("id", Value::from(h.id));
                    o.insert("handle", Value::String(h.handle.clone()));
                    o.insert("statement", Value::String(h.statement.clone()));
                    o.insert(
                        "image",
                        match h.image {
                            Some(id) => Value::from(id),
                            None => Value::Null,
                        },
                    );
                    o.insert("state", Value::String(h.state.label().to_string()));
                    o.insert("confidence", Value::from(f64::from(h.confidence)));
                    o.insert(
                        "evidence",
                        Value::Array(h.evidence.iter().map(evidence_to_json).collect()),
                    );
                    o.insert("created_seq", Value::from(h.created_seq));
                    o.insert("updated_seq", Value::from(h.updated_seq));
                    Value::Object(o)
                })
                .collect(),
        ),
    );
    m.insert(
        "notes",
        Value::Array(s.notes.iter().map(|n| Value::String(n.clone())).collect()),
    );
    Value::Object(m)
}

fn evidence_to_json(e: &Evidence) -> Value {
    let mut o = Map::new();
    o.insert("kind", Value::String(e.kind.label().to_string()));
    o.insert("source", Value::String(e.source.label().to_string()));
    o.insert(
        "at",
        match e.at {
            Some(a) => Value::String(format!("{:#x}", a.0)),
            None => Value::Null,
        },
    );
    o.insert(
        "span",
        match e.span {
            Some(s) => Value::from(s),
            None => Value::Null,
        },
    );
    o.insert("detail", Value::String(e.detail.clone()));
    o.insert("weight", Value::from(f64::from(e.weight)));
    Value::Object(o)
}

fn bad(path: &str, msg: impl std::fmt::Display) -> CoreError {
    CoreError::Backend(format!("{path}: {msg}"))
}

/// Read a saved session.
///
/// Every field is checked: a version this build does not know, an unknown
/// state name, a confidence outside `[0, 1]`, an image path that climbs out of
/// its directory, or a number that does not fit the type it is read into all
/// fail the load rather than being replaced with a default.
pub fn load(path: &str) -> Result<SavedSession> {
    let bytes = std::fs::read(Path::new(path)).map_err(|e| CoreError::io(path.to_string(), e))?;
    let value = occ_json::parse_bytes(&bytes).map_err(|e| bad(path, e))?;
    let obj = occ_json::schema::Obj::new(&value).map_err(|e| bad(path, e))?;

    let version = obj
        .opt_u64("version")
        .map_err(|e| bad(path, e))?
        .ok_or_else(|| bad(path, "missing `version`"))?;
    if version < u64::from(OLDEST_VERSION) || version > u64::from(FORMAT_VERSION) {
        return Err(bad(
            path,
            format!(
                "unsupported session version {version}; this build reads {OLDEST_VERSION}..={FORMAT_VERSION}"
            ),
        ));
    }
    let version = version as u32;

    let mut images = Vec::new();
    if let Some(Value::Array(items)) = value.get("images") {
        for (n, item) in items.iter().enumerate() {
            let o = occ_json::schema::Obj::new(item)
                .map_err(|e| bad(path, format!("images[{n}]: {e}")))?;
            let id = o
                .opt_u64("id")
                .map_err(|e| bad(path, format!("images[{n}]: {e}")))?
                .ok_or_else(|| bad(path, format!("images[{n}]: missing `id`")))?;
            let id = u32::try_from(id).map_err(|_| {
                bad(
                    path,
                    format!("images[{n}]: id {id} does not fit in 32 bits"),
                )
            })?;
            let image_path = o
                .opt_str("path")
                .map_err(|e| bad(path, format!("images[{n}]: {e}")))?
                .ok_or_else(|| bad(path, format!("images[{n}]: missing `path`")))?;
            check_image_path(path, n, image_path)?;
            let sha256 = o
                .opt_str("sha256")
                .map_err(|e| bad(path, format!("images[{n}]: {e}")))?
                .unwrap_or("");
            check_sha256(path, n, sha256)?;
            images.push(SavedImage {
                id,
                path: image_path.to_string(),
                sha256: sha256.to_string(),
            });
        }
    }

    let mut hypotheses = Vec::new();
    if let Some(Value::Array(items)) = value.get("hypotheses") {
        for (n, item) in items.iter().enumerate() {
            let o = occ_json::schema::Obj::new(item)
                .map_err(|e| bad(path, format!("hypotheses[{n}]: {e}")))?;
            let id = o
                .opt_u64("id")
                .map_err(|e| bad(path, format!("hypotheses[{n}]: {e}")))?
                .ok_or_else(|| bad(path, format!("hypotheses[{n}]: missing `id`")))?;
            let handle = o
                .opt_str("handle")
                .map_err(|e| bad(path, format!("hypotheses[{n}]: {e}")))?
                .ok_or_else(|| bad(path, format!("hypotheses[{n}]: missing `handle`")))?;
            let statement = o
                .opt_str("statement")
                .map_err(|e| bad(path, format!("hypotheses[{n}]: {e}")))?
                .ok_or_else(|| bad(path, format!("hypotheses[{n}]: missing `statement`")))?;
            if statement.trim().is_empty() {
                return Err(bad(path, format!("hypotheses[{n}]: `statement` is empty")));
            }

            let image = match o
                .opt_u64("image")
                .map_err(|e| bad(path, format!("hypotheses[{n}]: {e}")))?
            {
                None => None,
                Some(id) => Some(u32::try_from(id).map_err(|_| {
                    bad(
                        path,
                        format!("hypotheses[{n}]: image {id} does not fit in 32 bits"),
                    )
                })?),
            };

            // Version 1 did not record the standing; those hypotheses restore
            // as proposed, which is what version 1 knew.
            let state = match version {
                1 => HypothesisState::Proposed,
                _ => {
                    let label = o
                        .opt_str("state")
                        .map_err(|e| bad(path, format!("hypotheses[{n}]: {e}")))?
                        .ok_or_else(|| bad(path, format!("hypotheses[{n}]: missing `state`")))?;
                    HypothesisState::from_label(label).ok_or_else(|| {
                        bad(
                            path,
                            format!(
                                "hypotheses[{n}]: `state` is not one of proposed, supported, \
                                 refuted, inconclusive: {label:?}"
                            ),
                        )
                    })?
                }
            };

            let confidence = match version {
                1 => 0.0,
                _ => {
                    let raw = o
                        .opt_f64("confidence")
                        .map_err(|e| bad(path, format!("hypotheses[{n}]: {e}")))?
                        .ok_or_else(|| {
                            bad(path, format!("hypotheses[{n}]: missing `confidence`"))
                        })?;
                    if !raw.is_finite() || !(0.0..=1.0).contains(&raw) {
                        return Err(bad(
                            path,
                            format!("hypotheses[{n}]: confidence {raw} is outside [0, 1]"),
                        ));
                    }
                    raw as f32
                }
            };

            let evidence = match version {
                1 => Vec::new(),
                _ => {
                    let mut out = Vec::new();
                    match o.opt_value("evidence") {
                        None | Some(Value::Null) => {}
                        Some(Value::Array(rows)) => {
                            for (k, row) in rows.iter().enumerate() {
                                out.push(evidence_from_json(
                                    path,
                                    format!("hypotheses[{n}].evidence[{k}]"),
                                    row,
                                )?);
                            }
                        }
                        Some(_) => {
                            return Err(bad(
                                path,
                                format!("hypotheses[{n}]: `evidence` is not an array"),
                            ));
                        }
                    }
                    out
                }
            };

            let created_seq = o
                .opt_u64("created_seq")
                .map_err(|e| bad(path, e))?
                .unwrap_or(0);
            let updated_seq = o
                .opt_u64("updated_seq")
                .map_err(|e| bad(path, e))?
                .unwrap_or(0);
            hypotheses.push(SavedHypothesis {
                id,
                handle: handle.to_string(),
                statement: statement.to_string(),
                image,
                state,
                confidence,
                evidence,
                created_seq,
                updated_seq,
            });
        }
    }

    let mut notes = Vec::new();
    if let Some(Value::Array(items)) = value.get("notes") {
        for item in items {
            let Some(s) = item.as_str() else {
                return Err(bad(path, "`notes` holds something that is not a string"));
            };
            notes.push(s.to_string());
        }
    }

    Ok(SavedSession {
        version,
        images,
        hypotheses,
        notes,
    })
}

/// Reject a path this build will not reopen from a session file.
///
/// A session file is data, not code: a path with a `..` component, an empty
/// path or one carrying a NUL byte is a path that resolves somewhere the
/// caller did not name when the session was saved.
fn check_image_path(session: &str, n: usize, p: &str) -> Result<()> {
    if p.trim().is_empty() {
        return Err(bad(session, format!("images[{n}]: `path` is empty")));
    }
    if p.contains('\0') {
        return Err(bad(
            session,
            format!("images[{n}]: `path` carries a NUL byte"),
        ));
    }
    if escapes(Path::new(p)) {
        return Err(bad(
            session,
            format!("images[{n}]: `path` climbs out of its directory with `..`: {p:?}"),
        ));
    }
    Ok(())
}

/// True when the `..` components take the path above where it started.
///
/// `scripts/../tests/x` is inside its tree and legitimate; `../../etc/shadow`
/// is not. Only the ones that run past the start of the path are refused,
/// which is what makes the check about escape rather than about spelling.
fn escapes(p: &Path) -> bool {
    let mut depth = 0i64;
    for c in p.components() {
        match c {
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return true;
                }
            }
            Component::Normal(_) => depth += 1,
            Component::RootDir | Component::Prefix(_) => depth = 0,
            Component::CurDir => {}
        }
    }
    false
}

fn check_sha256(session: &str, n: usize, s: &str) -> Result<()> {
    if s.is_empty() {
        return Ok(());
    }
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(bad(
            session,
            format!("images[{n}]: `sha256` is not 64 hex digits: {s:?}"),
        ));
    }
    Ok(())
}

fn evidence_from_json(session: &str, at: String, value: &Value) -> Result<Evidence> {
    let o = occ_json::schema::Obj::new(value).map_err(|e| bad(session, format!("{at}: {e}")))?;
    let kind_label = o
        .opt_str("kind")
        .map_err(|e| bad(session, format!("{at}: {e}")))?
        .ok_or_else(|| bad(session, format!("{at}: missing `kind`")))?;
    let kind = EvidenceKind::from_label(kind_label).ok_or_else(|| {
        bad(
            session,
            format!("{at}: `kind` is not a known observation kind: {kind_label:?}"),
        )
    })?;
    let source_label = o
        .opt_str("source")
        .map_err(|e| bad(session, format!("{at}: {e}")))?
        .ok_or_else(|| bad(session, format!("{at}: missing `source`")))?;
    let source = EvidenceSource::from_label(source_label).ok_or_else(|| {
        bad(
            session,
            format!("{at}: `source` is not a known stage: {source_label:?}"),
        )
    })?;
    let at_addr = match o
        .opt_str("at")
        .map_err(|e| bad(session, format!("{at}: {e}")))?
    {
        None | Some("") => None,
        Some(s) => Some(
            occ_core::addr::parse_addr(s)
                .ok_or_else(|| bad(session, format!("{at}: `at` is not an address: {s:?}")))?,
        ),
    };
    let span = o
        .opt_u64("span")
        .map_err(|e| bad(session, format!("{at}: {e}")))?;
    let detail = o
        .opt_str("detail")
        .map_err(|e| bad(session, format!("{at}: {e}")))?
        .ok_or_else(|| bad(session, format!("{at}: missing `detail`")))?
        .to_string();
    let weight = o
        .opt_f64("weight")
        .map_err(|e| bad(session, format!("{at}: {e}")))?
        .ok_or_else(|| bad(session, format!("{at}: missing `weight`")))?;
    if !weight.is_finite() || !(-1.0..=1.0).contains(&weight) {
        return Err(bad(
            session,
            format!("{at}: weight {weight} is outside [-1, 1]"),
        ));
    }
    Ok(Evidence {
        kind,
        source,
        at: at_addr.map(|a: Addr| a),
        span,
        detail,
        weight: weight as f32,
    })
}
