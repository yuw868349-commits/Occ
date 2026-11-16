//! Session persistence.
//!
//! What is saved is the *question*, not the answer: image paths with their
//! hashes, hypotheses, and notes. Analyses are recomputed on load, because a
//! cached analysis that silently disagrees with the file on disk is worse than
//! a two-second delay, and the hash check makes the disagreement visible.

use std::path::Path;

use occ_core::{CoreError, Result};
use occ_json::{Map, Value};

use crate::session::Session;

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

/// One hypothesis.
#[derive(Clone, Debug)]
pub struct SavedHypothesis {
    /// Id.
    pub id: u64,
    /// Handle string.
    pub handle: String,
    /// Claim under test.
    pub statement: String,
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
            })
            .collect();
        let saved = SavedSession {
            version: 1,
            images,
            hypotheses,
            notes: self.notes(),
        };
        let text = occ_json::to_string_pretty(&to_json(&saved));
        std::fs::write(Path::new(path), text)
            .map_err(|e| CoreError::io(path.to_string(), e))
    }
}

fn to_json(s: &SavedSession) -> Value {
    let mut m = Map::new();
    m.insert("version", Value::from(1u32));
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

/// Read a saved session.
pub fn load(path: &str) -> Result<SavedSession> {
    let bytes = std::fs::read(Path::new(path)).map_err(|e| CoreError::io(path.to_string(), e))?;
    let value = occ_json::parse_bytes(&bytes).map_err(|e| CoreError::Backend(format!("{path}: {e}")))?;
    let obj = occ_json::schema::Obj::new(&value)
        .map_err(|e| CoreError::Backend(format!("{path}: {e}")))?;
    let version = obj.opt_u64("version").ok().flatten().unwrap_or(1) as u32;
    let mut images = Vec::new();
    if let Some(Value::Array(items)) = value.get("images") {
        for item in items {
            let o = occ_json::schema::Obj::new(item)
                .map_err(|e| CoreError::Backend(format!("{path}: {e}")))?;
            images.push(SavedImage {
                id: o.opt_u64("id").ok().flatten().unwrap_or(0) as u32,
                path: o.opt_str("path").ok().flatten().unwrap_or("").to_string(),
                sha256: o.opt_str("sha256").ok().flatten().unwrap_or("").to_string(),
            });
        }
    }
    let mut hypotheses = Vec::new();
    if let Some(Value::Array(items)) = value.get("hypotheses") {
        for item in items {
            let o = occ_json::schema::Obj::new(item)
                .map_err(|e| CoreError::Backend(format!("{path}: {e}")))?;
            hypotheses.push(SavedHypothesis {
                id: o.opt_u64("id").ok().flatten().unwrap_or(0),
                handle: o.opt_str("handle").ok().flatten().unwrap_or("").to_string(),
                statement: o.opt_str("statement").ok().flatten().unwrap_or("").to_string(),
            });
        }
    }
    let mut notes = Vec::new();
    if let Some(Value::Array(items)) = value.get("notes") {
        for item in items {
            if let Some(s) = item.as_str() {
                notes.push(s.to_string());
            }
        }
    }
    Ok(SavedSession {
        version,
        images,
        hypotheses,
        notes,
    })
}
