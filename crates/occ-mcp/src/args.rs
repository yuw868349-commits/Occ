//! Typed access to tool arguments.
//!
//! Every accessor reports a sentence the model can act on — "`addr` is
//! required and must be a hex address" — rather than a JSON pointer. The
//! caller of a tool is a language model; an error it cannot understand is a
//! failed turn.

use occ_core::{Addr, Depth, Image};
use occ_json::Value;
use occ_session::session::Session;
use std::sync::Arc;

use occ_session::ImageFile;

/// Wrapper around the `arguments` object.
#[derive(Clone)]
pub struct Args {
    value: Value,
}

impl Args {
    /// Wrap a JSON object.
    pub fn new(value: &Value) -> Args {
        Args {
            value: value.clone(),
        }
    }

    fn get(&self, key: &str) -> Option<&Value> {
        self.value.get(key).filter(|v| !v.is_null())
    }

    /// Raw value for a key.
    pub fn raw(&self, key: &str) -> Option<&Value> {
        self.get(key)
    }

    /// Required string.
    pub fn req_str(&self, key: &str) -> Result<&str, String> {
        match self.get(key) {
            Some(Value::String(s)) => Ok(s),
            Some(_) => Err(format!("`{key}` must be a string")),
            None => Err(format!("`{key}` is required")),
        }
    }

    /// Optional string.
    pub fn opt_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }

    /// Optional unsigned integer.
    pub fn opt_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64())
    }

    /// Unsigned integer with a default.
    pub fn u64_or(&self, key: &str, default: u64) -> u64 {
        self.opt_u64(key).unwrap_or(default)
    }

    /// Unsigned integer clamped to a range.
    pub fn u64_clamped(&self, key: &str, default: u64, max: u64) -> u64 {
        self.opt_u64(key).unwrap_or(default).min(max)
    }

    /// Boolean with a default.
    pub fn bool_or(&self, key: &str, default: bool) -> bool {
        self.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    }

    /// Required address, accepting `0x...`, decimal, or a bare hex string.
    pub fn req_addr(&self, key: &str) -> Result<Addr, String> {
        let s = self.req_str(key)?;
        occ_core::addr::parse_addr(s)
            .ok_or_else(|| format!("`{key}` is not an address: {s:?} (expected 0x401000)"))
    }

    /// Optional address.
    pub fn opt_addr(&self, key: &str) -> Result<Option<Addr>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::String(s)) => occ_core::addr::parse_addr(s)
                .map(Some)
                .ok_or_else(|| format!("`{key}` is not an address: {s:?}")),
            Some(v) => v
                .as_u64()
                .map(|n| Some(Addr(n)))
                .ok_or_else(|| format!("`{key}` must be a hex string or an integer")),
        }
    }

    /// Render depth, defaulting to `detail`.
    pub fn depth(&self) -> Depth {
        match self.opt_str("depth") {
            Some("summary") => Depth::Summary,
            Some("raw") => Depth::Raw,
            _ => Depth::Detail,
        }
    }

    /// Pagination as `(offset, limit)`. Callers compute truncation as
    /// `total > offset + limit`, which is the only thing they need.
    pub fn page(&self, default_limit: usize, hard_max: usize) -> (usize, usize) {
        let offset = self.opt_u64("offset").unwrap_or(0) as usize;
        let limit = self
            .opt_u64("limit")
            .map(|v| v as usize)
            .unwrap_or(default_limit)
            .clamp(1, hard_max);
        (offset, limit)
    }

    /// Select the image this call applies to.
    ///
    /// An explicit `image` always wins. Otherwise a session with exactly one
    /// open image resolves implicitly; with several, the error lists the ids
    /// because guessing would analyse the wrong binary and look like a wrong
    /// answer rather than a missing argument.
    pub fn image(&self, session: &Session) -> Result<u32, String> {
        if let Some(id) = self.opt_u64("image") {
            let id = id as u32;
            if session.image(id).is_none() {
                return Err(format!(
                    "image {id} is not open; open images: {:?}",
                    session.image_ids()
                ));
            }
            return Ok(id);
        }
        match session.only_image() {
            Some(id) => Ok(id),
            None => {
                let ids = session.image_ids();
                if ids.is_empty() {
                    Err("no image is open; call binary.open first".into())
                } else {
                    Err(format!(
                        "several images are open; pass `image` (one of {ids:?})"
                    ))
                }
            }
        }
    }

    /// Fetch the selected image.
    pub fn file(&self, session: &Session) -> Result<Arc<ImageFile>, String> {
        let id = self.image(session)?;
        session
            .image(id)
            .ok_or_else(|| format!("image {id} disappeared from the session"))
    }
}

/// Section names in the image, for schemas and error messages.
pub fn section_names(image: &Image) -> Vec<String> {
    image.sections.iter().map(|s| s.name.clone()).collect()
}
