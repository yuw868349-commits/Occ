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
    ///
    /// A negative number, a float or a string is a caller mistake, not a
    /// missing argument: `-1` does not mean "use the default", so the value
    /// that was sent is named in the error instead of being dropped.
    pub fn opt_u64(&self, key: &str) -> Result<Option<u64>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Number(n)) => n.as_u64().map(Some).ok_or_else(|| {
                format!("`{key}` must be a non-negative integer, got {n}")
            }),
            Some(other) => Err(format!(
                "`{key}` must be a non-negative integer, got {}",
                other.type_name()
            )),
        }
    }

    /// Integer with a default and a closed range.
    ///
    /// Values below `min` and above `max` are both refused: a cap the caller
    /// was not told about turns "keep 500 paths" into "the tool silently kept
    /// 200" and the caller cannot tell the difference from having found only
    /// 200. The message names the bound so the retry is obvious.
    pub fn u64_in(&self, key: &str, default: u64, min: u64, max: u64) -> Result<u64, String> {
        let Some(v) = self.opt_u64(key)? else {
            return Ok(default);
        };
        if v < min || v > max {
            let bound = if min == 0 {
                format!("at most {max}")
            } else {
                format!("between {min} and {max}")
            };
            return Err(format!("`{key}` must be {bound}, got {v}"));
        }
        Ok(v)
    }

    /// Boolean with a default.
    pub fn bool_or(&self, key: &str, default: bool) -> Result<bool, String> {
        match self.get(key) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(other) => Err(format!("`{key}` must be true or false, got {}", other.type_name())),
        }
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

    /// A string that must be one of `allowed`, or `default` when absent.
    ///
    /// An unrecognised value is refused with the list, because a misspelt
    /// `sort` that quietly falls back to address order looks like a tool that
    /// ignored the request.
    pub fn one_of<'a>(
        &'a self,
        key: &str,
        default: &'a str,
        allowed: &[&str],
    ) -> Result<&'a str, String> {
        match self.get(key) {
            None => Ok(default),
            Some(Value::String(s)) if allowed.contains(&s.as_str()) => Ok(s),
            Some(Value::String(s)) => Err(format!(
                "`{key}` is {s:?}; use one of {}",
                allowed.join(", ")
            )),
            Some(other) => Err(format!("`{key}` must be a string, got {}", other.type_name())),
        }
    }

    /// Render depth, defaulting to `detail`.
    pub fn depth(&self) -> Result<Depth, String> {
        Ok(match self.one_of("depth", "detail", &["summary", "detail", "raw"])? {
            "summary" => Depth::Summary,
            "raw" => Depth::Raw,
            _ => Depth::Detail,
        })
    }

    /// Pagination as `(offset, limit)`. Callers compute truncation as
    /// `total > offset + limit`, which is the only thing they need.
    pub fn page(&self, default_limit: usize, hard_max: usize) -> Result<(usize, usize), String> {
        let offset = self.opt_u64("offset")?.unwrap_or(0);
        let offset = usize::try_from(offset)
            .map_err(|_| format!("`offset` does not fit in memory: {offset}"))?;
        let limit = self.u64_in("limit", default_limit as u64, 1, hard_max as u64)? as usize;
        Ok((offset, limit))
    }

    /// Select the image this call applies to.
    ///
    /// An explicit `image` always wins. Otherwise a session with exactly one
    /// open image resolves implicitly; with several, the error lists the ids
    /// because guessing would analyse the wrong binary and look like a wrong
    /// answer rather than a missing argument.
    pub fn image(&self, session: &Session) -> Result<u32, String> {
        if let Some(id) = self.opt_u64("image")? {
            // `as u32` would wrap 2^32 into image 0 and silently analyse the
            // wrong binary.
            let id = u32::try_from(id)
                .map_err(|_| format!("`image` is not an id this server issues: {id}"))?;
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
