//! Error types shared across the workspace.
//!
//! Written out by hand rather than derived. The `thiserror` crate would save
//! about forty lines and cost a dependency, and the `Display` text is part of
//! what a caller reads — having it in one `match` next to the variants makes
//! inconsistencies visible, where a pile of `#[error("...")]` attributes hides
//! them behind the enum.

use std::fmt;

/// Convenience alias used throughout the workspace.
pub type Result<T, E = CoreError> = std::result::Result<T, E>;

/// Failures that can occur in the domain layer.
///
/// Parsing failures carry the byte offset where the interpretation broke down:
/// "malformed import table" is useless to an analyst, "import table at 0x2a0
/// claims 4096 descriptors but the section ends at 0x2b4" is actionable.
#[derive(Debug)]
#[non_exhaustive]
pub enum CoreError {
    /// A handle did not match the documented grammar.
    BadHandle {
        /// The offending input.
        input: String,
        /// Comma-separated list of accepted kinds.
        kinds: &'static str,
    },

    /// A handle was well formed but names nothing in the current session.
    DanglingHandle {
        /// The handle that failed to resolve.
        handle: String,
        /// Why it does not resolve.
        detail: String,
    },

    /// An address fell outside every mapped region.
    Unmapped {
        /// The address, rendered as hex.
        addr: String,
        /// Image identifier.
        image: u32,
    },

    /// The file's container format is recognised but unsupported.
    UnsupportedFormat(String),

    /// The container parsed but its contents are inconsistent.
    Malformed {
        /// Byte offset into the file.
        offset: u64,
        /// What contradicted itself.
        detail: String,
    },

    /// A requested capability is compiled out or unavailable on this host.
    Unavailable(String),

    /// A parser rejected the input without a usable offset.
    ///
    /// Kept distinct from [`CoreError::Malformed`] because the distinction is
    /// diagnostic: `Malformed` means *we* found the inconsistency and can say
    /// where, while `Backend` means the failure came from below with less
    /// context than we would like.
    Backend(String),

    /// The operation exceeded a caller-supplied limit.
    LimitExceeded(String),

    /// Filesystem failure, with the path attached.
    Io {
        /// Path involved.
        path: String,
        /// Underlying error.
        source: std::io::Error,
    },

    /// A system call failed.
    Sys(occ_sys::SysError),

    /// A session-level lookup found nothing by that name or address.
    ///
    /// Distinct from [`CoreError::DanglingHandle`]: a dangling handle is a
    /// handle that used to resolve or was expected to, while this is any
    /// lookup — a string search, a symbol name, a function address.
    NotFound {
        /// What was being looked for, with the caller's spelling.
        what: String,
    },

    /// An image id is not open in this session.
    NoSuchImage {
        /// The id that was requested.
        id: u32,
    },

    /// An internal invariant was violated. Always a bug in this crate, never
    /// in the input.
    Internal(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::BadHandle { input, kinds } => write!(
                f,
                "malformed handle {input:?}: expected `<kind>:<value>` where kind is one of {kinds}"
            ),
            CoreError::DanglingHandle { handle, detail } => {
                write!(f, "dangling handle {handle}: {detail}")
            }
            CoreError::Unmapped { addr, image } => {
                write!(f, "address {addr} is not mapped by image {image}")
            }
            CoreError::UnsupportedFormat(what) => write!(f, "unsupported format: {what}"),
            CoreError::Malformed { offset, detail } => {
                write!(f, "malformed binary at offset {offset:#x}: {detail}")
            }
            CoreError::Unavailable(what) => write!(f, "capability unavailable: {what}"),
            CoreError::Backend(what) => write!(f, "backend failure: {what}"),
            CoreError::LimitExceeded(what) => write!(f, "limit exceeded: {what}"),
            CoreError::Io { path, source } => write!(f, "io error on {path}: {source}"),
            CoreError::Sys(e) => write!(f, "{e}"),
            CoreError::NotFound { what } => write!(f, "not found: {what}"),
            CoreError::NoSuchImage { id } => {
                write!(f, "image {id} is not open in this session")
            }
            CoreError::Internal(detail) => write!(f, "internal error: {detail}"),
        }
    }
}

impl std::error::Error for CoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CoreError::Io { source, .. } => Some(source),
            CoreError::Sys(e) => Some(e),
            _ => None,
        }
    }
}

impl From<occ_sys::SysError> for CoreError {
    fn from(e: occ_sys::SysError) -> Self {
        CoreError::Sys(e)
    }
}

impl CoreError {
    /// Attach a path to an [`std::io::Error`].
    pub fn io(path: impl Into<String>, source: std::io::Error) -> Self {
        CoreError::Io {
            path: path.into(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_errors_name_the_offset() {
        let e = CoreError::Malformed {
            offset: 0x2a0,
            detail: "import table claims 4096 descriptors".into(),
        };
        let text = e.to_string();
        assert!(text.contains("0x2a0"), "{text}");
        assert!(text.contains("4096"), "{text}");
    }

    #[test]
    fn io_errors_chain_to_the_cause() {
        let e = CoreError::io(
            "/nope",
            std::io::Error::new(std::io::ErrorKind::NotFound, "gone"),
        );
        assert!(e.to_string().contains("/nope"));
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn sys_errors_convert_and_display() {
        let e: CoreError = occ_sys::SysError {
            op: "ptrace(PTRACE_GETREGS)",
            code: 3,
        }
        .into();
        assert!(e.to_string().contains("ESRCH"), "{e}");
    }
}
