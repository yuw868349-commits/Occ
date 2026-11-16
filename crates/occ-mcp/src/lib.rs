//! MCP server for `occ`.
//!
//! The server speaks JSON-RPC 2.0 over stdio or HTTP, exposes a tool registry,
//! and keeps one [`occ_session::Session`] alive for the lifetime of the
//! process so handles stay resolvable across requests.


#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod args;
pub mod protocol;
pub mod server;
pub mod tools;

/// Server name reported in `initialize`.
pub const SERVER_NAME: &str = "occ";
/// Version reported in `initialize`.
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
