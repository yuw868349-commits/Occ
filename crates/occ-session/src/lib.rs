//! Session state, handle resolution, persistence, diffing and pattern scanning.
//!
//! This crate is the layer an MCP server talks to. It owns the images, keeps
//! the handles stable across requests, and turns analysis results into the
//! envelope every tool returns.


#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod dbginfo;
pub mod diff;
pub mod pattern;
pub mod persist;
pub mod render;
pub mod session;

pub use session::{ImageFile, Session, Target, TraceRecord};
