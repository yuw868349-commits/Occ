//! JSON-RPC 2.0 and MCP framing.
//!
//! Two transports are supported and both are implemented here rather than
//! pulled in:
//!
//! * **stdio** — one JSON object per line, which is what MCP specifies for
//!   subprocess transports. A `Content-Length: N` header block is also accepted
//!   on input, because several clients inherited that framing from LSP and
//!   send it by mistake; rejecting it would produce a connection that works
//!   with one client and silently fails with another.
//! * **HTTP** — one POST per message, response as `application/json`. This is
//!   the request/response half of the Streamable HTTP transport and is what
//!   every client uses for tool calls.
//!
//! Nothing here buffers more than one message, and nothing trusts a declared
//! length: a 2 GiB `Content-Length` from a hostile client is rejected rather
//! than allocated.

use std::io::{self, BufRead, Write};

use occ_json::{Map, Value};

/// Largest message accepted, in bytes.
///
/// 8 MiB is far above any legitimate request (the biggest is a disassembly
/// range, a few hundred KiB) and far below what would exhaust memory.
pub const MAX_MESSAGE: usize = 8 * 1024 * 1024;

/// JSON-RPC error codes used by this server.
pub mod code {
    /// Invalid JSON.
    pub const PARSE_ERROR: i32 = -32700;
    /// Valid JSON, not a valid request.
    pub const INVALID_REQUEST: i32 = -32600;
    /// Unknown method.
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Bad parameters.
    pub const INVALID_PARAMS: i32 = -32602;
    /// Handler failure.
    pub const INTERNAL_ERROR: i32 = -32603;
    /// Tool executed and failed; the message is for the model to read.
    pub const TOOL_ERROR: i32 = -32000;
}

/// An incoming message.
#[derive(Clone, Debug)]
pub enum Incoming {
    /// A request expecting a response.
    Request {
        /// Id, echoed back.
        id: Value,
        /// Method name.
        method: String,
        /// Parameters.
        params: Value,
    },
    /// A notification expecting no response.
    Notification {
        /// Method name.
        method: String,
        /// Parameters.
        params: Value,
    },
    /// A response to something this server sent. Ignored, but accepted.
    Response,
}

/// Parse one message body.
pub fn parse_message(body: &str) -> Result<Incoming, (Value, i32, String)> {
    let value = occ_json::parse(body)
        .map_err(|e| (Value::Null, code::PARSE_ERROR, format!("invalid JSON: {e}")))?;
    let obj = match &value {
        Value::Object(o) => o,
        _ => {
            return Err((
                Value::Null,
                code::INVALID_REQUEST,
                "message must be a JSON object".into(),
            ));
        }
    };
    let id = obj.get("id").cloned().unwrap_or(Value::Null);
    let method = match obj.get("method") {
        Some(Value::String(s)) => s.clone(),
        Some(_) => {
            return Err((
                id,
                code::INVALID_REQUEST,
                "`method` must be a string".into(),
            ));
        }
        None => {
            // No method and an id: this is a response to a request we sent
            // (there are none), or a malformed message. Either way there is
            // nothing to do and nothing to answer.
            return Ok(Incoming::Response);
        }
    };
    let params = obj.get("params").cloned().unwrap_or(Value::Null);
    Ok(if id.is_null() {
        Incoming::Notification { method, params }
    } else {
        Incoming::Request { id, method, params }
    })
}

/// Build a success response.
pub fn success(id: Value, result: Value) -> Value {
    let mut m = Map::new();
    m.insert("jsonrpc", "2.0");
    m.insert("id", id);
    m.insert("result", result);
    Value::Object(m)
}

/// Build an error response.
pub fn error(id: Value, code: i32, message: impl Into<String>, data: Option<Value>) -> Value {
    let mut e = Map::new();
    e.insert("code", i64::from(code));
    e.insert("message", message.into());
    if let Some(d) = data {
        e.insert("data", d);
    }
    let mut m = Map::new();
    m.insert("jsonrpc", "2.0");
    m.insert("id", id);
    m.insert("error", Value::Object(e));
    Value::Object(m)
}

/// Read one message.
///
/// Accepts both framings and enforces [`MAX_MESSAGE`]. Returns `Ok(None)` at
/// end of input, which is a normal disconnect, not an error.
pub fn read_message<R: BufRead>(reader: &mut R) -> io::Result<Option<String>> {
    let mut first = String::new();
    loop {
        first.clear();
        let n = reader.read_line(&mut first)?;
        if n == 0 {
            return Ok(None);
        }
        if !first.trim().is_empty() {
            break;
        }
    }
    if first.to_ascii_lowercase().starts_with("content-length:") {
        let len: usize = first
            .split_once(':')
            .and_then(|(_, v)| v.trim().parse().ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad Content-Length"))?;
        if len > MAX_MESSAGE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "message exceeds MAX_MESSAGE",
            ));
        }
        // Consume the rest of the header block.
        loop {
            let mut h = String::new();
            let n = reader.read_line(&mut h)?;
            if n == 0 {
                return Ok(None);
            }
            if h == "\r\n" || h == "\n" {
                break;
            }
        }
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        return String::from_utf8(buf)
            .map(Some)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "message is not UTF-8"));
    }
    if first.len() > MAX_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "message exceeds MAX_MESSAGE",
        ));
    }
    Ok(Some(first.trim_end().to_string()))
}

/// Write one message, newline delimited.
pub fn write_message<W: Write>(writer: &mut W, value: &Value) -> io::Result<()> {
    let text = occ_json::to_string(value);
    writer.write_all(text.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_requests_and_notifications() {
        let r = parse_message(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#).unwrap();
        assert!(matches!(r, Incoming::Request { .. }));
        let n = parse_message(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).unwrap();
        assert!(matches!(n, Incoming::Notification { .. }));
    }

    #[test]
    fn rejects_garbage_with_a_code() {
        let (_, c, _) = parse_message("{").unwrap_err();
        assert_eq!(c, code::PARSE_ERROR);
        let (_, c, _) = parse_message("[]").unwrap_err();
        assert_eq!(c, code::INVALID_REQUEST);
    }

    #[test]
    fn reads_newline_framing() {
        let data = b"{\"a\":1}\n{\"b\":2}\n";
        let mut r = &data[..];
        assert_eq!(read_message(&mut r).unwrap().as_deref(), Some("{\"a\":1}"));
        assert_eq!(read_message(&mut r).unwrap().as_deref(), Some("{\"b\":2}"));
        assert!(read_message(&mut r).unwrap().is_none());
    }

    #[test]
    fn reads_content_length_framing() {
        let data = b"Content-Length: 7\r\n\r\n{\"a\":1}";
        let mut r = &data[..];
        assert_eq!(read_message(&mut r).unwrap().as_deref(), Some("{\"a\":1}"));
    }

    #[test]
    fn refuses_oversized_length() {
        let data = b"Content-Length: 999999999\r\n\r\n";
        let mut r = &data[..];
        assert!(read_message(&mut r).is_err());
    }

    #[test]
    fn responses_echo_the_id() {
        let v = success(Value::from(7u64), Value::Null);
        let text = occ_json::to_string(&v);
        assert!(text.contains("\"id\":7"));
        assert!(text.contains("\"jsonrpc\":\"2.0\""));
    }
}
