//! Request dispatch and the two transports.
//!
//! One function, [`handle`], turns a message body into an optional response.
//! Both transports share it, so a bug in tool dispatch is reproducible over
//! stdio and HTTP without a second code path.
//!
//! The HTTP transport implements the request/response half of Streamable HTTP:
//! `POST /mcp` with a JSON-RPC body, answered with `application/json`. It is
//! deliberately not a general web server — no keep-alive pipelining beyond one
//! request per connection, no chunked encoding, no TLS. Those belong in front
//! of it, not in it.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use occ_json::{Map, Value};
use occ_session::session::Session;

use crate::args::Args;
use crate::protocol::{self, Incoming, code};
use crate::{SERVER_NAME, SERVER_VERSION, tools};
use occ_session::obj;

/// Protocol versions this server can speak, newest first.
const PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// Handle one message body. `None` means "send nothing" (notification or
/// response).
pub fn handle(session: &Session, body: &str) -> Option<Value> {
    let incoming = match protocol::parse_message(body) {
        Ok(m) => m,
        Err((id, c, msg)) => return Some(protocol::error(id, c, msg, None)),
    };
    match incoming {
        Incoming::Response => None,
        Incoming::Notification { .. } => None,
        Incoming::Request { id, method, params } => {
            let result = dispatch(session, &method, &params);
            Some(match result {
                Ok(v) => protocol::success(id, v),
                Err((c, msg)) => protocol::error(id, c, msg, None),
            })
        }
    }
}

fn dispatch(session: &Session, method: &str, params: &Value) -> Result<Value, (i32, String)> {
    match method {
        "initialize" => Ok(initialize(params)),
        "ping" => Ok(Value::Object(Map::new())),
        "tools/list" => {
            let mut m = Map::new();
            m.insert("tools", tools::list_json());
            Ok(Value::Object(m))
        }
        "tools/call" => call_tool(session, params),
        "resources/list" => {
            let mut m = Map::new();
            m.insert("resources", Value::Array(Vec::new()));
            Ok(Value::Object(m))
        }
        "prompts/list" => {
            let mut m = Map::new();
            m.insert("prompts", Value::Array(Vec::new()));
            Ok(Value::Object(m))
        }
        other => Err((
            code::METHOD_NOT_FOUND,
            format!("unknown method {other:?}; supported: initialize, ping, tools/list, tools/call"),
        )),
    }
}

fn initialize(params: &Value) -> Value {
    let requested = params
        .get("protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or(PROTOCOL_VERSIONS[0]);
    // Echo the client's version when it is one we know; otherwise offer the
    // newest. A client that sees its own version assumes compatibility, so
    // confirming a version we do not speak would be a lie.
    let version = if PROTOCOL_VERSIONS.contains(&requested) {
        requested
    } else {
        PROTOCOL_VERSIONS[0]
    };
    obj!(
        "protocolVersion" => version,
        "capabilities" => obj!(
            "tools" => obj!("listChanged" => false),
        ),
        "serverInfo" => obj!(
            "name" => SERVER_NAME,
            "version" => SERVER_VERSION,
            "title" => "occ — binary reverse engineering for AI clients",
        ),
        "instructions" => "Handle-first binary analysis. Call binary.open once, then address everything by handle \
(fn:0x401000, str:0x402000, insn:0x401004). Responses carry a `handle` you can reuse and a `next` \
list of follow-ups. Conclusions include evidence and a confidence value; heuristics are labelled.",
    )
}

fn call_tool(session: &Session, params: &Value) -> Result<Value, (i32, String)> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or((code::INVALID_PARAMS, "missing `name`".to_string()))?;
    let def = tools::find(name).ok_or((
        code::INVALID_PARAMS,
        format!("unknown tool {name:?}; call tools/list to see the catalogue"),
    ))?;
    let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);
    if !matches!(arguments, Value::Object(_) | Value::Null) {
        return Err((
            code::INVALID_PARAMS,
            "`arguments` must be an object".into(),
        ));
    }
    let args = Args::new(&arguments);
    match (def.handler)(session, &args) {
        Ok(value) => Ok(tool_result(&value, false)),
        // Tool failures are returned as a result with `isError`, not as a
        // JSON-RPC error: the model is supposed to read the message and retry,
        // and most clients surface protocol errors as a dead connection.
        Err(msg) => Ok(tool_result(
            &obj!(
                "ok" => false,
                "error" => msg,
                "tool" => def.name,
            ),
            true,
        )),
    }
}

fn tool_result(value: &Value, is_error: bool) -> Value {
    let text = occ_json::to_string_pretty(value);
    let mut m = Map::new();
    m.insert(
        "content",
        Value::Array(vec![obj!(
            "type" => "text",
            "text" => text,
        )]),
    );
    m.insert("structuredContent", value.clone());
    m.insert("isError", is_error);
    Value::Object(m)
}

/// Serve MCP over stdin/stdout until end of input.
pub fn serve_stdio() -> io::Result<()> {
    let session = Session::new();
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut out = stdout.lock();
    while let Some(body) = protocol::read_message(&mut reader)? {
        if let Some(response) = handle(&session, &body) {
            protocol::write_message(&mut out, &response)?;
        }
    }
    Ok(())
}

/// Serve MCP over HTTP.
///
/// `POST /mcp` takes one JSON-RPC message and returns one response. `GET
/// /health` reports liveness for supervisors. Everything else is 404, because
/// a server that answers unrelated paths invites a proxy to cache them.
pub fn serve_http(addr: &str) -> io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let session = Arc::new(Session::new());
    eprintln!("occ listening on http://{addr}/mcp");
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let session = Arc::clone(&session);
        // One thread per connection: tool calls are CPU-bound and independent,
        // and the session already serialises its own mutations.
        std::thread::spawn(move || {
            let _ = serve_connection(stream, &session);
        });
    }
    Ok(())
}

fn serve_connection(mut stream: TcpStream, session: &Session) -> io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(60)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':')
            && k.eq_ignore_ascii_case("content-length")
        {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    if content_length > protocol::MAX_MESSAGE {
        return respond(&mut stream, 413, "text/plain", b"message too large");
    }

    if method == "GET" && path == "/health" {
        return respond(&mut stream, 200, "text/plain", b"ok");
    }
    if method != "POST" || !path.starts_with("/mcp") {
        return respond(&mut stream, 404, "text/plain", b"not found");
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    let body = String::from_utf8_lossy(&body).into_owned();
    match handle(session, &body) {
        Some(v) => respond(
            &mut stream,
            200,
            "application/json",
            occ_json::to_string(&v).as_bytes(),
        ),
        None => respond(&mut stream, 202, "text/plain", b"accepted"),
    }
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        202 => "Accepted",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(session: &Session, body: &str) -> Value {
        handle(session, body).expect("a response")
    }

    #[test]
    fn initialize_echoes_a_known_version() {
        let s = Session::new();
        let r = call(
            &s,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
        );
        let v = r.get("result").unwrap();
        assert_eq!(v.get("protocolVersion").unwrap().as_str(), Some("2024-11-05"));
        assert!(v.get("capabilities").is_some());
    }

    #[test]
    fn unknown_method_is_an_error() {
        let s = Session::new();
        let r = call(&s, r#"{"jsonrpc":"2.0","id":2,"method":"nope"}"#);
        assert_eq!(
            r.get("error").unwrap().get("code").unwrap().as_i64(),
            Some(i64::from(code::METHOD_NOT_FOUND))
        );
    }

    #[test]
    fn tools_list_is_non_empty() {
        let s = Session::new();
        let r = call(&s, r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#);
        let tools = r
            .get("result")
            .unwrap()
            .get("tools")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(tools.len() > 10);
        assert!(tools.iter().any(|t| t.get("name").and_then(|n| n.as_str()) == Some("binary.open")));
    }

    #[test]
    fn tool_errors_come_back_as_iserror() {
        let s = Session::new();
        let r = call(
            &s,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"session.info","arguments":{}}}"#,
        );
        let result = r.get("result").unwrap();
        assert_eq!(result.get("isError").and_then(|v| v.as_bool()), Some(false));
        assert!(result.get("structuredContent").is_some());
    }

    #[test]
    fn notifications_produce_no_response() {
        let s = Session::new();
        assert!(handle(&s, r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());
    }
}
