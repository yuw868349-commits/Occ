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
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
            format!(
                "unknown method {other:?}; supported: initialize, ping, tools/list, tools/call"
            ),
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
        return Err((code::INVALID_PARAMS, "`arguments` must be an object".into()));
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

/// Configuration for the HTTP transport.
///
/// The default is loopback only and no token: a local MCP client can start
/// the server and talk to it without ceremony. Binding any other interface is
/// a different decision, because the tool set can read and write arbitrary
/// files and run programs. That requires both an explicit `--allow-remote` and
/// a bearer token, so a misconfigured command line cannot silently expose the
/// machine.
pub struct HttpConfig {
    /// Address to bind, e.g. `127.0.0.1:8787` or `:8787`.
    pub addr: String,
    /// Bearer token required on `/mcp`. Empty means no authentication.
    pub token: Option<String>,
    /// True when a non-loopback address may be bound.
    pub allow_remote: bool,
    /// Maximum simultaneous connections.
    pub max_connections: usize,
}

impl Default for HttpConfig {
    fn default() -> Self {
        HttpConfig {
            addr: "127.0.0.1:8787".into(),
            token: None,
            allow_remote: false,
            max_connections: 64,
        }
    }
}

/// Parse and validate the bind address.
///
/// `:8787` is accepted as shorthand for loopback, matching the README, because
/// the empty host in a URL means "this host". A bare port is not accepted: at
/// the command line it is ambiguous with an image path.
pub fn bind_address(addr: &str) -> Result<SocketAddr, String> {
    let text = if addr.starts_with(':') {
        format!("127.0.0.1{addr}")
    } else if let Some(port) = addr.strip_prefix("localhost:") {
        format!("127.0.0.1:{port}")
    } else {
        addr.to_string()
    };
    let parsed: SocketAddr = text.parse().map_err(|_| {
        format!(
            "`{addr}` is not an IP socket address; use IP:PORT such as 127.0.0.1:8787, localhost:8787 or :8787"
        )
    })?;
    Ok(parsed)
}

/// Refuse a configuration that would expose the tool set without a gate.
fn check_bind(addr: SocketAddr, cfg: &HttpConfig) -> Result<(), String> {
    if addr.ip().is_loopback() {
        return Ok(());
    }
    if !cfg.allow_remote {
        return Err(format!(
            "refusing to bind {addr}: it is not loopback. Pass --allow-remote to accept remote connections"
        ));
    }
    match cfg.token.as_deref() {
        Some(t) if t.len() >= 16 => Ok(()),
        Some(_) => Err("a remote bind requires --token of at least 16 characters".into()),
        None => {
            Err("a remote bind requires a bearer token; pass --token or set OCC_HTTP_TOKEN".into())
        }
    }
}

/// Constant-time comparison for the bearer token.
///
/// A short-circuiting `==` leaks the length of the matching prefix through
/// timing. The threat here is a local process guessing a loopback token; that
/// is small, but the comparison is cheap and the alternative has no upside.
fn token_matches(expected: &str, supplied: &str) -> bool {
    let a = expected.as_bytes();
    let b = supplied.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Serve MCP over HTTP.
///
/// `POST /mcp` takes one JSON-RPC message and returns one response. `GET
/// /health` reports liveness for supervisors. Everything else is 404.
pub fn serve_http(cfg: &HttpConfig) -> io::Result<()> {
    let addr = bind_address(&cfg.addr).map_err(io::Error::other)?;
    check_bind(addr, cfg).map_err(io::Error::other)?;
    let listener = TcpListener::bind(addr)?;
    let session = Arc::new(Session::new());
    let active = Arc::new(AtomicUsize::new(0));
    eprintln!("occ listening on http://{addr}/mcp");
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        // Reserve the slot first, then check what the count was before the
        // increment. Loading first and incrementing second let two accept
        // loops both pass the test at the limit and run one thread over it.
        if active.fetch_add(1, Ordering::Relaxed) >= cfg.max_connections {
            active.fetch_sub(1, Ordering::Relaxed);
            let mut stream = stream;
            let _ = respond(&mut stream, 503, "text/plain", b"too many connections");
            continue;
        }
        let session = Arc::clone(&session);
        let active = Arc::clone(&active);
        let token = cfg.token.clone();
        let loopback = addr.ip().is_loopback();
        std::thread::spawn(move || {
            let _guard = ConnectionGuard(active);
            let _ = serve_connection(stream, &session, token.as_deref(), loopback);
        });
    }
    Ok(())
}

/// Decrements the live connection count when the thread ends, including on
/// an early return from a malformed request.
struct ConnectionGuard(Arc<AtomicUsize>);

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Read one request and answer it.
///
/// The parser is intentionally small: one request per connection, a bounded
/// header block, and a `Content-Length` body. That is all the Streamable HTTP
/// transport needs, and every limit here is a thing an unauthenticated caller
/// could otherwise use to allocate without bound.
fn serve_connection(
    mut stream: TcpStream,
    session: &Session,
    token: Option<&str>,
    loopback: bool,
) -> io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(60)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");

    let mut content_length: Option<usize> = None;
    let mut host: Option<String> = None;
    let mut origin: Option<String> = None;
    let mut authorization: Option<String> = None;
    let mut header_bytes = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        header_bytes += line.len();
        if header_bytes > 8 * 1024 {
            return respond(&mut stream, 431, "text/plain", b"request headers too large");
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim();
            let value = v.trim();
            if key.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().ok();
            } else if key.eq_ignore_ascii_case("host") {
                host = Some(value.to_string());
            } else if key.eq_ignore_ascii_case("origin") {
                origin = Some(value.to_string());
            } else if key.eq_ignore_ascii_case("authorization") {
                authorization = Some(value.to_string());
            }
        }
    }

    // Browser requests carry Origin. None of the MCP clients do, and a
    // loopback server without a token must not be reachable by a web page
    // through DNS rebinding. Refusing every browser origin is the simplest
    // rule that cannot be bypassed by a specially crafted one.
    if origin.is_some() {
        return respond(
            &mut stream,
            403,
            "text/plain",
            b"browser origins are not accepted",
        );
    }
    if loopback
        && let Some(host) = host.as_deref()
        && !host_is_loopback(host)
    {
        return respond(&mut stream, 403, "text/plain", b"host is not loopback");
    }

    if method == "GET" && path == "/health" {
        return respond(&mut stream, 200, "text/plain", b"ok");
    }
    if method != "POST" || path != "/mcp" {
        return respond(&mut stream, 404, "text/plain", b"not found");
    }

    if let Some(expected) = token {
        let supplied = authorization
            .as_deref()
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if !token_matches(expected, supplied) {
            return respond(
                &mut stream,
                401,
                "text/plain",
                b"missing or invalid bearer token",
            );
        }
    }

    let Some(content_length) = content_length else {
        return respond(&mut stream, 411, "text/plain", b"content-length required");
    };
    if content_length > protocol::MAX_MESSAGE {
        return respond(&mut stream, 413, "text/plain", b"message too large");
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

/// True when a `Host` header names a loopback interface.
fn host_is_loopback(host: &str) -> bool {
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host.rsplit_once(':').map_or(host, |(h, _)| h)
    };
    name.eq_ignore_ascii_case("localhost")
        || name
            .parse::<IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        202 => "Accepted",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        411 => "Length Required",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
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
        assert_eq!(
            v.get("protocolVersion").unwrap().as_str(),
            Some("2024-11-05")
        );
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
        assert!(
            tools
                .iter()
                .any(|t| t.get("name").and_then(|n| n.as_str()) == Some("binary.open"))
        );
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
    fn a_leading_colon_means_loopback() {
        assert_eq!(
            bind_address(":8787").unwrap(),
            "127.0.0.1:8787".parse().unwrap()
        );
        assert!(bind_address("8787").is_err());
        assert_eq!(
            bind_address("localhost:8787").unwrap(),
            "127.0.0.1:8787".parse().unwrap()
        );
    }

    #[test]
    fn a_remote_bind_needs_an_explicit_opt_in_and_a_token() {
        let remote: SocketAddr = "0.0.0.0:8787".parse().unwrap();
        let mut cfg = HttpConfig {
            addr: "0.0.0.0:8787".into(),
            ..HttpConfig::default()
        };
        assert!(
            check_bind(remote, &cfg)
                .unwrap_err()
                .contains("allow-remote")
        );
        cfg.allow_remote = true;
        assert!(check_bind(remote, &cfg).unwrap_err().contains("token"));
        cfg.token = Some("short".into());
        assert!(check_bind(remote, &cfg).unwrap_err().contains("16"));
        cfg.token = Some("0123456789abcdef".into());
        assert!(check_bind(remote, &cfg).is_ok());
        let loopback: SocketAddr = "127.0.0.1:8787".parse().unwrap();
        assert!(check_bind(loopback, &HttpConfig::default()).is_ok());
    }

    #[test]
    fn token_comparison_is_exact() {
        assert!(token_matches("0123456789abcdef", "0123456789abcdef"));
        assert!(!token_matches("0123456789abcdef", "0123456789abcde"));
        assert!(!token_matches("0123456789abcdef", "1123456789abcdef"));
    }

    #[test]
    fn loopback_host_validation_accepts_local_names_only() {
        assert!(host_is_loopback("localhost:8787"));
        assert!(host_is_loopback("127.0.0.1:8787"));
        assert!(host_is_loopback("[::1]:8787"));
        assert!(!host_is_loopback("evil.example:8787"));
        assert!(!host_is_loopback("0.0.0.0:8787"));
    }

    fn one_request(addr: SocketAddr, bearer: Option<&str>) -> String {
        use std::io::{Read, Write};
        let mut stream = TcpStream::connect(addr).unwrap();
        let auth = bearer
            .map(|t| format!("Authorization: Bearer {t}\r\n"))
            .unwrap_or_default();
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req = format!(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            addr.port(),
            body.len()
        );
        stream.write_all(req.as_bytes()).unwrap();
        let mut out = String::new();
        stream.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn a_request_without_the_token_gets_401() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let session = Arc::new(Session::new());
        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                let _ = serve_connection(stream, &session, Some("0123456789abcdef"), true);
            }
        });
        let reply = one_request(addr, None);
        assert!(reply.starts_with("HTTP/1.1 401"), "{reply}");
    }

    #[test]
    fn a_request_with_the_token_reaches_the_protocol() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let session = Arc::new(Session::new());
        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                let _ = serve_connection(stream, &session, Some("0123456789abcdef"), true);
            }
        });
        let reply = one_request(addr, Some("0123456789abcdef"));
        assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
        assert!(reply.contains("jsonrpc"), "{reply}");
    }

    #[test]
    fn a_browser_origin_is_refused() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let session = Arc::new(Session::new());
        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                let _ = serve_connection(stream, &session, None, true);
            }
        });
        use std::io::{Read, Write};
        let mut stream = TcpStream::connect(addr).unwrap();
        let req = "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://evil.example\r\nContent-Length: 2\r\n\r\n{}";
        stream.write_all(req.as_bytes()).unwrap();
        let mut out = String::new();
        stream.read_to_string(&mut out).unwrap();
        assert!(out.starts_with("HTTP/1.1 403"), "{out}");
    }

    #[test]
    fn notifications_produce_no_response() {
        let s = Session::new();
        assert!(
            handle(
                &s,
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#
            )
            .is_none()
        );
    }
}
