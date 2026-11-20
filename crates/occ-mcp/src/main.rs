//! `occ` entry point: the MCP server.
//!
//! Default transport is stdio, which is what an MCP client spawns. `--http
//! <addr>` serves the same tool set over HTTP for clients that cannot spawn a
//! subprocess.

use std::process::ExitCode;

const HELP: &str = "\
Usage:\n  \
  occ                       speak MCP over stdin/stdout\n  \
  occ --http ADDR           serve POST /mcp and GET /health (ADDR may be :8787)\n\n\
HTTP options:\n  \
  --allow-remote            permit a non-loopback bind (requires a token)\n  \
  --token TOKEN             require `Authorization: Bearer TOKEN` on /mcp\n  \
  --max-connections N       cap simultaneous connections (default 64)\n\n\
The token may also come from OCC_HTTP_TOKEN. Without it, HTTP is loopback-only and unauthenticated: any local process can read files and run programs through the tool set.\n";

fn fail(msg: &str) -> ExitCode {
    eprintln!("occ: {msg}");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut http: Option<String> = None;
    let mut allow_remote = false;
    let mut token = std::env::var("OCC_HTTP_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let mut max_connections = 64usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--http" => match args.next() {
                Some(addr) => http = Some(addr),
                None => return fail("--http needs an address, e.g. --http 127.0.0.1:8787"),
            },
            "--allow-remote" => allow_remote = true,
            "--token" => match args.next() {
                Some(t) if !t.is_empty() => token = Some(t),
                _ => return fail("--token needs a value"),
            },
            "--max-connections" => match args.next().and_then(|n| n.parse::<usize>().ok()) {
                Some(n) if (1..=4096).contains(&n) => max_connections = n,
                _ => return fail("--max-connections needs an integer between 1 and 4096"),
            },
            "--help" | "-h" => {
                println!("occ {}\n{HELP}", occ_mcp::SERVER_VERSION);
                return ExitCode::SUCCESS;
            }
            "--version" => {
                println!("{}", occ_mcp::SERVER_VERSION);
                return ExitCode::SUCCESS;
            }
            other => return fail(&format!("unknown argument {other}; try --help")),
        }
    }
    let result = match http {
        Some(addr) => {
            let cfg = occ_mcp::server::HttpConfig {
                addr,
                token,
                allow_remote,
                max_connections,
            };
            occ_mcp::server::serve_http(&cfg)
        }
        None => {
            if allow_remote || max_connections != 64 {
                return fail("--allow-remote and --max-connections only apply with --http");
            }
            occ_mcp::server::serve_stdio()
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("occ: {e}");
            ExitCode::FAILURE
        }
    }
}
