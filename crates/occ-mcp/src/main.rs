//! `occ` entry point: the MCP server.
//!
//! Default transport is stdio, which is what an MCP client spawns. `--http
//! <addr>` serves the same tool set over HTTP for clients that cannot spawn a
//! subprocess.

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut http: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--http" => match args.next() {
                Some(addr) => http = Some(addr),
                None => {
                    eprintln!("--http needs an address, e.g. --http 127.0.0.1:8787");
                    return ExitCode::from(2);
                }
            },
            "--help" | "-h" => {
                println!(
                    "occ {}\n\n\
                     Usage:\n  occ                speak MCP over stdin/stdout\n  \
                     occ --http ADDR    serve POST /mcp and GET /health\n",
                    occ_mcp::SERVER_VERSION
                );
                return ExitCode::SUCCESS;
            }
            "--version" => {
                println!("{}", occ_mcp::SERVER_VERSION);
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}; try --help");
                return ExitCode::from(2);
            }
        }
    }
    let result = match http {
        Some(addr) => occ_mcp::server::serve_http(&addr),
        None => occ_mcp::server::serve_stdio(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("occ: {e}");
            ExitCode::FAILURE
        }
    }
}
