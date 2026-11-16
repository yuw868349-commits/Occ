#!/usr/bin/env python3
"""Drive the occ MCP server over stdio the way a client does.

Not a unit test: it talks to the built binary through the same JSON-RPC wire
format an MCP client uses, so a regression in framing or dispatch shows up
here even when the Rust tests pass.
"""
import json
import os
import subprocess
import sys

BIN = sys.argv[1] if len(sys.argv) > 1 else "target/debug/occ"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "/bin/ls"


class Client:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.next_id = 1

    def send(self, method, params=None):
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.next_id += 1
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("server closed the connection")
        return json.loads(line)

    def call(self, name, **arguments):
        r = self.send("tools/call", {"name": name, "arguments": arguments})
        if "error" in r:
            raise RuntimeError(f"{name}: {r['error']}")
        result = r["result"]
        if result.get("isError"):
            raise RuntimeError(f"{name}: {result['structuredContent']}")
        return result["structuredContent"]

    def call_error(self, name, **arguments):
        """Call a tool expecting a refusal, returning the message."""
        r = self.send("tools/call", {"name": name, "arguments": arguments})
        result = r.get("result", {})
        if not result.get("isError"):
            raise RuntimeError(f"{name}: expected an error, got {result}")
        content = result.get("content") or []
        return content[0].get("text", "") if content else ""

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=5)


def check(cond, label):
    print(("ok   " if cond else "FAIL ") + label)
    if not cond:
        raise SystemExit(1)


def main():
    c = Client(BIN)
    init = c.send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    check(init["result"]["serverInfo"]["name"] == "occ", "initialize")

    tools = c.send("tools/list")["result"]["tools"]
    names = [t["name"] for t in tools]
    check(len(names) >= 20, f"tools/list exposes {len(names)} tools")
    for required in ("binary.open", "code.functions", "code.xrefs", "scan.pattern"):
        check(required in names, f"tool {required} present")

    opened = c.call("binary.open", path=TARGET)
    image = opened["data"]["image"]
    check(opened["data"]["coverage"]["functions"] > 0, f"recovered {opened['data']['coverage']['functions']} functions")

    info = c.call("binary.info", image=image)
    check(info["data"]["identity"]["arch"] in ("x86_64", "aarch64", "riscv64"), "arch reported")

    funcs = c.call("code.functions", image=image, sort="insns", limit=5)
    check(len(funcs["data"]["functions"]) == 5, "code.functions paginates")
    handle = funcs["data"]["functions"][0]["handle"]

    fn = c.call("code.function", image=image, target=handle, depth="summary")
    check(fn["handle"] == handle, "code.function echoes the handle")
    check(fn["data"]["function"]["blocks"] >= 1, "function has blocks")

    dis = c.call("code.disasm", image=image, target=handle, count=5)
    check(len(dis["data"]["insns"]) == 5, "code.disasm returns 5 instructions")

    strings = c.call("binary.strings", image=image, pattern="lib", limit=3)
    check("strings" in strings["data"], "binary.strings returns hits")

    ent = c.call("binary.entropy", image=image)
    check(len(ent["data"]["sections"]) > 0, "entropy per section")

    rules = """
rule libc_path {
  strings:
    $a = "/lib"
  condition:
    $a
}
"""
    scan = c.call("scan.pattern", image=image, rules=rules)
    check(len(scan["data"]["matches"]) >= 1, "pattern matched a string")

    h = c.call("hypothesis.create", image=image, target=handle, statement="this is main")
    hid = h["data"]["id"]
    ev = c.call("hypothesis.evidence", id=hid, detail="entry is the largest function", support=True, weight=0.8)
    check(ev["data"]["verdict"] in ("confirmed", "inconclusive"), "hypothesis evaluated")

    debug_checks(c)
    sym_checks(c)
    wasm_checks(c)
    trace_checks(c)
    switch_checks(c)

    saved = "/tmp/occ-smoke-session.json"
    c.call("session.save", path=saved)
    restored = c.call("session.restore", path=saved)
    check("warnings" in restored["data"], "session restored")

    c.close()
    print("\nall smoke checks passed")


DWARF_SOURCE = r"""
#include <stdio.h>
struct point { int x, y; };
struct shape { struct point a, b; unsigned char kind; };
int area(struct shape *s) { return (s->b.x - s->a.x) * (s->b.y - s->a.y); }
int main(int argc, char **argv) {
    struct shape s = {{0, 0}, {3, 4}, 1};
    printf("%d %s\n", area(&s), argv[0]);
    return argc;
}
"""


def debug_checks(c):
    """Debug-info tools, on a binary this script builds with -g."""
    if subprocess.run(["gcc", "--version"], capture_output=True).returncode != 0:
        print("skip debug checks (no gcc)")
        return
    src = "/tmp/occ-dwarf-smoke.c"
    exe = "/tmp/occ-dwarf-smoke"
    with open(src, "w") as fh:
        fh.write(DWARF_SOURCE)
    built = subprocess.run(["gcc", "-g", "-O1", "-o", exe, src], capture_output=True)
    if built.returncode != 0:
        raise AssertionError(f"the debug fixture failed to build:\n{built.stderr.decode()}")

    img = c.call("binary.open", path=exe)["data"]["image"]
    info = c.call("debug.info", image=img)
    check(len(info["data"]["units"]) == 1, "debug.info reports one unit")
    check(any(f.endswith(".c") for f in info["data"]["source_files"]), "debug.info lists source files")

    funcs = c.call("debug.functions", image=img)
    names = [f["name"] for f in funcs["data"]["functions"]]
    check("main" in names and "area" in names, "debug.functions lists main and area")

    fn = c.call("debug.function", image=img, target="main")
    check(len(fn["data"]["function"]["params"]) == 2, "debug.function finds two parameters")

    low = fn["data"]["function"]["low"]
    line = c.call("debug.line", image=img, at=low)
    check(line["data"]["lines"][0]["file"].endswith(".c"), "debug.line resolves a file")
    check(line["data"]["lines"][0]["line"] > 0, "debug.line resolves a line number")

    code = c.call("code.decompile", image=img, target="area")
    check("u64 area(" in code["data"]["code"], "code.decompile renders a signature")
    check("return" in code["data"]["code"], "code.decompile renders a return")
    check(code["handle"].startswith("fn:"), "code.decompile echoes a function handle")

    main_code = c.call("code.decompile", image=img, target="main")["data"]["code"]
    check("printf" in main_code, "code.decompile names an imported callee")

    var = c.call("debug.variables", image=img, at=low)
    check(any(v["live"] for v in var["data"]["variables"]), "debug.variables finds a live variable")

    types = c.call("debug.types", image=img, filter="shape")
    check(any(t["name"] == "shape" for t in types["data"]["types"]), "debug.types finds a struct")

    # The same program stripped must refuse with a message that says what is
    # missing, rather than an empty success.
    stripped = "/tmp/occ-dwarf-smoke-stripped"
    subprocess.run(["gcc", "-O1", "-s", "-o", stripped, src], check=True)
    simg = c.call("binary.open", path=stripped)["data"]["image"]
    msg = c.call_error("debug.info", image=simg)
    check("DWARF" in msg, "debug.info refuses without DWARF")

    # Decompilation does not depend on debug info, so the stripped build must
    # still produce a body.
    biggest = c.call("code.functions", image=simg, sort="insns", limit=1)["data"]["functions"][0]
    bare = c.call("code.decompile", image=simg, target=biggest["handle"])
    check("{" in bare["data"]["code"], "code.decompile works without DWARF")


SYM_SOURCE = r"""
#include <stdio.h>
long pick(long a, long b) { if (a > b) return a; return b; }
long count(long n) { long s = 0; while (s < n) s += 1; return s; }
int main(int argc, char **argv) {
    printf("%ld %s\n", pick(argc, 3), argv[0]);
    return (int)count(argc);
}
"""


def sym_checks(c):
    """Symbolic execution and taint, on a binary built for the purpose."""
    if subprocess.run(["gcc", "--version"], capture_output=True).returncode != 0:
        print("skip sym checks (no gcc)")
        return
    src = "/tmp/occ-sym-smoke.c"
    exe = "/tmp/occ-sym-smoke"
    with open(src, "w") as fh:
        fh.write(SYM_SOURCE)
    built = subprocess.run(["gcc", "-O1", "-g", "-o", exe, src], capture_output=True)
    if built.returncode != 0:
        raise AssertionError(f"the symbol-execution fixture failed to build:\n{built.stderr.decode()}")

    img = c.call("binary.open", path=exe)["data"]["image"]

    # `count` has a real branch and a loop, so it exercises forking, solving and
    # the unroll bound in one call.
    ex = c.call("code.explore", image=img, target="count", unroll=2, max_paths=12)
    paths = ex["data"]["paths"]
    check(len(paths) >= 2, "code.explore forks on the comparison")
    check(ex["data"]["feasible"] >= 1, "code.explore finds a feasible path")
    check(all(p["outcome"] in ("return", "truncated") for p in paths),
          "code.explore only reports returns and bounds")
    solved = [p for p in paths if p.get("inputs")]
    check(len(solved) >= 1, "code.explore solves inputs")
    check(any("rdi" in p["inputs"] for p in solved), "a solved input names a register")
    check(any(p["outcome"] == "truncated" for p in paths),
          "code.explore truncates a loop instead of hiding it")

    # `pick` compiles to a conditional move: no branch, so one path. The tool
    # must not invent a fork to look busy.
    one = c.call("code.explore", image=img, target="pick")
    check(len(one["data"]["paths"]) == 1, "code.explore does not fork a cmov")

    conc = c.call("code.explore", image=img, target="pick", mode="concrete",
                  inputs={"rdi": 5, "rsi": 3})
    check(len(conc["data"]["paths"]) == 1, "concrete mode follows one path")
    check(conc["data"]["paths"][0]["outcome"] == "return", "concrete mode returns")
    check(int(conc["data"]["paths"][0]["returns"], 0) == 5,
          "concrete mode computes the result")

    bare = c.call("code.explore", image=img, target="pick", mode="concrete")
    check(bare["data"]["paths"][0]["outcome"] == "missing_inputs",
          "concrete mode reports a missing input")
    check(bare["data"]["paths"][0].get("missing"), "the missing input is named")

    taint = c.call("code.taint", image=img, target="count")
    sinks = {f["sink"] for f in taint["data"]["flows"]}
    check("branch" in sinks, "code.taint reports a tainted branch")
    branch = [f for f in taint["data"]["flows"] if f["sink"] == "branch"][0]
    check(branch["path"] and len(branch["path"]) >= 1, "a flow carries a witness path")
    check(taint["data"]["converged"], "code.taint reaches a fixpoint")

    main_taint = c.call("code.taint", image=img, target="main", sources=["rsi"])
    printf = [f for f in main_taint["data"]["flows"] if "printf" in f["sink"]]
    check(bool(printf), "code.taint traces argv into printf")
    check(any(f["arg"] == 3 for f in printf), "the tainted argument index is reported")

    # Nothing tainted when nothing is a source: the answer must be empty, not
    # wrong, and the report must say the analysis ran.
    quiet = c.call("code.taint", image=img, target="count", sources=["r15"])
    check(quiet["data"]["flows"] == [], "code.taint reports no flow for an unused source")
    check(quiet["data"]["blocks"] >= 1, "the empty answer still reports its reach")


def wasm_checks(c):
    """Instruction-level view of a WebAssembly module."""
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "fixture.wasm")
    if not os.path.exists(path):
        print("skip wasm checks (no fixture)")
        return
    img = c.call("binary.open", path=path)["data"]["image"]
    info = c.call("binary.info", image=img)
    check(info["data"]["identity"]["arch"] == "wasm", "binary.info reports wasm")

    funcs = c.call("code.functions", image=img)
    names = {f["name"] for f in funcs["data"]["functions"]}
    check({"add", "sum"} <= names, "wasm function names come from the name section")

    dis = c.call("code.disasm", image=img, target="sum", count=200)
    texts = [i["text"] for i in dis["data"]["insns"]]
    check(any(t.startswith("i32.load") for t in texts), "wasm decodes a memory operand")
    check(any(t.startswith("br_if") for t in texts), "wasm decodes a conditional branch")
    check(any(t.startswith("loop") for t in texts), "wasm decodes the loop frame")

    blocks = c.call("code.blocks", image=img, target="sum")["data"]["blocks"]
    check(len(blocks) >= 4, "wasm control flow splits into blocks")
    starts = {b["start"] for b in blocks}
    br = [i for i in dis["data"]["insns"] if i["text"].startswith("br_if")]
    check(all(i.get("target") in starts for i in br if i.get("target")),
          "wasm branch targets land on block starts")



TRACE_SOURCE = r"""
#include <unistd.h>

__attribute__((noinline)) static int helper(int x) { return x + 1; }

int main(int argc, char **argv) {
    char c = (char)('a' + helper(argc));
    if (argc > 5) {
        c = 'z';
    }
    write(1, &c, 1);
    return 4;
}
"""


def trace_checks(c):
    """Running the program, then querying what the run saw."""
    if subprocess.run(["cc", "--version"], capture_output=True).returncode != 0:
        print("skip trace checks (no cc)")
        return
    src = "/tmp/occ-trace-smoke.c"
    exe = "/tmp/occ-trace-smoke"
    with open(src, "w") as fh:
        fh.write(TRACE_SOURCE)
    built = subprocess.run(
        ["cc", "-O0", "-fno-inline", "-g", "-o", exe, src], capture_output=True
    )
    if built.returncode != 0:
        raise AssertionError(f"the trace fixture failed to build:\n{built.stderr.decode()}")

    img = c.call("binary.open", path=exe)["data"]["image"]
    funcs = c.call("code.functions", image=img, limit=64)["data"]["functions"]
    helper = next((f["handle"] for f in funcs if f["name"] == "helper"), None)
    check(helper is not None, "trace fixture exposes helper")

    run = c.call("trace.run", image=img)
    handle = run["handle"]
    check(handle.startswith("trace:"), "trace.run returns a trace handle")
    check(run["data"]["run"]["end"].startswith("exited 4"), "trace.run reports the exit code")
    check(run["data"]["run"]["truncated"] is False, "trace.run was not truncated")
    check(run["data"]["coverage"]["executed_functions"] >= 2, "trace.run covers several functions")
    names = [f["name"] for f in run["data"]["coverage"]["functions"]]
    check("helper" in names, "trace.run covers helper")

    syscalls = [row["name"] for row in run["data"]["syscalls"]["rows"]]
    check("write" in syscalls, "trace.run records the write syscall")
    check(run["confidence"] >= 0.8, "trace.run reports its confidence")
    check(len(run["evidence"]) >= 1, "trace.run carries evidence")

    short = c.call("trace.get", trace=handle, target=helper)
    check(len(short["data"]["blocks"]) >= 1, "trace.get reports helper blocks")
    check(
        any(b["entered"] for b in short["data"]["blocks"]),
        "helper's block was entered",
    )

    main_fn = next((f["handle"] for f in funcs if f["name"] == "main"), None)
    check(main_fn is not None, "trace fixture exposes main")
    body = c.call("trace.get", trace=handle, target=main_fn)["data"]["blocks"]
    entered = [b for b in body if b["entered"]]
    check(len(body) >= 3 and len(entered) >= 1, "trace.get reports main's blocks")
    check(len(entered) < len(body), "trace.get shows a block main never entered")

    # Six arguments make argc 7, so `argc > 5` is true in the second run. The
    # difference between the two is the evidence that a trace follows the
    # program's input rather than the static shape of its control flow.
    wide = c.call("trace.run", image=img, argv=["1"] * 6)
    check(wide["handle"] != handle, "a second run gets its own handle")
    wide_get = c.call("trace.get", trace=wide["handle"], target=main_fn)
    wide_entered = [b for b in wide_get["data"]["blocks"] if b["entered"]]
    check(
        len(wide_entered) == len(entered) + 1,
        f"the extra arguments enter exactly one more block ({len(entered)} -> {len(wide_entered)})",
    )

    again = c.call("trace.get", trace=handle)
    check(again["handle"] == handle, "trace.get echoes the trace handle")
    check(again["data"]["coverage"]["executed_functions"] >= 2, "trace.get repeats the coverage")

SWITCH_SOURCE = r"""
volatile int sink;
int classify(int c) {
    switch (c) {
    case 0: sink = 1; return 10;
    case 1: sink = 2; return 21;
    case 2: sink = 3; return 32;
    case 3: sink = 4; return 43;
    case 4: sink = 5; return 54;
    case 5: sink = 6; return 65;
    case 6: sink = 7; return 76;
    default: sink = 9; return -1;
    }
}
int main(int argc, char **argv) { return classify(argc - 1); }
"""


def switch_checks(c):
    """A switch compiled to a jump table, on a binary built for the purpose."""
    if subprocess.run(["gcc", "--version"], capture_output=True).returncode != 0:
        print("skip switch checks (no gcc)")
        return
    src = "/tmp/occ-switch-smoke.c"
    exe = "/tmp/occ-switch-smoke"
    with open(src, "w") as fh:
        fh.write(SWITCH_SOURCE)
    # The bodies write to a volatile so the compiler keeps them distinct; a
    # switch whose cases all return a constant folds into arithmetic and there
    # is no table left to find.
    built = subprocess.run(["gcc", "-O2", "-fno-inline", "-o", exe, src], capture_output=True)
    if built.returncode != 0:
        raise AssertionError(f"the switch fixture failed to build:\n{built.stderr.decode()}")

    img = c.call("binary.open", path=exe)["data"]["image"]
    fn = c.call("code.function", image=img, target="classify")
    tables = fn["data"]["function"].get("jump_tables")
    check(bool(tables), "the dispatch table is reported")
    table = tables[0]
    check(table["count"] >= 7, f"the table has {table['count']} entries")
    check(len(table["targets"]) == table["count"], "every entry has a target")
    check(table["index_register"] is not None, "the dispatch names its index register")

    code = c.call("code.decompile", image=img, target="classify")["data"]["code"]
    check("switch (" in code, "the pseudo-C carries a switch")
    missing = [i for i in range(7) if f"case {i:#x}:" not in code]
    check(not missing, f"every case is listed, missing {missing}")
    check("return" in code, "a case body is rendered")

if __name__ == "__main__":
    main()
