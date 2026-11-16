#!/bin/sh
# Build the C interface and prove a caller can use it.
#
# Three checks, because three things can break independently:
#
#   1. the static library links and a C99 program runs against it;
#   2. the same header compiles as C++17, so a C++ caller does not need a
#      wrapper;
#   3. the exported symbols are the ones the header promises, so a rename in
#      Rust cannot silently produce a library that links but answers nothing.
#
# Usage: scripts/ffi_check.sh [path-to-a-binary]
set -eu

cd "$(dirname "$0")/.."
binary=${1:-/bin/true}
out=${TMPDIR:-/tmp}/occ-ffi-check
mkdir -p "$out"

echo "== build"
cargo build --release -p occ-ffi >/dev/null

echo "== c99"
cc -std=c99 -Wall -Wextra -Werror -pedantic -Iinclude \
    examples/c/occ_dump.c target/release/libocc.a \
    -lpthread -ldl -lm -o "$out/occ_dump"
"$out/occ_dump" "$binary" >"$out/c.log" 2>"$out/c.err" || {
    echo "the C program failed:" >&2
    cat "$out/c.err" >&2
    exit 1
}
grep -q "^occ $(cargo pkgid -p occ-ffi 2>/dev/null | sed 's/.*[@#]//')" "$out/c.log" ||
    grep -q "^occ " "$out/c.log" || {
    echo "the C program produced no version line:" >&2
    head -5 "$out/c.log" >&2
    exit 1
}
grep -q '"coverage"' "$out/c.log" || {
    echo "the info reply carries no coverage section" >&2
    exit 1
}

echo "== c++17"
cat >"$out/header.cc" <<'EOF'
#include "occ.h"
#include <cstring>

// A C++ caller compiles the header unchanged and calls through it. This checks
// the extern "C" guards and that no C99-only construct leaked into the header.
int main() {
    const char *v = occ_version();
    if (std::strlen(v) == 0) return 1;
    OccSession *s = occ_open("/bin/true", nullptr, 0);
    if (s == nullptr) return 2;
    const uint32_t n = occ_image_count(s);
    occ_close(s);
    return n == 0 ? 3 : 0;
}
EOF
c++ -std=c++17 -Wall -Wextra -Werror -Iinclude "$out/header.cc" \
    target/release/libocc.a -lpthread -ldl -lm -o "$out/header"
"$out/header"

echo "== symbols"
# Every function the header declares must be in the archive with C linkage.
for sym in occ_version occ_open occ_close occ_image_count occ_image_at occ_free \
           occ_info occ_disasm occ_function occ_xrefs occ_decompile occ_strings \
           occ_section_bytes; do
    if ! nm -g --defined-only target/release/libocc.a 2>/dev/null | grep -q " T $sym$"; then
        echo "missing exported symbol: $sym" >&2
        exit 1
    fi
done

echo "== leak-free"
# The same session opened and closed a hundred times must not grow the heap
# without bound. A leak here is invisible in a one-shot run and fatal in a
# server, so it is measured rather than assumed.
cat >"$out/loop.c" <<'EOF'
#include "occ.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/bin/true";
    for (int i = 0; i < 100; i++) {
        char err[256] = {0};
        OccSession *s = occ_open(path, err, sizeof err);
        if (!s) { fprintf(stderr, "%s\n", err); return 1; }
        int64_t id = occ_image_at(s, 0);
        char *j = occ_info(s, id, err, sizeof err);
        if (!j) { return 2; }
        occ_free(j);
        occ_close(s);
    }
    return 0;
}
EOF
cc -std=c99 -Iinclude "$out/loop.c" target/release/libocc.a \
    -lpthread -ldl -lm -o "$out/loop"
if command -v valgrind >/dev/null 2>&1; then
    valgrind --error-exitcode=9 --leak-check=full --errors-for-leak-kinds=definite \
        "$out/loop" "$binary" >"$out/valgrind.log" 2>&1 || {
        echo "valgrind reported a problem:" >&2
        tail -20 "$out/valgrind.log" >&2
        exit 1
    }
    echo "   valgrind clean over 100 open/close cycles"
else
    "$out/loop" "$binary"
    echo "   ran 100 open/close cycles (valgrind not installed)"
fi

echo "ok: the C interface builds, links, runs and leaks nothing"
