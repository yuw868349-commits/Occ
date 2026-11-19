#!/usr/bin/env python3
"""Build `tests/fixtures/fixture.wasm3`, the hand-written WebAssembly case.

Every other WebAssembly fixture in this repository comes out of a C
compiler, and a C compiler does not emit every shape the bytecode can
take. One shape it never emits is a branch to the top of its own loop:
`continue` is always folded into a conditional around the rest of the
body, so a `br 0` in the middle of a loop body — the encoding that means
`continue` and nothing else — never appears in compiled output. The
decoder still has to read it, and the decompiler has to say it as the
keyword, so there is a fixture that contains it.

The module, in the text format:

    (module
      (func (export "spin") (param i32) (result i32)
        (local i32)
        block
          loop
            local.get 0
            i32.eqz
            br_if 1        ;; out of the block, which is out of the loop
            local.get 1
            i32.const 3
            i32.eq
            br_if 0        ;; back to the top of the loop: `continue`
            local.get 1
            i32.const 1
            i32.add
            local.set 1
            br 0           ;; back to the top: the loop's own edge
          end
        end
        local.get 1))

The `name` custom section is written too, so the function is called
`spin` in the output rather than `func_0`.

There is no WAT assembler in the build environment — `wat2wasm` is not
installed and the fixture has to build without one — so the sections are
written out here and the result is checked in beside this script.
"""

import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# One local of type i32, then the block and the loop.
BODY = bytes(
    [
        0x01,
        0x01,
        0x7F,  # local declarations: one group, one i32
        0x02,
        0x40,  # block, no result
        0x03,
        0x40,  # loop, no result
        0x20,
        0x00,  # local.get 0
        0x45,  # i32.eqz
        0x0D,
        0x01,  # br_if 1
        0x20,
        0x01,  # local.get 1
        0x41,
        0x03,  # i32.const 3
        0x46,  # i32.eq
        0x0D,
        0x00,  # br_if 0
        0x20,
        0x01,  # local.get 1
        0x41,
        0x01,  # i32.const 1
        0x6A,  # i32.add
        0x21,
        0x01,  # local.set 1
        0x0C,
        0x00,  # br 0
        0x0B,  # end loop
        0x0B,  # end block
        0x20,
        0x01,  # local.get 1
        0x0B,  # end function
    ]
)


def uleb(value):
    """An unsigned LEB128, which is how every length and index is written."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def section(kind, payload):
    return bytes([kind]) + uleb(len(payload)) + payload


def name_of(text):
    """The payload of a `name` custom section naming function 0."""
    entry = uleb(0) + uleb(len(text)) + text  # function 0 is `text`
    content = uleb(1) + entry  # one entry
    subsection = bytes([1]) + uleb(len(content)) + content
    return uleb(4) + b"name" + subsection


def build():
    # The type section: one type, (i32) -> i32.
    types = uleb(1) + b"\x60" + uleb(1) + b"\x7f" + uleb(1) + b"\x7f"
    # The function section: one function, of type 0.
    funcs = uleb(1) + uleb(0)
    # The export section: "spin" is function 0.
    export = uleb(1) + uleb(4) + b"spin" + b"\x00" + uleb(0)
    # The code section: one body.
    code = uleb(1) + uleb(len(BODY)) + BODY

    module = b"\0asm\x01\0\0\0"
    module += section(1, types)
    module += section(3, funcs)
    module += section(7, export)
    module += section(10, code)
    module += section(0, name_of(b"spin"))
    return module


def main():
    module = build()
    out = HERE / "tests" / "fixtures" / "fixture.wasm3"
    out.write_bytes(module)
    print(f"{out}: {len(module)} bytes")


if __name__ == "__main__":
    main()
