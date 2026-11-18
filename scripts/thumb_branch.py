#!/usr/bin/env python3
"""Measure the branch and compare-and-branch classes of the 16-bit set.

The decoder and the two references disagree about *which way* a compare-and-
branch goes.  Rather than argue from memory of the architecture manual, the
question is settled by measurement: every encoding of the class is read with
both references, the destination is turned back into a displacement from the
program counter, and the displacement is fitted against the bits.

The assembler is the third voice: it is asked to encode a branch to a known
distance, and the bits it chooses say what the distance field is.

Usage:
    thumb_branch.py --out /tmp/occ-thumb-table/branch16.json
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capstone as cs

OBJCOPY = "/usr/lib/llvm-23/bin/llvm-objcopy"
OBJDUMP = "/usr/lib/llvm-23/bin/llvm-objdump"
TRIPLE = "thumbv7-linux-gnueabi"
MCTARGET = "/usr/lib/llvm-23/bin/llvm-mc"

# The address every halfword is read at.
ADDR = 0x1000


def capstone_detail(hw, addr=ADDR):
    """Read one halfword with capstone, structurally."""
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    md.detail = True
    code = hw.to_bytes(2, "little")
    for i in md.disasm(code, addr, count=1):
        if i.size != 2:
            return None
        ops = []
        for o in i.operands:
            if o.type == cs.arm.ARM_OP_REG:
                ops.append(("r", i.reg_name(o.reg)))
            elif o.type == cs.arm.ARM_OP_IMM:
                ops.append(("i", o.imm))
            else:
                ops.append(("?", str(o.type)))
        return (i.mnemonic, ops)
    return None


def objdump_table(words, scratch):
    """Read a stream of halfwords with llvm-objdump, each in its own slot."""
    stream = []
    for w in words:
        stream.append(w)
        stream.append(0xBF00)  # nop, so that a wide encoding cannot
                                # swallow the next slot
    blob = b"".join(w.to_bytes(2, "little") for w in stream)
    subprocess.run(["rm", "-rf", scratch], check=False)
    subprocess.run(["mkdir", "-p", scratch], check=True)
    raw = f"{scratch}/r.bin"
    with open(raw, "wb") as f:
        f.write(blob)
    obj = f"{scratch}/a.o"
    subprocess.run(
        [OBJCOPY, "--input-target=binary", "--output-target=elf32-littlearm",
         "--rename-section", ".data=.text,code", raw, obj],
        check=True, capture_output=True)
    res = subprocess.run(
        [OBJDUMP, "-d", f"--triple={TRIPLE}", obj],
        check=True, capture_output=True, text=True).stdout
    table = {}
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+[0-9a-f ]+\s*(.*)$", line)
        if m:
            table[int(m.group(1), 16)] = m.group(2).strip()
    return table


def objdump_imm(text):
    """The displacement llvm-objdump prints as `@ imm = #0x..`."""
    m = re.search(r"@\s*imm\s*=\s*(#-?0x[0-9a-f]+|#-?\d+)", text or "")
    if not m:
        return None
    body = m.group(1)[1:]
    return int(body, 16) if "0x" in body else int(body, 10)


def measure_class(words, scratch):
    """Read every encoding of a class with both references.

    Each encoding is given a slot of its own, so that a wide encoding
    cannot swallow the next one, and the slot's address is recorded.
    """
    table = objdump_table(words, scratch)
    out = []
    for i, hw in enumerate(words):
        addr = i * 4
        csr = capstone_detail(hw, addr)
        od = table.get(addr)
        out.append({"hw": hw, "addr": addr, "capstone": csr, "objdump": od})
    return out


def fit(rows, mnemonics):
    """Fit the displacement against the bits, for one class of encoding.

    For every encoding that the references agree is a branch of the given
    kind, the displacement — the destination minus the program counter — is
    computed, and then the same displacement is predicted from the bits
    under each candidate rule.  The rule that predicts every displacement
    is the rule the architecture uses.
    """
    samples = []
    for r in rows:
        csr = r["capstone"]
        od = r["objdump"] or ""
        if csr is None:
            continue
        mn, ops = csr
        if mn not in mnemonics:
            continue
        dest = None
        for kind, val in ops:
            if kind == "i":
                dest = val
        if dest is None:
            continue
        # objdump must name the same operation, or the class is not settled.
        if not od.lower().startswith(mn):
            continue
        disp = dest - (r["addr"] + 4)
        samples.append({
            "hw": r["hw"],
            "disp": disp,
            "od_imm": objdump_imm(od),
            "od_text": od,
        })
    return samples


def rules():
    """The candidate rules for a compare-and-branch displacement.

    Each rule says how the six-bit field and the direction bit make the
    displacement.  They are written out rather than generated, because the
    point is to *compare* what the manual says against what the
    references do.
    """
    return {
        # The field is a plain unsigned count of halfwords, forwards.
        "fwd_u6": lambda hw, f6, i: f6 * 2,
        # The field is a plain unsigned count of halfwords, and the
        # direction bit turns it backwards.
        "bwd_u6": lambda hw, f6, i: -(f6 * 2) if i else f6 * 2,
        # The field is a signed count of halfwords, with the direction
        # bit as the sign.
        "sign6": lambda hw, f6, i: (-(f6 & 0x3F) if i else (f6 & 0x3F)) * 2,
        # The field is a signed count of halfwords, and the direction bit
        # subtracts a fixed bias from it.
        "sign6_bias": lambda hw, f6, i: (f6 * 2) - 128 if i else f6 * 2,
        # The field counts halfwords and the direction bit turns it
        # backwards with a four-byte bias, as the wide form does.
        "bwd_u6_bias": lambda hw, f6, i: -(f6 * 2) - 4 if i else f6 * 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/occ-thumb-table/branch16.json")
    ap.add_argument("--scratch", default="/tmp/occ-thumb-branch")
    args = ap.parse_args()

    words = list(range(0xB000, 0xB400)) + list(range(0xD000, 0xE000))
    rows = measure_class(words, args.scratch)
    samples = fit(rows, {"cbz", "cbnz"})

    # The displacement is compared against each candidate rule.  The
    # direction bit and the field are taken at the positions the
    # architecture manual gives them, and the rule that fits every sample
    # is the one that is printed.
    verdict = collections.Counter()
    detail = {}
    for s in samples:
        hw = s["hw"]
        field = (hw >> 3) & 0x1F
        dirbit = (hw >> 9) & 1
        # The whole six-bit field, with the direction bit as its top.
        f6 = ((hw >> 8) & 1) << 5 | field
        for name, rule in rules().items():
            if rule(hw, f6, dirbit) == s["disp"]:
                verdict[name] += 1
        detail.setdefault(hw, s)

    total = len(samples)
    print(f"{total} compare-and-branch encodings measured")
    for name, n in verdict.most_common():
        print(f"  {name:14s} explains {n}/{total}")

    # A sample of the readings themselves.
    print()
    for hw in list(detail)[:12]:
        s = detail[hw]
        print(f"  hw=0x{hw:04x} disp={s['disp']:+#x} objdump_imm={s['od_imm']}"
              f"  {s['od_text']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"total": total,
                   "verdict": dict(verdict),
                   "samples": samples[:2000]}, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
