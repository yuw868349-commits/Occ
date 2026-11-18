/*
 * The 32-bit ARM fixture.
 *
 * Compiled into `fixture.arm32.o` with:
 *
 *     clang --target=armv7-linux-gnueabi -marm -mfloat-abi=soft \
 *           -O1 -ffreestanding -fno-pic -fno-inline \
 *           -fno-asynchronous-unwind-tables -fno-unwind-tables \
 *           -c fixture.arm32.c -o fixture.arm32.o
 *
 * `-marm` keeps the file in A32. Thumb shares the architecture and not the
 * encoding: a `bl` is one four-byte instruction in one and two two-byte
 * instructions in the other, so a fixture that mixed them would test two
 * decoders at once and fail at the one that is wrong. Thumb has its own.
 *
 * `-mfloat-abi=soft` matters for the same reason: the hard-float ABI puts
 * arguments in the VFP register file (`s0`, `d0`), a second register bank
 * with its own rounding rules, and it is not lifted. Everything here stays
 * in the integer registers.
 *
 * `-fno-pic` is the choice the x86-32 fixture makes too: position
 * independent code reaches a global through a GOT base loaded from a literal
 * pool, and this fixture is about the code, not about the loader.
 *
 * Everything below is a different shape of the same two questions — where a
 * value lives, and what a conditional instruction does when it does not
 * execute — because those are what A32 does differently from the other
 * architectures this tool lifts.
 */

volatile int sink;

/* Two arguments in r0 and r1, added, returned in r0. */
int add(int a, int b) { return a + b; }

/* The same argument twice through a call: `mov r1, r0; b add`. */
__attribute__((noinline)) int twice(int x) { return add(x, x); }

/* `a * 3` is `add r0, r0, r0, lsl #1`: a shifted operand, not a multiply. */
int mix(int a, int b) { return add(a * 3, b + 1); }

/*
 * Two calls in one body, so the second is not a tail call and the frame has
 * to survive both: `push {r11, lr}` at the top and `pop {r11, pc}` at the
 * bottom.
 */
__attribute__((noinline)) int chain(int x) { return add(x, 1) + add(x, 2); }

/*
 * Conditional execution. A32 predicates individual instructions instead of
 * branching around them, so this arm's body is a `moveq` and a `streq`: a
 * statement that happens only when the condition holds. A lift that drops
 * the condition writes the global on every path.
 */
void set_if(int x) { if (x == 3) sink = 7; }

/*
 * Four arms, one argument, and a `movwlt` — a predicated 16-bit immediate
 * move, which shares its encoding with `tst` and is only a `movw` because of
 * the mask the decoder uses.
 */
int classify(int x) {
    if (x < 0) {
        sink = 1;
        return -1;
    }
    if (x == 0) {
        sink = 2;
        return 0;
    }
    if (x < 10) {
        sink = 3;
        return 1;
    }
    sink = 4;
    return 2;
}

/* A counting loop: `subs`/`bne`, and the counter in a callee-saved register. */
int sum_to(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += i;
    }
    return s;
}

/*
 * A post-indexed load in a loop: `ldr r3, [r2], #4` reads through the
 * pointer and moves it, and the two effects are one instruction.
 */
int first(const int *p, int n) {
    int best = p[0];
    for (int i = 1; i < n; i++) {
        if (p[i] > best) best = p[i];
    }
    return best;
}

/*
 * `ldrd`: two consecutive words in one instruction. The pair is aligned by
 * the ABI, and the two halves are two four-byte reads here because the IR's
 * widest load is the machine word.
 */
struct point {
    int x;
    int y;
};

int sum_point(const struct point *p) { return p->x + p->y; }

/* A conditional select, which the compiler writes as `cmp` and `movle`. */
int choose(int a, int b) { return a > b ? a : b; }

/* A global read and written, reached through `movw`/`movt`. */
int read_sink(void) { return sink; }

int bump(void) {
    sink = sink + 1;
    return sink;
}

/*
 * A 64-bit product from two 32-bit factors: `umull` writes both halves at
 * once, and the low half alone is not the product.
 */
unsigned long long wide_mul(unsigned a, unsigned b) {
    return (unsigned long long)a * b;
}

/* The signed form, whose high half is not the unsigned one's. */
long long signed_wide(int a, int b) { return (long long)a * b; }

/* `mla` and a scaled index: `base[i * 3 + 1]` is two shifts and a load. */
int indexed(int *base, int i) { return base[i * 3 + 1]; }

/* A three-factor product, which is where `mla` appears. */
int scaled(int a, int b, int c) { return a * b + c; }

/*
 * A call through a pointer that is not a tail call: `mov lr, pc; blx r3`
 * leaves a return address, so the call has to come back to this frame.
 */
int dispatch(int (*f)(int, int), int a, int b) {
    int r = f(a, b);
    return r + sink;
}
