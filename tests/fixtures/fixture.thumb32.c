/*
 * The 32-bit ARM fixture, in Thumb.
 *
 * Compiled into `fixture.thumb32.o` with:
 *
 *     clang --target=armv7-linux-gnueabi -mthumb -mfloat-abi=soft \
 *           -O1 -ffreestanding -fno-pic -fno-inline \
 *           -fno-asynchronous-unwind-tables -fno-unwind-tables \
 *           -c fixture.thumb32.c -o fixture.thumb32.o
 *
 * This is the same source as `fixture.arm32.c` in the other instruction set.
 * The two are kept apart because the encodings are different, not the
 * architecture: a `bl` is one four-byte instruction in A32 and two two-byte
 * instructions in Thumb, and a conditional block is a predicate in one and
 * an `it` in the other. A fixture that mixed them would test two decoders
 * at once and fail at the one that is wrong.
 *
 * Thumb's conditionals are the reason this file exists on its own. In A32
 * every instruction carries its own condition; in Thumb the condition lives
 * in an `it` instruction that governs the four after it, and a lift that
 * reads the instructions without the block writes a store that should have
 * happened on one path only onto every path.
 *
 * `-mfloat-abi=soft` matters for the same reason it does in the A32
 * fixture: the hard-float ABI puts arguments in the VFP register file
 * (`s0`, `d0`), a second register bank with its own rounding rules, and it
 * is not lifted. Everything here stays in the integer registers.
 *
 * `-fno-pic` is the choice the x86-32 fixture makes too: position
 * independent code reaches a global through a GOT base loaded from a literal
 * pool, and this fixture is about the code, not about the loader.
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

/*
 * An `it` block with an else. Thumb has no predicated instructions: the
 * condition lives in an `it` that governs the four instructions after it,
 * and the second of those four is the *else* case. A lift that reads the
 * block as "everything under one condition" turns a choice into a
 * sequence.
 */
int either(int x) { return x > 3 ? x : -x; }

/*
 * A compare-and-branch on zero. `cbz` and `cbnz` have no A32 equivalent
 * that a compiler picks, so they are Thumb's own.
 */
int nonzero(int x) { return x != 0 ? 1 : 0; }

/*
 * The high registers. Thumb has two banks: r0 to r7 are the ones an
 * eight-bit instruction can name, and r8 to r15 need the wide form. A lift
 * that reads `r8` as an unknown register loses the value.
 */
int high(int x) {
    volatile int *p = &sink;
    *p = x;
    return sink;
}

/*
 * A byte and a halfword, sign- and zero-extended. These are Thumb's own
 * instructions and the extension is part of the instruction, not a separate
 * operation.
 */
int widened(unsigned char c, short s) { return c + s; }

/*
 * The parallel additions and subtractions.
 *
 * These work on the bytes of a word, or on its half words, and each lane
 * is worked on by itself: the carry of one lane never reaches the next.
 * A lift that reads them as one ordinary addition says the wrong thing
 * about every lane but the last.
 */
unsigned parallel_add8(unsigned a, unsigned b) {
    unsigned r;
    __asm__("sadd8 %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

unsigned parallel_qadd16(unsigned a, unsigned b) {
    unsigned r;
    __asm__("qadd16 %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

/*
 * The halving form. The result is half of the sum, and the bit that was
 * carried out of the lane survives the halving, so it is not the same as
 * adding and then halving.
 */
unsigned parallel_shadd8(unsigned a, unsigned b) {
    unsigned r;
    __asm__("shadd8 %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

/*
 * The unsigned halving form. The lanes are read as unsigned numbers and
 * the halving is done on the difference before it is cut back down, so a
 * lane that went below zero is halved as a negative number.
 */
unsigned parallel_uhsub16(unsigned a, unsigned b) {
    unsigned r;
    __asm__("uhsub16 %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

/*
 * The exchanged forms, which take the halves of the two operands the
 * other way round: the high half of the result comes from the high half
 * of the first operand and the low half of the second, and the low half
 * of the result comes from the low half of the first and the high half
 * of the second.
 */
unsigned exchange_sasx(unsigned a, unsigned b) {
    unsigned r;
    __asm__("sasx %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

unsigned exchange_uhasx(unsigned a, unsigned b) {
    unsigned r;
    __asm__("uhasx %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

/*
 * The reversals. Each throws away something: the byte swap throws away
 * the order, the half swap throws away the order inside each half, and
 * the bit reversal throws away everything but the number of ones.
 */
unsigned reverse_bytes(unsigned a) {
    unsigned r;
    __asm__("rev %0, %1" : "=r"(r) : "r"(a));
    return r;
}

unsigned reverse_halves(unsigned a) {
    unsigned r;
    __asm__("rev16 %0, %1" : "=r"(r) : "r"(a));
    return r;
}

unsigned reverse_half_signed(unsigned a) {
    unsigned r;
    __asm__("revsh %0, %1" : "=r"(r) : "r"(a));
    return r;
}

unsigned reverse_bits(unsigned a) {
    unsigned r;
    __asm__("rbit %0, %1" : "=r"(r) : "r"(a));
    return r;
}

/*
 * The extended additions: a half word of the second operand is extended
 * and added to the first. The extension is part of the instruction, and
 * reading it as a plain addition loses the half of the value the
 * instruction threw away.
 */
unsigned extended_add_signed(unsigned a, unsigned b) {
    unsigned r;
    __asm__("sxtah %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

unsigned extended_add_halves(unsigned a, unsigned b) {
    unsigned r;
    __asm__("uxtab16 %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

/*
 * The saturating additions, and the doubled form, which adds twice the
 * second operand. The result stops at the ends of the range.
 */
unsigned saturating_add(unsigned a, unsigned b) {
    unsigned r;
    __asm__("qadd %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

unsigned saturating_add_double(unsigned a, unsigned b) {
    unsigned r;
    __asm__("qdadd %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
