/*
 * The 32-bit x86 fixture.
 *
 * Compiled into `fixture.x86-32.o` with:
 *
 *     gcc -m32 -O1 -ffreestanding -fno-pic -fno-inline \
 *         -fno-asynchronous-unwind-tables -fno-unwind-tables \
 *         -c fixture.x86-32.c -o fixture.x86-32.o
 *
 * `-fno-pic` matters: the default on a distribution toolchain is position
 * independent, and a 32-bit PIC global access goes through a
 * `call __x86.get_pc_thunk` and a GOT base the decompiler does not model.
 * `-fno-inline` keeps `twice` calling `add`, which is the only way a test
 * sees how a call's arguments are pushed. The unwind tables are turned off
 * because a `PC32` relocation inside `.eh_frame` points at a section the
 * loader places far away, and an unpatched one is noise in a fixture that
 * exists to test code.
 *
 * Everything here is chosen to be a different shape of the same problem —
 * where an argument comes from — because that is the one thing 32-bit x86
 * does differently from every other architecture this tool lifts.
 */

volatile int sink;

/* Two arguments, no frame pointer: both are read straight off the stack. */
int add(int a, int b) { return a + b; }

/* One argument, passed twice: two pushes of the same value. */
__attribute__((noinline)) int twice(int x) { return add(x, x); }

/* Two different arguments, so the two pushes must not collapse into one. */
int mix(int a, int b) { return add(a * 3, b + 1); }

/* Signed, equality and "less than ten" branches over one argument. */
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

/* A counting loop, with the counter in a register the callee must preserve. */
int sum_to(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += i;
    }
    return s;
}

/*
 * A global load with no size keyword in front of it. The decoder writes
 * `mov eax, ds:0x...` and leaves the width to the destination register, so a
 * lift that defaults to the machine word reads four bytes too many and turns
 * `return sink;` into a load of the next global as well.
 */
int read_sink(void) { return sink; }

/* The same address read and written in one function, at one width. */
int bump(void) {
    sink = sink + 1;
    return sink;
}

/*
 * Callee cleanup. `ret $8` removes the two arguments after popping the
 * return address, so the instruction states the arity even though the body
 * only ever reads the first one — a caller still has to push both.
 */
__attribute__((naked)) int only_first(int a, int b) {
    __asm__("movl 4(%esp), %eax\n\tret $8");
}
