/* Compiled with: clang --target=wasm32 -O1 -nostdlib -Wl,--no-entry \
 *   -Wl,--export-all -o fixture.wasm2 fixture.wasm2.c
 *
 * The control-flow fixture. The other WebAssembly fixture is a straight
 * line and a loop; this one has the parts that make a stack machine hard:
 *
 *   - a call that is not inlined, so the call edge and the argument order
 *     are the module's own;
 *   - a switch, which the compiler lays out as a branch table with one
 *     label per case;
 *   - a store through a pointer, which is a memory write with an address
 *     the module computes;
 *   - a floating point operation, whose operands are bit patterns;
 *   - a loop with an early exit, which nests a block inside a block and
 *     branches out of both.
 */
__attribute__((noinline)) int add(int a, int b) { return a + b; }

int sum(const int *p, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s = add(s, p[i]);
    return s;
}

int store(int *p, int v) {
    *p = v;
    return v;
}

int pick(int x) {
    switch (x) {
    case 3:
        return 10;
    case 17:
        return 20;
    case 42:
        return 30;
    case 91:
        return 40;
    default:
        return -1;
    }
}

float fadd(float a, float b) { return a + b; }

int countdown(int n) {
    int total = 0;
    while (n > 0) {
        if (n == 3)
            break;
        /* The recurrence is deliberately not a sum: a compiler that folds
         * a countdown into a closed form leaves no loop behind, and the
         * loop is what this fixture is here for. */
        total = total * 2 + n;
        n--;
    }
    return total;
}

/* A loop that leaves from the inside: the branch goes out of the loop
 * rather than back to its start. */
int find(const int *p, int n, int v) {
    for (int i = 0; i < n; i++) {
        if (p[i] == v)
            return i;
    }
    return -1;
}

/* A loop that skips: the branch goes back to the start without running the
 * rest of the body. With nothing but the sum below it, a compiler folds the
 * branch into a select, which is a different shape and is covered on its
 * own. */
int skip(const int *p, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] & 1)
            continue;
        s += p[i];
    }
    return s;
}

/* The same loop with a side effect under the branch. A store cannot be
 * moved above the test, so the branch stays a branch. */
int skip_if(const int *p, int n, int *seen) {
    int s = 0;
    *seen = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] & 1)
            continue;
        (*seen)++;
        s += p[i];
    }
    return s;
}
