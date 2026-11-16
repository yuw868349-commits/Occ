/* Compiled with: clang --target=wasm32 -O1 -nostdlib -Wl,--no-entry \
 *   -Wl,--export-all -o fixture.wasm fixture.wasm.c
 *
 * Two functions, a loop, a memory load and a call: enough for the decoder to
 * have to resolve a forward branch, a loop back edge and a call edge.
 */
int add(int a, int b) { return a + b; }

int sum(const int *p, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s = add(s, p[i]);
    return s;
}
