/* PE fixture for occ end-to-end tests.
 *
 * The shape is chosen so that every analysis stage has something to chew on:
 *   - a switch with five dense cases  -> jump table (absolute 64-bit entries)
 *   - calls into kernel32/user32      -> IAT-bound imports
 *   - string literals                 -> string xrefs
 *   - a recursive function            -> non-trivial call graph
 *   - a function pointer table        -> address-taken code
 */
#include <windows.h>
#include <stdio.h>
#include <string.h>

static const char *const BANNER = "occ/pe fixture: decode round trip";

static int classify(unsigned int op) {
    switch (op) {
    case 1: return 11;
    case 2: return 22;
    case 3: return 33;
    case 4: return 44;
    case 5: return 55;
    default: return -1;
    }
}

static int accumulate(const int *values, int n) {
    if (n <= 0) {
        return 0;
    }
    return values[0] + accumulate(values + 1, n - 1);
}

typedef int (*op_fn)(unsigned int);
static op_fn handlers[] = {classify};

int main(int argc, char **argv) {
    int values[5] = {1, 2, 3, 4, 5};
    int total = accumulate(values, 5);
    int bucket = handlers[0]((unsigned int)argc);

    printf("%s\n", BANNER);
    printf("total=%d bucket=%d len=%u\n", total, bucket,
           (unsigned)strlen(BANNER));

    if (total > 100) {
        MessageBoxA(NULL, BANNER, "occ", MB_OK);
    }
    return classify(3) + bucket;
}
