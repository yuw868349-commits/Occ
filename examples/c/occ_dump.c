/*
 * occ_dump — a complete caller of the C interface.
 *
 * Build:
 *
 *     cargo build --release -p occ-ffi
 *     cc -std=c99 -Wall -Wextra -Iinclude examples/c/occ_dump.c \
 *        target/release/libocc.a -lpthread -ldl -lm -o occ_dump
 *
 * Run:
 *
 *     ./occ_dump /bin/ls                 # every image, with its summary
 *     ./occ_dump /bin/ls main            # plus one decompiled function
 *
 * The point of this file is that it contains no JSON parser. A caller that
 * wants fields can hand the document to whatever it already uses; a caller that
 * only wants the rendered body can pull it out with the small extractor below,
 * which is enough for one field and honest about being no more than that.
 */

#include "occ.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void complain(const char *what, const char *err) {
    fprintf(stderr, "occ_dump: %s: %s\n", what, err && *err ? err : "no detail");
}

/*
 * Copy the value of a top-level string field into `out`, unescaping the four
 * sequences occ-json emits for a string that spans lines. Returns 0 on success.
 *
 * This exists to show that a caller does not need a dependency to use the
 * library; it is not a general JSON reader and does not claim to be one.
 */
static int field_string(const char *json, const char *key, char *out, size_t cap) {
    char pat[64];
    int n = snprintf(pat, sizeof pat, "\"%s\":\"", key);
    if (n <= 0 || (size_t)n >= sizeof pat) {
        return -1;
    }
    const char *p = strstr(json, pat);
    if (!p) {
        return -1;
    }
    p += (size_t)n;
    size_t o = 0;
    while (*p && *p != '"' && o + 1 < cap) {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
            case 'n': out[o++] = '\n'; break;
            case 't': out[o++] = '\t'; break;
            case 'r': out[o++] = '\r'; break;
            case '"': out[o++] = '"'; break;
            case '\\': out[o++] = '\\'; break;
            case '/': out[o++] = '/'; break;
            default: out[o++] = *p; break;
            }
            p++;
        } else {
            out[o++] = *p++;
        }
    }
    out[o] = '\0';
    return 0;
}

/* Print a document, replacing a very long one with a size notice. */
static void show(const char *label, char *json) {
    size_t len = strlen(json);
    if (len > 4000) {
        printf("%s: %zu bytes (first 4000 shown)\n%.4000s\n...\n", label, len, json);
    } else {
        printf("%s\n%s\n", label, json);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <binary> [function]\n", argv[0]);
        return 2;
    }

    char err[512] = {0};
    OccSession *s = occ_open(argv[1], err, sizeof err);
    if (!s) {
        complain("open", err);
        return 1;
    }

    uint32_t images = occ_image_count(s);
    printf("occ %s · %u image(s) · %s\n", occ_version(), images, argv[1]);

    for (uint32_t i = 0; i < images; i++) {
        int64_t id = occ_image_at(s, i);
        if (id < 0) {
            fprintf(stderr, "occ_dump: image %u vanished\n", i);
            continue;
        }
        printf("\n=== image %u (id %lld)\n", i, (long long)id);

        char *info = occ_info(s, id, err, sizeof err);
        if (!info) {
            complain("info", err);
            continue;
        }
        show("info", info);
        occ_free(info);
    }

    if (argc >= 3) {
        int64_t id = occ_image_at(s, 0);
        char *dec = occ_decompile(s, id, argv[2], err, sizeof err);
        if (!dec) {
            complain("decompile", err);
        } else {
            static char code[1 << 16];
            if (field_string(dec, "code", code, sizeof code) == 0) {
                printf("\n=== %s\n%s\n", argv[2], code);
            } else {
                show("decompile", dec);
            }
            occ_free(dec);
        }

        char *refs = occ_xrefs(s, id, argv[2], err, sizeof err);
        if (refs) {
            show("xrefs", refs);
            occ_free(refs);
        } else {
            complain("xrefs", err);
        }
    }

    occ_close(s);
    return 0;
}
