/*
 * occ.h — C interface to occ, a binary analysis toolkit.
 *
 * Build the library:
 *
 *     cargo build --release -p occ-ffi
 *
 * Link a program:
 *
 *     cc -std=c99 -Iinclude prog.c target/release/libocc.a \
 *        -lpthread -ldl -lm -o prog
 *
 * Everything is C99 and compiles as C++17 without changes. There is no
 * dependency, no code generator, and no build script: the header is written by
 * hand so that a caller needs a C compiler and nothing else.
 *
 * ---------------------------------------------------------------------------
 * What a call returns
 * ---------------------------------------------------------------------------
 *
 * Every function that answers a question returns a NUL-terminated JSON
 * document, or NULL. The two cases mean different things and a caller that
 * conflates them will report a bad argument as a fact about the binary:
 *
 *   NULL            The call did not happen. `err` holds the reason: a null
 *                   pointer, invalid UTF-8, an image id that is not open, or a
 *                   panic caught at the language boundary.
 *
 *   non-NULL JSON   The call happened. The document carries "ok" and the
 *                   payload. A document with "error": true means the analysis
 *                   declined the question — an address outside every function,
 *                   a target that matched no symbol.
 *
 * `err` may be NULL when the caller does not want the detail. When it is not
 * NULL it is always NUL-terminated, and a message longer than `err_len - 1`
 * bytes is truncated on a character boundary rather than refused.
 *
 * ---------------------------------------------------------------------------
 * Ownership
 * ---------------------------------------------------------------------------
 *
 * Strings: every non-NULL return is owned by the caller and released with
 * occ_free(). A string stays valid after occ_close(), because it is a copy and
 * not a view into the session. Freeing the same pointer twice is undefined, as
 * it is for free().
 *
 * Session: occ_open() returns a session that must be released with
 * occ_close(). Closing a session releases its images, not the strings already
 * handed out.
 *
 * ---------------------------------------------------------------------------
 * Threads
 * ---------------------------------------------------------------------------
 *
 * A session serialises its own mutations behind a lock, so several threads may
 * call into one OccSession at once, including occ_open() on the same session.
 * Two rules are the caller's to keep:
 *
 *   1. Do not call occ_close() while any other thread is inside a call on the
 *      same session.
 *   2. Do not free a string while another thread is reading it.
 *
 * occ_version() touches no state and may be called from anywhere, including
 * before occ_open().
 *
 * ---------------------------------------------------------------------------
 * Limits
 * ---------------------------------------------------------------------------
 *
 * occ_disasm() caps a single call at 100000 instructions, occ_strings() at
 * 20000 strings and occ_xrefs() at 10000 references. A clamped reply sets
 * "truncated": true and explains the ceiling in "caveats". The caps exist so a
 * caller cannot hold the session lock for minutes by asking for a billion of
 * something.
 */

#ifndef OCC_H
#define OCC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* An open session. Opaque: the layout is Rust's business. */
typedef struct OccSession OccSession;

/* ------------------------------------------------------------------------- */
/* Lifetime                                                                   */
/* ------------------------------------------------------------------------- */

/*
 * The library version, for example "0.1.0". Static storage: do not free, and
 * the pointer stays valid for the life of the process.
 */
const char *occ_version(void);

/*
 * Open an image and return a session, or NULL with `err` filled.
 *
 * An `ar` archive opens as one image per member, so a caller must not assume
 * one image. Use occ_image_count() and occ_image_at() to walk them.
 */
OccSession *occ_open(const char *path, char *err, size_t err_len);

/* Close a session and release its images. Passing NULL does nothing. */
void occ_close(OccSession *s);

/* Number of images in the session. Answers 0 for a NULL session. */
uint32_t occ_image_count(const OccSession *s);

/*
 * The image id at `index`, or -1 when the index is past the end. Image ids are
 * what every other function takes; they are not indices and need not be dense.
 */
int64_t occ_image_at(const OccSession *s, uint32_t index);

/* Release a string returned by any function below. NULL is allowed. */
void occ_free(char *p);

/* ------------------------------------------------------------------------- */
/* Queries                                                                    */
/* ------------------------------------------------------------------------- */

/* Structure of one image: format, architecture, sections, imports, coverage. */
char *occ_info(OccSession *s, int64_t image, char *err, size_t err_len);

/*
 * Decode `count` instructions starting at `addr`.
 *
 * Bytes no architecture recognises come back as one-byte `.byte` directives,
 * which is what keeps the listing aligned after a gap. The reply names the
 * address after the last decoded instruction in "next_addr".
 */
char *occ_disasm(OccSession *s, int64_t image, uint64_t addr, uint32_t count,
                 char *err, size_t err_len);

/*
 * One recovered function: extent, blocks with their edges, callers, callees,
 * and any jump tables behind its computed branches.
 *
 * `target` is a symbol name, a "fn:0x401000" handle, or a bare address such as
 * "0x401000" or "401000".
 */
char *occ_function(OccSession *s, int64_t image, const char *target, char *err,
                   size_t err_len);

/* Cross references to a symbol name or address: who calls, jumps, reads, writes. */
char *occ_xrefs(OccSession *s, int64_t image, const char *target, char *err,
                size_t err_len);

/*
 * Decompile one function to pseudo-C.
 *
 * The reply carries "unstructured": the number of constructs that needed a
 * goto. Treat it as part of the answer — a body with 17 unstructured jumps is a
 * partial rendering and says so.
 */
char *occ_decompile(OccSession *s, int64_t image, const char *target, char *err,
                    size_t err_len);

/* Strings found in the image, in address order. */
char *occ_strings(OccSession *s, int64_t image, char *err, size_t err_len);

/*
 * Raw bytes of one section, hex encoded, for a caller that wants its own
 * parser. Answers NULL when the section does not exist or has no file bytes
 * (`.bss` typically has none).
 */
char *occ_section_bytes(OccSession *s, int64_t image, const char *name, char *err,
                        size_t err_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OCC_H */
