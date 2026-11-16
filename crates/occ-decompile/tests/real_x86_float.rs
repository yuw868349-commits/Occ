//! The x86-64 floating-point, atomic and string paths, checked against a
//! real binary.
//!
//! The unit tests lift instruction text. This file compiles the same
//! constructs with the system compiler and checks what comes back, which is
//! the only way to catch the mismatch between what the decoder prints and
//! what the lifter expects: a decoder that prints `fdivp st(1), st(0)` and a
//! lifter that reads the operands the other way round both look right on
//! their own.
//!
//! Every check is on the decompiled text, not on the intermediate program,
//! because the text is what a user reads.

use occ_analyze::{Analysis, AnalysisOptions, View};
use occ_core::Addr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

/// The constructs under test.
///
/// The x87 one is the interesting one: `a * b + a / b` compiles to a stack
/// shuffle that no straightforward reading of the manual gets right.
const SOURCE: &str = r#"
#include <string.h>
#include <emmintrin.h>
#include <stdatomic.h>

volatile int sink;
double gdouble;
float gfloat;

float sse_sum(const float *a, const float *b, int n) {
    __m128 va = _mm_loadu_ps(a);
    __m128 vb = _mm_loadu_ps(b);
    __m128 v = _mm_add_ps(va, vb);
    float out[4];
    _mm_storeu_ps(out, v);
    return out[0] + out[1] + out[2] + out[3];
}

long double x87_ld(long double a, long double b) {
    long double r = a * b + a / b;
    return r;
}

void copy(char *d, const char *s, unsigned long n) {
    memcpy(d, s, n);
}

void fill(char *d, int c, unsigned long n) {
    memset(d, c, n);
}

/* The compiler turns a `memcpy` call into a call, so the string
   instructions are asked for directly. */
void copy_rep(char *d, const char *s, unsigned long n) {
    __asm__ volatile ("rep movsb" : "+D"(d), "+S"(s), "+c"(n) :: "memory");
}

void copy_repq(char *d, const char *s, unsigned long n) {
    __asm__ volatile ("rep movsq" : "+D"(d), "+S"(s), "+c"(n) :: "memory");
}

void fill_rep(char *d, int c, unsigned long n) {
    __asm__ volatile ("rep stosb" : "+D"(d), "+c"(n) : "a"(c) : "memory");
}

int atomic_add(int *p) {
    return atomic_fetch_add(p, 3);
}

int atomic_cas(int *p, int expected) {
    int e = expected;
    return atomic_compare_exchange_strong(p, &e, 7);
}

int framed(int a, int b) {
    char buf[32];
    int local = a * b;
    for (int i = 0; i < 10; i++) local += i;
    memcpy(buf, "hello", 6);
    sink = local + buf[0];
    return local;
}

int main(void) { return framed(2, 3); }
"#;

/// Counter behind the file name, so two tests compiling at the same time do
/// not fight over one path.
static SEQ: AtomicU32 = AtomicU32::new(0);

/// The compiled fixture, built once for the whole file.
static BINARY: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Compile the fixture and return the binary.
fn build() -> PathBuf {
    BINARY
        .get_or_init(|| {
            let n = SEQ.fetch_add(1, Ordering::SeqCst);
            let dir = std::env::temp_dir()
                .join(format!("occ-x86-float-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("create scratch directory");
            let src = dir.join("unit.c");
            std::fs::write(&src, SOURCE).expect("write source");
            let out = dir.join("unit");
            let output = Command::new("cc")
                .args(["-O2", "-fno-inline", "-o"])
                .arg(&out)
                .arg(&src)
                .output()
                .expect("the system compiler runs");
            assert!(
                output.status.success(),
                "the fixture must compile:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            out
        })
        .clone()
}

/// A loaded binary and everything the analyses need.
struct Loaded {
    analysis: Analysis,
    image: occ_core::Image,
}

fn load(path: &Path) -> Loaded {
    let data = std::fs::read(path).expect("read binary");
    let image = occ_bin::load(&data, &path.display().to_string(), occ_core::ImageId(0))
        .expect("parse binary");
    let view = View::new(&image, &data);
    let analysis = Analysis::build(&view, &AnalysisOptions::default());
    Loaded { analysis, image }
}

/// The function with this name.
fn function(loaded: &Loaded, name: &str) -> occ_core::Function {
    loaded
        .analysis
        .functions
        .iter()
        .find(|f| f.display_name() == name)
        .unwrap_or_else(|| {
            panic!(
                "{name} is recovered; the binary has {:?}",
                loaded
                    .analysis
                    .functions
                    .iter()
                    .map(|f| f.display_name())
                    .collect::<Vec<_>>()
            )
        })
        .clone()
}

/// The resolver for call targets.
fn name_resolver(loaded: &Loaded) -> impl Fn(Addr) -> String + '_ {
    let analysis = &loaded.analysis;
    move |a: Addr| {
        analysis
            .function_at(a)
            .map(|f| f.display_name())
            .unwrap_or_else(|| format!("sub_{:x}", a.get()))
    }
}

/// The decompiled pseudo-C of one function.
fn decompile(loaded: &Loaded, f: &occ_core::Function) -> String {
    let name_of = name_resolver(loaded);
    let d = occ_decompile::decompile(
        &loaded.image,
        f,
        &occ_decompile::Options::default(),
        &name_of,
    );
    occ_decompile::render::render(&d)
}

#[test]
fn the_x87_stack_survives_a_real_compilation() {
    let path = build();
    let loaded = load(&path);
    let text = decompile(&loaded, &function(&loaded, "x87_ld"));
    println!("{text}");
    // The two arguments arrive as ten-byte slots and the arithmetic is
    // the machine's: the top of the stack is the left operand.
    assert!(text.contains("f80_mul("), "{text}");
    assert!(text.contains("f80_div("), "{text}");
    assert!(text.contains("f80_add("), "{text}");
    // Nothing was skipped.
    assert!(!text.contains("not lifted"), "{text}");
    assert!(!text.contains("unparsed"), "{text}");
}

#[test]
fn the_vector_path_survives_a_real_compilation() {
    let path = build();
    let loaded = load(&path);
    let text = decompile(&loaded, &function(&loaded, "sse_sum"));
    println!("{text}");
    assert!(text.contains("f32_add("), "{text}");
    assert!(!text.contains("not lifted"), "{text}");
    assert!(!text.contains("unparsed"), "{text}");
}

#[test]
fn the_atomic_path_survives_a_real_compilation() {
    let path = build();
    let loaded = load(&path);
    let add = decompile(&loaded, &function(&loaded, "atomic_add"));
    println!("{add}");
    // The old value is read once and used twice: for the sum that goes
    // into memory and for the value the function returns, which is the
    // value the memory had before. Reading it again after the store
    // would return the new value.
    let returned = add
        .lines()
        .find(|l| l.trim_start().starts_with("return "))
        .expect("the function returns something");
    assert!(
        !returned.contains('*'),
        "the result is the old value, not a re-read of memory:\n{add}"
    );
    assert!(
        add.contains("+ 3"),
        "the addend is the constant:\n{add}"
    );
    let cas = decompile(&loaded, &function(&loaded, "atomic_cas"));
    println!("{cas}");
    // The exchange is conditional: the memory is written only when the
    // accumulator matches what is there.
    assert!(cas.contains("cmpxchg"), "{cas}");
    assert!(!cas.contains("not lifted"), "{cas}");
    assert!(!cas.contains("unparsed"), "{cas}");
}

#[test]
fn the_string_path_survives_a_real_compilation() {
    let path = build();
    let loaded = load(&path);
    for name in ["copy_rep", "copy_repq", "fill_rep"] {
        let text = decompile(&loaded, &function(&loaded, name));
        println!("{text}");
        assert!(!text.contains("not lifted"), "{name}:\n{text}");
        assert!(!text.contains("unparsed"), "{name}:\n{text}");
    }
    // The count runs out and both pointers advance by the number of
    // elements moved. The copy itself is written as the one operation it
    // is, so the memory effect is not lost.
    let copy = decompile(&loaded, &function(&loaded, "copy_rep"));
    assert!(copy.contains("memcpy("), "{copy}");
    assert!(copy.contains("rdi") && copy.contains("rsi"), "{copy}");
    // The count is in elements, so a wider element moves more bytes.
    let copyq = decompile(&loaded, &function(&loaded, "copy_repq"));
    assert!(copyq.contains("memcpy("), "{copyq}");
    assert!(copyq.contains("* 8"), "{copyq}");
    let fill = decompile(&loaded, &function(&loaded, "fill_rep"));
    assert!(fill.contains("memset("), "{fill}");
}
