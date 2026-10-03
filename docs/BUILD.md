# Build

## Toolchain

Occ requires C++23.

| Compiler | Minimum |
|---|---|
| Clang | 18 |
| GCC | 14 |

The reference build uses Clang with libc++ from the LLVM release tree. The
reason is the static link: libc++'s static archives are self-contained and
are what the shipped binary is built against. libstdc++ works dynamically
and is usable for development. See "Static linking" below for the exact
flags.

CMake 3.20 or newer. Ninja is recommended but not required.

## Configure and build

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Output is `build/occ`, a single statically linked executable.

## Pointing at a libc++ tree

The installed libc++ is located automatically for the common cases. When
it is installed somewhere non-standard — a release tarball under `/opt`,
for instance — pass the root:

```
cmake -S . -B build -G Ninja \
  -DOCC_LIBCXX_ROOT=/opt/LLVM-23.1.2-Linux-X64 \
  -DOCC_GCC_LIB_DIR=/usr/lib/gcc/x86_64-linux-gnu/14
```

`occ doctor` prints both paths so a build can be reproduced from a working
binary.

## Static linking

Fully static, including libc++. The flags, in the order they matter:

```
-stdlib=libc++
-static
--unwindlib=libgcc
-I${OCC_LIBCXX_ROOT}/include/x86_64-unknown-linux-gnu/c++/v1
-L${OCC_LIBCXX_ROOT}/lib/x86_64-unknown-linux-gnu
-lc++ -lc++abi
```

Three things about this are worth writing down because each failed first:

**`--unwindlib=libgcc` is required.** libc++abi needs `_Unwind_*` symbols.
Both LLVM's libunwind and GCC's libgcc_eh provide them, and linking both
produces a wall of duplicate symbol errors that does not name the cause.
Picking one is the fix. libgcc is chosen because it is already present
where the toolchain is.

**`__gcc_personality_v0` is referenced from static libc.** It lives in
`libgcc_eh`. With `--unwindlib=libgcc` the link resolves it without an
explicit `-lgcc_eh`.

**Do not add `-lc++experimental`.** It pulls in a second definition of
several facilities that overlap with libc++ proper.

Verify the result:

```
file build/occ
# build/occ: ELF 64-bit LSB executable, x86-64, statically linked, ...
ldd build/occ
# not a dynamic executable
```

## Tests

```
cmake -S . -B build -G Ninja -DOCC_ENABLE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests are built into the static configuration. They run against the same
isolation code the binary ships, which is the point.

## Fuzz harnesses

Sanitizers and libFuzzer cannot be combined with a fully static link, so
the fuzz build is separate and dynamic:

```
cmake -S . -B build-fuzz -G Ninja \
  -DOCC_ENABLE_FUZZ=ON \
  -DOCC_ENABLE_TESTS=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer,address,undefined"
cmake --build build-fuzz
```

Nineteen hours of a corpus is not the same as correctness, but it is more
than zero, and zero is what a parser without a harness has.

## Sanitizer build of the test suite

```
cmake -S . -B build-asan -G Ninja \
  -DOCC_ENABLE_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

## Memory

A full build of a C++23 tree with sanitizers is not small. On a host with
2 GiB and a swap file, limit parallelism:

```
cmake --build build -j2
```

Without a swap file the linker will be killed on a 2 GiB host. This is
recorded because it happened.

## Reproducing a reported environment

`COMPAT.md` lists hosts this has been built and run on, with the kernel
release string, the toolchain paths, and the `occ doctor` output for each.
A build that differs from every entry there is not wrong, it is just not
one of the verified cases.
