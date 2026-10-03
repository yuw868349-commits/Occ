cmake_minimum_required(VERSION 3.20)

project(occ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- toolchain
#
# The reference build uses Clang with libc++ from an LLVM release tree and
# links fully static. The search is done in the toolchain, not nailed to one
# filesystem layout, so a distribution-packaged libc++ works too.

if(NOT DEFINED OCC_LIBCXX_ROOT)
    foreach(candidate /usr/lib/llvm-23 /usr/lib/llvm-22 /usr/lib/llvm-21
                      /usr/lib/llvm-20 /usr/lib/llvm-19 /usr/lib/llvm-18
                      /usr/lib/llvm-17 /opt/LLVM-23.1.2-Linux-X64)
        if(EXISTS "${candidate}/include/c++/v1/print" OR
           EXISTS "${candidate}/include/x86_64-unknown-linux-gnu/c++/v1/print")
            set(OCC_LIBCXX_ROOT "${candidate}" CACHE PATH "libc++ root")
            break()
        endif()
    endforeach()
endif()

set(OCC_USE_LIBCXX ON CACHE BOOL "Build against libc++ instead of libstdc++")
set(OCC_STATIC_LINK ON CACHE BOOL "Link the main binary statically")

if(OCC_USE_LIBCXX AND OCC_LIBCXX_ROOT)
    message(STATUS "occ: libc++ root ${OCC_LIBCXX_ROOT}")

    foreach(suffix /include/x86_64-unknown-linux-gnu/c++/v1
                   /include/c++/v1)
        if(EXISTS "${OCC_LIBCXX_ROOT}${suffix}/print")
            include_directories(SYSTEM "${OCC_LIBCXX_ROOT}${suffix}")
            set(OCC_LIBCXX_INCLUDE "${OCC_LIBCXX_ROOT}${suffix}")
        endif()
    endforeach()

    foreach(suffix /lib/x86_64-unknown-linux-gnu /lib)
        if(EXISTS "${OCC_LIBCXX_ROOT}${suffix}/libc++.a" OR
           EXISTS "${OCC_LIBCXX_ROOT}${suffix}/libc++.so")
            link_directories("${OCC_LIBCXX_ROOT}${suffix}")
            set(OCC_LIBCXX_LIBDIR "${OCC_LIBCXX_ROOT}${suffix}")
        endif()
    endforeach()

    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
    set(OCC_CXX_LIBS c++ c++abi)
else()
    message(STATUS "occ: libc++ not found, falling back to libstdc++")
    set(OCC_CXX_LIBS)
endif()

# When several GCC installations are present, Clang 19 and later warn that a
# future release will change which one it prefers. The warning is informative
# but it fires on every translation unit, and this build treats warnings as
# errors. Naming the directory makes the choice explicit and the build quiet.
if(OCC_USE_LIBCXX)
    foreach(ver 15 14)
        if(EXISTS "/usr/lib/gcc/x86_64-linux-gnu/${ver}")
            add_compile_options(
                "--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/${ver}")
            set(OCC_GCC_LIB_DIR "/usr/lib/gcc/x86_64-linux-gnu/${ver}")
            break()
        endif()
    endforeach()
endif()

# libc++abi needs _Unwind_*. Both LLVM's libunwind and GCC's libgcc_eh
# provide it. Linking both produces duplicate-symbol errors that do not name
# the cause, so exactly one is chosen. libgcc is chosen where a GCC runtime
# is present, which it is wherever a C++ toolchain is installed.
if(OCC_USE_LIBCXX AND OCC_STATIC_LINK)
    add_link_options(--unwindlib=libgcc)
endif()

if(OCC_STATIC_LINK)
    add_link_options(-static)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion
                    -Wsign-conversion -Wcast-qual -Wdouble-promotion
                    -Wformat=2 -Wundef -Wnull-dereference -Wimplicit-fallthrough)

# ---------------------------------------------------------------- options

option(OCC_ENABLE_TESTS "Build unit tests" ON)
option(OCC_ENABLE_FUZZ "Build libFuzzer harnesses" OFF)
option(OCC_WERROR "Treat warnings as errors" ON)

if(OCC_WERROR)
    add_compile_options(-Werror)
endif()

if(OCC_ENABLE_FUZZ)
    add_compile_options(-fsanitize=fuzzer-no-link,address,undefined
                        -fno-omit-frame-pointer)
    add_link_options(-fsanitize=fuzzer,address,undefined)
    # A sanitized build cannot be a static build; libFuzzer supplies main().
    set(OCC_STATIC_LINK OFF)
endif()

# ---------------------------------------------------------------- core

add_library(occ_core STATIC
    src/syscall/syscall.cpp
    src/syscall/capability.cpp
    src/util/log.cpp
    src/util/string.cpp
    src/util/errno.cpp
    src/util/fs.cpp
    src/probe/doctor.cpp
    src/isolation/seccomp.cpp
)

target_include_directories(occ_core PUBLIC include)
target_compile_options(occ_core PRIVATE -fno-exceptions -fno-rtti)

# ---------------------------------------------------------------- binary

add_executable(occ
    src/main.cpp
    src/cmd_check.cpp
    src/cmd_doctor.cpp
    src/cmd_run.cpp
    src/cmd_attach.cpp
    src/cmd_stop.cpp
)

target_link_libraries(occ PRIVATE occ_core)
if(OCC_CXX_LIBS)
    target_link_libraries(occ PRIVATE ${OCC_CXX_LIBS})
endif()

# ---------------------------------------------------------------- tests

if(OCC_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(OCC_ENABLE_FUZZ)
    add_subdirectory(fuzz)
endif()
