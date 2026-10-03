#pragma once

// Target format detection.
//
// `occ run` has to decide which engine a path belongs to before it can do
// anything with it, and the decision has to be made from the file itself
// rather than from its name. A file called `app.apk` that is an ELF binary
// and a file called `image` that is an Android package are both possible,
// and inferring the format from the extension produces a confident wrong
// answer in both cases.
//
// The detector reads the bytes. Each format is identified by a signature at
// a fixed offset plus a set of structural constraints that a coincidence
// would have to satisfy as well. The result carries the evidence that
// produced it, because a user who disagrees with the detection needs to see
// which bytes it was based on.

#include <cstdint>
#include <string>
#include <vector>

#include "occ/util/span.h"

namespace occ::parser {

enum class Format : std::uint8_t {
    Unknown,
    // An ELF object. The class, endianness, machine and type are reported
    // separately, because they determine what the exe engine has to do with
    // it: a 64-bit little-endian x86-64 executable is the case it handles
    // natively, and everything else is a refusal with a reason rather than
    // a crash.
    Elf,
    // An Android package: a zip whose first entry is AndroidManifest.xml,
    // stored uncompressed.
    Apk,
    // A zip with a classes.dex member. Treated as a code container, not as
    // an Android package, because a plain jar looks like this too.
    Zip,
    // A Mach-O or PE image. Recognised so that the refusal can name the
    // format instead of reporting a corrupt ELF.
    MachO,
    Pe,
};

[[nodiscard]] const char* format_name(Format f) noexcept;

enum class ElfClass : std::uint8_t { None, Elf32, Elf64 };
enum class ElfEndian : std::uint8_t { None, Little, Big };

// The ELF machine field, for the values the engine distinguishes. Anything
// else is reported as its raw value.
enum class ElfMachine : std::uint16_t {
    None = 0,
    X86 = 3,
    Arm = 40,
    X86_64 = 62,
    AArch64 = 183,
    RiscV = 243,
};

// The ELF type field.
enum class ElfType : std::uint16_t {
    None = 0,
    Rel = 1,       // relocatable object
    Exec = 2,      // executable
    Dyn = 3,       // shared object or position-independent executable
    Core = 4,      // core dump
};

[[nodiscard]] const char* elf_class_name(ElfClass c) noexcept;
[[nodiscard]] const char* elf_endian_name(ElfEndian e) noexcept;
[[nodiscard]] const char* elf_machine_name(ElfMachine m) noexcept;
[[nodiscard]] const char* elf_type_name(ElfType t) noexcept;

// Everything the detection learned. A field is left at its default when the
// format does not define it, and `evidence` holds the offsets and sizes that
// produced the verdict.
struct Detection {
    Format format = Format::Unknown;

    // ELF fields. Also populated for a bare program header, which is what
    // an Android OAT file starts with.
    ElfClass elf_class = ElfClass::None;
    ElfEndian elf_endian = ElfEndian::None;
    ElfMachine elf_machine = ElfMachine::None;
    ElfType elf_type = ElfType::None;

    // Set when the file begins with a program header instead of an ELF
    // header, which is how Android stores its ahead-of-time compiled output.
    bool bare_program_header = false;

    std::uint64_t file_size = 0;

    // Human-readable lines describing what was matched. Kept short: this is
    // what `occ check` prints, and a detection that needs a paragraph is a
    // detection that is guessing.
    std::vector<std::string> evidence;
};

// Detects the format of a file. Never fails: a path that cannot be read is
// reported as Format::Unknown with the reason in the evidence.
[[nodiscard]] Detection detect_file(const std::string& path) noexcept;

// Detects from bytes already in memory. Used by the tests and by the
// container path, which may be looking at a file it can only reach through
// a descriptor.
[[nodiscard]] Detection detect_bytes(ByteSpan bytes) noexcept;

} // namespace occ::parser
