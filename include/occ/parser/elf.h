#pragma once

// ELF image reader.
//
// The loader reads an ELF file and turns it into a description of what has
// to be mapped where. It does not perform the mapping: that is the engine's
// job, and keeping the two apart means the parser can be tested against
// files without any of them being executed.
//
// The reader is complete for ELF64 little-endian x86-64, which is the case
// the exe engine runs, and it reports what it cannot handle rather than
// reading a file it only half understands. A partial parse of an ELF file
// is worse than no parse: the fields that were skipped are exactly the ones
// that would have changed the answer.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "occ/util/span.h"

namespace occ::parser {

// The ELF types this reader understands. They mirror the ident block and
// header fields but are scoped to this namespace so that a caller does not
// have to include the whole detection API to read a header.
enum class ElfClass : std::uint8_t;
enum class ElfEndian : std::uint8_t;
enum class ElfMachine : std::uint16_t;
enum class ElfType : std::uint16_t;

// p_type
enum class SegmentType : std::uint32_t {
    Null = 0,
    Load = 1,
    Dynamic = 2,
    Interp = 3,
    Note = 4,
    Shlib = 5,
    Phdr = 6,
    Tls = 7,
    GnuEhFrame = 0x6474e550,
    GnuStack = 0x6474e551,
    GnuRelro = 0x6474e552,
    GnuProperty = 0x6474e553,
};

// p_flags
inline constexpr std::uint32_t kPfExec = 1;
inline constexpr std::uint32_t kPfWrite = 2;
inline constexpr std::uint32_t kPfRead = 4;

// PT_GNU_STACK and PT_GNU_RELRO are the two program headers that carry
// security-relevant information rather than a loadable region.
struct StackRequest {
    bool present = false;
    bool executable = false;
    std::uint64_t size = 0;
};

struct ProgramHeader {
    SegmentType type = SegmentType::Null;
    std::uint32_t flags = 0;
    // Offsets are into the file; addresses are where the segment wants to
    // live in the process.
    std::uint64_t offset = 0;
    std::uint64_t vaddr = 0;
    std::uint64_t paddr = 0;
    std::uint64_t filesz = 0;
    std::uint64_t memsz = 0;
    std::uint64_t align = 0;

    [[nodiscard]] bool readable() const noexcept {
        return (flags & kPfRead) != 0;
    }
    [[nodiscard]] bool writable() const noexcept {
        return (flags & kPfWrite) != 0;
    }
    [[nodiscard]] bool executable() const noexcept { return (flags & kPfExec) != 0; }

    // The part of the segment that exists in the file but not in memory is
    // the tail of .bss, and it has to be zeroed rather than copied.
    [[nodiscard]] std::uint64_t zero_tail() const noexcept {
        return memsz > filesz ? memsz - filesz : 0;
    }
};

// Where a loadable segment has to be placed. One DesiredMapping is produced
// per PT_LOAD program header, after the headers have been sorted and the
// overlaps resolved.
struct DesiredMapping {
    std::uint64_t file_offset = 0;
    std::uint64_t vaddr = 0;
    std::uint64_t filesz = 0;
    // The region from vaddr + filesz to vaddr + memsz has to be zeroed. It
    // may span more than one page, and the page boundary that separates it
    // from the file-backed part is the one the kernel rounds to.
    std::uint64_t memsz = 0;
    std::uint32_t flags = 0;
};

enum class LoadError : std::uint8_t {
    None,
    NotElf,
    UnsupportedClass,
    UnsupportedEndian,
    UnsupportedMachine,
    TruncatedHeader,
    TruncatedProgramHeaders,
    NoLoadSegments,
    BadAlignment,
    SegmentOutOfFile,
};

[[nodiscard]] const char* load_error_name(LoadError e) noexcept;

// The parsed image. Reading is done entirely from a byte range, so the
// caller decides where the bytes came from.
class ElfImage {
public:
    // Parses `bytes`. On failure the returned object is in the NotElf state
    // and `error()` names the reason.
    [[nodiscard]] static ElfImage parse(ByteSpan bytes) noexcept;

    [[nodiscard]] bool ok() const noexcept { return error_ == LoadError::None; }
    [[nodiscard]] LoadError error() const noexcept { return error_; }
    [[nodiscard]] const std::string& error_detail() const noexcept {
        return detail_;
    }

    [[nodiscard]] std::uint16_t machine() const noexcept { return machine_; }
    [[nodiscard]] std::uint16_t type() const noexcept { return type_; }
    [[nodiscard]] std::uint64_t entry() const noexcept { return entry_; }
    [[nodiscard]] std::uint64_t phoff() const noexcept { return phoff_; }
    [[nodiscard]] std::uint16_t phnum() const noexcept { return phnum_; }
    [[nodiscard]] std::uint16_t phentsize() const noexcept {
        return phentsize_;
    }

    [[nodiscard]] const std::vector<ProgramHeader>& headers() const noexcept {
        return headers_;
    }
    [[nodiscard]] const std::vector<DesiredMapping>& mappings() const noexcept {
        return mappings_;
    }
    [[nodiscard]] const StackRequest& stack() const noexcept { return stack_; }

    // The lowest virtual address any PT_LOAD segment occupies. The loader
    // uses this to decide whether the image is non-relocatable, which is
    // what distinguishes an ET_EXEC from a PIE.
    [[nodiscard]] std::uint64_t lowest_vaddr() const noexcept {
        return lowest_vaddr_;
    }
    [[nodiscard]] std::uint64_t highest_vaddr() const noexcept {
        return highest_vaddr_;
    }
    [[nodiscard]] bool has_interpreter() const noexcept {
        return has_interpreter_;
    }
    [[nodiscard]] const std::string& interpreter() const noexcept {
        return interpreter_;
    }
    [[nodiscard]] bool has_gnu_relro() const noexcept { return has_relro_; }

    // True when the image asks for an executable stack. Reported rather
    // than silently honoured, because a target that needs one is a fact the
    // caller has to decide about.
    [[nodiscard]] bool wants_executable_stack() const noexcept {
        return stack_.present && stack_.executable;
    }

private:
    LoadError error_ = LoadError::NotElf;
    std::string detail_;

    std::uint16_t machine_ = 0;
    std::uint16_t type_ = 0;
    std::uint64_t entry_ = 0;
    std::uint64_t phoff_ = 0;
    std::uint16_t phnum_ = 0;
    std::uint16_t phentsize_ = 0;

    std::vector<ProgramHeader> headers_;
    std::vector<DesiredMapping> mappings_;
    StackRequest stack_;

    std::uint64_t lowest_vaddr_ = 0;
    std::uint64_t highest_vaddr_ = 0;
    bool has_interpreter_ = false;
    std::string interpreter_;
    bool has_relro_ = false;
};

// Page helpers. The loader works in pages because the kernel does, and
// rounding in one place rather than at each call site is what keeps the
// mapping sizes consistent.
inline constexpr std::uint64_t kPageSize = 4096;
[[nodiscard]] inline std::uint64_t page_floor(std::uint64_t v) noexcept {
    return v & ~(kPageSize - 1);
}
[[nodiscard]] inline std::uint64_t page_ceil(std::uint64_t v) noexcept {
    return (v + kPageSize - 1) & ~(kPageSize - 1);
}

} // namespace occ::parser
