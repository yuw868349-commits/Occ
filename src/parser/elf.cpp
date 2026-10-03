#include "occ/parser/elf.h"

#include "occ/util/string.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace occ::parser {

namespace {

// Field offsets in the ELF64 header, and the values the identification
// block has to hold for the 64-bit little-endian path. The reader is
// deliberately specific: it is the reader for the case the engine runs, and
// a general one would be a general one that is only tested on that case.
constexpr std::size_t kIdentSize = 16;
constexpr std::size_t kElf64HeaderSize = 64;
constexpr std::size_t kElf64PhdrSize = 56;

// Field offsets in the 64-bit header. The 32-bit layout puts e_phoff and
// e_phnum elsewhere, so these are named after the word size rather than
// shared with a reader for the other one.
constexpr std::size_t kOffType = 16;
constexpr std::size_t kOffMachine = 18;
constexpr std::size_t kOffEntry = 24;
constexpr std::size_t kOffPhoff = 32;
constexpr std::size_t kOffPhentsize = 54;
constexpr std::size_t kOffPhnum = 56;

constexpr std::uint16_t kEmX86_64 = 62;
constexpr std::uint16_t kEtRel = 1;
constexpr std::uint16_t kEtExec = 2;
constexpr std::uint16_t kEtDyn = 3;
constexpr std::uint16_t kEtCore = 4;

// Rounds a value up to a power-of-two boundary without the arithmetic that
// overflows when the value is near the top of the address space. The
// expression (v + a - 1) & ~(a - 1) overflows for a v within a of
// UINT64_MAX, and a malformed file is exactly where such a value comes
// from.
std::uint64_t align_up_safe(std::uint64_t v, std::uint64_t a) noexcept {
    if (a <= 1) {
        return v;
    }
    const std::uint64_t rem = v & (a - 1);
    if (rem == 0) {
        return v;
    }
    const std::uint64_t delta = a - rem;
    if (v > UINT64_MAX - delta) {
        return UINT64_MAX;
    }
    return v + delta;
}

bool is_power_of_two(std::uint64_t v) noexcept {
    return v != 0 && (v & (v - 1)) == 0;
}

std::uint16_t rd16(ByteSpan b, std::size_t off) noexcept {
    if (off + 2 > b.size()) {
        return 0;
    }
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(b[off]) |
        (static_cast<std::uint16_t>(b[off + 1]) << 8));
}

std::uint32_t rd32(ByteSpan b, std::size_t off) noexcept {
    if (off + 4 > b.size()) {
        return 0;
    }
    return static_cast<std::uint32_t>(b[off]) |
           (static_cast<std::uint32_t>(b[off + 1]) << 8) |
           (static_cast<std::uint32_t>(b[off + 2]) << 16) |
           (static_cast<std::uint32_t>(b[off + 3]) << 24);
}

std::uint64_t rd64(ByteSpan b, std::size_t off) noexcept {
    if (off + 8 > b.size()) {
        return 0;
    }
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(b[off + static_cast<std::size_t>(i)])
             << (8 * i);
    }
    return v;
}

std::string decimal(std::uint64_t v) {
    std::string out;
    append_uint(out, v);
    return out;
}

std::string hex_value(std::uint64_t v) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    if (v == 0) {
        return "0x0";
    }
    while (v != 0) {
        out.push_back(digits[v & 0xf]);
        v >>= 4;
    }
    out.push_back('x');
    out.push_back('0');
    std::reverse(out.begin(), out.end());
    return out;
}

} // namespace

const char* load_error_name(LoadError e) noexcept {
    switch (e) {
    case LoadError::None:
        return "none";
    case LoadError::NotElf:
        return "not an ELF object";
    case LoadError::UnsupportedClass:
        return "not a 64-bit object";
    case LoadError::UnsupportedEndian:
        return "not little-endian";
    case LoadError::UnsupportedMachine:
        return "not x86-64";
    case LoadError::TruncatedHeader:
        return "the header is truncated";
    case LoadError::TruncatedProgramHeaders:
        return "the program header table is truncated";
    case LoadError::NoLoadSegments:
        return "there is no loadable segment";
    case LoadError::BadAlignment:
        return "a segment has an alignment that is not a power of two";
    case LoadError::SegmentOutOfFile:
        return "a segment extends past the end of the file";
    }
    return "unknown";
}

ElfImage ElfImage::parse(ByteSpan bytes) noexcept {
    ElfImage out;

    if (bytes.size() < kIdentSize || bytes[0] != 0x7f || bytes[1] != 'E' ||
        bytes[2] != 'L' || bytes[3] != 'F') {
        out.error_ = LoadError::NotElf;
        out.detail_ = "the ELF magic is absent";
        return out;
    }

    if (bytes[4] != 2) {
        out.error_ = LoadError::UnsupportedClass;
        out.detail_ = "EI_CLASS is " + decimal(bytes[4]) +
                      "; only 2 (ELFCLASS64) is handled";
        return out;
    }
    if (bytes[5] != 1) {
        out.error_ = LoadError::UnsupportedEndian;
        out.detail_ = "EI_DATA is " + decimal(bytes[5]) +
                      "; only 1 (ELFDATA2LSB) is handled";
        return out;
    }
    if (bytes[6] != 1) {
        out.error_ = LoadError::NotElf;
        out.detail_ = "EI_VERSION is " + decimal(bytes[6]) +
                      "; the only defined value is 1";
        return out;
    }

    if (bytes.size() < kElf64HeaderSize) {
        out.error_ = LoadError::TruncatedHeader;
        out.detail_ = "the file is " + decimal(bytes.size()) +
                      " bytes and a 64-bit header needs " +
                      decimal(kElf64HeaderSize);
        return out;
    }

    out.type_ = rd16(bytes, kOffType);
    out.machine_ = rd16(bytes, kOffMachine);
    out.entry_ = rd64(bytes, kOffEntry);
    out.phoff_ = rd64(bytes, kOffPhoff);
    out.phentsize_ = rd16(bytes, kOffPhentsize);
    out.phnum_ = rd16(bytes, kOffPhnum);

    if (out.machine_ != kEmX86_64) {
        out.error_ = LoadError::UnsupportedMachine;
        out.detail_ = "e_machine is " + decimal(out.machine_) +
                      "; only 62 (EM_X86_64) is handled";
        return out;
    }

    switch (out.type_) {
    case kEtExec:
    case kEtDyn:
        break;
    case kEtRel:
        out.error_ = LoadError::NoLoadSegments;
        out.detail_ = "e_type is ET_REL; a relocatable object is not a "
                      "program and has no program header table";
        return out;
    case kEtCore:
        out.error_ = LoadError::NoLoadSegments;
        out.detail_ = "e_type is ET_CORE; a core file is not a program";
        return out;
    default:
        out.error_ = LoadError::NoLoadSegments;
        out.detail_ = "e_type is " + decimal(out.type_) + ", which is "
                      "neither ET_EXEC nor ET_DYN";
        return out;
    }

    if (out.phnum_ == 0) {
        out.error_ = LoadError::NoLoadSegments;
        out.detail_ = "e_phnum is 0";
        return out;
    }

    // The entry size has to be the size this reader steps by. A file that
    // declares a different one is either corrupt or for an ABI this reader
    // does not implement, and both are reasons to stop rather than to walk
    // the table with the wrong stride.
    if (out.phentsize_ != kElf64PhdrSize) {
        out.error_ = LoadError::TruncatedProgramHeaders;
        out.detail_ = "e_phentsize is " + decimal(out.phentsize_) +
                      " and this reader steps by " +
                      decimal(kElf64PhdrSize);
        return out;
    }

    const std::uint64_t table_end =
        out.phoff_ + static_cast<std::uint64_t>(out.phnum_) * kElf64PhdrSize;
    if (out.phoff_ < kElf64HeaderSize || table_end > bytes.size()) {
        out.error_ = LoadError::TruncatedProgramHeaders;
        out.detail_ = "the table runs from " + hex_value(out.phoff_) + " to " +
                      hex_value(table_end) + " and the file is " +
                      decimal(bytes.size()) + " bytes";
        return out;
    }

    out.headers_.reserve(out.phnum_);
    for (std::uint16_t i = 0; i < out.phnum_; ++i) {
        const std::size_t base =
            static_cast<std::size_t>(out.phoff_) +
            static_cast<std::size_t>(i) * kElf64PhdrSize;

        ProgramHeader h;
        // The 64-bit program header: p_type, p_flags, p_offset, p_vaddr,
        // p_paddr, p_filesz, p_memsz, p_align. Note that flags come second,
        // before the offsets, which is not the order the names suggest.
        h.type = static_cast<SegmentType>(rd32(bytes, base + 0));
        h.flags = rd32(bytes, base + 4);
        h.offset = rd64(bytes, base + 8);
        h.vaddr = rd64(bytes, base + 16);
        h.paddr = rd64(bytes, base + 24);
        h.filesz = rd64(bytes, base + 32);
        h.memsz = rd64(bytes, base + 40);
        h.align = rd64(bytes, base + 48);

        switch (h.type) {
        case SegmentType::Load:
            break;
        case SegmentType::Interp:
            out.has_interpreter_ = true;
            break;
        case SegmentType::GnuStack:
            out.stack_.present = true;
            out.stack_.executable = h.executable();
            out.stack_.size = h.memsz;
            break;
        case SegmentType::GnuRelro:
            out.has_relro_ = true;
            break;
        default:
            break;
        }

        out.headers_.push_back(h);
    }

    // A PT_INTERP names the dynamic loader. Its contents are read here so
    // that the engine can report which interpreter the target expects
    // without having to keep the file open.
    if (out.has_interpreter_) {
        for (const auto& h : out.headers_) {
            if (h.type != SegmentType::Interp) {
                continue;
            }
            if (h.offset + h.filesz > bytes.size() || h.filesz == 0) {
                out.error_ = LoadError::SegmentOutOfFile;
                out.detail_ = "PT_INTERP runs past the end of the file";
                return out;
            }
            const std::size_t len = static_cast<std::size_t>(h.filesz);
            const char* p =
                reinterpret_cast<const char*>(bytes.data() + h.offset);
            // The segment is a NUL-terminated path, and the terminator is
            // part of filesz.
            std::size_t n = 0;
            while (n < len && p[n] != '\0') {
                ++n;
            }
            out.interpreter_.assign(p, n);
            break;
        }
    }

    // The mappings are derived from the PT_LOAD headers. The kernel's own
    // loader sorts them and merges what can be merged; this does the same,
    // because the alternative -- mapping each header separately -- produces
    // overlapping mappings whose permissions are whichever was mapped last.
    std::vector<ProgramHeader> loads;
    for (const auto& h : out.headers_) {
        if (h.type == SegmentType::Load) {
            loads.push_back(h);
        }
    }

    if (loads.empty()) {
        out.error_ = LoadError::NoLoadSegments;
        out.detail_ = "no program header has p_type PT_LOAD";
        return out;
    }

    std::sort(loads.begin(), loads.end(),
              [](const ProgramHeader& a, const ProgramHeader& b) {
                  return a.vaddr < b.vaddr;
              });

    for (const auto& h : loads) {
        if (h.align > 1 && !is_power_of_two(h.align)) {
            out.error_ = LoadError::BadAlignment;
            out.detail_ = "a PT_LOAD segment at " + hex_value(h.vaddr) +
                          " has p_align " + hex_value(h.align) +
                          ", which is not a power of two";
            return out;
        }
        // The congruent-offset rule from the ELF specification: a segment
        // whose filesz is non-zero has to satisfy offset = vaddr (mod
        // p_align), because the kernel maps whole pages and the file is
        // read from the rounded-down offset. A file that violates it would
        // be mapped with the wrong bytes at the start of the segment.
        if (h.filesz != 0 && h.align > 1 &&
            (h.offset % h.align) != (h.vaddr % h.align)) {
            out.error_ = LoadError::BadAlignment;
            out.detail_ = "a PT_LOAD segment at " + hex_value(h.vaddr) +
                          " has p_offset " + hex_value(h.offset) +
                          " and p_vaddr " + hex_value(h.vaddr) +
                          " that are not congruent modulo p_align " +
                          hex_value(h.align);
            return out;
        }
        if (h.offset + h.filesz > bytes.size()) {
            out.error_ = LoadError::SegmentOutOfFile;
            out.detail_ = "a PT_LOAD segment runs from " + hex_value(h.offset) +
                          " for " + hex_value(h.filesz) + " bytes and the "
                          "file is " + decimal(bytes.size()) + " bytes";
            return out;
        }
        if (h.memsz < h.filesz) {
            out.error_ = LoadError::SegmentOutOfFile;
            out.detail_ = "a PT_LOAD segment at " + hex_value(h.vaddr) +
                          " has p_memsz smaller than p_filesz";
            return out;
        }
    }

    out.lowest_vaddr_ = loads.front().vaddr;
    std::uint64_t high = 0;
    for (const auto& h : loads) {
        const std::uint64_t end = align_up_safe(h.vaddr, kPageSize) +
                                  align_up_safe(h.memsz, kPageSize);
        if (end > high) {
            high = end;
        }
    }
    out.highest_vaddr_ = high;

    for (const auto& h : loads) {
        DesiredMapping m;
        m.file_offset = h.offset;
        m.vaddr = h.vaddr;
        m.filesz = h.filesz;
        m.memsz = h.memsz;
        m.flags = h.flags;
        out.mappings_.push_back(m);
    }

    out.error_ = LoadError::None;
    return out;
}

} // namespace occ::parser
