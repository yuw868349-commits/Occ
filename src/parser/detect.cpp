#include "occ/parser/detect.h"

#include "occ/util/fs.h"
#include "occ/util/string.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace occ::parser {

namespace {

// ------------------------------------------------------------ byte reading

// Every multi-byte field in a file this project reads is little- or
// big-endian by its own declaration, so the reads are explicit rather than
// a cast through a packed struct. A cast would also be undefined behaviour
// through alignment and would silently do the wrong thing on a big-endian
// host, which is exactly the case the ELF endianness field exists to handle.
std::uint16_t read_le16(ByteSpan b, std::size_t off) noexcept {
    if (off + 2 > b.size()) {
        return 0;
    }
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(b[off]) |
        (static_cast<std::uint16_t>(b[off + 1]) << 8));
}

std::uint32_t read_le32(ByteSpan b, std::size_t off) noexcept {
    if (off + 4 > b.size()) {
        return 0;
    }
    return static_cast<std::uint32_t>(b[off]) |
           (static_cast<std::uint32_t>(b[off + 1]) << 8) |
           (static_cast<std::uint32_t>(b[off + 2]) << 16) |
           (static_cast<std::uint32_t>(b[off + 3]) << 24);
}

std::uint64_t read_le64(ByteSpan b, std::size_t off) noexcept {
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

bool magic_at(ByteSpan b, std::size_t off, const char* magic,
              std::size_t len) noexcept {
    if (off + len > b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < len; ++i) {
        if (b[off + i] != static_cast<std::uint8_t>(magic[i])) {
            return false;
        }
    }
    return true;
}

std::string hex_byte(std::uint8_t v) {
    static const char* digits = "0123456789abcdef";
    std::string out = "0x";
    out.push_back(digits[(v >> 4) & 0xf]);
    out.push_back(digits[v & 0xf]);
    return out;
}

std::string to_hex(std::uint64_t v) {
    static const char* digits = "0123456789abcdef";
    if (v == 0) {
        return "0x0";
    }
    std::string out;
    while (v != 0) {
        out.push_back(digits[v & 0xf]);
        v >>= 4;
    }
    out.push_back('x');
    out.push_back('0');
    for (std::size_t i = 0, j = out.size() - 1; i < j; ++i, --j) {
        const char t = out[i];
        out[i] = out[j];
        out[j] = t;
    }
    return out;
}

std::string decimal(std::uint64_t v) {
    std::string out;
    append_uint(out, v);
    return out;
}

// ---------------------------------------------------------------- ELF

// Offsets into the ELF identification block and the header that follows it.
// The identification block is the first sixteen bytes of every ELF file and
// has the same layout for both classes, which is what makes it readable
// before the class is known.
constexpr std::size_t kElfIdentSize = 16;
constexpr std::size_t kElfIdentClass = 4;
constexpr std::size_t kElfIdentData = 5;
constexpr std::size_t kElfIdentVersion = 6;
constexpr std::size_t kElfIdentOsAbi = 7;

// Minimum sizes are stated explicitly because a truncated file has to be
// rejected rather than read past. The 64-bit header is larger only because
// its addresses and offsets are eight bytes rather than four.
constexpr std::size_t kElf32HeaderSize = 52;
constexpr std::size_t kElf64HeaderSize = 64;

bool looks_like_elf_ident(ByteSpan b) noexcept {
    return b.size() >= kElfIdentSize && b[0] == 0x7f && b[1] == 'E' &&
           b[2] == 'L' && b[3] == 'F';
}

// Reads the class and endianness from the identification block. Returns
// false when either field holds a value the format does not define, which
// is the strongest available signal that the file only starts with the
// magic by coincidence.
bool read_ident(ByteSpan b, Detection& out) noexcept {
    switch (b[kElfIdentClass]) {
    case 1:
        out.elf_class = ElfClass::Elf32;
        break;
    case 2:
        out.elf_class = ElfClass::Elf64;
        break;
    default:
        out.evidence.push_back("e_ident[EI_CLASS] is " +
                               hex_byte(b[kElfIdentClass]) +
                               ", which is neither 1 nor 2");
        return false;
    }

    switch (b[kElfIdentData]) {
    case 1:
        out.elf_endian = ElfEndian::Little;
        break;
    case 2:
        out.elf_endian = ElfEndian::Big;
        break;
    default:
        out.evidence.push_back("e_ident[EI_DATA] is " +
                               hex_byte(b[kElfIdentData]) +
                               ", which is neither 1 nor 2");
        return false;
    }

    if (b[kElfIdentVersion] != 1) {
        out.evidence.push_back("e_ident[EI_VERSION] is " +
                               hex_byte(b[kElfIdentVersion]) +
                               ", the only defined value is 1");
        return false;
    }

    if (b[kElfIdentOsAbi] != 0 && b[kElfIdentOsAbi] != 3) {
        // 0 is System V, 3 is Linux. A GNU/Linux target is occasionally
        // marked as either. Anything else is unusual but is not a reason to
        // refuse the file, so it is recorded and the detection continues.
        out.evidence.push_back("e_ident[EI_OSABI] is " +
                               hex_byte(b[kElfIdentOsAbi]) +
                               " (neither System V nor Linux)");
    }

    return true;
}

void read_elf_header(ByteSpan b, Detection& out) noexcept {
    const bool wide = out.elf_class == ElfClass::Elf64;
    const bool little = out.elf_endian == ElfEndian::Little;
    const std::size_t need = wide ? kElf64HeaderSize : kElf32HeaderSize;
    if (b.size() < need) {
        out.evidence.push_back("the file is " + decimal(b.size()) +
                               " bytes, a header needs " + decimal(need));
        out.format = Format::Unknown;
        return;
    }

    // e_type and e_machine sit at the same offset in both classes, right
    // after the identification block. A big-endian file is returned from
    // above before this point, so the fields are read little-endian.
    const std::uint16_t type = read_le16(b, 16);
    const std::uint16_t machine = read_le16(b, 18);

    out.elf_type = static_cast<ElfType>(type);
    out.elf_machine = static_cast<ElfMachine>(machine);

    out.evidence.push_back(std::string("e_ident[EI_CLASS] = ") +
                           elf_class_name(out.elf_class));
    out.evidence.push_back(std::string("e_ident[EI_DATA] = ") +
                           elf_endian_name(out.elf_endian));
    out.evidence.push_back("e_type = " + std::string(elf_type_name(out.elf_type)));

    if (out.elf_machine == ElfMachine::X86_64) {
        out.evidence.push_back("e_machine = x86-64");
    } else if (out.elf_machine == ElfMachine::AArch64) {
        out.evidence.push_back("e_machine = aarch64");
    } else {
        out.evidence.push_back("e_machine = " +
                               decimal(static_cast<std::uint16_t>(machine)) +
                               " (not one the exe engine handles natively)");
    }

    // The program header table is what the engine needs in order to map the
    // image, so its absence is worth reporting here rather than at load
    // time.
    //
    // The two classes share none of these offsets, because e_entry is four
    // bytes in the 32-bit header and eight in the 64-bit one, which shifts
    // everything after it by four:
    //
    //            32-bit   64-bit
    //   e_phoff     28       32
    //   e_phnum     44       56
    //
    // Reading the 32-bit offset out of a 64-bit header lands on e_entry and
    // produces a plausible-looking number that is not the table's offset.
    const std::size_t phoff_off = wide ? 32 : 28;
    const std::size_t phnum_off = wide ? 56 : 44;
    const std::uint64_t phoff =
        wide ? read_le64(b, phoff_off) : read_le32(b, phoff_off);
    const std::uint16_t phnum = read_le16(b, phnum_off);

    if (!little) {
        // A big-endian file is reported but its header is not walked: the
        // exe engine does not handle one, and a half-parsed header would be
        // a more confusing answer than a refusal.
        out.evidence.push_back("the header is big-endian and was not walked");
        return;
    }

    if (phnum == 0) {
        out.evidence.push_back("e_phnum = 0; there is no program header "
                               "table and the image has no loadable "
                               "segments");
    } else {
        out.evidence.push_back("e_phnum = " + decimal(phnum) +
                               " segments at " + to_hex(phoff));
    }
}

// An Android OAT file begins with a program header rather than an ELF
// header. The first field of a program header is p_type, and PT_LOAD is 1,
// which is not a value that can be mistaken for ELF magic. The detection
// exists so that an OAT file is named rather than reported as a corrupt ELF.
bool looks_like_bare_program_header(ByteSpan b, Detection& out) noexcept {
    if (b.size() < 4) {
        return false;
    }
    const std::uint32_t p_type = read_le32(b, 0);
    if (p_type != 1) {
        return false;
    }
    // A 64-bit program header is 56 bytes and its p_flags field sits at
    // offset 4, which in practice is 4, 5 or 6. Requiring that narrows the
    // match far enough that a random four bytes are very unlikely to pass.
    if (b.size() >= 8) {
        const std::uint32_t flags = read_le32(b, 4);
        if (flags > 7) {
            return false;
        }
    }
    out.bare_program_header = true;
    out.elf_class = ElfClass::Elf64;
    out.elf_endian = ElfEndian::Little;
    out.evidence.push_back("the file starts with a program header "
                           "(p_type = PT_LOAD, p_flags = " +
                           to_hex(read_le32(b, 4)) +
                           "), which is the layout an Android OAT file uses");
    return true;
}

// ---------------------------------------------------------------- zip

// A zip archive ends with an end-of-central-directory record, but the first
// local file header is what identifies the first member, and the first
// member is what distinguishes an Android package from any other archive.
constexpr std::uint32_t kZipLocalHeaderMagic = 0x04034b50;

// The local file header is 30 bytes followed by the name and then the extra
// field, both of variable length.
constexpr std::size_t kZipLocalHeaderSize = 30;

struct ZipMember {
    std::string name;
    bool stored = false;
    std::uint32_t compressed_size = 0;
};

bool read_first_zip_member(ByteSpan b, ZipMember& out) noexcept {
    if (b.size() < kZipLocalHeaderSize) {
        return false;
    }
    if (read_le32(b, 0) != kZipLocalHeaderMagic) {
        return false;
    }

    // Fields of the local file header, per the PKZIP appnote. The local
    // header repeats the compression method and the sizes so that a reader
    // can extract a member without reading the central directory first.
    const std::uint16_t method = read_le16(b, 8);
    const std::uint32_t comp_size = read_le32(b, 18);
    const std::uint16_t name_len = read_le16(b, 26);
    const std::uint16_t extra_len = read_le16(b, 28);

    if (kZipLocalHeaderSize + name_len > b.size()) {
        return false;
    }

    out.name.assign(reinterpret_cast<const char*>(b.data() + kZipLocalHeaderSize),
                    name_len);
    // Method 0 is stored, 8 is deflate. Android requires the manifest to be
    // stored so that a package can be inspected without inflating anything.
    out.stored = method == 0;
    out.compressed_size = comp_size;
    (void)extra_len;
    return true;
}

} // namespace

const char* format_name(Format f) noexcept {
    switch (f) {
    case Format::Unknown:
        return "unknown";
    case Format::Elf:
        return "elf";
    case Format::Apk:
        return "apk";
    case Format::Zip:
        return "zip";
    case Format::MachO:
        return "mach-o";
    case Format::Pe:
        return "pe";
    }
    return "unknown";
}

const char* elf_class_name(ElfClass c) noexcept {
    switch (c) {
    case ElfClass::None:
        return "none";
    case ElfClass::Elf32:
        return "32-bit";
    case ElfClass::Elf64:
        return "64-bit";
    }
    return "none";
}

const char* elf_endian_name(ElfEndian e) noexcept {
    switch (e) {
    case ElfEndian::None:
        return "none";
    case ElfEndian::Little:
        return "little-endian";
    case ElfEndian::Big:
        return "big-endian";
    }
    return "none";
}

const char* elf_machine_name(ElfMachine m) noexcept {
    switch (m) {
    case ElfMachine::None:
        return "none";
    case ElfMachine::X86:
        return "x86";
    case ElfMachine::Arm:
        return "arm";
    case ElfMachine::X86_64:
        return "x86-64";
    case ElfMachine::AArch64:
        return "aarch64";
    case ElfMachine::RiscV:
        return "risc-v";
    }
    return "unknown";
}

const char* elf_type_name(ElfType t) noexcept {
    switch (t) {
    case ElfType::None:
        return "none";
    case ElfType::Rel:
        return "relocatable";
    case ElfType::Exec:
        return "executable";
    case ElfType::Dyn:
        return "shared object or pie";
    case ElfType::Core:
        return "core";
    }
    return "unknown";
}

Detection detect_bytes(ByteSpan bytes) noexcept {
    Detection out;
    out.file_size = bytes.size();

    if (bytes.size() < 4) {
        out.evidence.push_back("the file is shorter than any signature");
        return out;
    }

    // Order matters in two places. ELF is tested before the bare program
    // header because a real ELF header contains a program header table and
    // would otherwise be ambiguous in one direction only. Mach-O is tested
    // before ELF because one of its magics is the reverse byte order of an
    // ELF magic, and reading them in the other order would classify a
    // 32-bit big-endian Mach-O as an ELF file.
    if (magic_at(bytes, 0, "\xfe\xed\xfa\xce", 4) ||
        magic_at(bytes, 0, "\xfe\xed\xfa\xcf", 4) ||
        magic_at(bytes, 0, "\xce\xfa\xed\xfe", 4) ||
        magic_at(bytes, 0, "\xcf\xfa\xed\xfe", 4)) {
        out.format = Format::MachO;
        out.evidence.push_back("the first four bytes are a Mach-O magic");
        return out;
    }

    if (looks_like_elf_ident(bytes)) {
        if (!read_ident(bytes, out)) {
            out.evidence.push_back(
                "the ELF magic is present but the identification block is "
                "not valid, so this is not an ELF object");
            return out;
        }
        out.format = Format::Elf;
        read_elf_header(bytes, out);
        return out;
    }

    if (looks_like_bare_program_header(bytes, out)) {
        out.format = Format::Elf;
        return out;
    }

    // PE images begin with the DOS stub. The MZ signature alone is weak, so
    // the header offset is followed and the PE signature at that offset is
    // required as well.
    if (magic_at(bytes, 0, "MZ", 2)) {
        const std::uint32_t lfanew = read_le32(bytes, 0x3c);
        if (lfanew != 0 && lfanew < bytes.size() &&
            magic_at(bytes, lfanew, "PE\0\0", 4)) {
            out.format = Format::Pe;
            out.evidence.push_back("MZ stub with a PE signature at " +
                                   to_hex(lfanew));
            return out;
        }
        out.evidence.push_back("MZ signature without a PE header; this is a "
                               "DOS executable, not a Windows image");
        return out;
    }

    ZipMember member;
    if (read_first_zip_member(bytes, member)) {
        // An Android package is a zip whose first member is the manifest,
        // and the manifest is required to be stored rather than deflated so
        // that it can be read without inflating anything.
        if (member.name == "AndroidManifest.xml" && member.stored) {
            out.format = Format::Apk;
            out.evidence.push_back(
                "the first zip member is AndroidManifest.xml, stored "
                "uncompressed");
            return out;
        }
        if (member.name == "AndroidManifest.xml") {
            out.evidence.push_back(
                "the first zip member is AndroidManifest.xml but it is "
                "deflated; a valid package stores it");
        }
        out.format = Format::Zip;
        out.evidence.push_back("the first zip member is \"" + member.name +
                               "\"");
        return out;
    }

    out.evidence.push_back("no signature matched; the first bytes are " +
                           hex_byte(bytes[0]) + " " + hex_byte(bytes[1]) + " " +
                           hex_byte(bytes[2]) + " " + hex_byte(bytes[3]));
    return out;
}

Detection detect_file(const std::string& path) noexcept {
    auto bytes = fs::read_file_bytes(path);
    if (!bytes) {
        Detection out;
        out.evidence.push_back("cannot read " + path);
        return out;
    }
    Detection out = detect_bytes(ByteSpan{bytes->data(), bytes->size()});
    return out;
}

} // namespace occ::parser
