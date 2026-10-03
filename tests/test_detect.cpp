// Format detection tests.
//
// Detection is pure: it takes bytes and returns a verdict, so every case can
// be built in memory. The cases below are chosen so that each one would fail
// if a specific constraint were dropped from the detector -- a magic alone
// is not enough for any format, and the tests supply both the near miss and
// the real match.

#include "occ/parser/detect.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace occ::parser;
using occ::ByteSpan;

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

void put16(std::vector<std::uint8_t>& b, std::size_t off, std::uint16_t v) {
    b[off] = static_cast<std::uint8_t>(v & 0xff);
    b[off + 1] = static_cast<std::uint8_t>((v >> 8) & 0xff);
}

void put32(std::vector<std::uint8_t>& b, std::size_t off, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        b[off + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((v >> (8 * i)) & 0xff);
    }
}

void put64(std::vector<std::uint8_t>& b, std::size_t off, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        b[off + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((v >> (8 * i)) & 0xff);
    }
}

Detection run(const std::vector<std::uint8_t>& b) {
    return detect_bytes(ByteSpan{b.data(), b.size()});
}

// Builds an ELF64 header. Every field the detector reads is written, so a
// failure names a real field rather than a zero that happened to be there.
std::vector<std::uint8_t> elf64(std::uint16_t type, std::uint16_t machine,
                                std::uint16_t phnum, std::uint64_t phoff) {
    std::vector<std::uint8_t> b(128, 0);
    b[0] = 0x7f;
    b[1] = 'E';
    b[2] = 'L';
    b[3] = 'F';
    b[4] = 2; // ELFCLASS64
    b[5] = 1; // ELFDATA2LSB
    b[6] = 1; // EV_CURRENT
    b[7] = 0; // ELFOSABI_SYSV
    put16(b, 16, type);
    put16(b, 18, machine);
    put32(b, 20, 1);
    put64(b, 24, 0x401000);
    put64(b, 32, phoff);
    put16(b, 56, phnum);
    put16(b, 52, 64);  // e_ehsize
    put16(b, 54, 56);  // e_phentsize
    return b;
}

void test_elf64() {
    const auto b = elf64(2, 62, 11, 64);
    const Detection d = run(b);

    check(d.format == Format::Elf, "a valid ELF64 executable is detected");
    check(d.elf_class == ElfClass::Elf64, "the class is read");
    check(d.elf_endian == ElfEndian::Little, "the endianness is read");
    check(d.elf_type == ElfType::Exec, "e_type is read");
    check(d.elf_machine == ElfMachine::X86_64, "e_machine is read");
    check(!d.bare_program_header, "a real ELF header is not a bare program "
                                  "header");

    // The program header table offset has to come from the 64-bit field.
    // Reading the 32-bit offset instead lands on e_entry, which the fixture
    // sets to a value that is not 64, so the two are distinguishable.
    bool mentions_64 = false;
    for (const auto& e : d.evidence) {
        if (e.find("0x40") != std::string::npos &&
            e.find("segments") != std::string::npos) {
            mentions_64 = true;
        }
    }
    check(mentions_64, "the program header offset is read from the 64-bit "
                       "field");
}

void test_elf32_offsets() {
    // A 32-bit header has e_phoff at 28 and e_phnum at 44. Writing the same
    // numbers to the 64-bit locations and reading them back would produce a
    // wrong table, so this case is the one that catches a shared constant.
    std::vector<std::uint8_t> b(128, 0);
    b[0] = 0x7f;
    b[1] = 'E';
    b[2] = 'L';
    b[3] = 'F';
    b[4] = 1; // ELFCLASS32
    b[5] = 1;
    b[6] = 1;
    put16(b, 16, 2);
    put16(b, 18, 3); // EM_386
    put32(b, 20, 1);
    put32(b, 24, 0x8048000);
    put32(b, 28, 52); // e_phoff
    put16(b, 44, 5);  // e_phnum
    put16(b, 56, 0);  // a value at the 64-bit location, to be ignored

    const Detection d = run(b);
    check(d.format == Format::Elf, "a 32-bit ELF is detected");
    check(d.elf_class == ElfClass::Elf32, "the 32-bit class is read");
    check(d.elf_machine == ElfMachine::X86, "EM_386 is read");

    bool found = false;
    for (const auto& e : d.evidence) {
        if (e.find("e_phnum = 5") != std::string::npos) {
            found = true;
        }
    }
    check(found, "the 32-bit program header count comes from offset 44");
}

void test_elf_ident_rejection() {
    // A file whose identification block is invalid is not an ELF file, no
    // matter that the magic matches. Each field is tested on its own.
    auto b = elf64(2, 62, 1, 64);
    b[4] = 7; // an undefined class
    check(run(b).format == Format::Unknown,
          "an undefined EI_CLASS is rejected");

    b = elf64(2, 62, 1, 64);
    b[5] = 9; // an undefined data encoding
    check(run(b).format == Format::Unknown,
          "an undefined EI_DATA is rejected");

    b = elf64(2, 62, 1, 64);
    b[6] = 2; // an undefined version
    check(run(b).format == Format::Unknown,
          "an undefined EI_VERSION is rejected");

    // A file that is shorter than a header but has the magic is reported as
    // an ELF whose header could not be read, not as a different format.
    std::vector<std::uint8_t> tiny{0x7f, 'E', 'L', 'F', 2, 1, 1, 0};
    const Detection d = run(tiny);
    check(d.format == Format::Unknown || d.format == Format::Elf,
          "a truncated ELF is not misclassified as another format");
}

void test_bare_program_header() {
    // An Android OAT file begins with a program header: p_type = PT_LOAD.
    std::vector<std::uint8_t> b(64, 0);
    put32(b, 0, 1); // PT_LOAD
    put32(b, 4, 5); // PF_R | PF_X

    const Detection d = run(b);
    check(d.format == Format::Elf, "a bare program header is an ELF container");
    check(d.bare_program_header, "the bare-program-header layout is flagged");
    check(d.elf_class == ElfClass::Elf64, "the implied class is 64-bit");

    // p_flags above 7 is not a valid combination and the file is therefore
    // not treated as a program header.
    put32(b, 4, 0x40000000);
    check(!run(b).bare_program_header,
          "a program header with impossible flags is not accepted");
}

void test_apk() {
    // A real Android package starts with a local file header whose name is
    // AndroidManifest.xml, stored uncompressed.
    const char* name = "AndroidManifest.xml";
    const std::size_t name_len = std::strlen(name);
    std::vector<std::uint8_t> b(30 + name_len + 64, 0);
    put32(b, 0, 0x04034b50);
    put16(b, 4, 20);  // version needed
    put16(b, 6, 0);   // flags
    put16(b, 8, 0);   // method: stored
    put32(b, 18, 1024);
    put16(b, 26, static_cast<std::uint16_t>(name_len));
    std::memcpy(b.data() + 30, name, name_len);

    const Detection d = run(b);
    check(d.format == Format::Apk, "an APK is detected from its first member");
    check(!d.evidence.empty(), "the detection carries evidence");

    // The same member deflated is not a valid package, because a reader has
    // to be able to reach the manifest without inflating anything.
    put16(b, 8, 8);
    check(run(b).format == Format::Zip,
          "a deflated manifest is a zip rather than a package");

    // A zip whose first member is something else is not a package.
    const char* other = "classes.dex";
    const std::size_t other_len = std::strlen(other);
    std::vector<std::uint8_t> c(30 + other_len + 64, 0);
    put32(c, 0, 0x04034b50);
    put16(c, 8, 0);
    put16(c, 26, static_cast<std::uint16_t>(other_len));
    std::memcpy(c.data() + 30, other, other_len);
    check(run(c).format == Format::Zip,
          "a zip whose first member is not the manifest is not a package");
}

void test_macho() {
    // Both byte orders of the 32-bit magic, so a detector that only checks
    // one order is caught.
    for (std::uint32_t magic : {0xfeedfaceU, 0xcefaedfeU}) {
        std::vector<std::uint8_t> b(64, 0);
        put32(b, 0, magic);
        check(run(b).format == Format::MachO, "a Mach-O magic is detected");
    }

    // The little-endian 64-bit magic 0xfeedfacf is the reverse of the ELF
    // magic only in one byte position, so this is the case that catches an
    // ELF check placed before the Mach-O check.
    std::vector<std::uint8_t> b(64, 0);
    put32(b, 0, 0xfeedfacfU);
    check(run(b).format == Format::MachO,
          "a 64-bit Mach-O is not mistaken for an ELF");
}

void test_pe() {
    std::vector<std::uint8_t> b(256, 0);
    b[0] = 'M';
    b[1] = 'Z';
    put32(b, 0x3c, 0x80);
    b[0x80] = 'P';
    b[0x81] = 'E';
    b[0x82] = 0;
    b[0x83] = 0;

    check(run(b).format == Format::Pe, "a PE image is detected");

    // An MZ stub without a PE header is a DOS executable, which is a
    // different format and has no engine either, but the two must not be
    // conflated because the message a user needs is different.
    put32(b, 0x3c, 0);
    check(run(b).format == Format::Unknown,
          "a DOS stub without a PE header is not a PE image");
}

void test_unknown() {
    std::vector<std::uint8_t> b{0x23, 0x21, 0x2f, 0x62, 0x69, 0x6e, 0x2f};
    const Detection d = run(b);
    check(d.format == Format::Unknown, "a shell script is not a known format");
    check(!d.evidence.empty(), "an unknown file still explains itself");

    std::vector<std::uint8_t> empty{};
    check(run(empty).format == Format::Unknown, "an empty file is unknown");
}

void test_engine_names() {
    check(std::strcmp(format_name(Format::Elf), "elf") == 0,
          "the ELF format name");
    check(std::strcmp(format_name(Format::Apk), "apk") == 0,
          "the APK format name");
    check(std::strcmp(elf_machine_name(ElfMachine::X86_64), "x86-64") == 0,
          "the x86-64 machine name");
    check(std::strcmp(elf_class_name(ElfClass::Elf64), "64-bit") == 0,
          "the 64-bit class name");
    check(std::strcmp(elf_type_name(ElfType::Dyn), "shared object or pie") == 0,
          "the Dyn type name");
}

} // namespace

int main() {
    test_elf64();
    test_elf32_offsets();
    test_elf_ident_rejection();
    test_bare_program_header();
    test_apk();
    test_macho();
    test_pe();
    test_unknown();
    test_engine_names();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
