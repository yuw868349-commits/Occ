// ELF image reader tests.
//
// The reader takes bytes and returns a description of the mappings the
// image asks for, so every case can be assembled in memory. The fixtures
// write every field the reader touches, which means a failure points at a
// field rather than at a zero that happened to be in the buffer. The cases
// are the ones that would pass if a constraint were missing: a header whose
// program table runs off the end, a segment that is not congruent to its
// alignment, a segment whose memory size is smaller than its file size.

#include "occ/parser/elf.h"

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

// The 64-bit header. The program header table starts right after it at 64,
// which is where a real static binary puts it.
constexpr std::size_t kHeaderSize = 64;
constexpr std::size_t kPhdrSize = 56;

void put_ident(std::vector<std::uint8_t>& b, std::uint8_t cls, std::uint8_t data,
               std::uint8_t version) {
    b[0] = 0x7f;
    b[1] = 'E';
    b[2] = 'L';
    b[3] = 'F';
    b[4] = cls;
    b[5] = data;
    b[6] = version;
}

// Writes one 64-bit program header. `p_flags` comes before the offsets in
// the 64-bit layout, which is the detail a reader that mirrored the order of
// the names would get wrong.
void put_phdr(std::vector<std::uint8_t>& b, std::size_t index, std::uint32_t type,
              std::uint32_t flags, std::uint64_t offset, std::uint64_t vaddr,
              std::uint64_t filesz, std::uint64_t memsz, std::uint64_t align) {
    const std::size_t base = kHeaderSize + index * kPhdrSize;
    put32(b, base + 0, type);
    put32(b, base + 4, flags);
    put64(b, base + 8, offset);
    put64(b, base + 16, vaddr);
    put64(b, base + 24, 0); // p_paddr, unused on x86-64
    put64(b, base + 32, filesz);
    put64(b, base + 40, memsz);
    put64(b, base + 48, align);
}

// One PT_LOAD segment plus the file contents it points at. The segment
// starts at file offset 4096 with vaddr 0x400000 so that the offsets and the
// addresses are congruent modulo the 4096 alignment, which is the rule the
// kernel's loader relies on.
struct Image {
    std::vector<std::uint8_t> bytes;
    std::size_t payload_off = 4096;
};

Image base_image(std::uint16_t type = 2, std::uint16_t machine = 62,
                 std::uint16_t phnum = 1, std::uint64_t entry = 0x401000) {
    Image img;
    // Three pages, so a fixture can place a segment both at file offset 4096
    // and at file offset 8192. The congruence rule ties p_offset to p_vaddr
    // modulo p_align, and dividing the offsets into separate pages is what
    // lets a two-segment fixture keep two distinct addresses without
    // breaking the rule or overlapping the payloads.
    img.bytes.assign(12288, 0);
    put_ident(img.bytes, 2, 1, 1);
    put16(img.bytes, 16, type);
    put16(img.bytes, 18, machine);
    put32(img.bytes, 20, 1);
    put64(img.bytes, 24, entry);
    put64(img.bytes, 32, kHeaderSize);
    put16(img.bytes, 52, 64);          // e_ehsize
    put16(img.bytes, 54, kPhdrSize);   // e_phentsize
    put16(img.bytes, 56, phnum);       // e_phnum
    return img;
}

ElfImage parse(const Image& img) {
    return ElfImage::parse(ByteSpan{img.bytes.data(), img.bytes.size()});
}

// Returns the mappings of an image that is expected to have parsed. A
// fixture that fails to parse on its own returns nothing here rather than
// reaching into an empty vector, so a wrong fixture reports a failed check
// instead of taking the whole suite down with it.
const std::vector<DesiredMapping>& mappings_of(const ElfImage& e) {
    static const std::vector<DesiredMapping> empty;
    return e.ok() ? e.mappings() : empty;
}

void test_exec_image() {
    Image img = base_image(2, 62, 2, 0x401000);
    // The second segment lives at file offset 8192 and address 0x401000.
    // Both are congruent to 0 modulo the 4096 alignment, which the
    // congruence rule demands, so putting the payload anywhere else would
    // make this fixture a test of the rule instead of a test of the reader.
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
    put_phdr(img.bytes, 1, 1, 6, 8192, 0x401000, 128, 256, 4096);

    const ElfImage e = parse(img);
    check(e.ok(), "a two-segment ET_EXEC parses");
    check(e.type() == 2, "e_type is ET_EXEC");
    check(e.machine() == 62, "e_machine is EM_X86_64");
    check(e.entry() == 0x401000, "e_entry is read");
    check(e.phoff() == kHeaderSize,
          "e_phoff comes from offset 32, not the 32-bit offset 28");
    check(e.phnum() == 2, "e_phnum comes from offset 56, not 44");
    check(e.phentsize() == kPhdrSize, "e_phentsize is read");
    check(e.headers().size() == 2, "both headers are kept");
    check(mappings_of(e).size() == 2, "both PT_LOAD headers become mappings");

    // The flags sit before the offsets in the 64-bit layout. A reader that
    // read them from where the name order suggests would pick up the low
    // half of p_offset, which is 4096 here.
    check(e.headers()[0].flags == 5, "p_flags is read from offset 4");
    check(e.headers()[0].readable() && e.headers()[0].executable(),
          "R and X are set on the code segment");
    check(!e.headers()[0].writable(), "W is clear on the code segment");
    check(e.headers()[1].writable(), "W is set on the data segment");

    check(e.headers()[0].offset == 4096, "p_offset is read");
    check(e.headers()[0].vaddr == 0x400000, "p_vaddr is read");
    check(e.headers()[0].filesz == 512, "p_filesz is read");
    check(e.headers()[0].memsz == 512, "p_memsz is read");
    check(e.headers()[0].align == 4096, "p_align is read");

    check(e.lowest_vaddr() == 0x400000, "the lowest address is the first "
                                        "segment");
    // The second segment occupies 0x401000 for 256 bytes, so its end rounds
    // up to 0x402000.
    check(e.highest_vaddr() == 0x402000, "the highest address is rounded to "
                                         "a page");
    check(!e.has_interpreter(), "a static image has no interpreter");
    check(!e.has_gnu_relro(), "no PT_GNU_RELRO in this fixture");
    check(!e.wants_executable_stack(), "no PT_GNU_STACK in this fixture");
}

void test_bss_tail() {
    // p_memsz > p_filesz is the .bss tail. It has to be reported as a
    // separate length, because the loader has to zero it rather than copy it
    // from the file.
    Image img = base_image(2, 62, 1);
    put_phdr(img.bytes, 0, 1, 6, 4096, 0x400000, 1024, 8192, 4096);

    const ElfImage e = parse(img);
    check(e.ok(), "a segment with a bss tail parses");
    check(e.headers()[0].zero_tail() == 7168,
          "the zero tail is memsz minus filesz");
    check(mappings_of(e)[0].memsz == 8192, "the mapping keeps p_memsz");
    check(mappings_of(e)[0].filesz == 1024, "the mapping keeps p_filesz");

    // The tail spans pages, so the mapping reaches past one page boundary.
    check(page_ceil(0x400000 + 8192) == 0x402000,
          "the tail spans two pages past the file-backed part");
}

void test_no_zero_tail() {
    Image img = base_image(2, 62, 1);
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 4096, 4096, 4096);
    const ElfImage e = parse(img);
    check(e.ok(), "an equal-size segment parses");
    check(e.headers()[0].zero_tail() == 0, "there is no tail when the sizes "
                                          "are equal");
}

void test_memsz_smaller_than_filesz() {
    // A segment that claims fewer bytes in memory than it has in the file
    // cannot be mapped: the kernel would have to drop part of what it was
    // told to load.
    Image img = base_image(2, 62, 1);
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 4096, 1024, 4096);
    const ElfImage e = parse(img);
    check(!e.ok(), "p_memsz < p_filesz is rejected");
    check(e.error() == LoadError::SegmentOutOfFile,
          "the rejection names the segment");
}

void test_segment_out_of_file() {
    Image img = base_image(2, 62, 1);
    // The payload starts on the last page, so 8192 bytes of it run past the
    // end of the file.
    put_phdr(img.bytes, 0, 1, 5, 8192, 0x400000, 8192, 8192, 4096);
    const ElfImage e = parse(img);
    check(!e.ok(), "a segment past the end of the file is rejected");
    check(e.error() == LoadError::SegmentOutOfFile,
          "the rejection names the file overrun");
}

void test_bad_alignment() {
    Image img = base_image(2, 62, 1);
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 3000);
    const ElfImage e = parse(img);
    check(!e.ok(), "a non-power-of-two p_align is rejected");
    check(e.error() == LoadError::BadAlignment, "the rejection names the "
                                                "alignment");
}

void test_congruence_rule() {
    // offset 4096 and vaddr 0x400800 are both congruent to 0 modulo 4096,
    // but not to each other. The kernel maps from the page the offset rounds
    // to, so a segment that breaks the rule would be mapped with the wrong
    // bytes at its start.
    Image img = base_image(2, 62, 1);
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400800, 512, 512, 4096);
    const ElfImage e = parse(img);
    check(!e.ok(), "a segment violating offset = vaddr (mod align) is "
                   "rejected");
    check(e.error() == LoadError::BadAlignment,
          "the rejection names the congruence rule");
}

void test_align_of_one_is_allowed() {
    // p_align 1 means no alignment requirement, and the congruence rule does
    // not apply to it.
    Image img = base_image(2, 62, 1);
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400001, 512, 512, 1);
    const ElfImage e = parse(img);
    check(e.ok(), "p_align 1 imposes no congruence requirement");
}

void test_class_and_endian_rejection() {
    {
        Image img = base_image();
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        img.bytes[4] = 1; // ELFCLASS32
        const ElfImage e = parse(img);
        check(e.error() == LoadError::UnsupportedClass,
              "a 32-bit object is reported as the wrong class");
    }
    {
        Image img = base_image();
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        img.bytes[5] = 2; // ELFDATA2MSB
        const ElfImage e = parse(img);
        check(e.error() == LoadError::UnsupportedEndian,
              "a big-endian object is reported as the wrong endianness");
    }
    {
        Image img = base_image();
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        img.bytes[6] = 7;
        const ElfImage e = parse(img);
        check(e.error() == LoadError::NotElf,
              "an undefined EI_VERSION is not an ELF header");
    }
}

void test_machine_rejection() {
    Image img = base_image(2, 183); // EM_AARCH64
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
    const ElfImage e = parse(img);
    check(e.error() == LoadError::UnsupportedMachine,
          "a non-x86-64 object is reported as the wrong machine");
    check(e.error_detail().find("183") != std::string::npos,
          "the detail names the machine the file declares");
}

void test_type_rejection() {
    {
        Image img = base_image(1); // ET_REL
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        const ElfImage e = parse(img);
        check(e.error() == LoadError::NoLoadSegments,
              "an ET_REL object is not a program");
    }
    {
        Image img = base_image(4); // ET_CORE
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        const ElfImage e = parse(img);
        check(e.error() == LoadError::NoLoadSegments,
              "an ET_CORE file is not a program");
    }
    {
        Image img = base_image(7);
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        const ElfImage e = parse(img);
        check(e.error() == LoadError::NoLoadSegments,
              "an undefined e_type is not a program");
    }
}

void test_no_load_segments() {
    // A file whose only program header is a PT_NOTE has nothing to map.
    Image img = base_image(2, 62, 1);
    put_phdr(img.bytes, 0, 4, 4, 4096, 0x400000, 512, 512, 4096);
    const ElfImage e = parse(img);
    check(e.error() == LoadError::NoLoadSegments,
          "a file with no PT_LOAD header is rejected");
}

void test_truncated_header() {
    // The magic alone is not a header. A file cut off after the
    // identification block has to be reported as truncated rather than
    // parsed with zeros for the fields that were never read.
    std::vector<std::uint8_t> b(32, 0);
    put_ident(b, 2, 1, 1);
    const ElfImage e = ElfImage::parse(ByteSpan{b.data(), b.size()});
    check(e.error() == LoadError::TruncatedHeader,
          "a file cut off inside the header is rejected");
}

void test_truncated_table() {
    {
        // The table starts before the end of the header, which cannot
        // happen in a well-formed file.
        Image img = base_image(2, 62, 1);
        put64(img.bytes, 32, 8);
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        const ElfImage e = parse(img);
        check(e.error() == LoadError::TruncatedProgramHeaders,
              "a table overlapping the header is rejected");
    }
    {
        // Two entries declared but only room for one.
        Image img = base_image(2, 62, 2);
        img.bytes.resize(kHeaderSize + kPhdrSize);
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        const ElfImage e = parse(img);
        check(e.error() == LoadError::TruncatedProgramHeaders,
              "a table running past the end of the file is rejected");
    }
    {
        // A different entry size means a different stride, and walking the
        // table with the wrong stride reads the wrong bytes.
        Image img = base_image(2, 62, 1);
        put16(img.bytes, 54, 48);
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        const ElfImage e = parse(img);
        check(e.error() == LoadError::TruncatedProgramHeaders,
              "an unexpected e_phentsize is rejected");
    }
}

void test_interpreter() {
    // PT_INTERP holds a NUL-terminated path, and the terminator counts
    // toward p_filesz.
    Image img = base_image(3, 62, 2); // ET_DYN
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
    const char* interp = "/lib64/ld-linux-x86-64.so.2";
    const std::size_t len = std::strlen(interp) + 1;
    put_phdr(img.bytes, 1, 3, 4, 4096 + 512, 0x400200,
             static_cast<std::uint64_t>(len),
             static_cast<std::uint64_t>(len), 1);
    std::memcpy(img.bytes.data() + 4096 + 512, interp, len);

    const ElfImage e = parse(img);
    check(e.ok(), "an image with an interpreter parses");
    check(e.has_interpreter(), "the interpreter is noticed");
    check(e.interpreter() == interp,
          "the interpreter path is read up to the terminator");
    check(e.interpreter().size() == std::strlen(interp),
          "the terminator is not part of the path");
    check(e.type() == 3, "the ET_DYN type is kept");
}

void test_interp_out_of_file() {
    Image img = base_image(3, 62, 2);
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
    // PT_INTERP near the end of the file for more bytes than remain. The
    // reader copies the path out of the file rather than reporting an empty
    // string, so a truncated one has to be caught here.
    put_phdr(img.bytes, 1, 3, 4, 12224, 0x400200, 128, 128, 1);
    const ElfImage e = parse(img);
    check(e.error() == LoadError::SegmentOutOfFile,
          "a PT_INTERP past the end of the file is rejected");
}

void test_gnu_stack() {
    {
        Image img = base_image(2, 62, 2);
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        put_phdr(img.bytes, 1, 0x6474e551, 6, 0, 0, 0, 0x200000, 16);
        const ElfImage e = parse(img);
        check(e.ok(), "an image with PT_GNU_STACK parses");
        check(e.stack().present, "PT_GNU_STACK is noticed");
        check(!e.wants_executable_stack(),
              "a non-executable stack request is not a request for an "
              "executable one");
        check(e.stack().size == 0x200000, "the stack size request is kept");
    }
    {
        // p_flags with X set is the request for an executable stack, which
        // is a fact the caller has to decide about rather than one to
        // honour silently.
        Image img = base_image(2, 62, 2);
        put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
        put_phdr(img.bytes, 1, 0x6474e551, 7, 0, 0, 0, 0x200000, 16);
        const ElfImage e = parse(img);
        check(e.wants_executable_stack(),
              "an executable stack request is reported");
    }
}

void test_gnu_relro() {
    Image img = base_image(2, 62, 2);
    put_phdr(img.bytes, 0, 1, 5, 4096, 0x400000, 512, 512, 4096);
    put_phdr(img.bytes, 1, 0x6474e552, 4, 4096, 0x400000, 512, 512, 1);
    const ElfImage e = parse(img);
    check(e.ok(), "an image with PT_GNU_RELRO parses");
    check(e.has_gnu_relro(), "PT_GNU_RELRO is noticed");
    check(mappings_of(e).size() == 1,
          "a PT_GNU_RELRO header is not a mapping");
}

void test_sorting() {
    // The loader sorts the segments by address because a segment that
    // overlaps another has to be placed in address order for the later
    // permissions to be the ones that apply.
    Image img = base_image(2, 62, 2);
    put_phdr(img.bytes, 0, 1, 6, 8192, 0x401000, 128, 128, 4096);
    put_phdr(img.bytes, 1, 1, 5, 4096, 0x400000, 512, 512, 4096);

    const ElfImage e = parse(img);
    check(e.ok(), "segments in descending order parse");
    check(mappings_of(e)[0].vaddr == 0x400000,
          "the lowest address comes first after sorting");
    check(mappings_of(e)[1].vaddr == 0x401000, "the higher address follows");
    check(e.lowest_vaddr() == 0x400000,
          "the lowest address is taken after sorting, not from the first "
          "header read");
}

void test_page_helpers() {
    check(page_floor(0x400123) == 0x400000, "page_floor rounds down");
    check(page_ceil(0x400123) == 0x401000, "page_ceil rounds up");
    check(page_floor(0x400000) == 0x400000,
          "page_floor leaves an aligned value alone");
    check(page_ceil(0x400000) == 0x400000,
          "page_ceil leaves an aligned value alone");
    check(page_ceil(1) == kPageSize, "page_ceil rounds a small value up to "
                                     "one page");
}

void test_error_names() {
    check(std::strcmp(load_error_name(LoadError::None), "none") == 0,
          "the success name");
    check(std::strcmp(load_error_name(LoadError::BadAlignment),
                      "a segment has an alignment that is not a power of two") == 0,
          "the alignment name");
    check(std::strcmp(load_error_name(LoadError::SegmentOutOfFile),
                      "a segment extends past the end of the file") == 0,
          "the out-of-file name");
}

void test_truncated_buffer_is_not_elF() {
    std::vector<std::uint8_t> b(3, 0);
    const ElfImage e = ElfImage::parse(ByteSpan{b.data(), b.size()});
    check(!e.ok(), "three bytes are not an ELF file");
    check(e.error() == LoadError::NotElf, "a short buffer is not an ELF "
                                          "header");
}

} // namespace

int main() {
    test_exec_image();
    test_bss_tail();
    test_no_zero_tail();
    test_memsz_smaller_than_filesz();
    test_segment_out_of_file();
    test_bad_alignment();
    test_congruence_rule();
    test_align_of_one_is_allowed();
    test_class_and_endian_rejection();
    test_machine_rejection();
    test_type_rejection();
    test_no_load_segments();
    test_truncated_header();
    test_truncated_table();
    test_interpreter();
    test_interp_out_of_file();
    test_gnu_stack();
    test_gnu_relro();
    test_sorting();
    test_page_helpers();
    test_error_names();
    test_truncated_buffer_is_not_elF();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
