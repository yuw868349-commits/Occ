#include "occ/commands.h"

#include "occ/parser/detect.h"
#include "occ/util/log.h"
#include "occ/util/string.h"

#include <cstdio>
#include <string>
#include <string_view>

#include <unistd.h>

namespace occ {

namespace {

using parser::Detection;
using parser::ElfClass;
using parser::ElfEndian;
using parser::ElfMachine;
using parser::Format;

struct Cli {
    std::string path;
    bool use_ndjson = false;
};

// A deliberately small parser. There is no getopt here because the grammar
// is three tokens wide and the standard library's version brings in global
// state and an error convention that would have to be documented.
bool parse(const int argc, char** argv, Cli& out) {
    for (int i = 0; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--json") {
            out.use_ndjson = true;
        } else if (a == "--help" || a == "-h") {
            std::fprintf(
                stderr,
                "usage: occ check <path> [--json]\n"
                "\n"
                "Reports the format of a target and the engine occ would use "
                "for it.\n"
                "The exit status is 0 when the format is one an engine "
                "handles, 3 when it\n"
                "is recognised but not handled, and 4 when it is not "
                "recognised at all.\n");
            return false;
        } else if (!a.empty() && a[0] == '-') {
            log::error("unknown option");
            return false;
        } else if (out.path.empty()) {
            out.path = a;
        } else {
            log::error("check takes one path");
            return false;
        }
    }
    if (out.path.empty()) {
        log::error("check requires a path");
        return false;
    }
    return true;
}

// Which engine would run this. Every ELF goes to the exe engine, including
// the ones it does not handle: an engine that receives a target it cannot
// run produces a named refusal, and routing such a target nowhere would
// turn a specific error into a generic one.
const char* engine_for(const Detection& d) noexcept {
    switch (d.format) {
    case Format::Elf:
        return "exe";
    case Format::Apk:
        return "apk";
    case Format::Zip:
    case Format::MachO:
    case Format::Pe:
    case Format::Unknown:
        return nullptr;
    }
    return nullptr;
}

// The reason a format is not runnable, in one line. Kept separate from the
// engine decision because a caller reading the exit status needs to know
// which of "no engine exists" and "the engine refuses this file" applies.
std::string refusal(const Detection& d) {
    switch (d.format) {
    case Format::Unknown:
        return "no signature matched";
    case Format::Zip:
        return "a zip archive of code has no engine; point occ at a package "
               "or extract the dex";
    case Format::MachO:
        return "Mach-O images are not handled; occ runs Linux targets";
    case Format::Pe:
        return "PE images are not handled; occ runs Linux targets";
    case Format::Apk:
        return {};
    case Format::Elf:
        if (d.bare_program_header) {
            return "the file is an Android OAT image, which the apk engine "
                   "reads rather than the exe engine";
        }
        if (d.elf_class == ElfClass::Elf32) {
            return "32-bit ELF is not handled by the exe engine";
        }
        if (d.elf_endian == ElfEndian::Big) {
            return "big-endian ELF is not handled by the exe engine";
        }
        if (d.elf_machine != ElfMachine::X86_64) {
            return std::string("e_machine is ") +
                   parser::elf_machine_name(d.elf_machine) +
                   "; the exe engine handles x86-64 only";
        }
        if (d.elf_type == parser::ElfType::Rel) {
            return "a relocatable object is not a program; link it first";
        }
        if (d.elf_type == parser::ElfType::Core) {
            return "a core file is not a program";
        }
        return {};
    }
    return {};
}

int exit_status_for(const Detection& d) noexcept {
    if (d.format == Format::Unknown) {
        return 4;
    }
    if (engine_for(d) == nullptr || !refusal(d).empty()) {
        return 3;
    }
    return 0;
}

void print_text(const std::string& path, const Detection& d) {
    std::string out;
    out += path;
    out += ": ";
    out += parser::format_name(d.format);
    out += "\n";

    out += "  size         ";
    append_uint(out, d.file_size);
    out += "\n";

    if (d.format == Format::Elf) {
        if (d.bare_program_header) {
            out += "  layout       bare program header (Android OAT)\n";
        } else {
            out += "  class        ";
            out += parser::elf_class_name(d.elf_class);
            out += "\n";
            out += "  endianness   ";
            out += parser::elf_endian_name(d.elf_endian);
            out += "\n";
            out += "  type         ";
            out += parser::elf_type_name(d.elf_type);
            out += "\n";
            out += "  machine      ";
            out += parser::elf_machine_name(d.elf_machine);
            out += "\n";
        }
    }

    const char* engine = engine_for(d);
    out += "  engine       ";
    out += engine != nullptr ? engine : "(none)";
    out += "\n";

    const std::string why = refusal(d);
    if (!why.empty()) {
        out += "  reason       ";
        out += why;
        out += "\n";
    }

    if (!d.evidence.empty()) {
        out += "  evidence\n";
        for (const auto& e : d.evidence) {
            out += "    - ";
            out += e;
            out += "\n";
        }
    }

    (void)std::fwrite(out.data(), 1, out.size(), stdout);
}

void print_ndjson(const std::string& path, const Detection& d) {
    std::string out;
    out += "{\"path\":\"";
    append_json_escaped(out, path);
    out += "\",\"format\":\"";
    out += parser::format_name(d.format);
    out += "\",\"size\":";
    append_uint(out, d.file_size);

    if (d.format == Format::Elf) {
        out += ",\"elf_class\":\"";
        out += parser::elf_class_name(d.elf_class);
        out += "\",\"elf_endian\":\"";
        out += parser::elf_endian_name(d.elf_endian);
        out += "\",\"elf_type\":\"";
        out += parser::elf_type_name(d.elf_type);
        out += "\",\"elf_machine\":\"";
        out += parser::elf_machine_name(d.elf_machine);
        out += "\"";
    }

    const char* engine = engine_for(d);
    out += ",\"engine\":";
    if (engine == nullptr) {
        out += "null";
    } else {
        out += "\"";
        out += engine;
        out += "\"";
    }

    out += ",\"reason\":\"";
    append_json_escaped(out, refusal(d));
    out += "\"";

    out += ",\"evidence\":[";
    for (std::size_t i = 0; i < d.evidence.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += "\"";
        append_json_escaped(out, d.evidence[i]);
        out += "\"";
    }
    out += "]}\n";

    (void)std::fwrite(out.data(), 1, out.size(), stdout);
}

} // namespace

int cmd_check(int argc, char** argv) {
    Cli cli;
    if (!parse(argc, argv, cli)) {
        return 2;
    }

    // The default follows the same rule as the other commands: NDJSON when
    // stdout is not a terminal, the readable form when it is.
    if (!cli.use_ndjson) {
        cli.use_ndjson = ::isatty(1) == 0;
    }

    const Detection d = parser::detect_file(cli.path);

    if (cli.use_ndjson) {
        print_ndjson(cli.path, d);
    } else {
        print_text(cli.path, d);
    }

    return exit_status_for(d);
}

} // namespace occ
