#include "occ/doctor.h"
#include "occ/util/fs.h"
#include "occ/util/log.h"
#include "occ/util/string.h"

#include <cstdio>
#include <string>
#include <vector>

namespace occ {

namespace {

std::string status_text(CheckStatus s) {
    switch (s) {
    case CheckStatus::Ok:
        return "ok";
    case CheckStatus::Warn:
        return "warn";
    case CheckStatus::Fail:
        return "fail";
    }
    return "?";
}

// Human-readable table. Column widths are computed from content so the
// output stays aligned regardless of probe names.
void print_table(const DoctorReport& report) {
    std::size_t name_w = 0;
    std::size_t status_w = 0;
    for (const auto& c : report.checks) {
        if (c.name.size() > name_w) {
            name_w = c.name.size();
        }
        const auto s = status_text(c.status);
        if (s.size() > status_w) {
            status_w = s.size();
        }
    }

    for (const auto& c : report.checks) {
        const std::string s = status_text(c.status);
        std::string line;
        line.reserve(name_w + status_w + c.detail.size() + 8);
        line += c.name;
        line.append(name_w - c.name.size() + 2, ' ');
        line += s;
        line.append(status_w - s.size() + 2, ' ');
        line += c.detail;
        line += '\n';
        (void)std::fwrite(line.data(), 1, line.size(), stdout);
    }
}

// NDJSON, one object per check. Emitted when stdout is not a terminal.
void print_ndjson(const DoctorReport& report) {
    std::string out;
    for (const auto& c : report.checks) {
        out += "{\"check\":\"";
        append_json_escaped(out, c.name);
        out += "\",\"status\":\"";
        out += status_text(c.status);
        out += "\",\"detail\":\"";
        append_json_escaped(out, c.detail);
        out += "\"}\n";
    }
    (void)std::fwrite(out.data(), 1, out.size(), stdout);
}

} // namespace

int cmd_doctor(bool use_ndjson) {
    const DoctorReport report = run_doctor();

    if (use_ndjson) {
        print_ndjson(report);
    } else {
        print_table(report);
    }

    return report.exit_code();
}

} // namespace occ
