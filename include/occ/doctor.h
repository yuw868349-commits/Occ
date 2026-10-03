#pragma once

// Host capability probe.
//
// occ doctor answers one question: what will work on this machine, and what
// will not. It reports facts and does not decide policy. A caller that
// learns cgroup v2 is not writable can choose to proceed without limits;
// what it must not have to do is discover that at the moment of failure.

#include <string>
#include <vector>

namespace occ {

enum class CheckStatus {
    Ok,
    Warn,
    Fail,
};

struct Check {
    std::string name;
    CheckStatus status;
    std::string detail;
};

struct DoctorReport {
    std::vector<Check> checks;

    [[nodiscard]] bool has_failure() const noexcept;
    [[nodiscard]] int exit_code() const noexcept;
};

// Runs every probe. Each probe is independent: a failure in one does not
// skip the rest, because the report is more useful complete than short.
[[nodiscard]] DoctorReport run_doctor() noexcept;

} // namespace occ
