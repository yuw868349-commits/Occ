#include "occ/commands.h"
#include "occ/util/log.h"

namespace occ {

int cmd_check(int argc, char** argv) {
    (void)argv;
    (void)argv;
    if (argc < 1) {
        log::error("check requires a path");
        return 2;
    }
    log::error("format detection is not implemented yet");
    return 3;
}

} // namespace occ
