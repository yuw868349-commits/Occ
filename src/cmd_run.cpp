#include "occ/commands.h"
#include "occ/util/log.h"

namespace occ {

int cmd_run(int argc, char** argv) {
    (void)argv;
    (void)argv;
    if (argc < 1) {
        log::error("run requires a path");
        return 2;
    }
    log::error("the run path is not implemented yet");
    return 3;
}

} // namespace occ
