#include "occ/commands.h"
#include "occ/util/log.h"

namespace occ {

int cmd_stop(int argc, char** argv) {
    (void)argv;
    (void)argv;
    if (argc < 1) {
        log::error("stop requires a session id");
        return 2;
    }
    log::error("session handling is not implemented yet");
    return 3;
}

} // namespace occ
