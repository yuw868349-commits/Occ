#pragma once

namespace occ {

int cmd_check(int argc, char** argv);
int cmd_doctor(bool use_ndjson);
int cmd_run(int argc, char** argv);
int cmd_attach(int argc, char** argv);
int cmd_stop(int argc, char** argv);

} // namespace occ
