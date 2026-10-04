// Event stream tests.
//
// The stream is the machine-readable output of a run, so the tests are
// mostly about its shape rather than its content: that each value type
// encodes as JSON of the right kind, that a string containing JSON
// metacharacters cannot break the line, and that a field whose name the
// encoder chose cannot be shadowed by a caller. The cases below are the ones
// where a mistake would produce a stream that parses but says the wrong
// thing, which is worse than one that does not parse.

#include "occ/observer/event.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

using namespace occ;
using occ::obs::Event;
using occ::obs::EventKind;
using occ::obs::Writer;

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

// A writer whose sink is a pipe, so the tests read back the exact bytes a
// consumer would see. Encoding into a string and inspecting that would test
// the encoder but not the line framing, and the framing is half the format.
struct Captured {
    int fd_[2] = {-1, -1};
    Writer writer;

    Captured() {
        if (::pipe(fd_) != 0) {
            std::fprintf(stderr, "pipe failed\n");
            return;
        }
        writer.attach(fd_[1]);
    }

    ~Captured() {
        if (fd_[0] >= 0) {
            ::close(fd_[0]);
        }
        if (fd_[1] >= 0) {
            ::close(fd_[1]);
        }
    }

    std::vector<std::string> lines() {
        // The write end is closed so the read terminates. The writer is
        // discarded afterwards: a closed sink is exactly the state the
        // tests want to observe, not a mistake.
        ::close(fd_[1]);
        fd_[1] = -1;

        std::string all;
        char buffer[4096];
        for (;;) {
            const ssize_t n = ::read(fd_[0], buffer, sizeof(buffer));
            if (n <= 0) {
                break;
            }
            all.append(buffer, static_cast<std::size_t>(n));
        }

        std::vector<std::string> out;
        std::size_t start = 0;
        while (start < all.size()) {
            const std::size_t end = all.find('\n', start);
            if (end == std::string::npos) {
                break;
            }
            out.push_back(all.substr(start, end - start));
            start = end + 1;
        }
        return out;
    }
};

void test_kind_names() {
    check(std::strcmp(obs::event_kind_name(EventKind::SessionStart),
                      "session_start") == 0,
          "the session start name");
    check(std::strcmp(obs::event_kind_name(EventKind::ProcessExit),
                      "process_exit") == 0,
          "the process exit name");
    check(std::strcmp(obs::event_kind_name(EventKind::SyscallBlocked),
                      "syscall_blocked") == 0,
          "the blocked syscall name");
    check(std::strcmp(obs::event_kind_name(EventKind::Note), "note") == 0,
          "the note name");
}

void test_value_types() {
    Captured cap;
    auto& e = cap.writer.begin(EventKind::Note);
    e.add("count", static_cast<std::uint64_t>(42));
    e.add("delta", static_cast<std::int64_t>(-7));
    e.add("text", std::string_view{"hello"});
    e.add("flag", true);
    e.add("other", false);
    e.add_hex("addr", 0x401850);
    cap.writer.commit();

    const auto lines = cap.lines();
    check(lines.size() == 1, "one commit writes one line");
    if (lines.empty()) {
        return;
    }

    const std::string& line = lines[0];
    check(line.front() == '{' && line.back() == '}', "the line is an object");
    check(line.find("\"count\":42") != std::string::npos,
          "an unsigned integer is written as a number");
    check(line.find("\"delta\":-7") != std::string::npos,
          "a negative integer keeps its sign");
    check(line.find("\"text\":\"hello\"") != std::string::npos,
          "a string is quoted");
    check(line.find("\"flag\":true") != std::string::npos,
          "true is a JSON literal, not a string");
    check(line.find("\"other\":false") != std::string::npos,
          "false is a JSON literal, not a string");
    // The address is hexadecimal with the prefix and without padding. The
    // width passed to the writer is the number of nibbles it may use, so
    // the digits are what the value needs and nothing more.
    check(line.find("\"addr\":\"0x401850\"") != std::string::npos,
          "an address is hexadecimal with a prefix");
    check(line.find("0x0000000000401850") == std::string::npos,
          "an address is not padded with leading zeros");
}

void test_pointer_not_confused_with_bool() {
    // A const char* would convert to bool if the string overload did not
    // exist, and the event would carry a truth value where a name belongs.
    Captured cap;
    auto& e = cap.writer.begin(EventKind::SessionStart);
    e.add("name", obs::event_kind_name(EventKind::SessionStart));
    cap.writer.commit();

    const auto lines = cap.lines();
    check(lines.size() == 1, "the line is written");
    if (lines.empty()) {
        return;
    }
    check(lines[0].find("\"name\":\"session_start\"") != std::string::npos,
          "a bare pointer encodes as its string, not as true");
    check(lines[0].find("\"name\":true") == std::string::npos,
          "a bare pointer does not become a truth value");
}

void test_escaped_strings() {
    // A target's path is not under this program's control. A quote or a
    // backslash in it must not end the string early, and a control byte
    // must not be written raw into the line.
    Captured cap;
    auto& e = cap.writer.begin(EventKind::FileOpened);
    e.add("path", std::string_view{"a\"b\\c"});
    e.add("weird", std::string_view{"tab\there"});
    cap.writer.commit();

    const auto lines = cap.lines();
    check(lines.size() == 1, "the escaped line is written");
    if (lines.empty()) {
        return;
    }
    check(lines[0].find("\"path\":\"a\\\"b\\\\c\"") != std::string::npos,
          "a quote and a backslash are escaped");
    check(lines[0].find("\\t") != std::string::npos,
          "a control character is escaped");
    // The raw byte must not appear, or the line would break a consumer that
    // splits on it.
    check(lines[0].find('\t') == std::string::npos,
          "no raw tab is written into the line");
}

void test_begin_resets_fields() {
    // A second event must not inherit the first one's fields. A writer is
    // reused for every event in a run, so this is the difference between a
    // stream that describes one run and one that accumulates.
    Captured cap;
    {
        auto& e = cap.writer.begin(EventKind::Note);
        e.add("first", static_cast<std::uint64_t>(1));
        cap.writer.commit();
    }
    {
        auto& e = cap.writer.begin(EventKind::Note);
        e.add("second", static_cast<std::uint64_t>(2));
        cap.writer.commit();
    }

    const auto lines = cap.lines();
    check(lines.size() == 2, "two commits write two lines");
    if (lines.size() != 2) {
        return;
    }
    check(lines[0].find("\"first\"") != std::string::npos,
          "the first event carries its own field");
    check(lines[0].find("\"second\"") == std::string::npos,
          "the first event does not carry the second's field");
    check(lines[1].find("\"second\"") != std::string::npos,
          "the second event carries its own field");
    check(lines[1].find("\"first\"") == std::string::npos,
          "the second event does not inherit the first's field");
}

void test_every_event_has_kind_and_time() {
    Captured cap;
    {
        auto& e = cap.writer.begin(EventKind::Mapping);
        e.add("index", static_cast<std::uint64_t>(3));
        cap.writer.commit();
    }
    const auto lines = cap.lines();
    check(lines.size() == 1, "the event is written");
    if (lines.empty()) {
        return;
    }
    check(lines[0].find("\"kind\":\"mapping\"") != std::string::npos,
          "every event names its kind");
    check(lines[0].find("\"session\":") != std::string::npos,
          "every event names its session");
    check(lines[0].find("\"t\":") != std::string::npos,
          "every event carries a time");
}

void test_commit_without_begin() {
    // Committing without beginning must write nothing rather than emit an
    // empty object, which a consumer would have to special-case.
    Captured cap;
    cap.writer.commit();
    const auto lines = cap.lines();
    check(lines.empty(), "a commit with no begin writes nothing");
    check(cap.writer.events_written() == 0,
          "a commit with no begin counts nothing");
}

void test_note_shortcut() {
    Captured cap;
    cap.writer.note("something happened");
    const auto lines = cap.lines();
    check(lines.size() == 1, "the note is written");
    if (lines.empty()) {
        return;
    }
    check(lines[0].find("\"kind\":\"note\"") != std::string::npos,
          "the note has the note kind");
    check(lines[0].find("\"text\":\"something happened\"") != std::string::npos,
          "the note carries its text");
}

void test_sink_on_closed_descriptor() {
    // The writer is attached to a closed descriptor when the event stream is
    // not wanted, and every emit has to become a no-op instead of failing.
    Writer writer;
    writer.attach(-1);
    writer.note("this goes nowhere");
    check(writer.events_written() == 0,
          "a write to a closed sink does not count as written");
    check(writer.failed(), "a write to a closed sink is reported as a failure");
}

} // namespace

int main() {
    test_kind_names();
    test_value_types();
    test_pointer_not_confused_with_bool();
    test_escaped_strings();
    test_begin_resets_fields();
    test_every_event_has_kind_and_time();
    test_commit_without_begin();
    test_note_shortcut();
    test_sink_on_closed_descriptor();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
