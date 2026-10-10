#include "../tools/crossthink/crossthink-protocol.h"

#include <cstdio>
#include <stdexcept>

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        for (const char * text : {
                "<ct:peek/>", "<ct:inbox/>", "<ct:send>My plan</ct:send>",
                "\r\n<ct:inbox/>\r\n", " \t\n<ct:peek/> \t\r\n",
                "<ct:send>First line\nSecond line</ct:send>\n<ct:inbox/>",
                "<ct:send>\n<ct:inbox/>\n```c\n</ct:send>\n<ct:peek/>",
                "<ct:peek/> \t\r\n\t \r\n<ct:send></ct:send>\n<ct:inbox/>"}) {
            check(crossthink_misplaced_thought_commands(text), "complete command-only answer not recognized");
        }
        for (const char * text : {
                "", " \t\r\n", "The answer is 42.",
                "I'll check <ct:inbox/>.", "<ct:inbox/> means check the inbox.",
                "<ct:send>Plan</ct:send>\nNow continue the task.",
                "First, send this:\n<ct:send>Plan</ct:send>",
                "`<ct:inbox/>`", "\"<ct:inbox/>\"", "> <ct:inbox/>",
                "```xml\n<ct:inbox/>\n```", "~~~\n<ct:send>Plan</ct:send>\n~~~",
                "    <ct:inbox/>", "\t<ct:inbox/>", "\n <ct:inbox/>",
                "<ct:peek/>\n <ct:inbox/>", "<ct:peek/> <ct:inbox/>",
                "<ct:peek/><ct:inbox/>", "<ct:peek/>\n```\n<ct:inbox/>\n```",
                "<ct:inbox>", "<ct:peek />", "<ct:send/>",
                "<ct:send>Plan", "<ct:send>Plan<ct:send>",
                "<ct:inbox/>\n<ct:send>Plan", "<ct:send>Plan</ct:send>extra",
                "<ct:send>Plan</ct:send></ct:send>"}) {
            check(!crossthink_misplaced_thought_commands(text), "ordinary or malformed answer misrecognized");
        }
        const std::string open = "<ct:send>";
        const std::string close = "</ct:send>";
        const auto max_payload = server_thought_scanner::max_payload;
        check(crossthink_misplaced_thought_commands(open + std::string(max_payload, 'x') + close),
            "maximum text payload rejected");
        check(!crossthink_misplaced_thought_commands(open + std::string(max_payload + 1, 'x') + close),
            "oversized text payload recognized");
        check(crossthink_misplaced_thought_commands(open + std::string(max_payload, '"') + close),
            "maximum escaped payload rejected");
        const auto controls = server_thought_scanner::max_escaped_payload / 6;
        check(crossthink_misplaced_thought_commands(open + std::string(controls, '\x01') + close),
            "bounded control-byte payload rejected");
        check(!crossthink_misplaced_thought_commands(open + std::string(controls + 1, '\x01') + close),
            "oversized escaped payload recognized");
        check(!crossthink_misplaced_thought_commands("<ct:peek/>\n" + open +
            std::string(max_payload + 1, 'x') + close), "valid prefix hid oversized later command");
        std::string commands;
        for (size_t count = 0; count < 32; ++count) {
            commands += "<ct:inbox/>\n";
        }
        check(crossthink_misplaced_thought_commands(commands), "bounded command sequence rejected");
        check(!crossthink_misplaced_thought_commands(commands + "<ct:peek/>"),
            "oversized command sequence recognized");
        std::puts("crossthink protocol tests: passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "crossthink protocol test: %s\n", error.what());
        return 1;
    }
}
