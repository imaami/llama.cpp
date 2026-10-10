#include "../tools/crossthink/crossthink-protocol.h"

#include <cstdio>
#include <stdexcept>

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_command_repair() {
    crossthink_repaired_command result;
    for (const char * name : {"peek", "inbox"}) {
        const std::string command = name;
        for (const std::string & text : {
                "<ct:" + command + "/>",
                "\r\n<ct:" + command + "/> \t\n",
                "<tool_call><function=ct:" + command + "></function></tool_call>",
                "<tool_call>\n<function=ct:" + command + ">\n</function>\n</tool_call>"}) {
            check(crossthink_repair_thought_command(text, result), "complete no-argument command not repaired");
            check(result.command == command && result.payload.empty(), "no-argument command changed");
        }
    }
    for (const std::string & payload : {
            std::string("Plan"), std::string("\n  Plan\r\n\t"),
            std::string("a &lt; b && a > 0; \"quotes\" \\ backslash\n```c\n#include <stdio.h>\n```"),
            std::string(server_thought_scanner::max_payload, 'x'),
            std::string(server_thought_scanner::max_payload, '"'),
            std::string(server_thought_scanner::max_escaped_payload / 6, '\x01')}) {
        for (const std::string & text : {
                "<ct:send>" + payload + "</ct:send>",
                "\n<ct:send>" + payload + "</ct:send>\r\n",
                "<tool_call>\n<function=ct:send>" + payload + "</ct:send>",
                "<tool_call><function=ct:send>" + payload + "</ct:send></function></tool_call>",
                "<tool_call>\n<function=ct:send>\n<parameter=message>" + payload +
                    "</parameter>\n</function>\n</tool_call>"}) {
            check(crossthink_repair_thought_command(text, result), "complete send command not repaired");
            check(result.command == "send" && result.payload == payload, "send repair changed payload bytes");
        }
    }
    // This observed hybrid has its own ct closer but no native outer closers.
    const std::string hybrid = "<tool_call>\n<function=ct:send>\nPlanning the Mandelbrot renderer split.\n</ct:send>";
    check(crossthink_repair_thought_command(hybrid, result) &&
        result.payload == "\nPlanning the Mandelbrot renderer split.\n", "observed hybrid not repaired verbatim");
    for (const char * text : {
            "", " ", "<ct:send>Plan", "<ct:send>Plan<ct:send>", "<ct:send/>", "<ct:peek />",
            "  <ct:inbox/>", "\n\t<ct:inbox/>", "`<ct:inbox/>`", "\"<ct:inbox/>\"", "> <ct:inbox/>",
            "```xml\n<ct:inbox/>\n```", "~~~\n<ct:send>Plan</ct:send>\n~~~",
            "I'll send:\n<ct:send>Plan</ct:send>", "<ct:send>Plan</ct:send>\nDone.",
            "<ct:send>Plan</ct:send>\n<ct:inbox/>", "<ct:peek/><ct:inbox/>",
            "<ct:send><ct:peek/></ct:send>", "<ct:send><tool_call>calculator</tool_call></ct:send>",
            "<ct:send><function=calculator></function></ct:send>",
            "<ct:send><think>Plan</think></ct:send>", "<ct:send><|im_end|></ct:send>",
            "<tool_call>\n<function=ct:send>\n<parameter=</ct:send>\n</function>\n</tool_call>",
            "<tool_call><function=ct:send>Plan</function></tool_call>",
            "<tool_call><function=ct:send>Plan</ct:send></function>",
            "<tool_call><function=ct:send>Plan</ct:send></tool_call>",
            "<tool_call><function=ct:send>Plan</ct:send>extra</function></tool_call>",
            "<tool_call><function=ct:send>Plan</ct:send></function></tool_call>extra",
            "<tool_call><function=ct:send><parameter=message>Plan</ct:send></function></tool_call>",
            "<tool_call><function=ct:send><parameter=message>Plan</parameter>",
            "<tool_call><function=ct:send><parameter=text>Plan</parameter></function></tool_call>",
            "<tool_call><function=ct:send><parameter=message>x</parameter><parameter=message>y</parameter></function></tool_call>",
            "<tool_call><function=ct:send><parameter=message>x<parameter=message>y</parameter></function></tool_call>",
            "<tool_call><function=ct:send><parameter=message><ct:inbox/></parameter></function></tool_call>",
            "<tool_call><function=ct:peek>argument</function></tool_call>",
            "<tool_call><function=ct:peek>", "<tool_call><function=ct:peek></function>",
            "<tool_call><function=ct:inbox><parameter=message>x</parameter></function></tool_call>",
            "<tool_call><function=ct:send></ct:send><function=ct:inbox></function></tool_call>",
            "<tool_call><function=ct:peek></function></tool_call>\n<tool_call><function=calculator></function></tool_call>",
            "<tool_call><function=calculator><parameter=expression>1+1</parameter></function></tool_call>",
            "<tool_call><function=ct:unknown></function></tool_call>",
            "<tool_call>{\"name\":\"ct:send\",\"arguments\":{\"message\":\"hello\"}}</tool_call>"}) {
        result = {"stale", "must not survive failure"};
        check(!crossthink_repair_thought_command(text, result), "ambiguous or ordinary text repaired as a command");
        check(result.command.empty() && result.payload.empty(), "failed repair retained a stale command");
    }
    for (const std::string & payload : {std::string(), std::string(" \t\r\n\f\v")}) {
        for (const std::string & text : {
                "<ct:send>" + payload + "</ct:send>",
                "<tool_call><function=ct:send>" + payload + "</ct:send>",
                "<tool_call><function=ct:send><parameter=message>" + payload +
                    "</parameter></function></tool_call>"}) {
            check(!crossthink_repair_thought_command(text, result), "empty message repaired for delivery");
        }
    }
    for (const std::string & payload : {
            std::string(server_thought_scanner::max_payload + 1, 'x'),
            std::string(server_thought_scanner::max_escaped_payload / 6 + 1, '\x01')}) {
        check(!crossthink_repair_thought_command("<ct:send>" + payload + "</ct:send>", result),
            "oversized direct payload repaired");
        check(!crossthink_repair_thought_command("<tool_call><function=ct:send><parameter=message>" + payload +
            "</parameter></function></tool_call>", result), "oversized wrapped payload repaired");
    }
}

int main() {
    try {
        test_command_repair();
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
