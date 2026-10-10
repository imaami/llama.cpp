#pragma once

#include "../server/server-thought.h"

#include <string>

struct crossthink_repaired_command {
    std::string command;
    std::string payload;
};

// Only a complete, single command can cross from the native channel into thought handling.
inline bool crossthink_repair_thought_command(const std::string & text, crossthink_repaired_command & result) {
    result = {};
    if (text.size() > server_thought_scanner::max_payload + 512) {
        return false;
    }
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || (first && text[first - 1] != '\n' && text[first - 1] != '\r')) {
        return false;
    }
    std::string input = text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
    std::string command;
    std::string payload;
    const auto consume = [](std::string & value, const std::string & prefix) {
        if (value.compare(0, prefix.size(), prefix)) {
            return false;
        }
        value.erase(0, prefix.size());
        return true;
    };
    const auto whitespace = [](std::string & value) {
        const auto position = value.find_first_not_of(" \t\r\n");
        value.erase(0, position == std::string::npos ? value.size() : position);
    };
    const auto outer_end = [&](std::string value, bool optional) {
        whitespace(value);
        if (optional && value.empty()) {
            return true;
        }
        if (!consume(value, "</function>")) {
            return false;
        }
        whitespace(value);
        if (!consume(value, "</tool_call>")) {
            return false;
        }
        whitespace(value);
        return value.empty();
    };
    if (input == "<ct:peek/>" || input == "<ct:inbox/>") {
        command = input == "<ct:peek/>" ? "peek" : "inbox";
    } else if (consume(input, "<ct:send>")) {
        const auto close = input.find("</ct:send>");
        if (close == std::string::npos || close + sizeof("</ct:send>") - 1 != input.size()) {
            return false;
        }
        command = "send";
        payload = input.substr(0, close);
    } else if (consume(input, "<tool_call>")) {
        whitespace(input);
        for (const char * name : {"send", "peek", "inbox"}) {
            if (consume(input, std::string("<function=ct:") + name + '>')) {
                command = name;
                break;
            }
        }
        if (command.empty()) {
            return false;
        }
        if (command == "send") {
            std::string parameter = input;
            whitespace(parameter);
            if (consume(parameter, "<parameter=message>")) {
                const auto close = parameter.find("</parameter>");
                if (close == std::string::npos ||
                        !outer_end(parameter.substr(close + sizeof("</parameter>") - 1), false)) {
                    return false;
                }
                payload = parameter.substr(0, close);
            } else {
                const auto close = input.find("</ct:send>");
                if (close == std::string::npos ||
                        !outer_end(input.substr(close + sizeof("</ct:send>") - 1), true)) {
                    return false;
                }
                payload = input.substr(0, close);
            }
        } else if (!outer_end(input, false)) {
            return false;
        }
    } else {
        return false;
    }
    if (command == "send" && payload.find_first_not_of(" \t\r\n\f\v") == std::string::npos) {
        return false;
    }
    // Structural tags inside a repaired payload may indicate nested or incomplete calls.
    for (const char * marker : {"<ct:", "</ct:", "<tool_call", "</tool_call", "<function", "</function",
            "<parameter", "</parameter", "<think", "</think", "<|"}) {
        if (payload.find(marker) != std::string::npos) {
            return false;
        }
    }
    server_thought_scanner scanner;
    scanner.thinking = true;
    scanner.feed(command == "send" ? "<ct:send>" + payload + "</ct:send>" : "<ct:" + command + "/>");
    if (scanner.command != command || !scanner.error.empty() || scanner.payload != payload) {
        return false;
    }
    result = {command, payload};
    return true;
}

// Recognize a misplaced command-only answer for protocol repair, never execution.
inline bool crossthink_misplaced_thought_commands(const std::string & text) {
    size_t offset = text.find_first_not_of(" \t\r\n");
    if (offset == std::string::npos) {
        return false;
    }
    constexpr size_t max_commands = 32;
    constexpr size_t max_command_bytes = server_thought_scanner::max_payload +
        sizeof("<ct:send></ct:send>") - 1;
    for (size_t count = 0; count < max_commands; ++count) {
        // Whitespace on blank lines is harmless; indentation on a command line is not.
        if (offset && text[offset - 1] != '\n' && text[offset - 1] != '\r') {
            return false;
        }
        if (text.compare(offset, sizeof("<ct:peek/>") - 1, "<ct:peek/>") &&
                text.compare(offset, sizeof("<ct:inbox/>") - 1, "<ct:inbox/>") &&
                text.compare(offset, sizeof("<ct:send>") - 1, "<ct:send>")) {
            return false;
        }
        server_thought_scanner scanner;
        scanner.thinking = true;
        const auto start = offset;
        while (offset < text.size() && offset - start < max_command_bytes && scanner.command.empty()) {
            scanner.feed(text.substr(offset++, 1));
        }
        if (scanner.command.empty() || !scanner.error.empty()) {
            return false;
        }
        offset = text.find_first_not_of(" \t\r\n", offset);
        if (offset == std::string::npos) {
            return true;
        }
    }
    return false;
}
