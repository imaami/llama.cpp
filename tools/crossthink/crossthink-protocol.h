#pragma once

#include "../server/server-thought.h"

#include <string>

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
