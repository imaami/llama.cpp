#pragma once

#include <string>

struct crossthink_speech_guard {
    void reset() {
        state = lead;
    }

    void feed(const std::string & text) {
        for (unsigned char c : text) {
            if (c == '\n' || c == '\r') {
                state = lead;
            } else if (state == lead) {
                if (c == '[') {
                    state = bracket;
                } else if (c != ' ' && c != '\t' && c != '\r') {
                    state = body;
                }
            } else {
                state = body;
            }
        }
    }

    std::string grammar() const {
        const char * root = state == lead ? "lead" : state == bracket ? "bracket" : "body";
        return std::string("root ::= ") + root + "\n" +
            "lead ::= \"\" | [ \\t\\r\\n] lead | \"[\" bracket | [^ \\t\\r\\n\\x5b] body\n"
            "bracket ::= \"\" | [\\r\\n] lead | [^AB\\r\\n] body\n"
            "body ::= \"\" | [\\r\\n] lead | [^\\r\\n] body\n";
    }

    static bool substantive(const std::string & text) {
        for (size_t offset = 0; offset < text.size();) {
            const unsigned char c = text[offset];
            if (c < 0x80) {
                if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
                    return true;
                }
                ++offset;
                continue;
            }
            const unsigned char next = offset + 1 < text.size() ? text[offset + 1] : 0;
            const unsigned char third = offset + 2 < text.size() ? text[offset + 2] : 0;
            if (c == 0xc2 && (next == 0x85 || next == 0xa0)) {
                offset += 2;
                continue;
            }
            if ((c == 0xe2 && next == 0x80 && ((third >= 0x80 && third <= 0x8a) ||
                    third == 0x93 || third == 0x94 || third == 0xa2 || third == 0xa6 ||
                    third == 0xa8 || third == 0xa9 || third == 0xaf)) ||
                    (c == 0xe2 && next == 0x81 && third == 0x9f) ||
                    (c == 0xe3 && next == 0x80 && third == 0x80)) {
                offset += 3;
                continue;
            }
            return true;
        }
        return false;
    }

private:
    enum phase { lead, bracket, body } state = lead;
};
