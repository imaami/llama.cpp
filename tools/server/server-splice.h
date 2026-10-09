#pragma once

#include <cstddef>
#include <string>

// Inspect the complete raw token piece before deciding whether its end is a splice point.
struct server_splice_scanner {
    char fence = 0;
    size_t fence_size = 0;
    char line_marker = 0;
    size_t line_marker_size = 0;
    unsigned line_indent = 0;
    bool line_marker_possible = true;
    bool line_marker_done = false;
    bool line_marker_tail_space = true;
    bool line_backtick_info = false;
    bool line_space = true;
    bool seen_newline = false;
    bool pending_cr = false;
    bool paragraph = false;
    unsigned sentence = 0;
    bool generated_content = false;

    void begin_generation() {
        generated_content = false;
        paragraph = false;
    }

    static bool space(unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    }

    void newline() {
        const bool was_fenced = fence != 0;
        if (fence) {
            if (line_marker == fence && line_marker_size >= fence_size && line_marker_tail_space) {
                fence = 0;
                fence_size = 0;
            }
        } else if (line_marker_size >= 3 && !(line_marker == '`' && line_backtick_info)) {
            fence = line_marker;
            fence_size = line_marker_size;
        }
        paragraph = !was_fenced && !fence && line_space && seen_newline;
        if (was_fenced || fence) {
            sentence = 0;
        } else if (sentence == 1) {
            sentence = 2;
        }
        seen_newline = true;
        line_marker = 0;
        line_marker_size = 0;
        line_indent = 0;
        line_marker_possible = true;
        line_marker_done = false;
        line_marker_tail_space = true;
        line_backtick_info = false;
        line_space = true;
    }

    void feed(const std::string & piece) {
        for (unsigned char c : piece) {
            if (pending_cr) {
                pending_cr = false;
                newline();
                if (c == '\n') {
                    continue;
                }
            }
            if (c == '\r') {
                pending_cr = true;
                continue;
            }
            if (c == '\n') {
                newline();
                continue;
            }
            if (!space(c)) {
                generated_content = true;
                paragraph = false;
                line_space = false;
            }

            if (line_marker_possible && !line_marker) {
                if (c == ' ' && line_indent < 3) {
                    ++line_indent;
                } else if (c == '`' || c == '~') {
                    line_marker = c;
                    line_marker_size = 1;
                } else {
                    line_marker_possible = false;
                }
            } else if (line_marker) {
                if (!line_marker_done && c == line_marker) {
                    ++line_marker_size;
                } else {
                    line_marker_done = true;
                    line_marker_tail_space &= space(c);
                    line_backtick_info |= c == '`';
                }
            }

            if (c == '.' || c == '!' || c == '?') {
                sentence = 1;
            } else if (space(c)) {
                if (sentence == 1) {
                    sentence = 2;
                }
            } else if (sentence != 1 || (c != '\"' && c != '\'' && c != ')' && c != ']' && c != '}')) {
                sentence = 0;
            }
        }
    }

    const char * boundary(bool allow_sentence) const {
        if (!generated_content || pending_cr || fence || line_marker_size >= 3) {
            return "";
        }
        if (paragraph) {
            return "paragraph";
        }
        return allow_sentence && sentence == 2 ? "sentence" : "";
    }
};
