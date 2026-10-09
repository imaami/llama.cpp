#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace server_token_wire {

static constexpr char magic[] = "LLMTOK01";
static constexpr size_t metadata_max = 64 * 1024;
static constexpr size_t header_size = 16;

struct packet {
    std::string metadata;
    std::vector<llama_token> tokens;
};

inline uint32_t read_u32(const char * data) {
    const auto * p = reinterpret_cast<const unsigned char *>(data);
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

inline void append_u32(std::string & data, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        data.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

inline packet decode(const std::string & body, int32_t n_vocab) {
    if (body.size() < header_size || body.compare(0, 8, magic, 8)) {
        throw std::invalid_argument("invalid LLMTOK01 header");
    }
    const uint32_t n_metadata = read_u32(body.data() + 8);
    const uint32_t n_tokens = read_u32(body.data() + 12);
    if (n_metadata > metadata_max || n_metadata > body.size() - header_size) {
        throw std::invalid_argument("invalid LLMTOK01 metadata length");
    }
    const size_t offset = header_size + n_metadata;
    const size_t bytes = body.size() - offset;
    if (bytes % 4 || bytes / 4 != n_tokens) {
        throw std::invalid_argument("invalid LLMTOK01 token payload length");
    }
    if (n_vocab <= 0) {
        throw std::invalid_argument("invalid vocabulary size");
    }

    packet result;
    result.metadata = body.substr(header_size, n_metadata);
    result.tokens.reserve(n_tokens);
    for (size_t i = offset; i < body.size(); i += 4) {
        const uint32_t token = read_u32(body.data() + i);
        if (token >= static_cast<uint32_t>(n_vocab)) {
            throw std::invalid_argument("LLMTOK01 token ID outside vocabulary");
        }
        result.tokens.push_back(static_cast<llama_token>(token));
    }
    return result;
}

inline std::string encode(const std::string & metadata, const std::vector<llama_token> & tokens) {
    if (metadata.size() > metadata_max || tokens.size() > std::numeric_limits<uint32_t>::max() ||
            tokens.size() > (std::string().max_size() - header_size - metadata.size()) / 4) {
        throw std::length_error("LLMTOK01 packet too large");
    }
    std::string result(magic, 8);
    result.reserve(header_size + metadata.size() + tokens.size() * 4);
    append_u32(result, static_cast<uint32_t>(metadata.size()));
    append_u32(result, static_cast<uint32_t>(tokens.size()));
    result.append(metadata);
    for (llama_token token : tokens) {
        if (token < 0) {
            throw std::invalid_argument("negative LLMTOK01 token ID");
        }
        append_u32(result, static_cast<uint32_t>(token));
    }
    return result;
}

} // namespace server_token_wire
