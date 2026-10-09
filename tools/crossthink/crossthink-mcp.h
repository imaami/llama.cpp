#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using crossthink_mcp_json = nlohmann::ordered_json;
using crossthink_mcp_headers = std::map<std::string, std::string>;

struct crossthink_mcp_config {
    std::string name;
    std::string url;
    crossthink_mcp_headers headers;
};

std::vector<crossthink_mcp_config> parse_crossthink_mcp_config(const crossthink_mcp_json & config);

class crossthink_tool_service {
public:
    virtual ~crossthink_tool_service() = default;
    // Rearm cancellation once at the start of a private assistant turn.
    virtual void begin_turn() {}
    // OpenAI function definitions; names are unique across all configured servers.
    virtual crossthink_mcp_json tools() const = 0;
    // Return the MCP CallToolResult, including content, structuredContent and isError.
    virtual crossthink_mcp_json call(const std::string & name, const crossthink_mcp_json & arguments) = 0;
    virtual void cancel() = 0;
};

// Byte transport seam: each post is independent and may be cancelled from another thread.
class crossthink_mcp_http {
public:
    virtual ~crossthink_mcp_http() = default;
    virtual void begin_turn() {}
    virtual void post(const crossthink_mcp_headers & headers, const std::string & body,
            const std::function<void(int, const crossthink_mcp_headers &)> & response,
            const std::function<bool(const char *, size_t)> & receive) = 0;
    virtual void cancel() = 0;
};

using crossthink_mcp_http_factory = std::function<std::unique_ptr<crossthink_mcp_http>(
        const crossthink_mcp_config &, int timeout_seconds)>;

std::unique_ptr<crossthink_tool_service> crossthink_mcp_service(
        const std::vector<crossthink_mcp_config> & configs, int timeout_seconds = 60,
        crossthink_mcp_http_factory http_factory = {});
