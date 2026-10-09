#include "crossthink-mcp.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace {

constexpr size_t max_response_bytes = 4 * 1024 * 1024;
constexpr size_t max_tool_bytes = 256 * 1024;
constexpr size_t max_catalogue_bytes = 1024 * 1024;
constexpr size_t max_tools = 256;

std::string lowercase(std::string text) {
    for (char & c : text) {
        if (c >= 'A' && c <= 'Z') {
            c += 'a' - 'A';
        }
    }
    return text;
}

std::string header_value(const crossthink_mcp_headers & headers, const std::string & name) {
    for (const auto & header : headers) {
        if (lowercase(header.first) == name) {
            return header.second;
        }
    }
    return {};
}

bool header_name_valid(const std::string & text) {
    if (text.empty()) {
        return false;
    }
    for (unsigned char c : text) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || std::string("!#$%&'*+-.^_`|~").find(c) != std::string::npos)) {
            return false;
        }
    }
    return true;
}

struct endpoint {
    std::string origin;
    std::string path;
};

endpoint parse_endpoint(const std::string & url) {
    size_t start = url.compare(0, 7, "http://") == 0 ? 7 :
            url.compare(0, 8, "https://") == 0 ? 8 : 0;
    if (!start || url.size() > 8192 || url.find('#') != std::string::npos) {
        throw std::runtime_error("MCP URL must be an http:// or https:// endpoint without a fragment");
    }
    for (unsigned char c : url) {
        if (c <= 0x20 || c == 0x7f || c == '\\') {
            throw std::runtime_error("invalid character in MCP URL");
        }
    }
    size_t end = url.find_first_of("/?", start);
    if (end == std::string::npos) {
        end = url.size();
    }
    if (end == start || url.substr(start, end - start).find('@') != std::string::npos) {
        throw std::runtime_error("MCP URL requires a host and must not contain credentials");
    }
    endpoint result{url.substr(0, end), end == url.size() ? "/" : url.substr(end)};
    if (result.path[0] == '?') {
        result.path.insert(result.path.begin(), '/');
    }
    return result;
}

class http_transport final : public crossthink_mcp_http {
public:
    http_transport(const crossthink_mcp_config & config, int timeout_seconds) :
            address(parse_endpoint(config.url)), timeout_seconds(timeout_seconds) {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
        if (address.origin.compare(0, 8, "https://") == 0) {
            throw std::runtime_error("HTTPS MCP requires a build with OpenSSL support");
        }
#endif
    }

    void post(const crossthink_mcp_headers & headers, const std::string & body,
            const std::function<void(int, const crossthink_mcp_headers &)> & response,
            const std::function<bool(const char *, size_t)> & receive) override {
        const uint64_t generation = cancelled.load();
        auto client = std::make_shared<httplib::Client>(address.origin);
        client->set_connection_timeout(std::min(timeout_seconds, 5));
        client->set_read_timeout(timeout_seconds);
        client->set_write_timeout(timeout_seconds);
        client->set_max_timeout(timeout_seconds * 1000);
        client->set_payload_max_length(max_response_bytes);
        client->set_follow_location(false);
        client->set_keep_alive(false);
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (cancel_latched || cancelled.load() != generation) {
                throw std::runtime_error("MCP request cancelled");
            }
            active.push_back(client);
        }
        struct active_guard {
            http_transport & owner;
            std::shared_ptr<httplib::Client> client;
            ~active_guard() {
                std::lock_guard<std::mutex> lock(owner.mutex);
                auto & clients = owner.active;
                clients.erase(std::remove(clients.begin(), clients.end(), client), clients.end());
            }
        } guard{*this, client};

        httplib::Request request;
        request.method = "POST";
        request.path = address.path;
        request.body = body;
        for (const auto & header : headers) {
            request.headers.emplace(header.first, header.second);
        }
        bool finished = false;
        std::exception_ptr failure;
        request.response_handler = [&](const httplib::Response & result) {
            try {
                crossthink_mcp_headers received_headers;
                for (const auto & header : result.headers) {
                    received_headers.emplace(lowercase(header.first), header.second);
                }
                response(result.status, received_headers);
                return true;
            } catch (...) {
                failure = std::current_exception();
                return false;
            }
        };
        request.content_receiver = [&](const char * data, size_t length, uint64_t, uint64_t) {
            try {
                if (cancelled.load() != generation) {
                    return false;
                }
                finished = !receive(data, length);
                return !finished;
            } catch (...) {
                failure = std::current_exception();
                return false;
            }
        };
        auto result = client->send(request);
        if (failure) {
            std::rethrow_exception(failure);
        }
        if (cancelled.load() != generation) {
            throw std::runtime_error("MCP request cancelled");
        }
        if (!result && !finished) {
            throw std::runtime_error("MCP HTTP request failed: " + httplib::to_string(result.error()));
        }
    }

    void begin_turn() override {
        std::lock_guard<std::mutex> lock(mutex);
        cancel_latched = false;
    }

    void cancel() override {
        std::lock_guard<std::mutex> lock(mutex);
        cancel_latched = true;
        ++cancelled;
        for (auto & client : active) {
            client->stop();
        }
    }

private:
    endpoint address;
    int timeout_seconds;
    std::atomic<uint64_t> cancelled{0};
    std::mutex mutex;
    bool cancel_latched = false;
    std::vector<std::shared_ptr<httplib::Client>> active;
};

class rpc_response {
public:
    rpc_response(uint64_t id, std::function<void(const crossthink_mcp_json &)> server_request) :
            id(id), server_request(std::move(server_request)) {}

    void headers(int status, const crossthink_mcp_headers & headers) {
        if (status != 200) {
            throw std::runtime_error("MCP HTTP status " + std::to_string(status) +
                    "; expected a Streamable HTTP MCP endpoint (legacy SSE is unsupported)");
        }
        std::string content_type = lowercase(header_value(headers, "content-type"));
        const size_t separator = content_type.find(';');
        content_type.resize(separator == std::string::npos ? content_type.size() : separator);
        while (!content_type.empty() && content_type.back() == ' ') {
            content_type.pop_back();
        }
        if (content_type == "text/event-stream") {
            streaming = true;
        } else if (content_type != "application/json") {
            throw std::runtime_error("MCP response must be application/json or text/event-stream");
        }
    }

    bool feed(const char * data, size_t length) {
        if (length > max_response_bytes - bytes) {
            throw std::runtime_error("MCP response exceeds 4 MiB");
        }
        bytes += length;
        if (!streaming) {
            buffer.append(data, length);
            return true;
        }
        for (size_t i = 0; i < length && !complete; ++i) {
            const char c = data[i];
            if (c == '\r') {
                line();
                carriage_return = true;
            } else if (c == '\n') {
                if (!carriage_return) {
                    line();
                }
                carriage_return = false;
            } else {
                carriage_return = false;
                buffer += c;
            }
        }
        return !complete;
    }

    crossthink_mcp_json finish() {
        if (!streaming) {
            message(buffer);
        }
        if (!complete) {
            throw std::runtime_error("MCP response ended without a matching JSON-RPC result; request was not retried");
        }
        if (result.contains("error")) {
            const auto & error = result.at("error");
            const std::string text = error.is_object() && error.contains("message") && error.at("message").is_string() ?
                    error.at("message").get<std::string>() : "invalid JSON-RPC error";
            throw std::runtime_error("MCP error: " + text.substr(0, 1024));
        }
        return result.at("result");
    }

private:
    void line() {
        if (first_line) {
            first_line = false;
            if (buffer.compare(0, 3, "\xef\xbb\xbf") == 0) {
                buffer.erase(0, 3);
            }
        }
        if (buffer.empty()) {
            if (!event_data.empty()) {
                event_data.pop_back();
                if (!event_data.empty()) {
                    message(event_data);
                }
            }
            event_data.clear();
        } else if (buffer.compare(0, 5, "data:") == 0) {
            const size_t start = buffer.size() > 5 && buffer[5] == ' ' ? 6 : 5;
            event_data.append(buffer, start, std::string::npos);
            event_data += '\n';
        } else if (buffer == "data") {
            event_data += '\n';
        }
        buffer.clear();
    }

    void message(const std::string & text) {
        auto value = crossthink_mcp_json::parse(text, [](int depth, crossthink_mcp_json::parse_event_t,
                crossthink_mcp_json &) {
            if (depth > 64) {
                throw std::runtime_error("MCP JSON nesting exceeds 64 levels");
            }
            return true;
        });
        if (!value.is_object() || value.value("jsonrpc", "") != "2.0") {
            throw std::runtime_error("invalid MCP JSON-RPC message");
        }
        if (value.contains("method")) {
            if (!value.at("method").is_string() || value.contains("result") || value.contains("error")) {
                throw std::runtime_error("invalid MCP server message");
            }
            if (value.contains("id")) {
                if (++server_requests > 16 || (!value.at("id").is_string() && !value.at("id").is_number_integer())) {
                    throw std::runtime_error("invalid or excessive MCP server requests");
                }
                server_request(value);
            }
            return;
        }
        if (!value.contains("id") || !value.at("id").is_number_integer() || value.at("id") != id ||
                value.contains("result") == value.contains("error")) {
            throw std::runtime_error("MCP JSON-RPC response ID or envelope mismatch");
        }
        if (complete) {
            throw std::runtime_error("duplicate MCP JSON-RPC response");
        }
        result = std::move(value);
        complete = true;
    }

    uint64_t id;
    std::function<void(const crossthink_mcp_json &)> server_request;
    size_t bytes = 0;
    size_t server_requests = 0;
    bool streaming = false;
    bool carriage_return = false;
    bool first_line = true;
    bool complete = false;
    std::string buffer;
    std::string event_data;
    crossthink_mcp_json result;
};

struct mcp_connection {
    crossthink_mcp_config config;
    std::unique_ptr<crossthink_mcp_http> http;
    uint64_t next_id = 1;
    std::string session;
    std::string protocol;
    bool initialized = false;

    crossthink_mcp_headers headers() const {
        auto result = config.headers;
        result["Accept"] = "application/json, text/event-stream";
        result["Content-Type"] = "application/json";
        if (!session.empty()) {
            result["MCP-Session-Id"] = session;
        }
        if (!protocol.empty()) {
            result["MCP-Protocol-Version"] = protocol;
        }
        return result;
    }

    void notify(const crossthink_mcp_json & message) {
        bool accepted = false;
        http->post(headers(), message.dump(), [&](int status, const crossthink_mcp_headers &) {
            if (status != 202) {
                throw std::runtime_error("MCP notification or response rejected: HTTP " + std::to_string(status));
            }
            accepted = true;
        }, [](const char *, size_t length) {
            if (length) {
                throw std::runtime_error("MCP HTTP 202 response must have an empty body");
            }
            return true;
        });
        if (!accepted) {
            throw std::runtime_error("MCP notification received no HTTP response");
        }
    }

    crossthink_mcp_json rpc(const std::string & method, const crossthink_mcp_json & params) {
        const uint64_t id = next_id++;
        rpc_response parser(id, [&](const crossthink_mcp_json & request) {
            crossthink_mcp_json reply{{"jsonrpc", "2.0"}, {"id", request.at("id")}};
            if (request.at("method") == "ping") {
                reply["result"] = crossthink_mcp_json::object();
            } else {
                reply["error"] = {{"code", -32601}, {"message", "Client method not supported"}};
            }
            notify(reply);
        });
        const crossthink_mcp_json request{{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}};
        http->post(headers(), request.dump(), [&](int status, const crossthink_mcp_headers & received) {
            if (status == 404 && !session.empty()) {
                session.clear();
                protocol.clear();
                initialized = false;
                throw std::runtime_error("MCP session expired; this request was not retried");
            }
            parser.headers(status, received);
            if (method == "initialize") {
                session = header_value(received, "mcp-session-id");
                if (session.size() > 1024 || std::any_of(session.begin(), session.end(), [](unsigned char c) {
                    return c < 0x21 || c > 0x7e;
                })) {
                    throw std::runtime_error("invalid MCP session ID");
                }
            }
        }, [&](const char * data, size_t length) { return parser.feed(data, length); });
        return parser.finish();
    }

    void initialize() {
        if (initialized) {
            return;
        }
        session.clear();
        protocol.clear();
        auto result = rpc("initialize", {{"protocolVersion", "2025-11-25"},
                {"capabilities", crossthink_mcp_json::object()},
                {"clientInfo", {{"name", "llama-crossthink"}, {"version", "1"}}}});
        protocol = result.at("protocolVersion").get<std::string>();
        if (protocol != "2025-11-25" && protocol != "2025-06-18" && protocol != "2025-03-26") {
            throw std::runtime_error("unsupported MCP protocol version: " + protocol.substr(0, 64));
        }
        if (!result.contains("capabilities") || !result.at("capabilities").is_object() ||
                !result.at("capabilities").contains("tools") || !result.at("capabilities").at("tools").is_object()) {
            throw std::runtime_error("MCP server does not advertise tools");
        }
        notify({{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}});
        initialized = true;
    }
};

std::string tool_alias(const std::string & server, const std::string & tool) {
    std::string identity = server + '\0' + tool;
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : identity) {
        hash = (hash ^ c) * 1099511628211ull;
    }
    std::string name = server + "_" + tool;
    for (char & c : name) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_')) {
            c = '_';
        }
    }
    return "mcp_" + name.substr(0, 32) + "_" + std::to_string(hash);
}

class tool_service final : public crossthink_tool_service {
public:
    tool_service(const std::vector<crossthink_mcp_config> & configs, int timeout_seconds,
            const crossthink_mcp_http_factory & factory) {
        size_t catalogue_bytes = 0;
        for (const auto & config : configs) {
            auto connection = std::make_unique<mcp_connection>();
            connection->config = config;
            connection->http = factory ? factory(config, timeout_seconds) :
                    std::make_unique<http_transport>(config, timeout_seconds);
            if (!connection->http) {
                throw std::runtime_error("MCP HTTP factory returned no transport");
            }
            connection->initialize();
            std::set<std::string> cursors;
            std::string cursor;
            size_t pages = 0;
            do {
                if (++pages > 64) {
                    throw std::runtime_error("MCP tool catalogue exceeds 64 pages");
                }
                crossthink_mcp_json params = crossthink_mcp_json::object();
                if (!cursor.empty()) {
                    params["cursor"] = cursor;
                }
                const auto result = connection->rpc("tools/list", params);
                const auto & listed = result.at("tools");
                if (!listed.is_array()) {
                    throw std::runtime_error("MCP tools/list result is not an array");
                }
                for (const auto & item : listed) {
                    const std::string name = item.at("name").get<std::string>();
                    if (name.empty() || name.size() > 128 || !item.at("inputSchema").is_object()) {
                        throw std::runtime_error("invalid MCP tool name or input schema");
                    }
                    if (item.contains("execution") && item.at("execution").value("taskSupport", "") == "required") {
                        throw std::runtime_error("MCP tools requiring asynchronous tasks are unsupported");
                    }
                    const std::string alias = tool_alias(config.name, name);
                    if (!bindings.emplace(alias, binding{connections.size(), name}).second) {
                        throw std::runtime_error("duplicate MCP tool name");
                    }
                    crossthink_mcp_json definition{{"type", "function"}, {"function", {
                            {"name", alias}, {"description", config.name + "/" + name + "\n" + item.value("description", "")},
                            {"parameters", item.at("inputSchema")}}}};
                    catalogue_bytes += definition.dump().size();
                    if (catalogue_bytes > max_catalogue_bytes || definitions.size() >= max_tools) {
                        throw std::runtime_error("MCP tool catalogue exceeds 256 tools or 1 MiB");
                    }
                    definitions.push_back(std::move(definition));
                }
                cursor = result.value("nextCursor", "");
                if (cursor.size() > 4096 || (!cursor.empty() && !cursors.insert(cursor).second)) {
                    throw std::runtime_error("MCP tools/list has an invalid or repeated cursor");
                }
            } while (!cursor.empty());
            connections.push_back(std::move(connection));
        }
        std::sort(definitions.begin(), definitions.end(), [](const crossthink_mcp_json & a, const crossthink_mcp_json & b) {
            return a.at("function").at("name").get<std::string>() < b.at("function").at("name").get<std::string>();
        });
    }

    crossthink_mcp_json tools() const override {
        return definitions;
    }

    crossthink_mcp_json call(const std::string & name, const crossthink_mcp_json & arguments) override {
        check_cancelled();
        const auto found = bindings.find(name);
        if (found == bindings.end()) {
            throw std::runtime_error("unknown MCP tool: " + name.substr(0, 128));
        }
        if (!arguments.is_object() || arguments.dump().size() > max_tool_bytes) {
            throw std::runtime_error("MCP tool arguments must be an object of at most 256 KiB");
        }
        auto & connection = *connections[found->second.connection];
        connection.initialize();
        check_cancelled();
        auto result = connection.rpc("tools/call", {{"name", found->second.name}, {"arguments", arguments}});
        check_cancelled();
        if (!result.is_object() || !result.contains("content") || !result.at("content").is_array() ||
                (result.contains("isError") && !result.at("isError").is_boolean())) {
            throw std::runtime_error("invalid MCP tool result");
        }
        if (result.dump().size() > max_tool_bytes) {
            throw std::runtime_error("MCP tool result exceeds 256 KiB");
        }
        return result;
    }

    void begin_turn() override {
        std::lock_guard<std::mutex> lock(cancellation_mutex);
        for (auto & connection : connections) {
            connection->http->begin_turn();
        }
        cancelled = false;
    }

    void cancel() override {
        std::lock_guard<std::mutex> lock(cancellation_mutex);
        cancelled = true;
        for (auto & connection : connections) {
            connection->http->cancel();
        }
    }

private:
    void check_cancelled() const {
        if (cancelled.load()) {
            throw std::runtime_error("MCP tool turn cancelled");
        }
    }

    struct binding {
        size_t connection;
        std::string name;
    };
    std::vector<std::unique_ptr<mcp_connection>> connections;
    std::map<std::string, binding> bindings;
    crossthink_mcp_json definitions = crossthink_mcp_json::array();
    std::mutex cancellation_mutex;
    std::atomic<bool> cancelled{false};
};

} // namespace

std::vector<crossthink_mcp_config> parse_crossthink_mcp_config(const crossthink_mcp_json & config) {
    if (!config.is_object() || !config.contains("mcpServers") || !config.at("mcpServers").is_object()) {
        throw std::runtime_error("MCP config requires an mcpServers object");
    }
    const auto & servers = config.at("mcpServers");
    if (servers.size() > 16) {
        throw std::runtime_error("MCP config exceeds 16 servers");
    }
    std::vector<crossthink_mcp_config> result;
    for (auto entry = servers.begin(); entry != servers.end(); ++entry) {
        const auto & value = entry.value();
        if (!value.is_object()) {
            throw std::runtime_error("MCP server configuration must be an object");
        }
        if (value.value("disabled", false)) {
            continue;
        }
        if (entry.key().empty() || entry.key().size() > 256 || !value.contains("url") || !value.at("url").is_string()) {
            throw std::runtime_error("each MCP server requires a name and HTTP url; stdio is unsupported");
        }
        crossthink_mcp_config server{entry.key(), value.at("url").get<std::string>(), {}};
        parse_endpoint(server.url);
        if (value.contains("headers")) {
            const auto & headers = value.at("headers");
            if (!headers.is_object() || headers.size() > 64) {
                throw std::runtime_error("MCP headers must be an object with at most 64 entries");
            }
            std::set<std::string> names;
            for (auto header = headers.begin(); header != headers.end(); ++header) {
                const std::string name = lowercase(header.key());
                if (!header_name_valid(name) || name.size() > 256 || !header.value().is_string() ||
                        !names.insert(name).second || name == "host" || name == "content-length" ||
                        name == "transfer-encoding" || name == "connection" || name == "accept" ||
                        name == "content-type" || name == "mcp-session-id" || name == "mcp-protocol-version") {
                    throw std::runtime_error("invalid, duplicate or reserved MCP HTTP header");
                }
                const std::string text = header.value().get<std::string>();
                if (text.size() > 8192 || std::any_of(text.begin(), text.end(), [](unsigned char c) {
                    return (c < 0x20 && c != '\t') || c == 0x7f;
                })) {
                    throw std::runtime_error("invalid MCP HTTP header value");
                }
                server.headers.emplace(header.key(), text);
            }
        }
        result.push_back(std::move(server));
    }
    return result;
}

std::unique_ptr<crossthink_tool_service> crossthink_mcp_service(
        const std::vector<crossthink_mcp_config> & configs, int timeout_seconds,
        crossthink_mcp_http_factory http_factory) {
    if (timeout_seconds < 1 || timeout_seconds > 3600) {
        throw std::runtime_error("MCP timeout must be between 1 and 3600 seconds");
    }
    return std::make_unique<tool_service>(configs, timeout_seconds, http_factory);
}
