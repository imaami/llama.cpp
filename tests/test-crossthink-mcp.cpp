#include "crossthink-mcp.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = crossthink_mcp_json;

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        throw std::runtime_error("line " + std::to_string(line) + ": " + expression);
    }
}

#define CHECK(condition) check(bool(condition), #condition, __LINE__)

struct exchange {
    std::string method;
    int status = 200;
    crossthink_mcp_headers headers{{"Content-Type", "application/json"}};
    std::string body;
    std::function<void(const crossthink_mcp_headers &, const json &)> inspect;
    size_t stride = 1;
    std::function<void()> completed;
};

struct script {
    std::vector<exchange> exchanges;
    size_t next = 0;
    int cancellations = 0;
    int turns = 0;
    bool cancel_latched = false;
};

class fake_http final : public crossthink_mcp_http {
public:
    explicit fake_http(std::shared_ptr<script> state) : state(std::move(state)) {}

    void post(const crossthink_mcp_headers & headers, const std::string & body,
            const std::function<void(int, const crossthink_mcp_headers &)> & response,
            const std::function<bool(const char *, size_t)> & receive) override {
        if (state->cancel_latched) {
            throw std::runtime_error("fake MCP HTTP request cancelled");
        }
        CHECK(state->next < state->exchanges.size());
        const exchange item = state->exchanges[state->next++];
        const json request = json::parse(body);
        CHECK(request.value("method", "") == item.method);
        CHECK(headers.at("Accept") == "application/json, text/event-stream");
        CHECK(headers.at("Content-Type") == "application/json");
        if (item.inspect) {
            item.inspect(headers, request);
        }
        response(item.status, item.headers);
        for (size_t offset = 0; offset < item.body.size(); offset += item.stride) {
            if (!receive(item.body.data() + offset, std::min(item.stride, item.body.size() - offset))) {
                break;
            }
        }
        if (item.completed) {
            item.completed();
        }
    }

    void begin_turn() override {
        state->cancel_latched = false;
        ++state->turns;
    }

    void cancel() override {
        state->cancel_latched = true;
        ++state->cancellations;
    }

private:
    std::shared_ptr<script> state;
};

exchange reply(const std::string & method, uint64_t id, const json & result) {
    exchange value;
    value.method = method;
    value.body = json{{"jsonrpc", "2.0"}, {"id", id}, {"result", result}}.dump();
    return value;
}

json tool(const std::string & name) {
    return {{"name", name}, {"description", "A test tool"}, {"inputSchema", {{"type", "object"}}}};
}

std::shared_ptr<script> initialize_script(const json & tools = json::array({tool("calculate")})) {
    auto state = std::make_shared<script>();
    auto init = reply("initialize", 1, {{"protocolVersion", "2025-06-18"}, {"capabilities", {{"tools", json::object()}}}});
    init.headers["Mcp-Session-Id"] = "test-session";
    init.inspect = [](const crossthink_mcp_headers & headers, const json & request) {
        CHECK(!headers.count("MCP-Session-Id"));
        CHECK(!headers.count("MCP-Protocol-Version"));
        CHECK(request.at("params").at("protocolVersion") == "2025-11-25");
        CHECK(request.at("params").at("capabilities").empty());
    };
    state->exchanges.push_back(init);
    exchange notification;
    notification.method = "notifications/initialized";
    notification.status = 202;
    notification.inspect = [](const crossthink_mcp_headers & headers, const json & request) {
        CHECK(headers.at("MCP-Session-Id") == "test-session");
        CHECK(headers.at("MCP-Protocol-Version") == "2025-06-18");
        CHECK(!request.contains("id"));
    };
    state->exchanges.push_back(notification);
    state->exchanges.push_back(reply("tools/list", 2, {{"tools", tools}}));
    return state;
}

std::unique_ptr<crossthink_tool_service> service(const std::shared_ptr<script> & state) {
    return crossthink_mcp_service({{"workbench", "http://localhost:9000/mcp", {}}}, 60,
            [state](const crossthink_mcp_config &, int timeout) {
                CHECK(timeout == 60);
                return std::make_unique<fake_http>(state);
            });
}

template<class Function>
void must_throw(Function function, const std::string & expected) {
    try {
        function();
    } catch (const std::exception & error) {
        CHECK(std::string(error.what()).find(expected) != std::string::npos);
        return;
    }
    CHECK(false);
}

std::string alias(crossthink_tool_service & client) {
    return client.tools().at(0).at("function").at("name").get<std::string>();
}

void test_json_session_and_tool_result() {
    auto state = initialize_script();
    auto call = reply("tools/call", 3, {{"content", json::array({{{"type", "text"}, {"text", "42"}}})},
            {"structuredContent", {{"answer", 42}}}, {"isError", false}});
    call.inspect = [](const crossthink_mcp_headers & headers, const json & request) {
        CHECK(headers.at("MCP-Session-Id") == "test-session");
        CHECK(headers.at("MCP-Protocol-Version") == "2025-06-18");
        CHECK(request.at("params").at("name") == "calculate");
        CHECK(request.at("params").at("arguments").at("expression") == "6*7");
    };
    state->exchanges.push_back(call);
    auto client = service(state);
    CHECK(alias(*client).find("mcp_workbench_calculate_") == 0);
    CHECK(client->tools().at(0).at("function").at("parameters").at("type") == "object");
    const auto result = client->call(alias(*client), {{"expression", "6*7"}});
    CHECK(result.at("structuredContent").at("answer") == 42);
    CHECK(!result.at("isError").get<bool>());
    CHECK(state->next == state->exchanges.size());
    client->cancel();
    CHECK(state->cancellations == 1);
}

void test_sse_fragmentation_and_ping() {
    for (size_t stride : {size_t(1), size_t(2), size_t(7), size_t(4096)}) {
        auto state = initialize_script();
        exchange call;
        call.method = "tools/call";
        call.headers = {{"content-type", "text/event-stream; charset=utf-8"}};
        call.stride = stride;
        call.body = "\xef\xbb\xbf: heartbeat\r\nid: prime\r\ndata:\r\n\r\n"
                    "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n\n"
                    "data: {\"jsonrpc\":\"2.0\",\"id\":\"ping-id\",\"method\":\"ping\"}\r\r"
                    "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":3,\n"
                    "data: \"result\":{\"content\":[{\"type\":\"text\",\"text\":\"bad expression\"}],\"isError\":true}}\n\n";
        state->exchanges.push_back(call);
        exchange ping;
        ping.status = 202;
        ping.inspect = [](const crossthink_mcp_headers & headers, const json & request) {
            CHECK(headers.at("MCP-Session-Id") == "test-session");
            CHECK(request.at("id") == "ping-id");
            CHECK(request.at("result").is_object());
            CHECK(!request.contains("method"));
        };
        state->exchanges.push_back(ping);
        auto client = service(state);
        CHECK(client->call(alias(*client), json::object()).at("isError").get<bool>());
        CHECK(state->next == state->exchanges.size());
    }
}

void test_pagination_and_stable_names() {
    auto state = initialize_script(json::array({tool("z_compile")}));
    state->exchanges.back() = reply("tools/list", 2,
            {{"tools", json::array({tool("z_compile")})}, {"nextCursor", "page2"}});
    auto next = reply("tools/list", 3, {{"tools", json::array({tool("a_calculate")})}});
    next.inspect = [](const crossthink_mcp_headers &, const json & request) {
        CHECK(request.at("params").at("cursor") == "page2");
    };
    state->exchanges.push_back(next);
    auto client = service(state);
    auto reordered = service(initialize_script(json::array({tool("a_calculate"), tool("z_compile")})));
    CHECK(client->tools() == reordered->tools());

    auto repeated = initialize_script();
    repeated->exchanges.back() = reply("tools/list", 2, {{"tools", json::array()}, {"nextCursor", "same"}});
    repeated->exchanges.push_back(reply("tools/list", 3, {{"tools", json::array()}, {"nextCursor", "same"}}));
    must_throw([&] { service(repeated); }, "cursor");
}

void test_failures_are_not_retried() {
    for (const std::string body : {
            "{\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{}}",
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"error\":{\"code\":-32602,\"message\":\"bad arguments\"}}",
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"content\":7}}"}) {
        auto state = initialize_script();
        exchange call;
        call.method = "tools/call";
        call.body = body;
        state->exchanges.push_back(call);
        auto client = service(state);
        must_throw([&] { client->call(alias(*client), json::object()); }, "MCP");
        CHECK(state->next == 4);
    }
    auto state = initialize_script();
    exchange partial;
    partial.method = "tools/call";
    partial.headers = {{"Content-Type", "text/event-stream"}};
    partial.body = "data: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"content\":[]}}\n";
    state->exchanges.push_back(partial);
    auto client = service(state);
    must_throw([&] { client->call(alias(*client), json::object()); }, "not retried");
    CHECK(state->next == 4);

    auto expired = initialize_script();
    exchange missing;
    missing.method = "tools/call";
    missing.status = 404;
    expired->exchanges.push_back(missing);
    auto restarted = initialize_script();
    restarted->exchanges[0].body = reply("initialize", 4, {{"protocolVersion", "2025-06-18"},
            {"capabilities", {{"tools", json::object()}}}}).body;
    expired->exchanges.push_back(restarted->exchanges[0]);
    expired->exchanges.push_back(restarted->exchanges[1]);
    expired->exchanges.push_back(reply("tools/call", 5, {{"content", json::array()}}));
    auto recoverable = service(expired);
    must_throw([&] { recoverable->call(alias(*recoverable), json::object()); }, "expired");
    CHECK(expired->next == 4);
    CHECK(recoverable->call(alias(*recoverable), json::object()).at("content").empty());
    CHECK(expired->next == 7);
}

void test_bounds_and_config() {
    const auto valid = parse_crossthink_mcp_config({{"mcpServers", {
            {"calc", {{"url", "https://example.test/mcp?key=x"}, {"headers", {{"Authorization", "Bearer test"}}}}},
            {"disabled", {{"disabled", true}, {"command", "ignored"}}}}}});
    CHECK(valid.size() == 1);
    CHECK(valid[0].headers.at("Authorization") == "Bearer test");
    for (const std::string url : {"file:///tmp/mcp", "http://user:pass@host/mcp", "http://host/#fragment", "http://host/\r\n"}) {
        must_throw([&] { parse_crossthink_mcp_config({{"mcpServers", {{"bad", {{"url", url}}}}}}); }, "MCP");
    }
    for (const auto & header : std::vector<std::pair<std::string, std::string>>{
            {"Host", "example.test"}, {"mCp-SeSsIoN-Id", "wrong"}, {"Authorization", "token\r\nInjected: yes"}}) {
        must_throw([&] { parse_crossthink_mcp_config({{"mcpServers", {{"bad", {
                {"url", "http://localhost/mcp"}, {"headers", {{header.first, header.second}}}}}}}}); }, "MCP");
    }
    auto state = initialize_script();
    auto large = reply("tools/call", 3,
            {{"content", json::array({{{"type", "text"}, {"text", std::string(256 * 1024, 'x')}}})}});
    large.stride = 64 * 1024;
    state->exchanges.push_back(large);
    exchange huge;
    huge.method = "tools/call";
    huge.body.assign(4 * 1024 * 1024 + 1, 'x');
    huge.stride = huge.body.size();
    state->exchanges.push_back(huge);
    auto client = service(state);
    must_throw([&] { client->call(alias(*client), json::object()); }, "256 KiB");
    must_throw([&] { client->call(alias(*client), json::object()); }, "4 MiB");
    must_throw([&] { client->call("missing", json::object()); }, "unknown");
    must_throw([&] { client->call(alias(*client), json::array()); }, "arguments");
    CHECK(state->next == 5);
}

void test_sticky_cancellation() {
    auto state = initialize_script();
    state->exchanges.push_back(reply("tools/call", 3, {{"content", json::array()}}));
    auto client = service(state);
    client->cancel();
    must_throw([&] { client->call(alias(*client), json::object()); }, "cancelled");
    must_throw([&] { client->call(alias(*client), json::object()); }, "cancelled");
    CHECK(state->next == 3);
    client->begin_turn();
    CHECK(state->turns == 1);
    CHECK(client->call(alias(*client), json::object()).at("content").empty());
    CHECK(state->next == 4);

    for (bool cancel_after_notification : {false, true}) {
        auto reconnect = initialize_script();
        exchange missing;
        missing.method = "tools/call";
        missing.status = 404;
        reconnect->exchanges.push_back(missing);
        auto restarted = initialize_script();
        restarted->exchanges[0].body = reply("initialize", 4,
                {{"protocolVersion", "2025-06-18"}, {"capabilities", {{"tools", json::object()}}}}).body;
        reconnect->exchanges.push_back(restarted->exchanges[0]);
        reconnect->exchanges.push_back(restarted->exchanges[1]);
        reconnect->exchanges.push_back(reply("tools/call", 5, {{"content", json::array()}}));
        auto recoverable = service(reconnect);
        reconnect->exchanges[cancel_after_notification ? 5 : 4].completed = [&] { recoverable->cancel(); };
        must_throw([&] { recoverable->call(alias(*recoverable), json::object()); }, "expired");
        must_throw([&] { recoverable->call(alias(*recoverable), json::object()); }, "cancelled");
        CHECK(reconnect->next == (cancel_after_notification ? 6 : 5));
        must_throw([&] { recoverable->call(alias(*recoverable), json::object()); }, "cancelled");
        CHECK(reconnect->next == (cancel_after_notification ? 6 : 5));
        if (cancel_after_notification) {
            recoverable->begin_turn();
            CHECK(recoverable->call(alias(*recoverable), json::object()).at("content").empty());
            CHECK(reconnect->next == 7);
        }
    }
}

} // namespace

int main() {
    test_json_session_and_tool_result();
    test_sse_fragmentation_and_ping();
    test_pagination_and_stable_names();
    test_failures_are_not_retried();
    test_bounds_and_config();
    test_sticky_cancellation();
    std::cout << "crossthink MCP protocol tests: passed\n";
}
