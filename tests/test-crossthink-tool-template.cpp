#include "server-common.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_template(const char * path, const std::string & output, bool append = true) {
    std::ifstream file(path);
    check(file.good(), "could not open chat template");
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    server_chat_params options{};
    options.tmpls = common_chat_templates_init(nullptr, source);
    options.use_jinja = true;
    options.prefill_assistant = true;
    options.enable_thinking = true;
    options.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    const json tool = {
        {"type", "function"}, {"function", {
            {"name", "calculator"}, {"description", "Evaluate an expression"},
            {"parameters", {{"type", "object"}, {"properties", {
                {"expression", {{"type", "string"}}},
            }}, {"required", json::array({"expression"})}}},
        }},
    };
    json body = {
        {"messages", json::array({{{"role", "user"}, {"content", "Calculate 1 + 1"}}})},
        {"tools", json::array({tool})}, {"add_generation_prompt", true},
        {"chat_template_kwargs", {{"enable_thinking", true}}},
    };
    std::vector<raw_buffer> files;
    common_chat_session session;
    auto parameters = oaicompat_chat_params_parse(nullptr, body, options, files, session);
    check(session.prompt().find("calculator") != std::string::npos, "tool schema missing from prompt");
    const auto message = session.finish(common_chat_input(output));
    check(message.tool_calls.size() == 1, "native tool call was not parsed");
    check(message.reasoning_content.empty(), "private tail unexpectedly contains reasoning");
    check(message.tool_calls.front().name == "calculator", "wrong parsed tool name");
    check(json::parse(message.tool_calls.front().arguments).at("expression") == "1 + 1", "wrong parsed arguments");
    if (!append) {
        std::printf("tool parser %s: passed\n", path);
        return;
    }

    auto assistant = message.to_json_oaicompat();
    assistant["content"] = "";
    assistant.erase("reasoning_content");
    assistant["tool_calls"][0]["id"] = "call_a_1";
    body["messages"].push_back(assistant);
    body["add_generation_prompt"] = false;
    parameters = oaicompat_chat_params_parse(nullptr, body, options, files, session);
    const auto before = parameters.at("prompt").get<std::string>();
    const std::string end = "<|im_end|>";
    const auto boundary = before.rfind(end);
    check(boundary != std::string::npos, "complete assistant render omitted EOG");

    body["messages"].push_back({{"role", "tool"}, {"tool_call_id", "call_a_1"},
        {"name", "calculator"}, {"content", "2"}});
    body["add_generation_prompt"] = true;
    parameters = oaicompat_chat_params_parse(nullptr, body, options, files, session);
    const auto after = parameters.at("prompt").get<std::string>();
    check(after.compare(0, boundary + end.size(), before, 0, boundary + end.size()) == 0,
        "tool result changes the existing assistant prefix");
    const auto suffix = after.substr(boundary + end.size());
    check(suffix.front() == '\n', "tool suffix dropped the newline after the raw EOG token");
    check(suffix.find("<tool_response>\n2\n</tool_response>") != std::string::npos,
        "tool result missing from native suffix");
    const auto open = suffix.rfind("<think>");
    check(open != std::string::npos && suffix.find_first_not_of(" \t\r\n", open + 7) == std::string::npos,
        "tool result suffix does not reopen reasoning");

    // Default assistant prefill is retained unless a complete-message render was requested.
    body["messages"] = json::array({{{"role", "user"}, {"content", "Hello"}},
        {{"role", "assistant"}, {"content", "partial"}}});
    body.erase("add_generation_prompt");
    parameters = oaicompat_chat_params_parse(nullptr, body, options, files, session);
    const auto continued = parameters.at("prompt").get<std::string>();
    check(continued.size() >= 7 && continued.compare(continued.size() - 7, 7, "partial") == 0,
        "default assistant prefill changed");
    std::printf("tool template %s: passed\n", path);
}

int main() {
    try {
        test_template("models/templates/Qwen3.5-4B.jinja",
            "</think>\n\n<tool_call>\n<function=calculator>\n<parameter=expression>\n1 + 1\n</parameter>\n</function>\n</tool_call>");
        test_template("models/templates/Qwen-Qwen3-0.6B.jinja",
            "</think>\n\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"1 + 1\"}}\n</tool_call>", false);
        test_template("models/templates/Qwen-QwQ-32B.jinja",
            "</think>\n\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"1 + 1\"}}\n</tool_call>");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "tool template test: %s\n", error.what());
        return 1;
    }
}
