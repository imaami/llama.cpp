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

static void test_template(const char * path, const std::string & output, bool append = true, bool identity = false) {
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
    if (identity) {
        auto messages = json::array({{{"role", "system"},
            {"content", "You are agent A. Agent B is independent. The telepathic link is initially OFF."}}});
        for (const auto & message : body["messages"]) {
            messages.push_back(message);
        }
        body["messages"] = std::move(messages);
    }
    std::vector<raw_buffer> files;
    common_chat_session session;
    auto parameters = oaicompat_chat_params_parse(nullptr, body, options, files, session);
    check(session.prompt().find("calculator") != std::string::npos, "tool schema missing from prompt");
    check(!identity || session.prompt().find("You are agent A.") != std::string::npos,
        "agent identity missing from tool prompt");
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
    check(!identity || after.find("You are agent A.") != std::string::npos,
        "tool result continuation lost agent identity");
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

static void test_link_tool() {
    std::ifstream file("models/templates/Qwen3.5-4B.jinja");
    check(file.good(), "could not open link tool chat template");
    server_chat_params options{};
    options.tmpls = common_chat_templates_init(nullptr,
        std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()));
    options.use_jinja = true;
    options.enable_thinking = true;
    options.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const json body = {
        {"messages", json::array({
            {{"role", "system"}, {"content", "You are agent B. The telepathic link is initially OFF."}},
            {{"role", "user"}, {"content", "Think about the problem together."}},
        })},
        {"tools", json::array({{
            {"type", "function"}, {"function", {
                {"name", "think_with_telepathic_link"}, {"description", "Set the shared reasoning link"},
                {"parameters", {{"type", "object"}, {"properties", {
                    {"enabled", {{"type", "boolean"}}},
                    {"yield_until", {{"type", "string"}, {"enum", json::array({"fragment", "sentence", "paragraph"})}}},
                }}, {"required", json::array({"enabled"})}}},
            }},
        }})},
        {"add_generation_prompt", true}, {"chat_template_kwargs", {{"enable_thinking", true}}},
    };
    std::vector<raw_buffer> files;
    common_chat_session session;
    for (bool enabled : {false, true}) {
        auto request = body;
        oaicompat_chat_params_parse(nullptr, request, options, files, session);
        const auto output = std::string("</think>\n\n<tool_call>\n<function=think_with_telepathic_link>\n") +
            "<parameter=enabled>\n" + (enabled ? "true" : "false") + "\n</parameter>\n" +
            (enabled ? "<parameter=yield_until>\nparagraph\n</parameter>\n" : "") + "</function>\n</tool_call>";
        const auto parsed = session.finish(common_chat_input(output));
        check(parsed.tool_calls.size() == 1 && parsed.tool_calls.front().name == "think_with_telepathic_link",
            "native template did not parse the link tool");
        const auto arguments = json::parse(parsed.tool_calls.front().arguments);
        check(arguments.at("enabled").is_boolean() && arguments.at("enabled") == enabled,
            "native link tool lost its boolean enabled argument");
        check(enabled ? arguments.at("yield_until") == "paragraph" : !arguments.contains("yield_until"),
            "native link tool changed its optional yield argument");
    }
    std::puts("native link tool: passed");
}

int main() {
    try {
        test_template("models/templates/Qwen3.5-4B.jinja",
            "</think>\n\n<tool_call>\n<function=calculator>\n<parameter=expression>\n1 + 1\n</parameter>\n</function>\n</tool_call>");
        test_template("models/templates/Qwen3.5-4B.jinja",
            "</think>\n\n<tool_call>\n<function=calculator>\n<parameter=expression>\n1 + 1\n</parameter>\n</function>\n</tool_call>", true, true);
        test_template("models/templates/Qwen-Qwen3-0.6B.jinja",
            "</think>\n\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"1 + 1\"}}\n</tool_call>", false);
        test_template("models/templates/Qwen-QwQ-32B.jinja",
            "</think>\n\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"1 + 1\"}}\n</tool_call>");
        test_link_tool();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "tool template test: %s\n", error.what());
        return 1;
    }
}
