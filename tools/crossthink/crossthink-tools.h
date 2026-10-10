#pragma once

#include <nlohmann/json.hpp>

#include <string>

inline nlohmann::ordered_json crossthink_thought_tools() {
    using json = nlohmann::ordered_json;
    auto tools = json::array();
    const auto add = [&](const char * name, const char * description, const json & properties,
            const json & required) {
        tools.push_back({{"type", "function"}, {"function", {
            {"name", name}, {"description", description}, {"strict", true},
            {"parameters", {{"type", "object"}, {"properties", properties},
                {"required", required}, {"additionalProperties", false}}},
        }}});
    };
    add("send_thought", "Queue a thought message for your partner without interrupting them. "
        "They receive it when they check their inbox. Returns immediately; continue your own work.",
        {{"message", {{"type", "string"}, {"minLength", 1}, {"maxLength", 16384},
            {"description", "Useful task information to send, at most 16 KiB of UTF-8 text."}}}},
        json::array({"message"}));
    add("check_inbox", "Read and consume queued thought messages. The result appears inside your "
        "next reasoning block. An empty inbox returns immediately; do not wait or poll repeatedly.",
        json::object(), json::array());
    add("read_thoughts", "Read a verbatim snapshot of your partner's current or most recent reasoning, "
        "without interrupting them. Later reads return only new text. The result appears inside your "
        "next reasoning block and may end mid-word.", json::object(), json::array());
    add("ping_peer", "Send a brief fixed attention notification to your partner. It carries no message "
        "body; use send_thought for content. Use sparingly when unread mail needs attention. "
        "Returns immediately; neither agent waits for a reply.", json::object(), json::array());
    return tools;
}

inline std::string crossthink_thought_command(const std::string & name) {
    if (name == "send_thought") {
        return "send";
    }
    if (name == "check_inbox") {
        return "inbox";
    }
    if (name == "read_thoughts") {
        return "peek";
    }
    if (name == "ping_peer") {
        return "ping";
    }
    return {};
}

inline bool crossthink_thought_arguments(const std::string & name,
        const nlohmann::ordered_json & arguments, std::string & payload, std::string & error) {
    payload.clear();
    error.clear();
    if (crossthink_thought_command(name).empty()) {
        error = "Unknown thought tool.";
        return false;
    }
    if (!arguments.is_object()) {
        error = "Tool arguments must be an object.";
        return false;
    }
    if (name != "send_thought") {
        if (!arguments.empty()) {
            error = "This thought tool takes no arguments.";
            return false;
        }
        return true;
    }
    if (arguments.size() != 1 || !arguments.contains("message") || !arguments.at("message").is_string()) {
        error = "send_thought requires exactly one string argument named message.";
        return false;
    }
    const auto & message = arguments.at("message").get_ref<const std::string &>();
    if (message.empty() || message.find_first_not_of(" \t\r\n\f\v") == std::string::npos) {
        error = "Thought messages must not be empty.";
        return false;
    }
    if (message.size() > 16384) {
        error = "Thought messages must not exceed 16 KiB of UTF-8 text.";
        return false;
    }
    payload = message;
    return true;
}
