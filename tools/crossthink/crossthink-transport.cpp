#include "crossthink.h"

#include <cpp-httplib/httplib.h>
#include <sys/socket.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <utility>

using json = crossthink_json;

class unix_transport final : public crossthink_transport {
    httplib::Client client;
    int32_t n_vocab = 0;
    std::string initial_text;
    std::atomic<uint64_t> cancellation{0};

    std::string response(const httplib::Result & result) const {
        if (!result) {
            throw std::runtime_error("server request failed: " + httplib::to_string(result.error()));
        }
        if (result->status != 200) {
            throw std::runtime_error("server HTTP " + std::to_string(result->status) + ": " + result->body.substr(0, 4096));
        }
        return result->body;
    }

    json post(const char * path, const json & body) {
        return json::parse(response(client.Post(path, body.dump(), "application/json")));
    }

    std::vector<llama_token> tokenize(const std::string & text, bool special = false) {
        const auto result = post("/tokenize", {{"content", text}, {"add_special", special}, {"parse_special", true}});
        std::vector<llama_token> tokens;
        for (const auto & value : result.at("tokens")) {
            if (!value.is_number_integer()) {
                throw std::runtime_error("tokenizer returned a noninteger ID");
            }
            const auto token = value.get<int64_t>();
            if (token < 0 || token >= n_vocab) {
                throw std::runtime_error("tokenizer returned an out-of-range ID");
            }
            tokens.push_back(static_cast<llama_token>(token));
        }
        return tokens;
    }

    std::string render(const json & messages) {
        const auto result = post("/apply-template", {
            {"messages", messages}, {"chat_template_kwargs", {{"enable_thinking", true}}},
        });
        const auto text = result.at("prompt").get<std::string>();
        const auto opening = text.rfind("<think>");
        if (opening == std::string::npos || text.find_first_not_of(" \t\r\n", opening + 7) != std::string::npos) {
            throw std::runtime_error("chat template must end in an open <think> section; use --jinja and a thinking-enabled model");
        }
        return text;
    }

public:
    unix_transport(const std::string & path, const std::string & key) : client(path, 80) {
        client.set_address_family(AF_UNIX);
        client.set_connection_timeout(5);
        client.set_read_timeout(600);
        client.set_write_timeout(60);
        client.set_keep_alive(true);
        if (!key.empty()) {
            client.set_bearer_token_auth(key);
        }
    }

    json describe() override {
        auto result = json::parse(response(client.Get("/tokens/info")));
        const int64_t count = result.at("n_vocab").get<int64_t>();
        if (count <= 0 || count > INT32_MAX) {
            throw std::runtime_error("invalid vocabulary size");
        }
        n_vocab = static_cast<int32_t>(count);
        const auto close = tokenize("</think>");
        const auto open = tokenize("<think>");
        if (close.size() != 1 || open.size() != 1) {
            throw std::runtime_error("crossthink requires single-token <think> and </think> delimiters");
        }
        result["close_token"] = close.front();
        result["control_ids"] = json::array({open.front(), close.front()});
        for (const char * marker : {"<|im_start|>", "<|im_end|>"}) {
            const auto ids = tokenize(marker);
            if (ids.size() == 1) {
                result["control_ids"].push_back(ids.front());
            }
        }
        return result;
    }

    std::vector<llama_token> initial_prompt(const std::string & text) override {
        initial_text = text;
        return tokenize(render(json::array({{{"role", "user"}, {"content", text}}})), true);
    }

    std::vector<llama_token> next_user(const std::string & text) override {
        std::string sentinel = "CROSSTHINK_ASSISTANT_BOUNDARY_57c86b14";
        while (initial_text.find(sentinel) != std::string::npos || text.find(sentinel) != std::string::npos) {
            sentinel += "_";
        }
        const auto rendered = render(json::array({
            {{"role", "user"}, {"content", initial_text}},
            {{"role", "assistant"}, {"content", sentinel}},
            {{"role", "user"}, {"content", text}},
        }));
        const auto boundary = rendered.find(sentinel);
        if (boundary == std::string::npos || rendered.find(sentinel, boundary + sentinel.size()) != std::string::npos) {
            throw std::runtime_error("chat template did not preserve a unique assistant boundary");
        }
        return tokenize(rendered.substr(boundary + sentinel.size()));
    }

    json generate(const std::vector<llama_token> & prompt, const json & parameters,
            const std::function<bool(const server_token_wire::packet &)> & receive) override {
        const auto generation = cancellation.load();
        server_token_wire::stream_decoder decoder(n_vocab, parameters.at("n_predict").get<size_t>());
        json terminal;
        std::exception_ptr failure;
        bool cancelled = false;
        bool done = false;
        int status = 0;
        std::string error_body;
        httplib::Request request;
        request.method = "POST";
        request.path = "/completion/tokens";
        request.body = server_token_wire::encode(parameters.dump(), prompt);
        request.set_header("Content-Type", "application/octet-stream");
        request.response_handler = [&](const httplib::Response & result) {
            status = result.status;
            return generation == cancellation.load();
        };
        request.content_receiver = [&](const char * data, size_t size, uint64_t, uint64_t) {
            try {
                if (generation != cancellation.load()) {
                    cancelled = true;
                    return false;
                }
                if (status != 200) {
                    error_body.append(data, std::min(size, size_t(4096) - error_body.size()));
                    return true;
                }
                decoder.feed(data, size, [&](server_token_wire::packet packet) {
                    if (done) {
                        throw std::runtime_error("data after binary completion's final record");
                    }
                    if (cancelled) {
                        return;
                    }
                    const auto metadata = json::parse(packet.metadata);
                    const auto type = metadata.at("type").get<std::string>();
                    if (type == "tokens" && !packet.tokens.empty()) {
                        cancelled = !receive(packet);
                    } else if (type == "done" && packet.tokens.empty()) {
                        terminal = metadata;
                        done = true;
                    } else if (type == "error") {
                        throw std::runtime_error("server stream error: " + metadata.dump());
                    } else {
                        throw std::runtime_error("invalid binary completion record");
                    }
                });
                return !cancelled;
            } catch (...) {
                failure = std::current_exception();
                return false;
            }
        };
        const auto result = client.send(request);
        if (failure) {
            std::rethrow_exception(failure);
        }
        if (cancelled || generation != cancellation.load()) {
            return {};
        }
        if (status && status != 200) {
            throw std::runtime_error("server HTTP " + std::to_string(status) + ": " + error_body);
        }
        if (!result) {
            throw std::runtime_error("binary stream failed: " + httplib::to_string(result.error()));
        }
        decoder.finish();
        if (!done) {
            throw std::runtime_error("binary stream ended without its final record");
        }
        return terminal;
    }

    void cancel() override {
        ++cancellation;
        client.stop();
    }
};

std::unique_ptr<crossthink_transport> crossthink_unix_transport(
        const std::string & path, const std::string & key) {
    return std::make_unique<unix_transport>(path, key);
}
