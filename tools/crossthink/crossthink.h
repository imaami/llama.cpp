#pragma once

#include "server-token-wire.h"
#include "server-thought.h"
#include "crossthink-mcp.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using crossthink_json = nlohmann::ordered_json;

class crossthink_native_parse_error final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

    static bool is_response(int status, const crossthink_json & response) {
        if (status != 500 || !response.is_object()) {
            return false;
        }
        const auto error = response.find("error");
        if (error == response.end() || !error->is_object()) {
            return false;
        }
        return error->contains("code") && error->at("code").is_number_integer() && error->at("code") == 500 &&
            error->contains("type") && error->at("type") == "server_error" &&
            error->contains("message") && error->at("message") ==
                "The model produced output that does not match the expected peg-native format";
    }
};

struct crossthink_options {
    int32_t chunk_tokens = 512;
    int32_t sentence_after = 256;
    bool paragraph_splice = true;
    int32_t answer_tokens = 1024;
    uint32_t seed = 42;
    double temperature = 1.0;
    int32_t tool_tokens = 2048;
    int32_t max_tool_rounds = 8;
    int32_t protocol_retries = -1;
    bool strict_thought_protocol = false;
    bool telepathy = true;
    int32_t link_quantum = 64;
    int32_t private_quantum = 256;
    int32_t link_wait_tokens = 512;
};

class crossthink_transport {
public:
    virtual ~crossthink_transport() = default;
    // describe() adds close_token and control_ids to /tokens/info metadata.
    virtual crossthink_json describe() = 0;
    virtual std::vector<llama_token> initial_prompt(const std::string & text) = 0;
    virtual std::vector<llama_token> next_user(const std::string & text) = 0;
    virtual void configure_peer(const std::string &) { throw std::runtime_error("transport does not support peer identities"); }
    virtual std::vector<llama_token> text_tokens(const std::string &) { throw std::runtime_error("transport does not support text tokenization"); }
    virtual std::vector<llama_token> literal_tokens(const std::string & text) { return text_tokens(text); }
    virtual void configure_tools(const crossthink_json &) { throw std::runtime_error("transport does not support tools"); }
    virtual crossthink_json parse_tool_turn(const std::vector<llama_token> &) { throw std::runtime_error("transport does not support tool parsing"); }
    virtual std::vector<llama_token> tool_results(const crossthink_json &, const crossthink_json &) {
        throw std::runtime_error("transport does not support tool results");
    }
    virtual crossthink_json generate(const std::vector<llama_token> & prompt,
            const crossthink_json & parameters,
            const std::function<bool(const server_token_wire::packet &)> & receive) = 0;
    virtual void cancel() = 0;
};

std::unique_ptr<crossthink_transport> crossthink_unix_transport(
        const std::string & socket, const std::string & api_key);

class crossthink_session {
public:
    crossthink_session(std::array<std::unique_ptr<crossthink_transport>, 2> transports,
            const crossthink_options & options,
            std::array<std::unique_ptr<crossthink_tool_service>, 2> tools = {});
    ~crossthink_session();
    crossthink_session(const crossthink_session &) = delete;
    crossthink_session & operator=(const crossthink_session &) = delete;

    // Commands are asynchronous. A concurrent pending command is rejected.
    void command(const std::string & action, const std::string & text = {}, const std::string & target = "both");
    crossthink_json state();
    std::vector<crossthink_json> events_after(uint64_t cursor, bool wait = false);

private:
    struct thought_message {
        uint64_t id;
        size_t sender;
        std::string text;
    };
    struct token_sample {
        std::chrono::steady_clock::time_point time;
        uint64_t generated;
        uint64_t thinking;
    };
    struct peer_state {
        std::unique_ptr<crossthink_transport> transport;
        std::unique_ptr<crossthink_tool_service> tools;
        crossthink_json info;
        std::vector<llama_token> tape;
        std::vector<llama_token> inbox;
        std::vector<llama_token> controls;
        std::vector<llama_token> eog;
        uint64_t context_size = 0;
        uint64_t generated = 0;
        uint64_t imported = 0;
        uint64_t sequence = 0;
        uint64_t forced_splices = 0;
        uint64_t tool_calls = 0;
        int32_t tool_rounds = 0;
        uint64_t protocol_retries = 0;
        uint64_t protocol_repairs = 0;
        uint32_t generation_limit = 0;
        server_thought_scanner thought_scanner;
        uint64_t revision = 1;
        uint64_t thinking_tokens = 0;
        uint64_t thought_turn = 0;
        std::string thought_text;
        std::vector<llama_token> thought_tokens;
        size_t thought_visible_tokens = 0;
        uint64_t peek_turn = 0;
        size_t peek_tokens = 0;
        size_t peek_bytes = 0;
        std::deque<thought_message> mailbox;
        size_t mailbox_bytes = 0;
        std::deque<token_sample> samples;
        std::chrono::steady_clock::time_point sampling_started = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point request_started;
        std::chrono::steady_clock::time_point last_token;
        uint64_t request_generated = 0;
        std::string phase = "idle";
        std::string tool_status;
        std::string boundary;
        llama_token close_token = -1;
        bool active = false;
        bool thinking_open = false;
        bool answer_done = false;
        bool empty_response = false;
        bool segment_done = false;
        bool private_pending = false;
        bool force_answer = false;
    };

    struct pending_command {
        std::string action;
        std::string text;
        std::string resume_mode;
        int target = -1;
    };

    crossthink_options options;
    std::array<peer_state, 2> peers;
    std::array<std::thread, 2> workers;
    std::thread controller;
    std::mutex mutex;
    std::condition_variable changed;
    std::condition_variable event_changed;
    std::deque<crossthink_json> event_log;
    size_t event_bytes = 0;
    uint64_t next_event_id = 1;
    uint64_t epoch = 1;
    uint64_t round = 0;
    uint64_t exchanges = 0;
    std::string mode = "idle";
    std::string error;
    bool stopping = false;
    bool busy = false;
    pending_command pending;
    bool answer_requested = false;
    uint64_t next_message_id = 1;

    bool native_tools(const peer_state & peer) const { return options.telepathy || bool(peer.tools); }
    const char * completed_mode() const;
    bool blocked(size_t index) const;
    uint64_t generation_id(size_t index) const;
    void begin_thought(peer_state & peer);
    void begin_request(peer_state & peer, const char * phase);
    void count_tokens(peer_state & peer, uint64_t generated, uint64_t thinking);
    bool thought_command(size_t index, uint64_t generation_epoch, const crossthink_json & result,
            std::unique_lock<std::mutex> & lock);
    void telepathy_worker(size_t index);

    crossthink_json state_locked() const;
    void emit(crossthink_json event);
    void emit_state();
    void fail(const std::string & message);
    void drain(peer_state & peer);
    void rendezvous();
    bool tool_turn(size_t index, uint64_t generation_epoch, std::unique_lock<std::mutex> & lock);
    void worker(size_t index);
    void control();
    void apply(const pending_command & command);
    bool receive(size_t index, uint64_t generation_epoch, bool answer,
            size_t & received, const server_token_wire::packet & packet);
};
