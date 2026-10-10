#pragma once

#include "server-token-wire.h"
#include "crossthink-mcp.h"
#include "crossthink-speech.h"

#include <nlohmann/json.hpp>

#include <array>
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

struct crossthink_options {
    int32_t chunk_tokens = 512;
    int32_t sentence_after = 256;
    bool paragraph_splice = true;
    int32_t answer_tokens = 1024;
    uint32_t seed = 42;
    double temperature = 1.0;
    int32_t tool_tokens = 2048;
    int32_t max_tool_rounds = 8;
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
    void command(const std::string & action, const std::string & text = {});
    crossthink_json state();
    std::vector<crossthink_json> events_after(uint64_t cursor, bool wait = false);

private:
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
        uint32_t generation_limit = 0;
        uint64_t link_blank_tokens = 0;
        bool link_turn_content = false;
        crossthink_speech_guard speech_guard;
        std::string tool_status;
        std::string boundary;
        llama_token close_token = -1;
        bool active = false;
        bool thinking_open = false;
        bool answer_done = false;
        bool segment_done = false;
        bool private_pending = false;
    };

    struct pending_command {
        std::string action;
        std::string text;
        std::string resume_mode;
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
    bool link_enabled = false;
    bool link_pending = false;
    bool link_requested = false;
    int link_requester = -1;
    int link_speaker = -1;
    size_t link_next = 0;
    int private_owner = -1;
    int link_yield_to = -1;
    uint64_t link_yield_tokens = 0;
    std::string link_wait;
    std::string link_previous_boundary;
    std::array<std::vector<llama_token>, 2> link_labels;
    std::array<std::vector<llama_token>, 2> link_interrupt_labels;
    std::array<std::vector<llama_token>, 2> link_markers;

    bool native_tools(const peer_state & peer) const { return options.telepathy || bool(peer.tools); }
    void clear_link_wait();
    void clear_link_speech();
    void request_link(bool enabled, int requester, const std::string & wait = {});
    bool apply_link();
    void append_shared(const std::vector<llama_token> & tokens);
    void start_speaker(size_t index);
    void finish_link(const std::string & reason);
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
