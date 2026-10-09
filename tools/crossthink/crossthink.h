#pragma once

#include "server-token-wire.h"

#include <nlohmann/json.hpp>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
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
};

class crossthink_transport {
public:
    virtual ~crossthink_transport() = default;
    // describe() adds close_token and control_ids to /tokens/info metadata.
    virtual crossthink_json describe() = 0;
    virtual std::vector<llama_token> initial_prompt(const std::string & text) = 0;
    virtual std::vector<llama_token> next_user(const std::string & text) = 0;
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
            const crossthink_options & options);
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
        std::string boundary;
        llama_token close_token = -1;
        bool active = false;
        bool thinking_open = false;
        bool answer_done = false;
        bool segment_done = false;
    };

    struct pending_command {
        std::string action;
        std::string text;
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

    crossthink_json state_locked() const;
    void emit(crossthink_json event);
    void emit_state();
    void fail(const std::string & message);
    void drain(peer_state & peer);
    void rendezvous();
    void worker(size_t index);
    void control();
    void apply(const pending_command & command);
    bool receive(size_t index, uint64_t generation_epoch, bool answer,
            size_t & received, const server_token_wire::packet & packet);
};
