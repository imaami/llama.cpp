#include "../tools/crossthink/crossthink.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <utility>

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Predicate>
static void await(std::condition_variable & changed, std::unique_lock<std::mutex> & lock,
        Predicate predicate, const char * operation) {
    check(changed.wait_for(lock, std::chrono::seconds(10), predicate), std::string("timeout: ") + operation);
}

struct fake_step {
    std::vector<llama_token> tokens;
    bool done = false;
    crossthink_json result;
    std::string content = "piece";
};

struct fake_request {
    std::vector<llama_token> prompt;
    crossthink_json parameters;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<fake_step> steps;
    std::vector<llama_token> on_cancel;
    size_t consumed = 0;
    bool cancelled = false;
    bool returned = false;
    bool stale_accepted = false;

    void send(std::vector<llama_token> tokens, const std::string & content = "piece") {
        std::lock_guard<std::mutex> lock(mutex);
        steps.push_back({std::move(tokens), false, {}, content});
        changed.notify_all();
    }

    void finish(const std::string & stop_type = "limit", const std::string & boundary = "paragraph") {
        crossthink_json result = {{"type", "done"}, {"stop_type", stop_type}, {"truncated", false}};
        if (!boundary.empty()) {
            result["splice_boundary"] = boundary;
        }
        finish_result(std::move(result));
    }

    void finish_result(crossthink_json result) {
        std::lock_guard<std::mutex> lock(mutex);
        steps.push_back({{}, true, std::move(result)});
        changed.notify_all();
    }

    void wait_consumed(size_t count) {
        std::unique_lock<std::mutex> lock(mutex);
        await(changed, lock, [&] { return consumed >= count || returned; }, "consume fake token record");
        check(consumed >= count, "request ended before consuming token record");
    }

    void stale_on_cancel(std::vector<llama_token> tokens) {
        std::lock_guard<std::mutex> lock(mutex);
        on_cancel = std::move(tokens);
    }
};

struct fake_operation {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
    bool cancelled = false;
    bool throw_on_cancel = false;

    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released || cancelled; });
        if (cancelled && throw_on_cancel) {
            throw std::runtime_error("stale template operation failed after cancellation");
        }
    }

    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        await(changed, lock, [&] { return entered; }, "enter fake template operation");
    }

    void release(bool cancel = false) {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cancelled = cancelled || cancel;
        changed.notify_all();
    }
};

class fake_transport final : public crossthink_transport {
public:
    uint64_t context_size = 256;
    std::string fingerprint = "test-vocabulary";
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::shared_ptr<fake_request>> requests;
    std::vector<std::string> initial_texts;
    std::vector<std::string> user_texts;
    std::string configured_peer;
    crossthink_json configured_tools;
    crossthink_json parsed_assistant;
    std::vector<std::vector<llama_token>> parsed_turns;
    std::vector<crossthink_json> result_assistants;
    std::vector<crossthink_json> result_messages;
    std::shared_ptr<fake_operation> parse_gate;
    std::shared_ptr<fake_operation> results_gate;
    std::shared_ptr<fake_operation> literal_gate;
    std::shared_ptr<fake_operation> next_user_gate;

    crossthink_json describe() override {
        return {
            {"protocol", "LLMTOK01"}, {"fingerprint", fingerprint},
            {"n_vocab", 4096}, {"eog_ids", {3}}, {"context_size", context_size},
            {"close_token", 2}, {"control_ids", {1, 2, 3}}, {"splice", true}, {"splice_quantum", true}, {"tool_parse", true}, {"thought_commands", true},
        };
    }

    std::vector<llama_token> initial_prompt(const std::string & text) override {
        std::lock_guard<std::mutex> lock(mutex);
        initial_texts.push_back(text);
        return {1, 10, 11};
    }

    std::vector<llama_token> next_user(const std::string & text) override {
        std::shared_ptr<fake_operation> gate;
        {
            std::lock_guard<std::mutex> lock(mutex);
            user_texts.push_back(text);
            gate = next_user_gate;
        }
        if (gate) {
            gate->run();
        }
        return {3, 20, 21, 1};
    }

    void configure_tools(const crossthink_json & tools) override {
        std::lock_guard<std::mutex> lock(mutex);
        configured_tools = tools;
    }

    void configure_peer(const std::string & name) override {
        std::lock_guard<std::mutex> lock(mutex);
        configured_peer = name;
    }

    std::vector<llama_token> text_tokens(const std::string & text) override {
        std::vector<llama_token> tokens;
        for (unsigned char byte : text) {
            tokens.push_back(1000 + byte);
        }
        return tokens;
    }

    std::vector<llama_token> literal_tokens(const std::string & text) override {
        std::shared_ptr<fake_operation> gate;
        {
            std::lock_guard<std::mutex> lock(mutex);
            gate = literal_gate;
        }
        if (gate) {
            gate->run();
        }
        return text_tokens(text);
    }

    crossthink_json parse_tool_turn(const std::vector<llama_token> & tokens) override {
        crossthink_json result;
        std::shared_ptr<fake_operation> gate;
        {
            std::lock_guard<std::mutex> lock(mutex);
            parsed_turns.push_back(tokens);
            result = parsed_assistant;
            gate = parse_gate;
        }
        if (gate) {
            gate->run();
        }
        return result;
    }

    std::vector<llama_token> tool_results(const crossthink_json & assistant,
            const crossthink_json & results) override {
        std::shared_ptr<fake_operation> gate;
        {
            std::lock_guard<std::mutex> lock(mutex);
            result_assistants.push_back(assistant);
            result_messages.push_back(results);
            gate = results_gate;
        }
        if (gate) {
            gate->run();
        }
        return {40, 1};
    }

    crossthink_json generate(const std::vector<llama_token> & prompt,
            const crossthink_json & parameters,
            const std::function<bool(const server_token_wire::packet &)> & receive) override {
        auto request = std::make_shared<fake_request>();
        request->prompt = prompt;
        request->parameters = parameters;
        {
            std::lock_guard<std::mutex> lock(mutex);
            requests.push_back(request);
            changed.notify_all();
        }
        std::unique_lock<std::mutex> lock(request->mutex);
        for (;;) {
            request->changed.wait(lock, [&] { return request->cancelled || !request->steps.empty(); });
            if (request->cancelled) {
                const auto stale = request->on_cancel;
                lock.unlock();
                const bool accepted = !stale.empty() && receive(packet(stale));
                lock.lock();
                request->stale_accepted = accepted;
                request->returned = true;
                request->changed.notify_all();
                return {};
            }
            const auto step = std::move(request->steps.front());
            request->steps.pop_front();
            if (step.done) {
                request->returned = true;
                request->changed.notify_all();
                return step.result;
            }
            lock.unlock();
            const bool accepted = receive(packet(step.tokens, step.content));
            lock.lock();
            ++request->consumed;
            request->changed.notify_all();
            if (!accepted) {
                request->returned = true;
                request->changed.notify_all();
                return {};
            }
        }
    }

    void cancel() override {
        std::lock_guard<std::mutex> lock(mutex);
        if (parse_gate) {
            parse_gate->release(true);
        }
        if (results_gate) {
            results_gate->release(true);
        }
        if (literal_gate) {
            literal_gate->release(true);
        }
        if (next_user_gate) {
            std::lock_guard<std::mutex> gate_lock(next_user_gate->mutex);
            if (next_user_gate->entered) {
                next_user_gate->cancelled = true;
                next_user_gate->changed.notify_all();
            }
        }
        if (requests.empty()) {
            return;
        }
        auto & request = *requests.back();
        std::lock_guard<std::mutex> request_lock(request.mutex);
        request.cancelled = true;
        request.changed.notify_all();
    }

    std::shared_ptr<fake_request> request(size_t index) {
        std::unique_lock<std::mutex> lock(mutex);
        await(changed, lock, [&] { return requests.size() > index; }, "start fake generation");
        return requests[index];
    }

    size_t request_count() {
        std::lock_guard<std::mutex> lock(mutex);
        return requests.size();
    }

private:
    static server_token_wire::packet packet(const std::vector<llama_token> & tokens, const std::string & content = "piece") {
        return {crossthink_json{{"type", "tokens"}, {"content", content}}.dump(), tokens};
    }
};

struct fake_tool_call {
    std::string name;
    crossthink_json arguments;
    std::mutex mutex;
    std::condition_variable changed;
    crossthink_json result;
    bool finished = false;
    bool cancelled = false;
    bool returned = false;

    void finish(crossthink_json value) {
        std::lock_guard<std::mutex> lock(mutex);
        result = std::move(value);
        finished = true;
        changed.notify_all();
    }
};

class fake_tool_service final : public crossthink_tool_service {
public:
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::shared_ptr<fake_tool_call>> calls;

    crossthink_json tools() const override {
        return crossthink_json::array({{
            {"type", "function"}, {"function", {
                {"name", "calculator"}, {"description", "Evaluate an expression"},
                {"parameters", {{"type", "object"}, {"properties", {
                    {"expression", {{"type", "string"}}}}}, {"required", {"expression"}}}},
            }},
        }});
    }

    crossthink_json call(const std::string & name, const crossthink_json & arguments) override {
        auto request = std::make_shared<fake_tool_call>();
        request->name = name;
        request->arguments = arguments;
        {
            std::lock_guard<std::mutex> lock(mutex);
            calls.push_back(request);
            changed.notify_all();
        }
        std::unique_lock<std::mutex> lock(request->mutex);
        request->changed.wait(lock, [&] { return request->finished || request->cancelled; });
        request->returned = true;
        request->changed.notify_all();
        // A completed response may race cancellation; the engine must reject its stale epoch.
        return request->cancelled ? crossthink_json{{"content", {{{"type", "text"}, {"text", "stale"}}}}}
            : request->result;
    }

    void cancel() override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!calls.empty()) {
            auto & request = *calls.back();
            std::lock_guard<std::mutex> request_lock(request.mutex);
            request.cancelled = true;
            request.changed.notify_all();
        }
    }

    std::shared_ptr<fake_tool_call> request(size_t index) {
        std::unique_lock<std::mutex> lock(mutex);
        await(changed, lock, [&] { return calls.size() > index; }, "start fake tool call");
        return calls[index];
    }

    size_t call_count() {
        std::lock_guard<std::mutex> lock(mutex);
        return calls.size();
    }
};

struct fixture {
    std::array<fake_transport *, 2> peers;
    std::array<fake_tool_service *, 2> tools = {};
    std::unique_ptr<crossthink_session> session;
    uint64_t cursor = 0;

    fixture(int32_t chunk = 2, uint64_t context = 256, bool paragraph_splice = false,
            bool enable_tools = false, bool telepathy = false, int32_t max_tool_rounds = 8) {
        std::array<std::unique_ptr<crossthink_transport>, 2> transports;
        std::array<std::unique_ptr<crossthink_tool_service>, 2> services;
        for (size_t i = 0; i < peers.size(); ++i) {
            auto peer = std::make_unique<fake_transport>();
            peer->context_size = context;
            peers[i] = peer.get();
            transports[i] = std::move(peer);
            if (enable_tools) {
                auto service = std::make_unique<fake_tool_service>();
                tools[i] = service.get();
                services[i] = std::move(service);
                peers[i]->parsed_assistant = {
                    {"role", "assistant"}, {"content", ""}, {"tool_calls", crossthink_json::array({{
                        {"id", "call_calculator"}, {"type", "function"}, {"function", {
                            {"name", "calculator"}, {"arguments", "{\"expression\":\"6*7\"}"},
                        }},
                    }})},
                };
            }
        }
        crossthink_options options;
        options.chunk_tokens = chunk;
        options.answer_tokens = 8;
        options.tool_tokens = 8;
        options.paragraph_splice = paragraph_splice;
        options.sentence_after = 2;
        options.telepathy = telepathy;
        options.link_quantum = 2;
        options.private_quantum = 4;
        options.link_wait_tokens = 8;
        options.max_tool_rounds = max_tool_rounds;
        session = std::make_unique<crossthink_session>(std::move(transports), options, std::move(services));
    }

    template<typename Predicate>
    crossthink_json wait_state(Predicate predicate, const char * operation, bool allow_error = false) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        for (;;) {
            auto state = session->state();
            if (predicate(state)) {
                return state;
            }
            check(allow_error || state.at("mode") != "error", "unexpected engine error: " + state.dump());
            check(std::chrono::steady_clock::now() < deadline, std::string("timeout: ") + operation);
            for (const auto & event : session->events_after(cursor, true)) {
                cursor = event.at("id").get<uint64_t>();
            }
        }
    }

    crossthink_json wait_mode(const std::string & mode) {
        return wait_state([&](const crossthink_json & state) {
            return state.at("mode") == mode && !state.at("busy").get<bool>();
        }, mode.c_str(), mode == "error");
    }

    crossthink_json wait_peer(size_t index) {
        return wait_state([&](const crossthink_json & state) {
            const auto & peer = state.at("peers").at(index);
            return peer.at("waiting").get<bool>() && !peer.at("active").get<bool>();
        }, "wait for peer boundary");
    }

    void start(const std::string & text = "first question") {
        session->command("message", text);
        wait_mode("thinking");
    }

    void pause() {
        session->command("pause");
        for (size_t i = 0; i < peers.size(); ++i) {
            std::shared_ptr<fake_request> request;
            {
                std::lock_guard<std::mutex> lock(peers[i]->mutex);
                if (!peers[i]->requests.empty()) {
                    request = peers[i]->requests.back();
                }
            }
            if (request) {
                bool needs_token;
                {
                    std::lock_guard<std::mutex> lock(request->mutex);
                    needs_token = !request->returned && !request->consumed;
                }
                if (needs_token) {
                    request->send({static_cast<llama_token>(3000 + i)});
                }
                request->finish();
            }
        }
        wait_mode("paused");
    }
};

static size_t count(const std::vector<llama_token> & tokens, llama_token token) {
    return static_cast<size_t>(std::count(tokens.begin(), tokens.end(), token));
}

static bool ends_with(const std::vector<llama_token> & tokens, const std::vector<llama_token> & suffix) {
    return tokens.size() >= suffix.size() && std::equal(suffix.rbegin(), suffix.rend(), tokens.rbegin());
}

static void test_streaming_boundary_and_resume() {
    fixture f;
    f.start();
    auto a1 = f.peers[0]->request(0);
    auto b1 = f.peers[1]->request(0);
    a1->send({101});
    a1->wait_consumed(1);
    b1->send({201, 202});
    b1->wait_consumed(1);
    b1->finish();
    auto b2 = f.peers[1]->request(1);
    check(b1->prompt == std::vector<llama_token>({1, 10, 11}), "peer modified an active generation prompt");
    check(b2->prompt == std::vector<llama_token>({1, 10, 11, 201, 202, 101}),
            "peer did not import streamed tokens at its local quantum boundary");
    {
        std::lock_guard<std::mutex> lock(a1->mutex);
        check(!a1->returned, "source completed before peer consumed its first token");
    }
    a1->send({102});
    a1->wait_consumed(2);
    a1->finish();
    auto a2 = f.peers[0]->request(1);
    check(a2->prompt == std::vector<llama_token>({1, 10, 11, 101, 102, 201, 202}),
            "imported tokens echoed or arrived before the local generation boundary");
    f.pause();
    f.session->command("resume");
    f.wait_mode("thinking");
    auto a3 = f.peers[0]->request(2);
    auto b3 = f.peers[1]->request(2);
    for (const auto & request : {a3, b3}) {
        for (llama_token token : {101, 102, 201, 202, 3000, 3001}) {
            check(count(request->prompt, token) == 1, "pause/resume lost, echoed or replayed tokens");
        }
    }
    f.pause();
}

static void test_user_answer_and_reset(bool paragraph_splice) {
    fixture f(2, 256, paragraph_splice);
    f.start("question one");
    auto a1 = f.peers[0]->request(0);
    auto b1 = f.peers[1]->request(0);
    a1->send({101});
    b1->send({201});
    a1->wait_consumed(1);
    b1->wait_consumed(1);
    f.session->command("message", "question two");
    a1->finish();
    b1->finish();
    f.wait_mode("thinking");
    auto a2 = f.peers[0]->request(1);
    auto b2 = f.peers[1]->request(1);
    for (size_t i = 0; i < f.peers.size(); ++i) {
        const auto request = i ? b2 : a2;
        check(ends_with(request->prompt, {2, 3, 20, 21, 1}), "user intervention did not close reasoning before template suffix");
        check(count(request->prompt, 101) == 1 && count(request->prompt, 201) == 1,
                "user intervention lost or duplicated in-flight reasoning");
        std::lock_guard<std::mutex> lock(f.peers[i]->mutex);
        check(f.peers[i]->initial_texts == std::vector<std::string>{"question one"}, "unexpected initial template call");
        check(f.peers[i]->user_texts == std::vector<std::string>{"question two"}, "wrong intervention template text");
    }
    a2->send({102});
    b2->send({202});
    a2->wait_consumed(1);
    b2->wait_consumed(1);
    f.session->command("answer");
    a2->finish();
    b2->finish();
    f.wait_mode("answering");
    auto a3 = f.peers[0]->request(2);
    auto b3 = f.peers[1]->request(2);
    for (const auto & request : {a3, b3}) {
        check(request->prompt.back() == 2, "answer did not close reasoning");
        check(count(request->prompt, 102) == 1 && count(request->prompt, 202) == 1,
                "answer omitted queued reasoning");
        check(request->parameters.at("n_predict") == 8, "answer used reasoning quantum as its token budget");
        check(!request->parameters.contains("splice"), "answer inherited reasoning splice condition");
    }
    const auto before_answer = f.session->state();
    a3->send({301});
    b3->send({401});
    a3->wait_consumed(1);
    b3->wait_consumed(1);
    a3->finish("eos");
    b3->finish("eos");
    const auto answered = f.wait_mode("answered");
    for (size_t i = 0; i < f.peers.size(); ++i) {
        check(answered.at("peers").at(i).at("queued") == 0, "answer tokens fed back into peer reasoning");
        check(answered.at("peers").at(i).at("imported") == before_answer.at("peers").at(i).at("imported"),
                "answer imported peer answer tokens");
    }
    f.session->command("message", "question three");
    f.wait_mode("thinking");
    auto a4 = f.peers[0]->request(3);
    auto b4 = f.peers[1]->request(3);
    check(count(a4->prompt, 301) == 1 && !count(a4->prompt, 401), "A's next user turn mixed answer streams");
    check(count(b4->prompt, 401) == 1 && !count(b4->prompt, 301), "B's next user turn mixed answer streams");
    check(ends_with(a4->prompt, {301, 3, 20, 21, 1}), "answered turn got a second reasoning-close marker");
    a4->stale_on_cancel({901});
    b4->stale_on_cancel({902});
    f.session->command("reset");
    const auto reset = f.wait_mode("idle");
    check(reset.at("exchanges") == 0, "reset retained the previous exchange count");
    for (size_t i = 0; i < f.peers.size(); ++i) {
        for (const auto * field : {"tokens", "queued", "generated", "imported", "forced_splices"}) {
            check(reset.at("peers").at(i).at(field) == 0, std::string("reset retained ") + field);
        }
        check(!reset.at("peers").at(i).at("waiting").get<bool>(), "reset retained a completed paragraph");
    }
    for (const auto & request : {a4, b4}) {
        std::lock_guard<std::mutex> lock(request->mutex);
        check(request->returned && !request->stale_accepted, "old epoch accepted tokens after reset");
    }
    f.start("new session");
    check(f.peers[0]->request(4)->prompt == std::vector<llama_token>({1, 10, 11}), "reset contaminated A's new prompt");
    check(f.peers[1]->request(4)->prompt == std::vector<llama_token>({1, 10, 11}), "reset contaminated B's new prompt");
    f.pause();
}

static void test_paragraph_rendezvous() {
    fixture f(4, 256, true);
    f.start();
    auto a1 = f.peers[0]->request(0);
    auto b1 = f.peers[1]->request(0);
    for (const auto & request : {a1, b1}) {
        check(request->parameters.at("splice").at("sentence_after") == 2, "sentence fallback was not passed to server");
        check(request->parameters.at("n_predict") == 4, "paragraph hard ceiling was not passed to server");
    }
    a1->send({101, 102});
    b1->send({201});
    a1->wait_consumed(1);
    b1->wait_consumed(1);
    a1->finish();
    const auto waiting = f.wait_peer(0);
    check(waiting.at("exchanges") == 0, "incomplete pair counted as an exchange");
    for (size_t i = 0; i < f.peers.size(); ++i) {
        check(waiting.at("peers").at(i).at("imported") == 0, "unfinished peer paragraph was spliced");
        check(f.peers[i]->request_count() == 1, "peer generated beyond its pending paragraph");
    }
    check(waiting.at("peers").at(0).at("queued") == 1, "incoming stream was not buffered while awaiting peer boundary");
    b1->send({202});
    b1->finish("limit", "sentence");
    auto a2 = f.peers[0]->request(1);
    auto b2 = f.peers[1]->request(1);
    check(a2->prompt == std::vector<llama_token>({1, 10, 11, 101, 102, 201, 202}),
            "A did not receive B's complete paragraph after its own boundary");
    check(b2->prompt == std::vector<llama_token>({1, 10, 11, 201, 202, 101, 102}),
            "B did not receive A's complete paragraph after its own boundary");
    const auto exchanged = f.session->state();
    check(exchanged.at("exchanges") == 1, "pair was not counted as exactly one exchange");
    check(exchanged.at("peers").at(0).at("boundary") == "paragraph" &&
            exchanged.at("peers").at(1).at("boundary") == "sentence", "splice reasons were not retained");
    a2->send({103, 104, 105, 106});
    a2->finish("limit", "limit");
    f.wait_peer(0);
    check(f.peers[0]->request_count() == 2, "forced boundary bypassed rendezvous");
    b2->send({203});
    b2->finish();
    auto a3 = f.peers[0]->request(2);
    auto b3 = f.peers[1]->request(2);
    for (const auto & request : {a3, b3}) {
        for (llama_token token : {101, 102, 103, 104, 105, 106, 201, 202, 203}) {
            check(count(request->prompt, token) == 1, "rendezvous lost, echoed or replayed a reasoning token");
        }
    }
    const auto forced = f.session->state();
    check(forced.at("exchanges") == 2, "forced boundary did not complete the next exchange");
    check(forced.at("peers").at(0).at("forced_splices") == 1 &&
            forced.at("peers").at(1).at("forced_splices") == 0, "forced splice count is incorrect");
    f.pause();
}

static void test_paragraph_pause_and_resume() {
    fixture f(4, 256, true);
    f.start();
    auto a1 = f.peers[0]->request(0);
    auto b1 = f.peers[1]->request(0);
    a1->send({101});
    a1->finish();
    f.wait_peer(0);
    b1->send({201});
    b1->wait_consumed(1);
    f.session->command("pause");
    b1->send({202});
    b1->finish();
    const auto paused = f.wait_mode("paused");
    check(paused.at("exchanges") == 0, "pause unexpectedly spliced a pending pair");
    for (size_t i = 0; i < f.peers.size(); ++i) {
        check(paused.at("peers").at(i).at("waiting").get<bool>(), "pause lost completed paragraph state");
        check(paused.at("peers").at(i).at("imported") == 0, "pause consumed pending peer tokens");
        check(f.peers[i]->request_count() == 1, "paused peer started a new paragraph");
    }
    f.session->command("resume");
    f.wait_mode("thinking");
    auto a2 = f.peers[0]->request(1);
    auto b2 = f.peers[1]->request(1);
    for (const auto & request : {a2, b2}) {
        for (llama_token token : {101, 201, 202}) {
            check(count(request->prompt, token) == 1, "resume did not splice the pending pair exactly once");
        }
    }
    check(f.session->state().at("exchanges") == 1, "resume did not count the pending exchange");
    f.pause();
}

static void test_invalid_paragraph_result() {
    for (const std::string boundary : {"", "invalid", "limit", "error"}) {
        fixture f(4, 256, true);
        f.start();
        auto a1 = f.peers[0]->request(0);
        auto b1 = f.peers[1]->request(0);
        a1->send({101});
        a1->finish();
        f.wait_peer(0);
        b1->send({201});
        if (boundary == "error") {
            b1->finish_result({{"type", "error"}, {"message", "test transport failure"}});
        } else {
            b1->finish("limit", boundary);
        }
        const auto failed = f.wait_mode("error");
        check(failed.at("exchanges") == 0, "invalid completion was counted as a complete exchange");
        for (size_t i = 0; i < f.peers.size(); ++i) {
            check(failed.at("peers").at(i).at("imported") == 0, "invalid completion spliced partial reasoning");
            check(f.peers[i]->request_count() == 1, "generation continued after an invalid completion");
        }
        f.session->command("reset");
        const auto reset = f.wait_mode("idle");
        check(reset.at("exchanges") == 0, "reset retained exchange count after failure");
        for (const auto & peer : reset.at("peers")) {
            check(!peer.at("waiting").get<bool>() && peer.at("queued") == 0,
                    "reset retained failed exchange state");
        }
    }
}

static void test_paragraph_failure_during_command() {
    for (const std::string action : {"message", "answer"}) {
        fixture f(4, 256, true);
        f.start();
        auto a1 = f.peers[0]->request(0);
        auto b1 = f.peers[1]->request(0);
        a1->send({101});
        a1->finish();
        f.wait_peer(0);
        b1->send({201});
        b1->wait_consumed(1);
        f.session->command(action, action == "message" ? "intervention" : "");
        b1->finish("limit", "");
        const auto failed = f.wait_mode("error");
        check(failed.at("round") == 1 && failed.at("exchanges") == 0,
                "pending command committed a failed exchange");
        for (size_t i = 0; i < f.peers.size(); ++i) {
            check(failed.at("peers").at(i).at("imported") == 0, "pending command imported incomplete reasoning");
            check(f.peers[i]->request_count() == 1, "pending command started generation after failure");
            std::lock_guard<std::mutex> lock(f.peers[i]->mutex);
            check(f.peers[i]->user_texts.empty(), "failed pending message reached template preparation");
        }
        bool rejected = false;
        try {
            f.session->command("message", "try to continue");
        } catch (const std::logic_error &) {
            rejected = true;
        }
        check(rejected, "failed exchange accepted a new message without reset");
        f.session->command("reset");
        f.wait_mode("idle");
    }
}

static std::shared_ptr<fake_request> start_private_turn(fixture & f, size_t index) {
    auto reasoning = f.peers[index]->request(0);
    reasoning->send({static_cast<llama_token>(101 + index), 2});
    reasoning->finish("limit", "tool");
    auto request = f.peers[index]->request(1);
    check(request->prompt.back() == 2, "private assistant turn did not follow the reasoning-close token");
    check(request->parameters.at("n_predict") == 8, "private assistant turn has the wrong token budget");
    check(!request->parameters.contains("splice"), "private assistant turn inherited paragraph stopping");
    return request;
}

static void test_tool_result_is_private() {
    fixture f(4, 256, true, true);
    f.start();
    auto b1 = f.peers[1]->request(0);
    b1->send({201});
    b1->finish();
    f.wait_peer(1);
    auto a2 = start_private_turn(f, 0);
    a2->send({501, 3});
    a2->finish("eos", "");
    auto call = f.tools[0]->request(0);
    check(call->name == "calculator" && call->arguments == crossthink_json{{"expression", "6*7"}},
            "native tool arguments did not reach MCP intact");
    check(f.tools[1]->call_count() == 0, "peer executed the other model's tool call");
    const auto blocked = f.session->state();
    check(blocked.at("peers").at(1).at("queued") == 0, "incomplete pre-tool thought remained queued for the peer");
    check(blocked.at("peers").at(0).at("active").get<bool>(), "tool call released the active-turn guard");
    check(f.peers[1]->request_count() == 1, "waiting peer continued before tool user's paragraph");
    const crossthink_json result = {
        {"content", {{{"type", "text"}, {"text", "42"}}}},
        {"structuredContent", {{"value", 42}}}, {"isError", false},
    };
    call->finish(result);
    auto a3 = f.peers[0]->request(2);
    check(ends_with(a3->prompt, {3, 40, 1}), "tool response was not followed by a fresh thinking prefix");
    check(count(a3->prompt, 501) == 1 && count(a3->prompt, 3) == 1,
            "private assistant body was lost or its terminator was duplicated");
    {
        std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
        check(f.peers[0]->configured_tools == f.tools[0]->tools(), "discovered tools were not applied to the template");
        check(f.peers[0]->parsed_turns == std::vector<std::vector<llama_token>>{{2, 501, 3}},
                "native parser received an incomplete or contaminated assistant turn");
        check(f.peers[0]->result_assistants.size() == 1, "tool result template received multiple assistant turns");
        check(f.peers[0]->result_messages.size() == 1, "tool response was omitted or delivered twice");
        const auto & assistant_call = f.peers[0]->result_assistants.front().at("tool_calls").at(0);
        const auto & message = f.peers[0]->result_messages.front().at(0);
        check(assistant_call.at("id") == "call_calculator", "native tool-call ID was replaced during execution");
        check(assistant_call.at("function") == f.peers[0]->parsed_assistant.at("tool_calls").at(0).at("function"),
                "tool result template received a different function call");
        check(message.at("role") == "tool" && message.at("name") == "calculator" &&
                message.at("tool_call_id") == assistant_call.at("id"), "tool result was not matched to its native call ID");
        check(crossthink_json::parse(message.at("content").get<std::string>()) == result,
                "tool content or structured MCP result was lost before template rendering");
    }
    a3->send({102});
    a3->finish();
    auto a4 = f.peers[0]->request(3);
    auto b2 = f.peers[1]->request(1);
    check(b2->prompt == std::vector<llama_token>({1, 10, 11, 201, 102}),
            "private tool turn or incomplete reasoning leaked into the peer's paragraph splice");
    check(count(a4->prompt, 201) == 1 && count(a4->prompt, 101) == 1 && count(a4->prompt, 40) == 1,
            "tool user lost its private context or imported the peer twice");
    check(f.session->state().at("exchanges") == 1, "tool turn itself was counted as a paragraph exchange");
    f.pause();
}

static void test_reset_cancels_tool_call() {
    fixture f(4, 256, true, true);
    f.start();
    auto b1 = f.peers[1]->request(0);
    b1->send({201});
    b1->finish();
    f.wait_peer(1);
    auto a2 = start_private_turn(f, 0);
    a2->send({501, 3});
    a2->finish("eos", "");
    auto call = f.tools[0]->request(0);
    f.session->command("reset");
    const auto reset = f.wait_mode("idle");
    {
        std::lock_guard<std::mutex> lock(call->mutex);
        check(call->cancelled && call->returned, "reset did not interrupt the pending MCP call");
    }
    {
        std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
        check(f.peers[0]->result_messages.empty(), "cancelled MCP result reached the prompt template");
    }
    for (const auto & peer : reset.at("peers")) {
        check(peer.at("tool_calls") == 0 && peer.at("tool_status") == "", "reset retained tool-call state");
        check(peer.at("tokens") == 0 && peer.at("queued") == 0, "reset retained private tool tokens");
    }
    f.start("new question");
    check(f.peers[0]->request(2)->prompt == std::vector<llama_token>({1, 10, 11}),
            "late tool response contaminated A's new session");
    check(f.peers[1]->request(1)->prompt == std::vector<llama_token>({1, 10, 11}),
            "late tool response contaminated B's new session");
    f.pause();
}

static void test_incomplete_tool_turn_does_not_execute() {
    for (const std::string failure : {"limit", "truncated", "error"}) {
        fixture f(4, 256, true, true);
        f.start();
        auto b1 = f.peers[1]->request(0);
        b1->send({201});
        b1->finish();
        f.wait_peer(1);
        auto a2 = start_private_turn(f, 0);
        a2->send(failure == "limit" ? std::vector<llama_token>{501} : std::vector<llama_token>{501, 3});
        if (failure == "error") {
            a2->finish_result({{"type", "error"}, {"message", "broken stream"}});
        } else {
            a2->finish_result({{"type", "done"}, {"stop_type", failure == "limit" ? "limit" : "eos"},
                {"truncated", failure == "truncated"}});
        }
        f.wait_mode("error");
        check(f.tools[0]->call_count() == 0 && f.tools[1]->call_count() == 0,
                "invalid private assistant completion executed a tool");
        std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
        check(f.peers[0]->parsed_turns.empty() && f.peers[0]->result_messages.empty(),
                "invalid private assistant completion reached the tool parser or template");
    }
}

static void test_natural_final_answer_with_tools() {
    fixture f(4, 256, true, true);
    for (size_t i = 0; i < f.peers.size(); ++i) {
        std::lock_guard<std::mutex> lock(f.peers[i]->mutex);
        f.peers[i]->parsed_assistant = {{"role", "assistant"}, {"content", i ? "B answer" : "A answer"}};
    }
    f.start();
    auto b1 = f.peers[1]->request(0);
    b1->send({201});
    b1->finish();
    f.wait_peer(1);
    auto a2 = start_private_turn(f, 0);
    a2->send({501, 3});
    a2->finish("eos", "");
    auto b2 = f.peers[1]->request(1);
    check(b2->prompt.back() == 2, "other peer's final answer did not close its reasoning");
    check(!count(b2->prompt, 101) && !count(b2->prompt, 501), "natural final answer was spliced into peer reasoning");
    b2->send({601, 3});
    b2->finish("eos", "");
    f.wait_mode("answered");
    check(f.tools[0]->call_count() == 0 && f.tools[1]->call_count() == 0, "ordinary final answer executed an MCP tool");
    std::array<bool, 2> answer_seen = {};
    for (const auto & event : f.session->events_after(0)) {
        if (event.value("type", std::string()) == "answer") {
            if (event.value("peer", std::string()) == "A" && event.value("text", std::string()) == "A answer") {
                answer_seen[0] = true;
            }
            if (event.value("peer", std::string()) == "B" && event.value("text", std::string()) == "B answer") {
                answer_seen[1] = true;
            }
        }
    }
    check(answer_seen[0] && answer_seen[1], "native final answer content did not reach both answer panes");
}

static void test_reset_during_tool_template_operation() {
    for (bool rendering : {false, true}) {
        fixture f(4, 256, true, true);
        auto gate = std::make_shared<fake_operation>();
        gate->throw_on_cancel = true;
        {
            std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
            (rendering ? f.peers[0]->results_gate : f.peers[0]->parse_gate) = gate;
        }
        f.start();
        auto b1 = f.peers[1]->request(0);
        b1->send({201});
        b1->finish();
        f.wait_peer(1);
        auto a2 = start_private_turn(f, 0);
        a2->send({501, 3});
        a2->finish("eos", "");
        if (rendering) {
            f.tools[0]->request(0)->finish({{"content", {{{"type", "text"}, {"text", "42"}}}}});
        }
        gate->wait_entered();
        f.session->command("reset", "replacement question");
        const auto replacement = f.wait_mode("thinking");
        check(replacement.at("error") == "", "stale template exception failed the replacement session");
        check(f.peers[0]->request(2)->prompt == std::vector<llama_token>({1, 10, 11}) &&
                f.peers[1]->request(1)->prompt == std::vector<llama_token>({1, 10, 11}),
                "cancelled template operation contaminated a replacement prompt");
        check(f.tools[0]->call_count() == (rendering ? 1 : 0), "cancelled parser dispatched a tool call");
        {
            std::lock_guard<std::mutex> lock(gate->mutex);
            check(gate->cancelled, "reset did not cancel the pending template operation");
        }
        f.pause();
    }
}

static void test_peer_error_survives_private_final_answer() {
    fixture f(4, 256, true, true);
    auto gate = std::make_shared<fake_operation>();
    {
        std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
        f.peers[0]->parse_gate = gate;
        f.peers[0]->parsed_assistant = {{"role", "assistant"}, {"content", "A final answer"}};
    }
    f.start();
    auto b1 = f.peers[1]->request(0);
    b1->send({201});
    b1->wait_consumed(1);
    auto a2 = start_private_turn(f, 0);
    a2->send({501, 3});
    a2->finish("eos", "");
    gate->wait_entered();
    b1->finish_result({{"type", "error"}, {"message", "peer stream failed"}});
    const auto failed = f.wait_mode("error");
    gate->release();
    const auto finished = f.wait_state([](const crossthink_json & state) {
        return !state.at("peers").at(0).at("active").get<bool>();
    }, "finish private answer after peer failure", true);
    check(finished.at("mode") == "error" && finished.at("error") == failed.at("error"),
            "successful private answer cleared its peer's error and restarted generation");
    check(f.peers[0]->request_count() == 2 && f.peers[1]->request_count() == 1,
            "a private final answer started another request after peer failure");
}


static std::vector<crossthink_json> events_of(fixture & f, const std::string & type,
        const std::string & peer = {}) {
    std::vector<crossthink_json> result;
    for (const auto & event : f.session->events_after(0)) {
        if (event.value("type", std::string()) == type &&
                (peer.empty() || event.value("peer", std::string()) == peer)) {
            result.push_back(event);
        }
    }
    return result;
}

static bool request_is_active(const std::shared_ptr<fake_request> & request) {
    std::lock_guard<std::mutex> lock(request->mutex);
    return !request->returned && !request->cancelled;
}

static void finish_thought_command(const std::shared_ptr<fake_request> & request,
        const std::string & command, llama_token token, const std::string & payload = {}) {
    const std::string text = command == "send" ? "<ct:send>" + payload + "</ct:send>" : "<ct:" + command + "/>";
    request->send({token}, text);
    crossthink_json result = {{"type", "done"}, {"stop_type", "limit"}, {"truncated", false},
        {"splice_boundary", "thought_command"}, {"thought_command", command}};
    if (command == "send") {
        result["thought_payload"] = payload;
    }
    request->finish_result(std::move(result));
}

static void finish_thought_answer(fixture & f, size_t index, size_t request_index,
        const std::string & answer = "Final answer") {
    {
        std::lock_guard<std::mutex> lock(f.peers[index]->mutex);
        f.peers[index]->parsed_assistant = {{"role", "assistant"}, {"content", answer}};
    }
    auto reasoning = f.peers[index]->request(request_index);
    reasoning->send({2}, "</think>");
    reasoning->finish("limit", "tool");
    auto body = f.peers[index]->request(request_index + 1);
    check(!body->parameters.contains("splice"), "final answer retained reasoning command detection");
    body->send({static_cast<llama_token>(700 + index), 3}, answer);
    body->finish("eos", "");
    f.wait_state([&](const crossthink_json & state) {
        return state.at("peers").at(index).at("answer_done").get<bool>();
    }, "finish individual final answer");
}

static void test_thought_mode_parallel_and_catalogue() {
    check(crossthink_options{}.telepathy, "parallel thought tools are not the default mode");
    fixture f(4, 32768, true, false, true);
    for (size_t index = 0; index < f.peers.size(); ++index) {
        std::lock_guard<std::mutex> lock(f.peers[index]->mutex);
        check(f.peers[index]->configured_tools.empty(), "obsolete native link tool remained in the tool catalogue");
        check(f.peers[index]->configured_peer == (index ? "B" : "A"), "agent identity was not configured");
    }
    f.start();
    auto a = f.peers[0]->request(0);
    auto b = f.peers[1]->request(0);
    for (const auto & request : {a, b}) {
        const auto & splice = request->parameters.at("splice");
        check(splice.at("thought_commands").get<bool>() && splice.at("stop_on_think_close").get<bool>(),
                "reasoning request omitted thought commands or native closing detection");
        check(!splice.contains("token_after") && !splice.contains("sentence_after"),
                "independent reasoning retained periodic shared-floor stops");
        check(request->parameters.at("n_predict").get<int>() > 256,
                "independent reasoning retained the small private generation quantum");
        check(!request->parameters.contains("grammar"), "independent reasoning retained the shared speech grammar");
    }
    a->send({101, 102}, "A independently reasons.");
    b->send({201, 202, 203}, "B independently reasons.");
    a->wait_consumed(1);
    b->wait_consumed(1);
    const auto state = f.session->state();
    for (const auto & peer : state.at("peers")) {
        check(peer.at("active").get<bool>(), "one model lost its stream while the other decoded");
        check(peer.at("imported") == 0 && peer.at("queued") == 0, "unsolicited reasoning was queued for the peer");
    }
    check(f.peers[0]->request_count() == 1 && f.peers[1]->request_count() == 1,
            "concurrent reasoning caused request restarts");
    f.pause();
    f.session->command("resume");
    f.wait_mode("thinking");
    auto a2 = f.peers[0]->request(1);
    auto b2 = f.peers[1]->request(1);
    check(count(a2->prompt, 101) == 1 && count(a2->prompt, 102) == 1 && !count(a2->prompt, 201),
            "A's private reasoning was lost, duplicated or mixed on resume");
    check(count(b2->prompt, 201) == 1 && count(b2->prompt, 203) == 1 && !count(b2->prompt, 101),
            "B's private reasoning was lost, duplicated or mixed on resume");
    f.pause();
}

static void test_thought_peek_is_incremental_and_nonintrusive() {
    fixture f(4, 32768, true, false, true);
    f.start();
    auto a0 = f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    const std::string first = "A proof starts with a partial wor";
    const std::string second = "d.\n\n```cpp\nint x = 1;\n```\n$\\alpha + 1$";
    b0->send({201, 202}, first);
    b0->wait_consumed(1);
    finish_thought_command(a0, "peek", 101);
    auto a1 = f.peers[0]->request(1);
    auto captures = events_of(f, "thought_capture", "A");
    check(captures.size() == 1 && captures[0].at("text") == first && captures[0].at("source") == "B",
            "first peek was not a verbatim snapshot of the ongoing peer thought");
    check(captures[0].at("offset") == 0, "first peek did not start at the beginning of the thought");
    check(count(a1->prompt, 201) == 1 && count(a1->prompt, 202) == 1,
            "peek did not inject the exact captured source token IDs");
    check(a1->prompt.back() != 2 && request_is_active(b0) && f.peers[1]->request_count() == 1,
            "peek closed reasoning or disturbed the source's ongoing generation");
    const auto first_results = events_of(f, "thought_result", "A");
    check(first_results.size() == 1 && first_results[0].at("text").get<std::string>().find(first) != std::string::npos,
            "peek did not expose its marked result to the receiver's reasoning view");
    b0->send({203, 204}, second);
    b0->wait_consumed(2);
    finish_thought_command(a1, "peek", 102);
    auto a2 = f.peers[0]->request(2);
    captures = events_of(f, "thought_capture", "A");
    check(captures.size() == 2 && captures[1].at("text") == second &&
            captures[1].at("offset") == first.size() && captures[1].at("turn") == captures[0].at("turn"),
            "successive peek replayed, skipped or reformatted the source text");
    for (llama_token token : {201, 202, 203, 204}) {
        check(count(a2->prompt, token) == 1, "incremental peeks duplicated or omitted captured tokens");
    }
    finish_thought_command(a2, "peek", 103);
    auto a3 = f.peers[0]->request(3);
    captures = events_of(f, "thought_capture", "A");
    check(captures.size() == 3 && captures[2].at("text") == "" && captures[2].at("offset") == first.size() + second.size(),
            "peek without new source text did not preserve the cursor");
    for (llama_token token : {201, 202, 203, 204}) {
        check(count(a3->prompt, token) == 1, "empty incremental peek replayed an old capture");
    }
    check(request_is_active(b0) && f.peers[1]->request_count() == 1,
            "repeated reads interrupted or restarted the source model");
    f.pause();
}

static void test_thought_peek_latest_turn_and_cursor_reset() {
    fixture f(4, 32768, true, false, true);
    f.start();
    auto a0 = f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    b0->send({201}, "Completed reasoning.");
    b0->wait_consumed(1);
    finish_thought_answer(f, 1, 0, "Private final answer");
    check(request_is_active(a0), "peer's final answer interrupted independent reasoning");
    finish_thought_command(a0, "peek", 101);
    auto a1 = f.peers[0]->request(1);
    const auto before = events_of(f, "thought_capture", "A");
    check(before.size() == 1 && before[0].at("text") == "Completed reasoning.",
            "peek did not retain the most recent completed reasoning turn");
    check(!count(a1->prompt, 701) && !count(a1->prompt, 2), "peek imported a final answer or reasoning-close token");
    f.session->command("message", "A new question for B only", "B");
    f.wait_mode("thinking");
    auto b2 = f.peers[1]->request(2);
    check(request_is_active(a1), "starting the peer's next turn interrupted the requester");
    b2->send({202}, "Fresh thought from its beginning.");
    b2->wait_consumed(1);
    finish_thought_command(a1, "peek", 102);
    auto a2 = f.peers[0]->request(2);
    const auto after = events_of(f, "thought_capture", "A");
    check(after.size() == 2 && after[1].at("text") == "Fresh thought from its beginning." &&
            after[1].at("offset") == 0 && after[1].at("turn") != before[0].at("turn"),
            "peek cursor did not reset when its source began a new reasoning turn");
    check(count(a2->prompt, 201) == 1 && count(a2->prompt, 202) == 1,
            "peeking a new turn replayed old source tokens or omitted new ones");
    f.pause();
}

static void test_thought_send_and_inbox_are_opt_in() {
    fixture f(4, 32768, true, false, true);
    f.start();
    auto a0 = f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    const std::string payload = "mail sentinel: </think><ct:peek/><ct:inbox/>\n```cpp\nreturn 42;\n```";
    finish_thought_command(a0, "send", 101, payload);
    auto a1 = f.peers[0]->request(1);
    auto messages = events_of(f, "thought_message");
    check(messages.size() == 1 && messages[0].at("from") == "A" && messages[0].at("to") == "B" &&
            messages[0].at("text") == payload && messages[0].at("status") == "sent",
            "sending did not preserve message text and route it to the peer mailbox");
    check(request_is_active(b0) && f.peers[1]->request_count() == 1,
            "sending a thought interrupted its recipient");
    check(f.session->state().at("peers").at(1).at("imported") == 0,
            "unread mail was injected into the recipient's thought stream");
    finish_thought_command(b0, "inbox", 201);
    auto b1 = f.peers[1]->request(1);
    messages = events_of(f, "thought_message");
    check(messages.size() == 2 && messages[1].at("status") == "received" &&
            messages[1].at("message_id") == messages[0].at("message_id") && messages[1].at("text") == payload,
            "inbox did not acknowledge exactly the delivered message");
    auto results = events_of(f, "thought_result", "B");
    check(results.size() == 1 && results[0].at("text").get<std::string>().find(payload) != std::string::npos,
            "inbox result reformatted or omitted its message");
    const auto injected = f.peers[1]->text_tokens(results[0].at("text").get<std::string>());
    check(ends_with(b1->prompt, injected), "inbox result was not injected contiguously into the ongoing reasoning");
    check(count(b1->prompt, 2) == 0 && request_is_active(a1),
            "inbox text promoted a control marker or interrupted its sender");
    const size_t previous_prompt_size = b1->prompt.size();
    finish_thought_command(b1, "inbox", 202);
    auto b2 = f.peers[1]->request(2);
    messages = events_of(f, "thought_message");
    check(messages.size() == 2, "reading an empty inbox redelivered or executed quoted command text");
    results = events_of(f, "thought_result", "B");
    check(results.size() == 2 && results[1].at("text").get<std::string>().find(payload) == std::string::npos,
            "second inbox read replayed the delivered body");
    check(b2->prompt.size() > previous_prompt_size, "empty inbox did not return an explicit response");
    finish_thought_command(a1, "peek", 102);
    f.peers[0]->request(2);
    const auto capture = events_of(f, "thought_capture", "A");
    check(capture.size() == 1 && capture[0].at("text").get<std::string>().find("mail sentinel") == std::string::npos,
            "peek echoed previously injected mailbox contents as the source's own reasoning");
    f.pause();
}

static void test_thought_targeted_message_preserves_other_stream() {
    fixture f(4, 32768, true, false, true);
    f.start("initial task");
    auto a0 = f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    a0->send({101}, "A already thought this.");
    b0->send({201}, "B is still thinking.");
    a0->wait_consumed(1);
    b0->wait_consumed(1);
    a0->stale_on_cancel({901});
    f.session->command("message", "Only A should see this", "A");
    f.wait_mode("thinking");
    auto a1 = f.peers[0]->request(1);
    check(ends_with(a1->prompt, {2, 3, 20, 21, 1}) && count(a1->prompt, 101) == 1 && !count(a1->prompt, 901),
            "targeted intervention lost local tokens, accepted stale tokens, or omitted its user template");
    {
        std::lock_guard<std::mutex> lock(a0->mutex);
        check(a0->cancelled && a0->returned && !a0->stale_accepted, "targeted intervention did not cancel only its old request");
    }
    check(request_is_active(b0) && f.peers[1]->request_count() == 1,
            "targeting A cancelled or restarted B's ongoing generation");
    b0->send({202}, " More uninterrupted work.");
    b0->wait_consumed(2);
    {
        std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
        check(f.peers[0]->user_texts == std::vector<std::string>{"Only A should see this"},
                "targeted message did not reach A exactly once");
    }
    {
        std::lock_guard<std::mutex> lock(f.peers[1]->mutex);
        check(f.peers[1]->user_texts.empty(), "targeted message was rendered into the other agent's template");
    }
    check(f.session->state().at("peers").at(1).at("generated") == 2,
            "targeting A reset B's token counter or rejected its still-valid generation epoch");
    f.session->command("message", "Both should see this", "both");
    f.wait_mode("thinking");
    f.peers[0]->request(2);
    f.peers[1]->request(1);
    for (auto * peer : f.peers) {
        std::lock_guard<std::mutex> lock(peer->mutex);
        check(peer->user_texts.back() == "Both should see this", "broadcast message did not reach both templates");
    }
    f.pause();
}

static void test_thought_native_mcp_does_not_serialize_peer() {
    fixture f(4, 32768, true, true, true);
    f.start();
    auto b0 = f.peers[1]->request(0);
    b0->send({201}, "B is generating throughout A's tool call.");
    b0->wait_consumed(1);
    auto a1 = start_private_turn(f, 0);
    a1->send({501, 3}, "calculator call");
    a1->finish("eos", "");
    auto call = f.tools[0]->request(0);
    check(request_is_active(b0) && f.peers[1]->request_count() == 1,
            "A's native tool tail or MCP call serialized the other model");
    b0->send({202}, " Still independent.");
    b0->wait_consumed(2);
    check(f.session->state().at("peers").at(1).at("generated") == 2,
            "peer could not decode while an MCP request was pending");
    call->finish({{"content", {{{"type", "text"}, {"text", "42"}}}}});
    auto a2 = f.peers[0]->request(2);
    check(ends_with(a2->prompt, {3, 40, 1}) && !count(a2->prompt, 201),
            "MCP result lost its native template or imported unsolicited peer text");
    check(request_is_active(b0) && f.peers[1]->request_count() == 1,
            "resuming after an MCP result restarted the peer's stream");
    a2->send({102}, "Fresh reasoning after calculator result.");
    a2->wait_consumed(1);
    finish_thought_command(b0, "peek", 203);
    auto b1 = f.peers[1]->request(1);
    const auto captures = events_of(f, "thought_capture", "B");
    check(captures.size() == 1 && captures[0].at("text") == "Fresh reasoning after calculator result." &&
            count(b1->prompt, 102) == 1 && !count(b1->prompt, 101) && !count(b1->prompt, 501) && !count(b1->prompt, 40),
            "post-tool peek included a prior reasoning turn, native call, or tool response");
    f.pause();
}

static void test_thought_answers_never_execute_commands() {
    fixture f(4, 32768, true, false, true);
    f.start();
    auto b0 = f.peers[1]->request(0);
    const std::string answer = "Literal documentation: <ct:send>do not deliver this</ct:send> <ct:peek/> <ct:inbox/>";
    finish_thought_answer(f, 0, 0, answer);
    check(request_is_active(b0) && f.peers[1]->request_count() == 1,
            "finishing A forced B to stop its independent reasoning");
    check(events_of(f, "thought_message").empty() && events_of(f, "thought_result").empty(),
            "command syntax in a final answer triggered a thought tool");
    const auto answers = events_of(f, "answer", "A");
    check(answers.size() == 1 && answers[0].at("text") == answer,
            "native final answer content did not remain intact");
    finish_thought_answer(f, 1, 0, "B final answer");
    f.wait_mode("answered");
}

static void test_thought_reset_discards_stale_streams_and_mail() {
    fixture f(4, 32768, true, false, true);
    f.start();
    auto a0 = f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    b0->send({201}, "old private reasoning");
    b0->wait_consumed(1);
    finish_thought_command(a0, "send", 101, "old unread mail");
    auto a1 = f.peers[0]->request(1);
    a1->stale_on_cancel({901});
    b0->stale_on_cancel({902});
    f.session->command("reset");
    const auto reset = f.wait_mode("idle");
    for (const auto & peer : reset.at("peers")) {
        check(peer.at("tokens") == 0 && peer.at("generated") == 0 && peer.at("imported") == 0 &&
                peer.at("thinking_tokens") == 0, "reset retained context or rolling token counters");
    }
    for (const auto & request : {a1, b0}) {
        std::lock_guard<std::mutex> lock(request->mutex);
        check(request->returned && !request->stale_accepted, "reset accepted packets from an old generation epoch");
    }
    f.start("fresh question");
    auto a2 = f.peers[0]->request(2);
    auto b1 = f.peers[1]->request(1);
    check(a2->prompt == std::vector<llama_token>({1, 10, 11}) && b1->prompt == a2->prompt,
            "old captured or injected text contaminated the new session");
    finish_thought_command(b1, "inbox", 202);
    f.peers[1]->request(2);
    const auto results = events_of(f, "thought_result", "B");
    check(!results.empty() && results.back().at("text").get<std::string>().find("old unread mail") == std::string::npos,
            "reset failed to empty the reasoning-time mailbox");
    finish_thought_command(a2, "peek", 102);
    f.peers[0]->request(3);
    const auto captures = events_of(f, "thought_capture", "A");
    check(!captures.empty() && captures.back().at("text").get<std::string>().find("old private reasoning") == std::string::npos,
            "reset retained a source's prior thought capture");
    f.pause();
}

static void test_thought_peek_defers_incomplete_utf8_packets() {
    fixture f(4, 32768, true, false, true);
    f.start();
    auto a0 = f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    b0->send({201}, "");
    b0->wait_consumed(1);
    finish_thought_command(a0, "peek", 101);
    auto a1 = f.peers[0]->request(1);
    auto captures = events_of(f, "thought_capture", "A");
    check(captures.size() == 1 && captures[0].at("text") == "" && !count(a1->prompt, 201),
            "peek injected an incomplete UTF-8 token sequence before it was displayable");
    b0->send({202}, "é");
    b0->wait_consumed(2);
    finish_thought_command(a1, "peek", 102);
    auto a2 = f.peers[0]->request(2);
    captures = events_of(f, "thought_capture", "A");
    check(captures.size() == 2 && captures[1].at("text") == "é" && captures[1].at("offset") == 0,
            "UTF-8 buffering skipped bytes or advanced the peek cursor prematurely");
    check(count(a2->prompt, 201) == 1 && count(a2->prompt, 202) == 1,
            "completed UTF-8 snapshot omitted or duplicated its earlier raw token IDs");
    f.pause();
}

static void test_thought_cancellation_during_result_tokenization() {
    for (bool reset : {false, true}) {
        fixture f(4, 32768, true, false, true);
        f.start();
        auto a0 = f.peers[0]->request(0);
        auto b0 = f.peers[1]->request(0);
        b0->send({201}, "Capture that must not arrive after cancellation.");
        b0->wait_consumed(1);
        auto gate = std::make_shared<fake_operation>();
        gate->throw_on_cancel = true;
        {
            std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
            f.peers[0]->literal_gate = gate;
        }
        finish_thought_command(a0, "peek", 101);
        gate->wait_entered();
        if (reset) {
            f.session->command("reset", "Replacement after reset");
        } else {
            f.session->command("message", "Replacement for A only", "A");
        }
        f.wait_mode("thinking");
        auto a1 = f.peers[0]->request(1);
        check(!count(a1->prompt, 201), "cancelled tokenization committed its stale thought capture");
        check(events_of(f, "thought_result", "A").empty(), "cancelled capture emitted a stale injected result");
        if (reset) {
            f.peers[1]->request(1);
        } else {
            check(request_is_active(b0) && f.peers[1]->request_count() == 1,
                    "cancelling A's result tokenization interrupted B's still-active stream");
            b0->send({202}, " B keeps decoding.");
            b0->wait_consumed(2);
        }
        f.pause();
    }
}

static void test_thought_pause_waits_for_native_tool_boundary() {
    for (bool during_call : {false, true}) {
        fixture f(4, 32768, true, true, true);
        f.start();
        auto b0 = f.peers[1]->request(0);
        auto a1 = start_private_turn(f, 0);
        std::shared_ptr<fake_tool_call> call;
        if (during_call) {
            a1->send({501, 3}, "calculator call");
            a1->finish("eos", "");
            call = f.tools[0]->request(0);
        } else {
            a1->send({501}, "partial native calculator call");
            a1->wait_consumed(1);
        }
        f.session->command("pause");
        if (!during_call) {
            check(request_is_active(a1), "pause cancelled an incomplete native tool tail");
            a1->send({3}, "");
            a1->finish("eos", "");
            call = f.tools[0]->request(0);
        }
        {
            std::lock_guard<std::mutex> lock(call->mutex);
            check(!call->cancelled, "pause cancelled an already-dispatched MCP side effect");
        }
        call->finish({{"content", {{{"type", "text"}, {"text", "42"}}}}});
        f.wait_mode("paused");
        check(f.tools[0]->call_count() == 1 && f.peers[0]->request_count() == 2,
                "pause replayed the MCP call or started reasoning past its safe boundary");
        f.session->command("resume");
        f.wait_mode("thinking");
        auto a2 = f.peers[0]->request(2);
        f.peers[1]->request(1);
        check(ends_with(a2->prompt, {3, 40, 1}) && count(a2->prompt, 501) == 1 && f.tools[0]->call_count() == 1,
                "resuming after native tool pause lost or duplicated the completed call/result");
        f.pause();
    }
}

static void test_thought_targeted_initial_session_finishes_individually() {
    fixture f(4, 32768, true, false, true);
    f.session->command("message", "Only A should start", "A");
    f.wait_mode("thinking");
    f.peers[0]->request(0);
    check(f.peers[1]->request_count() == 0 && f.session->state().at("peers").at(1).at("tokens") == 0,
            "targeted initial message started the other agent");
    finish_thought_answer(f, 0, 0, "Only A answers");
    f.wait_mode("answered");
    check(f.peers[1]->request_count() == 0, "finishing the only active agent started an unsolicited peer answer");
}

static void test_thought_pause_drops_only_incomplete_utf8_suffix() {
    fixture f(4, 32768, true, false, true);
    f.start();
    f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    b0->send({201}, "Complete text. ");
    b0->send({202}, "");
    b0->wait_consumed(2);
    f.session->command("pause");
    f.wait_mode("paused");
    f.session->command("resume");
    f.wait_mode("thinking");
    auto a1 = f.peers[0]->request(1);
    auto b1 = f.peers[1]->request(1);
    check(count(b1->prompt, 201) == 1 && !count(b1->prompt, 202),
            "pause retained an incomplete UTF-8 suffix or discarded earlier complete output");
    b1->send({203}, "é");
    b1->wait_consumed(1);
    finish_thought_command(a1, "peek", 101);
    auto a2 = f.peers[0]->request(2);
    const auto captures = events_of(f, "thought_capture", "A");
    check(captures.size() == 1 && captures[0].at("text") == "Complete text. é" &&
            count(a2->prompt, 201) == 1 && !count(a2->prompt, 202) && count(a2->prompt, 203) == 1,
            "post-resume thought snapshot included a discarded partial code point");
    check(f.session->state().at("peers").at(1).at("generated") == 3,
            "discarding an unrendered suffix rewrote actual generation counters");
    f.pause();
}

static void test_thought_command_survives_pause_exactly_once() {
    for (bool completed : {false, true}) {
        fixture f(4, 32768, true, false, true);
        f.start();
        auto a0 = f.peers[0]->request(0);
        f.peers[1]->request(0);
        a0->send({101}, completed ? "<ct:send>across pause</ct:send>" : "<ct:se");
        a0->wait_consumed(1);
        f.session->command("pause");
        f.wait_mode("paused");
        f.session->command("resume");
        f.wait_mode("thinking");
        auto a1 = f.peers[0]->request(1);
        f.peers[1]->request(1);
        if (!completed) {
            check(a1->parameters.at("splice").value("thought_prefix", std::string()) == "<ct:se",
                    "pause discarded the pending thought-command prefix");
            a1->send({102}, "nd>across pause</ct:send>");
            a1->finish_result({{"type", "done"}, {"stop_type", "limit"}, {"truncated", false},
                {"splice_boundary", "thought_command"}, {"thought_command", "send"}, {"thought_payload", "across pause"}});
            f.peers[0]->request(2);
        }
        const auto messages = events_of(f, "thought_message");
        check(messages.size() == 1 && messages[0].at("status") == "sent" && messages[0].at("text") == "across pause",
                "pause lost a complete command or executed it more than once");
        const auto results = events_of(f, "thought_result", "A");
        check(results.size() == 1 && results[0].at("command") == "send", "completed thought command did not get exactly one acknowledgement");
        f.pause();
    }
}

static void test_thought_targeted_message_replaces_completed_native_terminator() {
    for (bool during_call : {false, true}) {
        fixture f(4, 32768, true, true, true);
        auto gate = std::make_shared<fake_operation>();
        f.start();
        if (!during_call) {
            gate->throw_on_cancel = true;
            std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
            f.peers[0]->parse_gate = gate;
        }
        auto b0 = f.peers[1]->request(0);
        auto a1 = start_private_turn(f, 0);
        a1->send({501, 3}, "complete native tool tail");
        a1->finish("eos", "");
        if (during_call) {
            f.tools[0]->request(0);
        } else {
            gate->wait_entered();
        }
        f.session->command("message", "Replace the interrupted tool turn", "A");
        f.wait_mode("thinking");
        auto a2 = f.peers[0]->request(2);
        check(ends_with(a2->prompt, {501, 3, 20, 21, 1}) && count(a2->prompt, 3) == 1,
                "targeted message appended a duplicate native end-of-turn marker");
        check(request_is_active(b0) && f.peers[1]->request_count() == 1,
                "replacing a native tool turn cancelled the other model's reasoning");
        check(f.tools[0]->call_count() == (during_call ? 1 : 0),
                "cancelled native parsing dispatched or replayed an MCP call");
        f.pause();
    }
}

static void test_thought_targeted_template_preserves_concurrent_peer_failure() {
    fixture f(4, 32768, true, false, true);
    f.start();
    auto a0 = f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    a0->send({101}, "Local work before intervention.");
    a0->wait_consumed(1);
    auto gate = std::make_shared<fake_operation>();
    {
        std::lock_guard<std::mutex> lock(f.peers[0]->mutex);
        f.peers[0]->next_user_gate = gate;
    }
    f.session->command("message", "Targeted update whose template is slow", "A");
    gate->wait_entered();
    b0->finish_result({{"type", "error"}, {"message", "B failed while A's template was processing"}});
    const auto failed = f.wait_state([](const crossthink_json & state) {
        return state.at("mode") == "error";
    }, "observe concurrent peer failure", true);
    gate->release();
    const auto settled = f.wait_mode("error");
    check(settled.at("error") == failed.at("error"),
            "targeted message template cleared or replaced the other model's error");
    check(f.peers[0]->request_count() == 1 && f.peers[1]->request_count() == 1,
            "targeted message restarted generation after a concurrent peer failure");
}

static void test_thought_metrics_count_generation_only() {
    fixture f(4, 32768, true, false, true);
    f.start();
    auto a0 = f.peers[0]->request(0);
    auto b0 = f.peers[1]->request(0);
    a0->send({101, 102}, "two locally generated tokens");
    b0->send({201, 202, 203}, "three locally generated tokens");
    a0->wait_consumed(1);
    b0->wait_consumed(1);
    const auto before = f.session->state();
    check(before.at("peers").at(0).at("thinking_tokens") == 2 &&
            before.at("peers").at(1).at("thinking_tokens") == 3,
            "thinking token counters did not count raw generated IDs");
    check(before.at("thinking_tokens") == 5, "combined thinking counter did not sum both agents");
    check(std::abs(before.at("thinking_tokens_per_second").get<double>() -
            before.at("peers").at(0).at("thinking_tokens_per_second").get<double>() -
            before.at("peers").at(1).at("thinking_tokens_per_second").get<double>()) < 1e-9,
            "combined thinking rate did not sum both agent rates from the same snapshot");
    for (const auto & peer : before.at("peers")) {
        check(peer.at("tokens_per_second").is_number() && peer.at("tokens_per_second").get<double>() > 0 &&
                peer.at("thinking_tokens_per_second").is_number() && peer.at("thinking_tokens_per_second").get<double>() > 0,
                "rolling token rates did not reflect newly generated tokens");
    }
    finish_thought_command(a0, "peek", 103);
    f.peers[0]->request(1);
    const auto after = f.session->state();
    check(after.at("peers").at(0).at("thinking_tokens") == 3 &&
            after.at("peers").at(1).at("thinking_tokens") == 3,
            "injected thought tokens were counted as newly generated thinking");
    f.pause();
}

int main() {
    const char * current_test = "initialization";
    try {
        current_test = "test_streaming_boundary_and_resume";
        test_streaming_boundary_and_resume();
        current_test = "test_user_answer_and_reset";
        test_user_answer_and_reset(false);
        current_test = "test_user_answer_and_reset";
        test_user_answer_and_reset(true);
        current_test = "test_paragraph_rendezvous";
        test_paragraph_rendezvous();
        current_test = "test_paragraph_pause_and_resume";
        test_paragraph_pause_and_resume();
        current_test = "test_invalid_paragraph_result";
        test_invalid_paragraph_result();
        current_test = "test_paragraph_failure_during_command";
        test_paragraph_failure_during_command();
        current_test = "test_tool_result_is_private";
        test_tool_result_is_private();
        current_test = "test_reset_cancels_tool_call";
        test_reset_cancels_tool_call();
        current_test = "test_incomplete_tool_turn_does_not_execute";
        test_incomplete_tool_turn_does_not_execute();
        current_test = "test_natural_final_answer_with_tools";
        test_natural_final_answer_with_tools();
        current_test = "test_reset_during_tool_template_operation";
        test_reset_during_tool_template_operation();
        current_test = "test_peer_error_survives_private_final_answer";
        test_peer_error_survives_private_final_answer();
        current_test = "test_thought_mode_parallel_and_catalogue";
        test_thought_mode_parallel_and_catalogue();
        current_test = "test_thought_peek_is_incremental_and_nonintrusive";
        test_thought_peek_is_incremental_and_nonintrusive();
        current_test = "test_thought_peek_latest_turn_and_cursor_reset";
        test_thought_peek_latest_turn_and_cursor_reset();
        current_test = "test_thought_send_and_inbox_are_opt_in";
        test_thought_send_and_inbox_are_opt_in();
        current_test = "test_thought_targeted_message_preserves_other_stream";
        test_thought_targeted_message_preserves_other_stream();
        current_test = "test_thought_native_mcp_does_not_serialize_peer";
        test_thought_native_mcp_does_not_serialize_peer();
        current_test = "test_thought_answers_never_execute_commands";
        test_thought_answers_never_execute_commands();
        current_test = "test_thought_reset_discards_stale_streams_and_mail";
        test_thought_reset_discards_stale_streams_and_mail();
        test_thought_peek_defers_incomplete_utf8_packets();
        current_test = "test_thought_cancellation_during_result_tokenization";
        test_thought_cancellation_during_result_tokenization();
        current_test = "test_thought_pause_waits_for_native_tool_boundary";
        test_thought_pause_waits_for_native_tool_boundary();
        current_test = "test_thought_targeted_initial_session_finishes_individually";
        test_thought_targeted_initial_session_finishes_individually();
        test_thought_pause_drops_only_incomplete_utf8_suffix();
        current_test = "test_thought_command_survives_pause_exactly_once";
        test_thought_command_survives_pause_exactly_once();
        current_test = "test_thought_targeted_message_replaces_completed_native_terminator";
        test_thought_targeted_message_replaces_completed_native_terminator();
        current_test = "test_thought_targeted_template_preserves_concurrent_peer_failure";
        test_thought_targeted_template_preserves_concurrent_peer_failure();
        current_test = "test_thought_metrics_count_generation_only";
        test_thought_metrics_count_generation_only();
        std::puts("crossthink tests: passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "crossthink tests (%s): %s\n", current_test, error.what());
        return 1;
    }
}
