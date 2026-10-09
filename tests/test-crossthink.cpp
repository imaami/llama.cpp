#include "../tools/crossthink/crossthink.h"

#include <algorithm>
#include <chrono>
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

    void send(std::vector<llama_token> tokens) {
        std::lock_guard<std::mutex> lock(mutex);
        steps.push_back({std::move(tokens), false, {}});
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
    crossthink_json configured_tools;
    crossthink_json parsed_assistant;
    std::vector<std::vector<llama_token>> parsed_turns;
    std::vector<crossthink_json> result_assistants;
    std::vector<crossthink_json> result_messages;
    std::shared_ptr<fake_operation> parse_gate;
    std::shared_ptr<fake_operation> results_gate;

    crossthink_json describe() override {
        return {
            {"protocol", "LLMTOK01"}, {"fingerprint", fingerprint},
            {"n_vocab", 4096}, {"eog_ids", {3}}, {"context_size", context_size},
            {"close_token", 2}, {"control_ids", {1, 2, 3}}, {"splice", true}, {"tool_parse", true},
        };
    }

    std::vector<llama_token> initial_prompt(const std::string & text) override {
        std::lock_guard<std::mutex> lock(mutex);
        initial_texts.push_back(text);
        return {1, 10, 11};
    }

    std::vector<llama_token> next_user(const std::string & text) override {
        std::lock_guard<std::mutex> lock(mutex);
        user_texts.push_back(text);
        return {3, 20, 21, 1};
    }

    void configure_tools(const crossthink_json & tools) override {
        std::lock_guard<std::mutex> lock(mutex);
        configured_tools = tools;
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
            const bool accepted = receive(packet(step.tokens));
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
    static server_token_wire::packet packet(const std::vector<llama_token> & tokens) {
        return {crossthink_json{{"type", "tokens"}, {"content", "piece"}}.dump(), tokens};
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

    fixture(int32_t chunk = 2, uint64_t context = 256, bool paragraph_splice = false, bool enable_tools = false) {
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

int main() {
    try {
        test_streaming_boundary_and_resume();
        test_user_answer_and_reset(false);
        test_user_answer_and_reset(true);
        test_paragraph_rendezvous();
        test_paragraph_pause_and_resume();
        test_invalid_paragraph_result();
        test_paragraph_failure_during_command();
        test_tool_result_is_private();
        test_reset_cancels_tool_call();
        test_incomplete_tool_turn_does_not_execute();
        test_natural_final_answer_with_tools();
        test_reset_during_tool_template_operation();
        test_peer_error_survives_private_final_answer();
        std::puts("crossthink tests: passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "crossthink tests: %s\n", error.what());
        return 1;
    }
}
