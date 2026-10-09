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
    std::string stop_type = "limit";
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
        steps.push_back({std::move(tokens), false, "limit"});
        changed.notify_all();
    }

    void finish(const std::string & stop_type = "limit") {
        std::lock_guard<std::mutex> lock(mutex);
        steps.push_back({{}, true, stop_type});
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

class fake_transport final : public crossthink_transport {
public:
    uint64_t context_size = 256;
    std::string fingerprint = "test-vocabulary";
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::shared_ptr<fake_request>> requests;
    std::vector<std::string> initial_texts;
    std::vector<std::string> user_texts;

    crossthink_json describe() override {
        return {
            {"protocol", "LLMTOK01"}, {"fingerprint", fingerprint},
            {"n_vocab", 4096}, {"eog_ids", {3}}, {"context_size", context_size},
            {"close_token", 2}, {"control_ids", {1, 2, 3}},
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
                return {{"type", "done"}, {"stop_type", step.stop_type}, {"truncated", false}};
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

private:
    static server_token_wire::packet packet(const std::vector<llama_token> & tokens) {
        return {crossthink_json{{"type", "tokens"}, {"content", "piece"}}.dump(), tokens};
    }
};

struct fixture {
    std::array<fake_transport *, 2> peers;
    std::unique_ptr<crossthink_session> session;
    uint64_t cursor = 0;

    fixture(int32_t chunk = 2, uint64_t context = 256) {
        std::array<std::unique_ptr<crossthink_transport>, 2> transports;
        for (size_t i = 0; i < peers.size(); ++i) {
            auto peer = std::make_unique<fake_transport>();
            peer->context_size = context;
            peers[i] = peer.get();
            transports[i] = std::move(peer);
        }
        crossthink_options options;
        options.chunk_tokens = chunk;
        options.answer_tokens = 8;
        session = std::make_unique<crossthink_session>(std::move(transports), options);
    }

    crossthink_json wait_mode(const std::string & mode) {
        for (;;) {
            auto state = session->state();
            if (state.at("mode") == mode && !state.at("busy").get<bool>()) {
                return state;
            }
            check(state.at("mode") != "error" || mode == "error", "unexpected engine error: " + state.dump());
            for (const auto & event : session->events_after(cursor, true)) {
                cursor = event.at("id").get<uint64_t>();
            }
        }
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

static void test_user_answer_and_reset() {
    fixture f;
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
    for (size_t i = 0; i < f.peers.size(); ++i) {
        for (const auto * field : {"tokens", "queued", "generated", "imported"}) {
            check(reset.at("peers").at(i).at(field) == 0, std::string("reset retained ") + field);
        }
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

int main() {
    try {
        test_streaming_boundary_and_resume();
        test_user_answer_and_reset();
        std::puts("crossthink tests: passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "crossthink tests: %s\n", error.what());
        return 1;
    }
}
