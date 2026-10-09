#include "crossthink.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

using json = crossthink_json;

static bool contains(const std::vector<llama_token> & tokens, llama_token token) {
    return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

static const char * peer_name(size_t index) {
    return index ? "B" : "A";
}

crossthink_session::crossthink_session(
        std::array<std::unique_ptr<crossthink_transport>, 2> transports,
        const crossthink_options & opts) : options(opts) {
    if (options.chunk_tokens < 1 || options.chunk_tokens > 4096 ||
            options.answer_tokens < 1 || options.answer_tokens > 65536) {
        throw std::invalid_argument("chunk/answer token limits are outside supported bounds");
    }
    for (size_t index = 0; index < peers.size(); ++index) {
        auto & peer = peers[index];
        peer.transport = std::move(transports[index]);
        if (!peer.transport) {
            throw std::invalid_argument("two transports are required");
        }
        peer.info = peer.transport->describe();
        const int64_t n_vocab = peer.info.at("n_vocab").get<int64_t>();
        const int64_t context = peer.info.at("context_size").get<int64_t>();
        if (peer.info.at("protocol") != server_token_wire::magic || n_vocab <= 0 ||
                n_vocab > INT32_MAX || context <= 0 || context > INT32_MAX) {
            throw std::runtime_error("invalid token protocol, vocabulary or context size");
        }
        peer.context_size = static_cast<uint64_t>(context);
        peer.close_token = peer.info.at("close_token").get<llama_token>();
        peer.controls = peer.info.at("control_ids").get<std::vector<llama_token>>();
        peer.eog = peer.info.at("eog_ids").get<std::vector<llama_token>>();
        peer.controls.insert(peer.controls.end(), peer.eog.begin(), peer.eog.end());
        peer.controls.push_back(peer.close_token);
        for (llama_token token : peer.controls) {
            if (token < 0 || token >= n_vocab) {
                throw std::runtime_error("invalid control token ID");
            }
        }
        if (contains(peer.eog, peer.close_token)) {
            throw std::runtime_error("</think> cannot also be an end-of-generation token");
        }
        std::sort(peer.controls.begin(), peer.controls.end());
        peer.controls.erase(std::unique(peer.controls.begin(), peer.controls.end()), peer.controls.end());
    }
    if (peers[0].info.at("fingerprint") != peers[1].info.at("fingerprint") ||
            peers[0].info.at("n_vocab") != peers[1].info.at("n_vocab") ||
            peers[0].eog != peers[1].eog || peers[0].close_token != peers[1].close_token ||
            peers[0].controls != peers[1].controls) {
        throw std::runtime_error("peers have incompatible tokenizer mappings or control tokens");
    }
    try {
        for (size_t index = 0; index < workers.size(); ++index) {
            workers[index] = std::thread(&crossthink_session::worker, this, index);
        }
        controller = std::thread(&crossthink_session::control, this);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        for (auto & thread : workers) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        throw;
    }
}

crossthink_session::~crossthink_session() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        ++epoch;
    }
    changed.notify_all();
    event_changed.notify_all();
    for (auto & peer : peers) {
        peer.transport->cancel();
    }
    controller.join();
    for (auto & thread : workers) {
        thread.join();
    }
}

json crossthink_session::state_locked() const {
    json result = {
        {"mode", mode}, {"busy", busy}, {"epoch", epoch}, {"round", round},
        {"error", error}, {"last_event_id", next_event_id - 1}, {"peers", json::array()},
    };
    for (size_t index = 0; index < peers.size(); ++index) {
        const auto & peer = peers[index];
        result["peers"].push_back({
            {"name", peer_name(index)}, {"tokens", peer.tape.size()}, {"queued", peer.inbox.size()},
            {"generated", peer.generated}, {"imported", peer.imported},
            {"context_size", peer.context_size}, {"active", peer.active},
        });
    }
    return result;
}

json crossthink_session::state() {
    std::lock_guard<std::mutex> lock(mutex);
    return state_locked();
}

void crossthink_session::emit(json event) {
    event["id"] = next_event_id++;
    event["epoch"] = epoch;
    event["round"] = round;
    event_bytes += event.dump().size();
    event_log.push_back(std::move(event));
    while (event_log.size() > 4096 || event_bytes > 4 * 1024 * 1024) {
        event_bytes -= event_log.front().dump().size();
        event_log.pop_front();
    }
    event_changed.notify_all();
}

void crossthink_session::emit_state() {
    emit({{"type", "state"}, {"state", state_locked()}});
}

std::vector<json> crossthink_session::events_after(uint64_t cursor, bool wait) {
    std::unique_lock<std::mutex> lock(mutex);
    if (wait) {
        event_changed.wait_for(lock, std::chrono::seconds(15), [&] {
            return stopping || next_event_id > cursor + 1;
        });
    }
    std::vector<json> result;
    if (!event_log.empty() && cursor < event_log.front().at("id").get<uint64_t>() - 1) {
        result.push_back({{"type", "gap"}, {"id", event_log.front().at("id").get<uint64_t>() - 1},
            {"message", "Older trace output expired from the bounded event buffer."}});
    }
    for (const auto & event : event_log) {
        if (event.at("id").get<uint64_t>() > cursor) {
            result.push_back(event);
        }
    }
    return result;
}

void crossthink_session::command(const std::string & action, const std::string & text) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (busy || stopping) {
            throw std::logic_error("another control operation is pending");
        }
        if (text.size() > 1024 * 1024) {
            throw std::invalid_argument("message exceeds 1 MiB");
        }
        if (action == "message") {
            if (text.empty()) {
                throw std::invalid_argument("message text must not be empty");
            }
            if (mode == "answering") {
                throw std::logic_error("wait for the answers before sending the next message");
            }
        } else if (action == "pause") {
            if (mode != "thinking") {
                throw std::logic_error("only an active thinking session can be paused");
            }
        } else if (action == "resume") {
            if (mode != "paused" || !error.empty() || peers[0].tape.empty() || !peers[0].thinking_open) {
                throw std::logic_error("there is no paused reasoning session to resume");
            }
        } else if (action == "answer") {
            if ((mode != "thinking" && mode != "paused") || peers[0].tape.empty()) {
                throw std::logic_error("start or resume a conversation before requesting answers");
            }
        } else if (action == "reset") {
            ++epoch;
        } else {
            throw std::invalid_argument("unknown control operation");
        }
        busy = true;
        mode = "paused";
        pending = {action, text};
        emit_state();
    }
    if (action == "reset") {
        for (auto & peer : peers) {
            peer.transport->cancel();
        }
    }
    changed.notify_all();
}

void crossthink_session::fail(const std::string & message) {
    error = message;
    mode = "error";
    emit({{"type", "error"}, {"message", message}});
    emit_state();
    changed.notify_all();
}

void crossthink_session::drain(peer_state & peer) {
    if (peer.tape.size() + peer.inbox.size() >= peer.context_size) {
        throw std::runtime_error("context full while importing peer tokens; reset the conversation");
    }
    peer.tape.insert(peer.tape.end(), peer.inbox.begin(), peer.inbox.end());
    peer.imported += peer.inbox.size();
    peer.inbox.clear();
    changed.notify_all();
}

bool crossthink_session::receive(size_t index, uint64_t generation_epoch, bool answer,
        size_t & received, const server_token_wire::packet & packet) {
    const json metadata = json::parse(packet.metadata);
    std::lock_guard<std::mutex> lock(mutex);
    if (stopping || epoch != generation_epoch) {
        return false;
    }
    auto & peer = peers[index];
    auto & other = peers[1 - index];
    const size_t maximum = answer ? options.answer_tokens : options.chunk_tokens;
    if (packet.tokens.size() > maximum - received) {
        throw std::runtime_error("server generated more tokens than requested");
    }
    received += packet.tokens.size();
    for (llama_token token : packet.tokens) {
        if (token < 0 || token >= peer.info.at("n_vocab").get<int64_t>()) {
            throw std::runtime_error("server returned a token outside the vocabulary");
        }
        if (contains(peer.controls, token) && !(answer && contains(peer.eog, token))) {
            throw std::runtime_error("server generated a forbidden reasoning/chat control token");
        }
    }
    for (llama_token token : packet.tokens) {
        // The next-user template supplies the assistant terminator exactly once.
        if (answer && contains(peer.eog, token)) {
            continue;
        }
        peer.tape.push_back(token);
        ++peer.generated;
        if (!answer) {
            other.inbox.push_back(token);
        }
    }
    const std::string content = metadata.value("content", std::string());
    if (!content.empty()) {
        emit({{"type", answer ? "answer" : "token"}, {"peer", peer_name(index)}, {"text", content}});
    }
    changed.notify_all();
    return true;
}

void crossthink_session::worker(size_t index) {
    auto & peer = peers[index];
    auto & other = peers[1 - index];
    std::unique_lock<std::mutex> lock(mutex);
    while (!stopping) {
        changed.wait(lock, [&] {
            return stopping || (!busy && (mode == "thinking" || (mode == "answering" && !peer.answer_done)));
        });
        if (stopping) {
            break;
        }
        const bool answer = mode == "answering";
        try {
            if (!answer) {
                drain(peer);
                const size_t queue_limit = static_cast<size_t>(options.chunk_tokens) * 4;
                const uint64_t reserve = options.chunk_tokens + queue_limit + options.answer_tokens + 2;
                if (peer.tape.size() + reserve >= peer.context_size) {
                    error = std::string(peer_name(index)) + ": context full for further crossthink; request answers or reset";
                    mode = "paused";
                    emit({{"type", "error"}, {"message", error}});
                    emit_state();
                    changed.notify_all();
                    continue;
                }
                // Reserve a whole quantum so callbacks never block each other.
                changed.wait(lock, [&] {
                    return stopping || busy || mode != "thinking" ||
                        other.inbox.size() + static_cast<size_t>(options.chunk_tokens) <= queue_limit;
                });
                if (stopping || busy || mode != "thinking") {
                    continue;
                }
                // New peer tokens may have arrived while waiting for outbound capacity.
                drain(peer);
            }
            const auto prompt = peer.tape;
            const uint64_t generation_epoch = epoch;
            const uint32_t seed = (uint64_t(options.seed) + index + 2 * (peer.sequence++ % UINT32_MAX)) % UINT32_MAX;
            json bias = json::array();
            for (llama_token token : peer.controls) {
                if (!answer || !contains(peer.eog, token)) {
                    bias.push_back(json::array({token, false}));
                }
            }
            const json parameters = {
                {"n_predict", answer ? options.answer_tokens : options.chunk_tokens},
                {"seed", seed}, {"temperature", options.temperature}, {"cache_prompt", true},
                {"stream", true}, {"return_content", true}, {"reasoning_budget_tokens", -1},
                {"logit_bias", std::move(bias)}, {"stop", json::array()},
            };
            peer.active = true;
            if (answer) {
                emit({{"type", "answer_start"}, {"peer", peer_name(index)}});
            }
            lock.unlock();
            size_t received = 0;
            json result;
            std::string failure;
            try {
                result = peer.transport->generate(prompt, parameters, [&](const server_token_wire::packet & packet) {
                    return receive(index, generation_epoch, answer, received, packet);
                });
            } catch (const std::exception & exception) {
                failure = exception.what();
            }
            lock.lock();
            peer.active = false;
            changed.notify_all();
            if (stopping || epoch != generation_epoch) {
                continue;
            }
            if (!failure.empty()) {
                throw std::runtime_error(failure);
            }
            if (!result.is_object() || result.value("type", std::string()) != "done" ||
                    result.value("truncated", false)) {
                throw std::runtime_error("completion ended without a valid final record, or truncated its context");
            }
            if (!answer && (!received || result.value("stop_type", std::string()) != "limit")) {
                throw std::runtime_error("reasoning ended before its quantum; control token suppression may be unsupported");
            }
            if (answer) {
                peer.answer_done = true;
                if (other.answer_done && !busy) {
                    mode = "answered";
                }
                if (result.value("stop_type", std::string()) == "limit") {
                    emit({{"type", "notice"}, {"message", std::string(peer_name(index)) + ": answer reached its token budget"}});
                }
            }
            emit_state();
        } catch (const std::exception & exception) {
            fail(std::string(peer_name(index)) + ": " + exception.what());
        }
    }
}

void crossthink_session::control() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stopping) {
        changed.wait(lock, [&] { return stopping || busy; });
        if (stopping) {
            break;
        }
        changed.wait(lock, [&] { return stopping || (!peers[0].active && !peers[1].active); });
        if (stopping) {
            break;
        }
        const auto command = pending;
        lock.unlock();
        try {
            apply(command);
        } catch (const std::exception & exception) {
            std::lock_guard<std::mutex> error_lock(mutex);
            fail(exception.what());
        }
        lock.lock();
        busy = false;
        pending = {};
        emit_state();
        changed.notify_all();
    }
}

void crossthink_session::apply(const pending_command & command) {
    std::array<std::vector<llama_token>, 2> prepared;
    const bool reset = command.action == "reset";
    const bool message = command.action == "message" || (reset && !command.text.empty());
    if (message) {
        for (size_t index = 0; index < peers.size(); ++index) {
            auto & peer = peers[index];
            prepared[index] = reset || peer.tape.empty()
                ? peer.transport->initial_prompt(command.text)
                : peer.transport->next_user(command.text);
            const uint64_t retained = reset ? 0 : peer.tape.size() + peer.inbox.size() + (peer.thinking_open ? 1 : 0);
            const uint64_t reserve = 5 * options.chunk_tokens + options.answer_tokens + 2;
            if (prepared[index].empty() || retained + prepared[index].size() + reserve >= peer.context_size) {
                throw std::runtime_error(std::string(peer_name(index)) + ": message does not fit with the reasoning/answer reserve; reset or use smaller budgets");
            }
        }
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (stopping) {
        return;
    }
    error.clear();
    if (reset) {
        for (auto & peer : peers) {
            peer.tape.clear();
            peer.inbox.clear();
            peer.generated = 0;
            peer.imported = 0;
            peer.sequence = 0;
            peer.thinking_open = false;
            peer.answer_done = false;
        }
        round = 0;
        mode = "idle";
        emit({{"type", "reset"}});
    }
    if (message) {
        ++round;
        for (size_t index = 0; index < peers.size(); ++index) {
            auto & peer = peers[index];
            drain(peer);
            if (peer.thinking_open) {
                peer.tape.push_back(peer.close_token);
            }
            peer.tape.insert(peer.tape.end(), prepared[index].begin(), prepared[index].end());
            peer.thinking_open = true;
            peer.answer_done = false;
        }
        emit({{"type", "user"}, {"text", command.text}});
        mode = "thinking";
    } else if (command.action == "answer") {
        for (auto & peer : peers) {
            drain(peer);
            if (peer.tape.size() + options.answer_tokens + 2 >= peer.context_size) {
                throw std::runtime_error("context full for answers; reset or use a smaller answer budget");
            }
        }
        for (auto & peer : peers) {
            if (peer.thinking_open) {
                peer.tape.push_back(peer.close_token);
                peer.thinking_open = false;
            }
            peer.answer_done = false;
        }
        mode = "answering";
    } else if (command.action == "resume") {
        mode = "thinking";
    }
}
