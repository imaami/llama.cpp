#include "crossthink.h"

#include <algorithm>
#include <chrono>
#include <set>
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
        const crossthink_options & opts,
        std::array<std::unique_ptr<crossthink_tool_service>, 2> tools) : options(opts) {
    if (options.chunk_tokens < 1 || options.chunk_tokens > 4096 ||
            options.sentence_after < 1 || options.sentence_after > 4096 ||
            options.answer_tokens < 1 || options.answer_tokens > 65536 ||
            options.tool_tokens < 1 || options.tool_tokens > 16384 ||
            options.max_tool_rounds < 1 || options.max_tool_rounds > 64) {
        throw std::invalid_argument("chunk/answer token limits are outside supported bounds");
    }
    if (bool(tools[0]) != bool(tools[1]) || (tools[0] && !options.paragraph_splice)) {
        throw std::invalid_argument("MCP requires paragraph rendezvous and a tool service for both peers");
    }
    for (size_t index = 0; index < peers.size(); ++index) {
        auto & peer = peers[index];
        peer.transport = std::move(transports[index]);
        if (!peer.transport) {
            throw std::invalid_argument("two transports are required");
        }
        peer.info = peer.transport->describe();
        peer.tools = std::move(tools[index]);
        if (peer.tools) {
            if (!peer.info.value("tool_parse", false) || peer.tools->tools().empty()) {
                throw std::runtime_error("MCP requires updated model servers and at least one discovered tool");
            }
            peer.transport->configure_tools(peer.tools->tools());
        }
        if (options.paragraph_splice && !peer.info.value("splice", false)) {
            throw std::runtime_error("paragraph splicing requires updated model servers with splice support");
        }
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
        if (peer.tools) { peer.tools->cancel(); }
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
        {"splice_mode", options.paragraph_splice ? "paragraph" : "fixed"}, {"exchanges", exchanges},
        {"tools", peers[0].tools ? peers[0].tools->tools().size() : 0},
    };
    for (size_t index = 0; index < peers.size(); ++index) {
        const auto & peer = peers[index];
        result["peers"].push_back({
            {"name", peer_name(index)}, {"tokens", peer.tape.size()}, {"queued", peer.inbox.size()},
            {"generated", peer.generated}, {"imported", peer.imported},
            {"context_size", peer.context_size}, {"active", peer.active},
            {"waiting", peer.segment_done}, {"boundary", peer.boundary}, {"forced_splices", peer.forced_splices},
            {"tool_calls", peer.tool_calls}, {"tool_status", peer.tool_status},
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
            if (options.paragraph_splice && mode == "error") {
                throw std::logic_error("reset the failed session before starting another message");
            }
        } else if (action == "pause") {
            if (mode != "thinking") {
                throw std::logic_error("only an active thinking session can be paused");
            }
        } else if (action == "resume") {
            if (mode != "paused" || !error.empty() || peers[0].tape.empty() ||
                    (peers[0].answer_done && peers[1].answer_done)) {
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
        if (action == "reset") {
            for (auto & peer : peers) {
                peer.transport->cancel();
                if (peer.tools) { peer.tools->cancel(); }
            }
        }
        emit_state();
    }
    changed.notify_all();
}

void crossthink_session::fail(const std::string & message) {
    error = message;
    mode = "error";
    for (auto & peer : peers) {
        if (peer.tools) { peer.tools->cancel(); }
    }
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

void crossthink_session::rendezvous() {
    if (!options.paragraph_splice || !peers[0].segment_done || !peers[1].segment_done) {
        return;
    }
    for (const auto & peer : peers) {
        if (peer.tape.size() + peer.inbox.size() >= peer.context_size) {
            throw std::runtime_error("context full at splice boundary; reset the conversation");
        }
    }
    for (auto & peer : peers) {
        drain(peer);
        peer.segment_done = false;
        peer.tool_rounds = 0;
    }
    ++exchanges;
    emit({{"type", "splice"}, {"exchange", exchanges},
        {"a_boundary", peers[0].boundary}, {"b_boundary", peers[1].boundary}});
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
        if (contains(peer.controls, token) && !(answer && contains(peer.eog, token)) &&
                !(!answer && peer.tools && token == peer.close_token)) {
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
        if (!answer && peer.tools && token == peer.close_token) {
            peer.thinking_open = false;
            other.inbox.clear();
        } else if (!answer && peer.thinking_open && !other.answer_done && mode != "answering") {
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

bool crossthink_session::tool_turn(size_t index, uint64_t generation_epoch, std::unique_lock<std::mutex> & lock) {
    auto & peer = peers[index];
    auto & other = peers[1 - index];
    if (stopping || epoch != generation_epoch || mode == "error") {
        return false;
    }
    peer.tools->begin_turn();
    const size_t budget = std::max(options.tool_tokens, options.answer_tokens);
    if (peer.tape.size() + budget + 1 >= peer.context_size) {
        throw std::runtime_error("context full for a tool/answer turn");
    }
    json bias = json::array();
    for (llama_token token : peer.controls) {
        if (!contains(peer.eog, token)) {
            bias.push_back(json::array({token, false}));
        }
    }
    const json parameters = {
        {"n_predict", budget}, {"temperature", options.temperature},
        {"seed", (uint64_t(options.seed) + index + 2 * (peer.sequence++ % UINT32_MAX)) % UINT32_MAX},
        {"stream", true}, {"return_content", false}, {"cache_prompt", true},
        {"reasoning_budget_tokens", -1}, {"logit_bias", bias}, {"stop", json::array()},
    };
    const auto prompt = peer.tape;
    std::vector<llama_token> output{peer.close_token};
    peer.tool_status = "generating tool call or answer";
    emit_state();
    lock.unlock();
    try {
        const auto result = peer.transport->generate(prompt, parameters, [&](const server_token_wire::packet & packet) {
            std::lock_guard<std::mutex> guard(mutex);
            if (stopping || epoch != generation_epoch || mode == "error") {
                return false;
            }
            if (packet.tokens.size() > budget - (output.size() - 1)) {
                throw std::runtime_error("private tool turn exceeded its token budget");
            }
            for (llama_token token : packet.tokens) {
                if (token < 0 || token >= peer.info.at("n_vocab").get<int64_t>() ||
                        (contains(peer.controls, token) && !contains(peer.eog, token))) {
                    throw std::runtime_error("invalid control token in private tool turn");
                }
            }
            output.insert(output.end(), packet.tokens.begin(), packet.tokens.end());
            peer.tape.insert(peer.tape.end(), packet.tokens.begin(), packet.tokens.end());
            peer.generated += packet.tokens.size();
            return true;
        });
        lock.lock();
        if (stopping || epoch != generation_epoch || mode == "error") {
            return false;
        }
        if (!result.is_object() || result.value("type", std::string()) != "done" ||
                result.value("truncated", false) || result.value("stop_type", std::string()) != "eos" ||
                output.size() < 2 || !contains(peer.eog, output.back())) {
            throw std::runtime_error("tool/answer turn did not finish; increase --tool-turn-tokens if it hit the limit");
        }
        lock.unlock();
        auto assistant = peer.transport->parse_tool_turn(output);
        const auto calls = assistant.value("tool_calls", json::array());
        if (!calls.is_array() || calls.size() > 32) {
            throw std::runtime_error("invalid number of tool calls in assistant turn");
        }
        lock.lock();
        if (stopping || epoch != generation_epoch || mode == "error") {
            return false;
        }
        if (calls.empty()) {
            peer.tape.pop_back(); // next_user() supplies the assistant terminator.
            peer.answer_done = true;
            peer.segment_done = false;
            peer.tool_status.clear();
            peer.inbox.clear();
            other.inbox.clear();
            emit({{"type", "answer_start"}, {"peer", peer_name(index)}});
            emit({{"type", "answer"}, {"peer", peer_name(index)}, {"text", assistant.value("content", std::string())}});
            if (!busy) {
                mode = other.answer_done ? "answered" : "answering";
            }
            changed.notify_all();
            return false;
        }
        if (++peer.tool_rounds > options.max_tool_rounds) {
            throw std::runtime_error("tool round limit reached before a shared paragraph; increase --max-tool-rounds or reset");
        }
        std::set<std::string> call_ids;
        for (auto & call : assistant["tool_calls"]) {
            call.at("function").at("name").get<std::string>();
            call.at("function").at("arguments");
            auto call_id = call.value("id", std::string());
            if (call_id.empty()) {
                call_id = "ct_" + std::to_string(generation_epoch) + "_" + peer_name(index) + "_" +
                    std::to_string(peer.tool_calls + call_ids.size() + 1);
                call["id"] = call_id;
            }
            if (!call_ids.insert(call_id).second) {
                throw std::runtime_error("duplicate tool call ID in assistant turn");
            }
        }
        json results = json::array();
        for (size_t i = 0; i < calls.size(); ++i) {
            auto & call = assistant["tool_calls"][i];
            const auto name = call.at("function").at("name").get<std::string>();
            const auto call_id = call.at("id").get<std::string>();
            ++peer.tool_calls;
            const auto arguments = call.at("function").at("arguments");
            peer.tool_status = name;
            emit({{"type", "tool_call"}, {"peer", peer_name(index)}, {"name", name},
                {"call_id", call_id}, {"arguments", arguments}});
            emit_state();
            lock.unlock();
            json response;
            try {
                const auto parsed = arguments.is_string() ? json::parse(arguments.get<std::string>()) : arguments;
                if (!parsed.is_object()) {
                    throw std::runtime_error("tool arguments must be a JSON object");
                }
                response = peer.tools->call(name, parsed);
            } catch (const std::exception & exception) {
                response = {{"isError", true}, {"content", json::array({{
                    {"type", "text"}, {"text", std::string("Tool request failed: ") + exception.what() +
                        ". A transport failure may occur after execution; do not blindly repeat a side effect."},
                }})}};
            }
            lock.lock();
            if (stopping || epoch != generation_epoch || mode == "error") {
                return false;
            }
            emit({{"type", "tool_result"}, {"peer", peer_name(index)}, {"name", name},
                {"call_id", call_id}, {"result", response}});
            std::string content = response.dump();
            if (content.size() > 65536) {
                size_t length = 65536;
                while (length && (static_cast<unsigned char>(content[length]) & 0xc0) == 0x80) { --length; }
                content.resize(length);
                content += "\n[Tool result truncated at 64 KiB]";
            }
            results.push_back({{"role", "tool"}, {"tool_call_id", call_id}, {"name", name}, {"content", content}});
        }
        lock.unlock();
        const auto suffix = peer.transport->tool_results(assistant, results);
        lock.lock();
        if (stopping || epoch != generation_epoch || mode == "error") {
            return false;
        }
        const uint64_t reserve = 2 * options.chunk_tokens + budget + options.answer_tokens + 2;
        if (suffix.empty() || peer.tape.size() + suffix.size() + reserve >= peer.context_size) {
            throw std::runtime_error("tool results do not fit the remaining context; reset or reduce token budgets");
        }
        peer.tape.insert(peer.tape.end(), suffix.begin(), suffix.end());
        peer.thinking_open = true;
        peer.tool_status.clear();
        return true;
    } catch (...) {
        if (!lock.owns_lock()) { lock.lock(); }
        peer.tool_status.clear();
        if (stopping || epoch != generation_epoch || mode == "error") { return false; }
        throw;
    }
}

void crossthink_session::worker(size_t index) {
    auto & peer = peers[index];
    auto & other = peers[1 - index];
    std::unique_lock<std::mutex> lock(mutex);
    while (!stopping) {
        changed.wait(lock, [&] {
            return stopping || (!busy && ((mode == "thinking" &&
                (!options.paragraph_splice || !peer.segment_done)) || (mode == "answering" && !peer.answer_done)));
        });
        if (stopping) {
            break;
        }
        const bool answer = mode == "answering";
        try {
            if (answer && peer.tools) {
                peer.active = true;
                if (peer.thinking_open) {
                    peer.tape.push_back(peer.close_token);
                    peer.thinking_open = false;
                }
                tool_turn(index, epoch, lock);
                peer.active = false;
                emit_state();
                changed.notify_all();
                continue;
            }
            if (!answer) {
                const size_t queue_limit = static_cast<size_t>(options.chunk_tokens) * 4;
                if (!options.paragraph_splice) {
                    drain(peer);
                }
                const uint64_t reserve = (options.paragraph_splice ? 2 * options.chunk_tokens
                    : options.chunk_tokens + queue_limit) + options.answer_tokens + 2 +
                    (peer.tools ? std::max(options.tool_tokens, options.answer_tokens) : 0);
                if (peer.tape.size() + reserve >= peer.context_size) {
                    error = std::string(peer_name(index)) + ": context full for further crossthink; request answers or reset";
                    mode = "paused";
                    emit({{"type", "error"}, {"message", error}});
                    emit_state();
                    changed.notify_all();
                    continue;
                }
                if (!options.paragraph_splice) {
                    // Reserve a whole quantum so callbacks never block each other.
                    changed.wait(lock, [&] {
                        return stopping || busy || mode != "thinking" ||
                            other.inbox.size() + static_cast<size_t>(options.chunk_tokens) <= queue_limit;
                    });
                    if (stopping || busy || mode != "thinking") {
                        continue;
                    }
                    drain(peer);
                }
            }
            const auto prompt = peer.tape;
            const uint64_t generation_epoch = epoch;
            const uint32_t seed = (uint64_t(options.seed) + index + 2 * (peer.sequence++ % UINT32_MAX)) % UINT32_MAX;
            json bias = json::array();
            for (llama_token token : peer.controls) {
                if ((!answer || !contains(peer.eog, token)) && !(peer.tools && token == peer.close_token)) {
                    bias.push_back(json::array({token, false}));
                }
            }
            json parameters = {
                {"n_predict", answer ? options.answer_tokens : options.chunk_tokens},
                {"seed", seed}, {"temperature", options.temperature}, {"cache_prompt", true},
                {"stream", true}, {"return_content", true}, {"reasoning_budget_tokens", -1},
                {"logit_bias", std::move(bias)}, {"stop", json::array()},
            };
            if (!answer && options.paragraph_splice) {
                parameters["splice"] = {{"sentence_after", options.sentence_after}};
                if (peer.tools) {
                    parameters["splice"]["stop_on_think_close"] = true;
                    parameters["preserved_tokens"] = json::array({"</think>"});
                }
            }
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
            if (stopping || epoch != generation_epoch || mode == "error") {
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
                throw std::runtime_error("reasoning ended unexpectedly; control token suppression may be unsupported");
            }
            if (!answer && options.paragraph_splice) {
                const auto boundary = result.value("splice_boundary", std::string());
                if (boundary == "tool" && peer.tools) {
                    if (peer.thinking_open || peer.tape.back() != peer.close_token) {
                        throw std::runtime_error("tool boundary did not end with the reasoning terminator");
                    }
                    peer.active = true;
                    tool_turn(index, generation_epoch, lock);
                    peer.active = false;
                    emit_state();
                    changed.notify_all();
                    continue;
                }
                if ((boundary != "paragraph" && boundary != "sentence" && boundary != "limit") ||
                        (boundary == "limit" && received != static_cast<size_t>(options.chunk_tokens))) {
                    throw std::runtime_error("completion did not report a valid splice boundary");
                }
                peer.boundary = boundary;
                peer.segment_done = true;
                if (boundary == "limit") {
                    ++peer.forced_splices;
                    emit({{"type", "notice"}, {"message", std::string(peer_name(index)) +
                        ": forced splice at the token ceiling; no clean boundary was found"}});
                }
                if (!busy && mode == "thinking") {
                    rendezvous();
                }
            } else if (answer) {
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
            peer.active = false;
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
        if (options.paragraph_splice && mode == "error" && pending.action != "reset") {
            busy = false;
            pending = {};
            emit_state();
            continue;
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
            const uint64_t reserve = (options.paragraph_splice ? 2 : 5) * options.chunk_tokens + options.answer_tokens + 2 +
                (peer.tools ? std::max(options.tool_tokens, options.answer_tokens) : 0);
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
            peer.segment_done = false;
            peer.boundary.clear();
            peer.forced_splices = 0;
            peer.tool_calls = 0;
            peer.tool_rounds = 0;
            peer.tool_status.clear();
        }
        round = 0;
        exchanges = 0;
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
            peer.segment_done = false;
            peer.tool_rounds = 0;
        }
        emit({{"type", "user"}, {"text", command.text}});
        mode = "thinking";
    } else if (command.action == "answer") {
        for (auto & peer : peers) {
            if (peer.answer_done) { continue; }
            drain(peer);
            if (peer.tape.size() + (peer.tools ? std::max(options.tool_tokens, options.answer_tokens)
                    : options.answer_tokens) + 2 >= peer.context_size) {
                throw std::runtime_error("context full for answers; reset or use a smaller answer budget");
            }
        }
        for (auto & peer : peers) {
            if (peer.answer_done) { continue; }
            if (peer.thinking_open) {
                peer.tape.push_back(peer.close_token);
                peer.thinking_open = false;
            }
            peer.answer_done = false;
            peer.segment_done = false;
        }
        mode = "answering";
    } else if (command.action == "resume") {
        rendezvous();
        mode = peers[0].answer_done || peers[1].answer_done ? "answering" : "thinking";
    } else if (command.action == "pause" && peers[0].answer_done && peers[1].answer_done) {
        mode = "answered";
    }
}
