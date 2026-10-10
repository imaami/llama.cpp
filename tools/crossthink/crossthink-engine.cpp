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

static constexpr const char * link_tool_name = "think_with_telepathic_link";

static json link_tool() {
    return {{"type", "function"}, {"function", {
        {"name", link_tool_name},
        {"description", "Enable or disable the shared telepathic reasoning channel for both twins. "
            "While enabled, both see the same live reasoning with coordinator-owned A/B speaker labels. "
            "Use yield_until to let your twin finish a fragment, sentence, or paragraph before you continue; "
            "the wait is bounded. Tool calls and final answers remain private."},
        {"parameters", {{"type", "object"}, {"properties", {
            {"enabled", {{"type", "boolean"}}},
            {"yield_until", {{"type", "string"}, {"enum", {"fragment", "sentence", "paragraph"}}}},
        }}, {"required", {"enabled"}}, {"additionalProperties", false}}},
    }}};
}

crossthink_session::crossthink_session(
        std::array<std::unique_ptr<crossthink_transport>, 2> transports,
        const crossthink_options & opts,
        std::array<std::unique_ptr<crossthink_tool_service>, 2> tools) : options(opts) {
    if (options.chunk_tokens < 1 || options.chunk_tokens > 4096 ||
            options.sentence_after < 1 || options.sentence_after > 4096 ||
            options.answer_tokens < 1 || options.answer_tokens > 65536 ||
            options.tool_tokens < 1 || options.tool_tokens > 16384 ||
            options.max_tool_rounds < 1 || options.max_tool_rounds > 64 ||
            options.link_quantum < 1 || options.link_quantum > 4096 ||
            options.link_wait_tokens < 1 || options.link_wait_tokens > 65536) {
        throw std::invalid_argument("chunk/answer token limits are outside supported bounds");
    }
    if (bool(tools[0]) != bool(tools[1]) || (tools[0] && !options.paragraph_splice && !options.telepathy)) {
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
        if (native_tools(peer)) {
            if (!peer.info.value("tool_parse", false) || (peer.tools && peer.tools->tools().empty())) {
                throw std::runtime_error("native tools require updated model servers and a nonempty MCP catalogue when configured");
            }
            json definitions = peer.tools ? peer.tools->tools() : json::array();
            if (options.telepathy) {
                peer.transport->configure_peer(peer_name(index));
                for (const auto & definition : definitions) {
                    if (definition.at("function").at("name") == link_tool_name) {
                        throw std::runtime_error("MCP tool name conflicts with the built-in telepathic link tool");
                    }
                }
                definitions.push_back(link_tool());
            }
            peer.transport->configure_tools(definitions);
        }
        if ((options.paragraph_splice || options.telepathy) && !peer.info.value("splice", false)) {
            throw std::runtime_error("paragraph splicing requires updated model servers with splice support");
        }
        if (options.telepathy && !peer.info.value("splice_quantum", false)) {
            throw std::runtime_error("telepathy requires updated model servers with UTF-8 quantum support");
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
    if (options.telepathy) {
        for (size_t index = 0; index < peers.size(); ++index) {
            const std::string name = peer_name(index);
            link_labels[index] = peers[0].transport->text_tokens("\n[" + name + "]: ");
            link_interrupt_labels[index] = peers[0].transport->text_tokens(
                "\n[" + name + " interrupts " + peer_name(1 - index) + "]: ");
        }
        link_markers[0] = peers[0].transport->text_tokens("\n[Telepathic link OFF: continue privately.]\n");
        link_markers[1] = peers[0].transport->text_tokens("\n[Telepathic link ON: shared reasoning follows.]\n");
        for (const auto * collection : {&link_labels, &link_interrupt_labels, &link_markers}) {
            for (const auto & tokens : *collection) {
                if (tokens.empty()) { throw std::runtime_error("empty telepathic marker tokenization"); }
                for (llama_token token : tokens) {
                    if (token < 0 || token >= peers[0].info.at("n_vocab").get<int64_t>() ||
                            contains(peers[0].controls, token)) {
                        throw std::runtime_error("invalid control token in telepathic marker");
                    }
                }
            }
        }
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
        {"tools", (peers[0].tools ? peers[0].tools->tools().size() : 0) + (options.telepathy ? 1 : 0)},
        {"telepathy", options.telepathy}, {"link_enabled", link_enabled},
        {"link_speaker", link_speaker < 0 ? json(nullptr) : json(peer_name(link_speaker))},
        {"link_wait", link_wait},
    };
    for (size_t index = 0; index < peers.size(); ++index) {
        const auto & peer = peers[index];
        result["peers"].push_back({
            {"name", peer_name(index)}, {"tokens", peer.tape.size()}, {"queued", peer.inbox.size()},
            {"generated", peer.generated}, {"imported", peer.imported},
            {"context_size", peer.context_size}, {"active", peer.active},
            {"waiting", peer.segment_done || (options.telepathy && link_yield_to == int(1 - index))}, {"boundary", peer.boundary}, {"forced_splices", peer.forced_splices},
            {"tool_calls", peer.tool_calls}, {"tool_status", peer.tool_status}, {"answer_done", peer.answer_done},
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
            if ((options.telepathy || options.paragraph_splice) && mode == "error") {
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
        } else if (action == "link_on" || action == "link_off") {
            if (!options.telepathy || (mode != "thinking" && mode != "paused") || peers[0].tape.empty()) {
                throw std::logic_error("link controls require an active telepathy conversation");
            }
            if (action == "link_on" && (peers[0].answer_done || peers[1].answer_done)) {
                throw std::logic_error("both twins must still be thinking to enable the link");
            }
        } else if (action == "reset") {
            ++epoch;
        } else {
            throw std::invalid_argument("unknown control operation");
        }
        busy = true;
        pending = {action, text, mode};
        mode = "paused";
        if (options.telepathy) { clear_link_wait(); }
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
    clear_link_wait();
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
    if (options.telepathy || !options.paragraph_splice || !peers[0].segment_done || !peers[1].segment_done) {
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
    if (stopping || epoch != generation_epoch || (options.telepathy && mode == "error")) {
        return false;
    }
    auto & peer = peers[index];
    auto & other = peers[1 - index];
    const size_t maximum = answer ? options.answer_tokens :
        (options.telepathy ? peer.generation_limit : options.chunk_tokens);
    if (packet.tokens.size() > maximum - received) {
        throw std::runtime_error("server generated more tokens than requested");
    }
    received += packet.tokens.size();
    for (llama_token token : packet.tokens) {
        if (token < 0 || token >= peer.info.at("n_vocab").get<int64_t>()) {
            throw std::runtime_error("server returned a token outside the vocabulary");
        }
        if (contains(peer.controls, token) && !(answer && contains(peer.eog, token)) &&
                !(!answer && native_tools(peer) && token == peer.close_token)) {
            throw std::runtime_error("server generated a forbidden reasoning/chat control token");
        }
    }
    const bool shared = options.telepathy && link_enabled && !answer;
    bool closed = false;
    if (options.telepathy) {
        if (!answer && !peer.thinking_open && !packet.tokens.empty()) {
            throw std::runtime_error("server streamed another packet after the private reasoning terminator");
        }
        for (llama_token token : packet.tokens) {
            if (closed) { throw std::runtime_error("server streamed tokens after the private reasoning terminator"); }
            closed = token == peer.close_token;
        }
        if (shared && other.tape.size() + packet.tokens.size() >= other.context_size) {
            throw std::runtime_error("peer context full while receiving telepathic tokens");
        }
    }
    for (llama_token token : packet.tokens) {
        // The next-user template supplies the assistant terminator exactly once.
        if (answer && contains(peer.eog, token)) {
            continue;
        }
        peer.tape.push_back(token);
        ++peer.generated;
        if (options.telepathy && !answer && token != peer.close_token) { peer.tool_rounds = 0; }
        if (!answer && native_tools(peer) && token == peer.close_token) {
            peer.thinking_open = false;
            other.inbox.clear();
            if (options.telepathy) {
                peer.private_pending = true;
                if (private_owner < 0) { private_owner = static_cast<int>(index); }
            }
        } else if (shared) {
            if (!other.thinking_open || other.answer_done) {
                throw std::runtime_error("shared token targeted a peer outside its thinking block");
            }
            other.tape.push_back(token);
            ++other.imported;
        } else if (!options.telepathy && !answer && peer.thinking_open && !other.answer_done && mode != "answering") {
            other.inbox.push_back(token);
        }
    }
    std::string content = metadata.value("content", std::string());
    if (options.telepathy && closed) {
        const auto end = content.find("</think>");
        if (end != std::string::npos) { content.resize(end); }
    }
    if (!content.empty()) {
        emit({{"type", shared ? "shared" : answer ? "answer" : "token"}, {"peer", peer_name(index)}, {"text", content}});
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
    if (peer.tools) { peer.tools->begin_turn(); }
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
            if (options.telepathy) { finish_link("final answer"); }
            peer.segment_done = false;
            peer.tool_status.clear();
            peer.inbox.clear();
            other.inbox.clear();
            emit({{"type", "answer_start"}, {"peer", peer_name(index)}});
            emit({{"type", "answer"}, {"peer", peer_name(index)}, {"text", assistant.value("content", std::string())}});
            if (!busy) {
                mode = other.answer_done ? "answered" :
                    (options.telepathy && !answer_requested ? "thinking" : "answering");
            }
            changed.notify_all();
            return false;
        }
        if (++peer.tool_rounds > options.max_tool_rounds) {
            throw std::runtime_error(options.telepathy
                ? "consecutive tool round limit reached without reasoning; increase --max-tool-rounds or reset"
                : "tool round limit reached before a shared paragraph; increase --max-tool-rounds or reset");
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
            json response;
            try {
                const auto parsed = arguments.is_string() ? json::parse(arguments.get<std::string>()) : arguments;
                if (!parsed.is_object()) {
                    throw std::runtime_error("tool arguments must be a JSON object");
                }
                if (options.telepathy && name == link_tool_name) {
                    if (!parsed.contains("enabled") || !parsed.at("enabled").is_boolean()) {
                        throw std::runtime_error("enabled must be a boolean");
                    }
                    const bool enabled = parsed.at("enabled").get<bool>();
                    const auto wait = parsed.value("yield_until", std::string());
                    if ((!wait.empty() && wait != "fragment" && wait != "sentence" && wait != "paragraph") ||
                            (!enabled && !wait.empty())) {
                        throw std::runtime_error("yield_until requires an enabled link and fragment, sentence, or paragraph");
                    }
                    if (enabled && (mode == "answering" || other.answer_done)) {
                        throw std::runtime_error("the linked thinking stage has ended; finish your separate answer");
                    }
                    request_link(enabled, static_cast<int>(index), wait);
                    response = {{"content", json::array({{{"type", "text"}, {"text",
                        std::string("Telepathic link ") + (enabled ? "ON" : "OFF") +
                        " requested. It takes effect at the next safe reasoning boundary; coordinator markers announce the change." +
                        (wait.empty() ? "" : " Your twin has the floor until the requested bounded boundary.")}}})}};
                } else {
                    if (!peer.tools) { throw std::runtime_error("unknown tool: " + name); }
                    lock.unlock();
                    response = peer.tools->call(name, parsed);
                    lock.lock();
                }
            } catch (const std::exception & exception) {
                if (!lock.owns_lock()) { lock.lock(); }
                response = {{"isError", true}, {"content", json::array({{
                    {"type", "text"}, {"text", std::string("Tool request failed: ") + exception.what() +
                        ". A transport failure may occur after execution; do not blindly repeat a side effect."},
                }})}};
            }
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
        const uint64_t reserve = (options.telepathy ? options.link_quantum + 128 : 2 * options.chunk_tokens) +
            budget + options.answer_tokens + 2;
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

void crossthink_session::clear_link_wait() {
    link_yield_to = -1;
    link_yield_tokens = 0;
    link_wait.clear();
}

void crossthink_session::request_link(bool enabled, int requester, const std::string & wait) {
    if (!enabled && !link_enabled) {
        link_pending = false;
        link_requested = false;
        clear_link_wait();
        changed.notify_all();
        return;
    }
    if (enabled && (peers[0].answer_done || peers[1].answer_done)) {
        throw std::logic_error("both twins must still be thinking to enable the link");
    }
    link_pending = true;
    link_requested = enabled;
    link_requester = requester;
    clear_link_wait();
    if (enabled && !wait.empty() && requester >= 0) {
        link_yield_to = 1 - requester;
        link_wait = wait;
    }
    changed.notify_all();
}

void crossthink_session::append_shared(const std::vector<llama_token> & tokens) {
    for (const auto & peer : peers) {
        if (peer.tape.size() + tokens.size() >= peer.context_size) {
            throw std::runtime_error("context full for the shared telepathic transcript");
        }
    }
    for (auto & peer : peers) {
        peer.tape.insert(peer.tape.end(), tokens.begin(), tokens.end());
    }
}

bool crossthink_session::apply_link() {
    if (!link_pending || peers[0].active || peers[1].active || private_owner >= 0 ||
            !peers[0].thinking_open || !peers[1].thinking_open ||
            peers[0].answer_done || peers[1].answer_done) {
        return false;
    }
    if (link_enabled != link_requested) {
        append_shared(link_markers[link_requested ? 1 : 0]);
        link_enabled = link_requested;
        link_speaker = -1;
        link_previous_boundary.clear();
        link_next = 0;
        emit({{"type", "link"}, {"enabled", link_enabled},
            {"peer", link_requester < 0 ? "coordinator" : peer_name(link_requester)},
            {"reason", "requested"}});
    }
    if (link_enabled && link_yield_to >= 0) {
        link_next = static_cast<size_t>(link_yield_to);
    }
    link_pending = false;
    emit_state();
    changed.notify_all();
    return true;
}

void crossthink_session::start_speaker(size_t index) {
    if (link_speaker == static_cast<int>(index)) { return; }
    const bool interrupted = link_speaker >= 0 && link_previous_boundary != "sentence" &&
        link_previous_boundary != "paragraph";
    append_shared(interrupted ? link_interrupt_labels[index] : link_labels[index]);
    const std::string text = "\n[" + std::string(peer_name(index)) +
        (interrupted ? " interrupts " + std::string(peer_name(1 - index)) : "") + "]: ";
    link_speaker = static_cast<int>(index);
    emit({{"type", "shared"}, {"peer", peer_name(index)}, {"text", text}, {"label", true}});
}

void crossthink_session::finish_link(const std::string & reason) {
    const bool was_enabled = link_enabled;
    link_enabled = false;
    link_pending = false;
    link_requested = false;
    link_requester = -1;
    link_speaker = -1;
    clear_link_wait();
    if (was_enabled) {
        // A completed answer has left its thinking block. Only the still-thinking
        // twin needs the marker; no answer or private tail crosses the link.
        for (auto & peer : peers) {
            if (peer.thinking_open && peer.tape.size() + link_markers[0].size() < peer.context_size) {
                peer.tape.insert(peer.tape.end(), link_markers[0].begin(), link_markers[0].end());
            }
        }
        emit({{"type", "link"}, {"enabled", false}, {"peer", "coordinator"}, {"reason", reason}});
    }
    changed.notify_all();
}

void crossthink_session::telepathy_worker(size_t index) {
    auto & peer = peers[index];
    auto & other = peers[1 - index];
    std::unique_lock<std::mutex> lock(mutex);
    while (!stopping) {
        changed.wait(lock, [&] {
            if (stopping) { return true; }
            if (busy || (mode != "thinking" && mode != "answering") || peer.answer_done) { return false; }
            if (private_owner >= 0) {
                return private_owner == static_cast<int>(index) && !other.active;
            }
            if (peer.private_pending) { return !other.active; }
            if (other.private_pending) { return false; }
            if (link_pending) { return !other.active; }
            return !link_enabled || (!other.active && link_next == index);
        });
        if (stopping) { break; }
        try {
            if (link_pending && private_owner < 0 && !peer.private_pending && !other.private_pending) {
                apply_link();
                if (link_pending || (link_enabled && link_next != index)) { continue; }
            }
            const bool answer = mode == "answering";
            if (answer || peer.private_pending || private_owner == static_cast<int>(index)) {
                // Tool and answer tails never run while another shared fragment is
                // being accepted. An off-link quantum already in flight may finish.
                if (other.active) { continue; }
                private_owner = static_cast<int>(index);
                peer.private_pending = false;
                peer.active = true;
                if (peer.thinking_open) {
                    peer.tape.push_back(peer.close_token);
                    peer.thinking_open = false;
                }
                tool_turn(index, epoch, lock);
                peer.active = false;
                private_owner = -1;
                link_speaker = -1;
                if (!stopping && mode != "error") {
                    apply_link();
                }
                emit_state();
                changed.notify_all();
                continue;
            }
            const uint64_t quantum = link_enabled && link_yield_to == static_cast<int>(index)
                ? std::min<uint64_t>(options.link_quantum, options.link_wait_tokens - link_yield_tokens)
                : options.link_quantum;
            const uint64_t budget = quantum + 8;
            const uint64_t reserve = budget + std::max(options.tool_tokens, options.answer_tokens) +
                options.answer_tokens + 2;
            const auto has_space = [&](const peer_state & target) {
                const size_t label = std::max(link_labels[index].size(), link_interrupt_labels[index].size());
                return target.tape.size() + reserve + label < target.context_size;
            };
            if (!has_space(peer) || (link_enabled && !has_space(other))) {
                error = std::string(peer_name(index)) + ": context full for further telepathy; request answers or reset";
                mode = "paused";
                clear_link_wait();
                emit({{"type", "error"}, {"message", error}});
                emit_state();
                changed.notify_all();
                continue;
            }
            const bool shared = link_enabled;
            if (shared) { start_speaker(index); }
            const auto prompt = peer.tape;
            const uint64_t generation_epoch = epoch;
            json bias = json::array();
            for (llama_token token : peer.controls) {
                if (token != peer.close_token) { bias.push_back(json::array({token, false})); }
            }
            json parameters = {
                {"n_predict", budget}, {"temperature", options.temperature},
                {"seed", (uint64_t(options.seed) + index + 2 * (peer.sequence++ % UINT32_MAX)) % UINT32_MAX},
                {"cache_prompt", true}, {"stream", true}, {"return_content", true},
                {"reasoning_budget_tokens", -1}, {"logit_bias", std::move(bias)}, {"stop", json::array()},
                {"preserved_tokens", json::array({"</think>"})},
                {"splice", {{"sentence_after", 1},
                    {"token_after", quantum}, {"stop_on_think_close", true}}},
            };
            peer.generation_limit = static_cast<uint32_t>(budget);
            peer.active = true;
            lock.unlock();
            size_t received = 0;
            json result;
            std::string failure;
            try {
                result = peer.transport->generate(prompt, parameters, [&](const server_token_wire::packet & packet) {
                    return receive(index, generation_epoch, false, received, packet);
                });
            } catch (const std::exception & exception) {
                failure = exception.what();
            }
            lock.lock();
            peer.active = false;
            changed.notify_all();
            if (stopping || epoch != generation_epoch || mode == "error") { continue; }
            if (!failure.empty()) { throw std::runtime_error(failure); }
            if (!result.is_object() || result.value("type", std::string()) != "done" ||
                    result.value("truncated", false) || !received ||
                    result.value("stop_type", std::string()) != "limit") {
                throw std::runtime_error("telepathic reasoning ended without a valid final boundary");
            }
            const auto boundary = result.value("splice_boundary", std::string());
            if (boundary == "tool") {
                if (peer.thinking_open || peer.tape.empty() || peer.tape.back() != peer.close_token) {
                    throw std::runtime_error("private tool boundary did not end with the reasoning terminator");
                }
                peer.private_pending = true;
                if (private_owner < 0) { private_owner = static_cast<int>(index); }
                if (shared && link_yield_to == static_cast<int>(index)) {
                    // A tool call does not let the current speaker evade a bounded
                    // yield. Count its preceding shared words, excluding </think>.
                    link_yield_tokens += received - 1;
                    if ((link_wait == "fragment" && received > 1) ||
                            link_yield_tokens >= static_cast<uint64_t>(options.link_wait_tokens)) {
                        clear_link_wait();
                        link_next = 1 - index;
                    }
                }
            } else {
                if (boundary != "paragraph" && boundary != "sentence" && boundary != "quantum" &&
                        (boundary != "limit" || received != budget)) {
                    throw std::runtime_error("completion did not report a valid telepathic boundary");
                }
                if (!peer.thinking_open) {
                    throw std::runtime_error("reasoning terminator was not reported as a private tool boundary");
                }
                peer.boundary = boundary;
                if (boundary == "paragraph") { peer.tool_rounds = 0; }
                if (boundary == "limit") { ++peer.forced_splices; }
                if (shared) {
                    ++exchanges;
                    link_previous_boundary = boundary;
                    bool keep_floor = false;
                    if (link_yield_to == static_cast<int>(index)) {
                        link_yield_tokens += received;
                        const bool done = link_wait == "fragment" || boundary == "paragraph" ||
                            (link_wait == "sentence" && boundary == "sentence") ||
                            link_yield_tokens >= static_cast<uint64_t>(options.link_wait_tokens);
                        if (done) { clear_link_wait(); }
                        else { keep_floor = true; }
                    }
                    link_next = keep_floor ? index : 1 - index;
                }
            }
            if (!busy && private_owner < 0) { apply_link(); }
            emit_state();
            changed.notify_all();
        } catch (const std::exception & exception) {
            peer.active = false;
            private_owner = -1;
            fail(std::string(peer_name(index)) + ": " + exception.what());
        }
    }
}

void crossthink_session::worker(size_t index) {
    if (options.telepathy) { telepathy_worker(index); return; }
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
        if ((options.telepathy || options.paragraph_splice) && mode == "error" && pending.action != "reset") {
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
            const uint64_t reserve = (options.telepathy ? options.link_quantum + 128 :
                (options.paragraph_splice ? 2 : 5) * options.chunk_tokens) + options.answer_tokens + 2 +
                (native_tools(peer) ? std::max(options.tool_tokens, options.answer_tokens) : 0);
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
            peer.private_pending = false;
            peer.boundary.clear();
            peer.forced_splices = 0;
            peer.tool_calls = 0;
            peer.tool_rounds = 0;
            peer.tool_status.clear();
        }
        round = 0;
        exchanges = 0;
        answer_requested = false;
        link_enabled = false;
        link_pending = false;
        link_requested = false;
        link_requester = -1;
        link_speaker = -1;
        link_next = 0;
        private_owner = -1;
        clear_link_wait();
        mode = "idle";
        emit({{"type", "reset"}});
    }
    if (message) {
        answer_requested = false;
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
            peer.private_pending = false;
        }
        if (options.telepathy) { private_owner = -1; }
        emit({{"type", "user"}, {"text", command.text}});
        mode = "thinking";
        if (options.telepathy) {
            clear_link_wait();
            link_speaker = -1;
            link_next = 0;
            if (link_enabled) { append_shared(link_markers[1]); }
            apply_link();
        }
    } else if (command.action == "answer") {
        if (options.telepathy) { answer_requested = true; finish_link("answers requested"); }
        for (auto & peer : peers) {
            if (peer.answer_done) { continue; }
            drain(peer);
            if (peer.tape.size() + (native_tools(peer) ? std::max(options.tool_tokens, options.answer_tokens)
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
        mode = peers[0].answer_done && peers[1].answer_done ? "answered" : "answering";
    } else if (command.action == "link_on" || command.action == "link_off") {
        // A final answer can arrive after command() validates the request but
        // before this quiescent boundary. Reject that late enable nonfatally.
        if (command.action == "link_on" && (peers[0].answer_done || peers[1].answer_done)) {
            emit({{"type", "notice"}, {"message", "The linked stage ended before the link could be enabled."}});
            mode = peers[0].answer_done && peers[1].answer_done ? "answered" : command.resume_mode;
            return;
        }
        request_link(command.action == "link_on", -1);
        apply_link();
        mode = peers[0].answer_done && peers[1].answer_done ? "answered" : command.resume_mode;
    } else if (command.action == "resume") {
        rendezvous();
        mode = (options.telepathy ? answer_requested : peers[0].answer_done || peers[1].answer_done) ? "answering" : "thinking";
    } else if (command.action == "pause" && peers[0].answer_done && peers[1].answer_done) {
        mode = "answered";
    }
}
