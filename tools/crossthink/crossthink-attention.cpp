#include "crossthink.h"

#include <algorithm>

using json = crossthink_json;

static const char * attention_peer(size_t index) {
    return index ? "B" : "A";
}

json crossthink_session::queue_attention(size_t sender) {
    const size_t recipient = 1 - sender;
    auto & peer = peers[recipient];
    const auto now = std::chrono::steady_clock::now();
    const bool coalesced = peer.attention_pending ||
        (peer.last_attention != std::chrono::steady_clock::time_point{} &&
            now - peer.last_attention < std::chrono::seconds(2));
    if (!coalesced) {
        peer.attention_pending = true;
        // Never cancel native output, MCP calls, or result preparation.
        if (!blocked(recipient) && mode == "thinking" && !answer_requested && !peer.force_answer &&
                peer.active && peer.thinking_open &&
                (peer.phase == "waiting_reasoning" || peer.phase == "reasoning")) {
            ++peer.revision;
            peer.transport->cancel();
        }
    }
    emit({{"type", "attention"}, {"from", attention_peer(sender)}, {"to", attention_peer(recipient)},
        {"status", coalesced ? "coalesced" : "queued"}, {"text", "Your partner requests your attention."}});
    changed.notify_all();
    return {{"success", true}, {"text", "Attention requested from your partner. Continue your work without waiting."}};
}

bool crossthink_session::deliver_attention(size_t index, uint64_t generation_epoch,
        std::unique_lock<std::mutex> & lock) {
    auto & peer = peers[index];
    const bool wake = peer.answer_done;
    const std::string text = "\n<ct:attention from=\"" + std::string(attention_peer(1 - index)) + "\">\n"
        "Your partner would like an update. Check your inbox when useful and keep working independently; "
        "no reply or turn-taking barrier is required.\n</ct:attention>\n";
    peer.phase = "attention";
    lock.unlock();
    std::vector<llama_token> tokens;
    std::string failure;
    try {
        tokens = wake ? peer.transport->next_user(text) : peer.transport->literal_tokens(text);
    } catch (const std::exception & exception) {
        failure = exception.what();
    }
    lock.lock();
    if (stopping || generation_id(index) != generation_epoch || mode == "error" || blocked(index)) {
        return false;
    }
    const size_t reserve = std::max(options.tool_tokens, options.answer_tokens) + options.answer_tokens + 128;
    if (failure.empty() && (tokens.empty() || peer.tape.size() + tokens.size() + reserve >= peer.context_size)) {
        failure = "attention notice does not fit the remaining context";
    }
    for (const auto token : tokens) {
        if (token < 0 || token >= peer.info.at("n_vocab").get<int64_t>() ||
                (!wake && std::find(peer.controls.begin(), peer.controls.end(), token) != peer.controls.end())) {
            failure = "invalid token in attention notice";
            break;
        }
    }
    if (!failure.empty()) {
        peer.attention_pending = false;
        emit({{"type", "notice"}, {"message", std::string(attention_peer(index)) + ": " + failure}});
        return false;
    }
    if (wake) {
        if (peer.thinking_open) { peer.tape.push_back(peer.close_token); }
        if (!peer.tape.empty() && std::find(peer.eog.begin(), peer.eog.end(), peer.tape.back()) != peer.eog.end()) {
            peer.tape.pop_back();
        }
        peer.answer_done = false;
        peer.empty_response = false;
        peer.thinking_open = true;
        peer.private_pending = false;
        peer.thought_scanner = {};
        peer.thought_scanner.thinking = true;
        mode = "thinking";
    }
    peer.tape.insert(peer.tape.end(), tokens.begin(), tokens.end());
    peer.imported += tokens.size();
    peer.attention_pending = false;
    peer.last_attention = std::chrono::steady_clock::now();
    ++peer.attention_received;
    emit({{"type", "attention_result"}, {"peer", attention_peer(index)}, {"text", text}});
    emit({{"type", "attention"}, {"from", attention_peer(1 - index)}, {"to", attention_peer(index)},
        {"status", "delivered"}, {"text", "Your partner requests your attention."}});
    return true;
}
