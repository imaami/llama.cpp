// Without arguments, test the codec. Pass gen-tiny-qwen35-mtp.py's GGUF to test routes without sockets.
#include "../tools/server/server-context.h"
#include "../tools/server/server-token-wire.h"
#include "../tools/server/server-splice.h"
#include "../tools/server/server-thought.h"
#include "../tools/crossthink/crossthink-speech.h"

#include <cstdio>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <utility>

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_codec() {
    const llama_tokens tokens = {0, 1, 258};
    const std::string encoded = server_token_wire::encode("{}", tokens);
    const auto decoded = server_token_wire::decode(encoded, 259);
    check(decoded.metadata == "{}" && decoded.tokens == tokens, "codec round trip");
    const std::vector<std::string> invalid = {
        "LLMTOK01", "BADTOK01" + encoded.substr(8), encoded + "x", encoded.substr(0, encoded.size() - 1),
        server_token_wire::encode("{}", {259}),
    };
    for (const auto & body : invalid) {
        bool rejected = false;
        try {
            server_token_wire::decode(body, 259);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "codec accepted malformed packet");
    }
    std::string oversized = encoded;
    oversized[8] = 1;
    oversized[9] = 0;
    oversized[10] = 1;
    bool rejected = false;
    try {
        server_token_wire::decode(oversized, 259);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "codec accepted oversized metadata");

    const std::string stream = encoded + server_token_wire::encode("{\"type\":\"done\"}", {});
    for (size_t stride = 1; stride <= stream.size(); ++stride) {
        server_token_wire::stream_decoder parser(259, tokens.size());
        std::vector<server_token_wire::packet> packets;
        for (size_t offset = 0; offset < stream.size(); offset += stride) {
            parser.feed(stream.data() + offset, std::min(stride, stream.size() - offset), [&](server_token_wire::packet p) {
                packets.push_back(std::move(p));
            });
        }
        parser.finish();
        check(packets.size() == 2 && packets[0].tokens == tokens && packets[1].tokens.empty(), "fragmented stream mismatch");
    }
    std::vector<std::string> invalid_streams = invalid;
    invalid_streams.push_back(oversized);
    invalid_streams.push_back(server_token_wire::encode("{}", {1, 2, 3, 4}));
    for (size_t size = 1; size < encoded.size(); ++size) {
        invalid_streams.push_back(encoded.substr(0, size));
    }
    for (const auto & body : invalid_streams) {
        rejected = false;
        try {
            server_token_wire::stream_decoder parser(259, 3);
            parser.feed(body.data(), body.size(), [](server_token_wire::packet) {});
            parser.finish();
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "stream decoder accepted malformed packet");
    }
}

static void test_splice_scanner() {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"First\n\n", "paragraph"}, {"First\r\n \t\r\n", "paragraph"},
        {"First\n\nNext", ""}, {"First\n\n \t", "paragraph"}, {"\n\n", ""},
        {"First. ", "sentence"}, {"First.)\" \t", "sentence"}, {"First. Next", ""},
        {"First\r\n\r", ""}, {"First.\r\n", "sentence"},
        {"```cpp\ncode\n\n", ""}, {"~~~cpp\ncode\n\n", ""},
        {"```cpp\ncode\n```\n\n", "paragraph"}, {"~~~cpp\ncode\n~~~\n\n", "paragraph"},
        {"````cpp\n```\n\n", ""}, {"```cpp\n```oops\n\n", ""},
        {"  ```cpp\ncode. \n\n", ""}, {"  ```cpp\ncode\n  ``` \n\n", "paragraph"},
        {"```info. ", ""}, {"``inline``\n\n", "paragraph"},
    };
    for (const auto & item : cases) {
        for (size_t stride = 1; stride <= item.first.size(); ++stride) {
            server_splice_scanner scanner;
            for (size_t offset = 0; offset < item.first.size(); offset += stride) {
                scanner.feed(item.first.substr(offset, stride));
            }
            check(scanner.boundary(true) == item.second, "splice scan mismatch: " + item.first);
        }
    }
    server_splice_scanner scanner;
    scanner.feed("Prompt.\n");
    scanner.begin_generation();
    scanner.feed("\n \t");
    check(std::string(scanner.boundary(true)).empty(), "prompt text counted as generated paragraph");
    scanner.feed("Next. ");
    check(std::string(scanner.boundary(false)).empty(), "sentence fallback was enabled too early");
    check(std::string(scanner.boundary(true)) == "sentence", "sentence fallback did not trigger");
    scanner = {};
    scanner.feed("```cpp\ncode");
    scanner.begin_generation();
    scanner.feed("\n\nmore. ");
    check(std::string(scanner.boundary(true)).empty(), "prompt fence state was lost");
    scanner.feed("\n```\n\n");
    check(std::string(scanner.boundary(true)) == "paragraph", "continued fence did not close");
}

static void test_speech_progress() {
    for (const std::string & text : {std::string(), std::string(" \t\r\n- . [] :"),
            std::string("\xe2\x80\x93\xe2\x80\x94\xe2\x80\xa2\xe2\x80\xa6"),
            std::string("\xc2\xa0\xe2\x80\x89\xe3\x80\x80")}) {
        check(!crossthink_speech_guard::substantive(text), "punctuation or whitespace counted as speech");
    }
    for (const std::string & text : {std::string("a"), std::string("7"), std::string("\xc3\xa9"),
            std::string("\xe6\x80\x9d")}) {
        check(crossthink_speech_guard::substantive(text), "ordinary speech did not count as progress");
    }
}

static void test_thought_scanner() {
    const std::vector<std::pair<std::string, std::string>> commands = {
        {"<ct:peek/>", "peek"}, {"<ct:inbox/>", "inbox"},
        {"<ct:send>hello\n\xe6\x80\x9d <ct:peek/> </think></ct:send>", "send"},
    };
    for (const auto & item : commands) {
        for (size_t split = 0; split < item.first.size(); ++split) {
            server_thought_scanner scanner;
            scanner.thinking = true;
            scanner.feed("ordinary prose\n" + item.first.substr(0, split));
            check(scanner.command.empty(), "partial thought command executed");
            server_thought_scanner resumed;
            resumed.thinking = true;
            resumed.restore_state(scanner.continuation_state());
            resumed.feed(scanner.continuation());
            resumed.feed(item.first.substr(split) + "trailing text<ct:inbox/>");
            check(resumed.command == item.second, "thought command split or first-command recognition failed");
            check(resumed.continuation().empty(), "completed command retained a continuation");
            if (item.second == "send") {
                check(resumed.payload == "hello\n\xe6\x80\x9d <ct:peek/> </think>", "thought payload was modified or nested command executed");
            }
        }
        server_thought_scanner inactive;
        inactive.feed(item.first);
        check(inactive.command.empty() && inactive.continuation().empty(), "command outside reasoning executed");
    }
    const std::vector<std::string> inert_commands = {
        "Maybe check inbox with <ct:inbox/>.", " <ct:peek/>", "\t<ct:peek/>", "- <ct:inbox/>",
        "`<ct:peek/>`", "<ct:nope><ct:peek/>",
        "```text\n<ct:inbox/>\n```", "~~~text\r\n<ct:inbox/>\r\n~~~",
        "  ```text\n<ct:send>example</ct:send>\n  ```",
        "````text\n```\n<ct:inbox/>\n````",
        "```text\n```oops\n<ct:inbox/>",
    };
    for (const auto & text : inert_commands) {
        for (size_t split = 0; split <= text.size(); ++split) {
            server_thought_scanner scanner;
            scanner.thinking = true;
            scanner.feed(text.substr(0, split));
            server_thought_scanner resumed;
            resumed.thinking = true;
            resumed.restore_state(scanner.continuation_state());
            resumed.feed(scanner.continuation());
            resumed.feed(text.substr(split));
            check(scanner.command.empty() && resumed.command.empty(), "quoted or inline thought command executed: " + text);
        }
    }
    for (const std::string & prefix : {std::string(), std::string("words\n"), std::string("words\r\n"),
            std::string("words\r"), std::string("```text\n<ct:inbox/>\n```\n"),
            std::string("  ~~~~text\r\n<ct:inbox/>\r\n  ~~~~\r\n"), std::string("```no`fence\n")}) {
        const auto text = prefix + "<ct:peek/>";
        for (size_t split = 0; split < text.size(); ++split) {
            server_thought_scanner scanner;
            scanner.thinking = true;
            scanner.feed(text.substr(0, split));
            check(scanner.command.empty(), "thought command fired before closing tag");
            server_thought_scanner resumed;
            resumed.thinking = true;
            resumed.restore_state(scanner.continuation_state());
            resumed.feed(scanner.continuation());
            resumed.feed(text.substr(split));
            check(resumed.command == "peek", "line/fence continuation lost a valid command: " + text);
        }
    }
    server_thought_scanner scanner;
    scanner.thinking = true;
    scanner.feed("malformed <ct:peek> <ct:send /> <ct:inbox / >\n<ct:pe");
    check(scanner.command.empty() && scanner.continuation() == "<ct:pe", "malformed syntax confused command prefix");
    scanner = {};
    scanner.thinking = true;
    scanner.feed("<ct:send>" + std::string(server_thought_scanner::max_payload + 7, 'x') + "</ct:");
    check(scanner.payload.size() == server_thought_scanner::max_payload, "oversized thought payload was not bounded");
    server_thought_scanner resumed;
    resumed.thinking = true;
    resumed.feed(scanner.continuation());
    resumed.feed("send>");
    check(resumed.command == "send" && resumed.payload.empty() && !resumed.error.empty(), "oversized thought continuation lost error");
    scanner = {};
    scanner.thinking = true;
    scanner.feed("<ct:send>" + std::string(server_thought_scanner::max_escaped_payload / 6 + 1, '\1') + "</ct:send>");
    check(scanner.command == "send" && scanner.payload.empty() && !scanner.error.empty(),
        "JSON escaping allowed a thought payload to exceed wire metadata limits");
}

struct streamed_result {
    json metadata;
    llama_tokens tokens;
    std::string content;
    std::vector<server_token_wire::packet> packets;
};

struct route_fixture {
    common_params params;
    server_context context;
    server_routes routes;
    std::thread worker;
    int32_t n_vocab = 0;

    route_fixture(const char * model, bool mtp, bool tool_template = false) : routes(params, context) {
        params.model.path = model;
        params.n_ctx = 512;
        params.n_batch = params.n_ubatch = 64;
        params.n_gpu_layers = 0;
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 2;
        params.n_parallel = 1;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.warmup = false;
        params.fit_params = false;
        params.ctx_shift = false;
        params.sampling.backend_sampling = false;
        if (tool_template) {
            std::ifstream file("models/templates/Qwen3.5-4B.jinja");
            check(file.good(), "tool chat template missing");
            params.chat_template.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }
        if (mtp) {
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            params.speculative.draft.n_max = 2;
            params.speculative.draft.backend_sampling = false;
        }
        check(context.load_model(params), "fixture load failed");
        routes.update_meta(context);
        n_vocab = context.get_meta().model_vocab_n_tokens;
        worker = std::thread([this] { context.start_loop(); });
    }

    ~route_fixture() {
        context.terminate();
        if (worker.joinable()) {
            worker.join();
        }
    }

    std::pair<int, std::string> call(const server_http_context::handler_t & handler, const std::string & body) {
        const std::function<bool()> stop = [] { return false; };
        const server_http_req request = {{}, {}, {}, {}, body, {}, stop};
        auto response = handler(request);
        check(response && !response->is_stream(), "unexpected streaming response");
        response->on_complete();
        return {response->status, response->data};
    }

    std::pair<json, llama_tokens> binary(const json & options, const llama_tokens & prompt) {
        const auto result = call(routes.post_completions_tokens, server_token_wire::encode(options.dump(), prompt));
        check(result.first == 200, "binary route: " + result.second);
        const auto packet = server_token_wire::decode(result.second, n_vocab);
        return {json::parse(packet.metadata), packet.tokens};
    }

    streamed_result binary_stream(json options, const llama_tokens & prompt, bool allow_error = false) {
        options["stream"] = true;
        const std::function<bool()> stop = [] { return false; };
        const server_http_req request = {{}, {}, {}, {}, server_token_wire::encode(options.dump(), prompt), {}, stop};
        auto response = routes.post_completions_tokens(request);
        check(response && response->status == 200 && response->is_stream(), "binary stream request failed");
        check(response->content_type == "application/octet-stream", "wrong stream content type");
        server_token_wire::stream_decoder parser(n_vocab, options.at("n_predict").get<size_t>());
        streamed_result result;
        bool done = false;
        bool next;
        do {
            std::string chunk;
            next = response->next(chunk);
            parser.feed(chunk.data(), chunk.size(), [&](server_token_wire::packet p) {
                check(!done, "packet after done");
                const json metadata = json::parse(p.metadata);
                const std::string type = metadata.at("type");
                if (type == "tokens") {
                    check(!p.tokens.empty(), "empty token frame");
                    result.tokens.insert(result.tokens.end(), p.tokens.begin(), p.tokens.end());
                    result.content += metadata.value("content", std::string());
                } else {
                    check((type == "done" || (allow_error && type == "error")) && p.tokens.empty(),
                        "invalid terminal frame: " + p.metadata);
                    result.metadata = metadata;
                    done = true;
                }
                result.packets.push_back(std::move(p));
            });
        } while (next);
        parser.finish();
        response->on_complete();
        check(done, "stream lacks done frame");
        return result;
    }

    json ordinary(json options, const llama_tokens & prompt) {
        options["prompt"] = prompt;
        const auto result = call(routes.post_completions, options.dump());
        check(result.first == 200, "ordinary route: " + result.second);
        return json::parse(result.second);
    }
};

static void test_routes(const char * model, bool mtp) {
    route_fixture fixture(model, mtp);
    auto & routes = fixture.routes;
    const auto info_response = fixture.call(routes.get_tokens_info, "");
    check(info_response.first == 200, "info route failed");
    const json info = json::parse(info_response.second);
    check(info.at("protocol") == "LLMTOK01" && info.at("n_vocab") == 259, "wrong fixture vocabulary");
    check(info.at("stream") == true, "stream support not advertised");
    check(info.at("splice") == true, "splice support not advertised");
    check(info.at("splice_quantum") == true, "quantum splice support not advertised");
    check(info.at("fingerprint") == json::parse(fixture.call(routes.get_tokens_info, "").second).at("fingerprint"), "unstable fingerprint");

    const llama_tokens prompt = {1, 100, 101, 102};
    const json options = {
        {"n_predict", 12}, {"temperature", 0}, {"seed", 7}, {"ignore_eos", true},
        {"cache_prompt", false}, {"return_tokens", true},
    };
    const json ordinary = fixture.ordinary(options, prompt);
    const auto binary = fixture.binary(options, prompt);
    check(binary.second == ordinary.at("tokens").get<llama_tokens>(), "binary/JSON token mismatch");
    check(binary.second.size() == 12, "wrong generated token count");
    check(binary.first.at("tokens_evaluated") == prompt.size(), "prompt IDs changed");
    const auto streamed = fixture.binary_stream(options, prompt);
    check(streamed.tokens == binary.second, "stream/nonstream token mismatch");
    check(streamed.metadata.at("tokens_predicted") == binary.second.size(), "stream token count mismatch");
    json no_content = options;
    no_content["return_content"] = false;
    for (const auto & packet : fixture.binary_stream(no_content, prompt).packets) {
        check(!json::parse(packet.metadata).contains("content"), "return_content false ignored");
    }
    if (mtp) {
        check(binary.first.at("timings").at("draft_n").get<int>() > 0, "MTP never drafted");
    }

    llama_tokens base = {1};
    for (llama_token token = 80; token < 112; ++token) {
        base.push_back(token);
    }
    json cached = options;
    cached["cache_prompt"] = true;
    llama_tokens own = base;
    own.insert(own.end(), {41, 42, 43});
    fixture.binary(cached, own);
    llama_tokens swapped = base;
    swapped.insert(swapped.end(), {51, 52, 53, 54});
    const auto changed = fixture.binary(cached, swapped);
    const auto fresh = fixture.ordinary(options, swapped);
    check(changed.second == fresh.at("tokens").get<llama_tokens>(), "cached reasoning replacement differs from fresh evaluation");

    for (const auto & marker : std::vector<std::pair<std::string, llama_token>>{{"q", 3 + 'q'}, {"<s>", 1}}) {
        json stopped = options;
        stopped["stop"] = json::array({marker.first});
        stopped["preserved_tokens"] = json::array({marker.first});
        stopped["logit_bias"] = json::array({json::array({marker.second, 1000})});
        const auto result = fixture.binary(stopped, prompt);
        check(result.second == llama_tokens{marker.second}, "raw stop token lost");
        check(result.first.at("content") == "" && result.first.at("stop_type") == "word", "stop text not removed");
        check(result.first.at("stopping_word") == marker.first, "wrong stopping marker");
        const auto streamed_stop = fixture.binary_stream(stopped, prompt);
        check(streamed_stop.tokens == result.second, "stream stop token lost or duplicated");
        check(streamed_stop.content.empty() && streamed_stop.metadata.at("stop_type") == "word", "stream stop text not removed");
    }

    json eos = options;
    eos["ignore_eos"] = false;
    eos["logit_bias"] = json::array({json::array({2, 1000})});
    const auto ended = fixture.binary_stream(eos, prompt);
    check(ended.tokens == llama_tokens{2} && ended.metadata.at("stop_type") == "eos", "stream lost EOG token");
    check(ended.content.empty(), "EOG should have no display text");

    json utf8 = options;
    utf8["ignore_eos"] = false;
    utf8["grammar"] = "root ::= \"\\u00e9\"";
    const auto utf8_plain = fixture.binary(utf8, prompt);
    const auto utf8_stream = fixture.binary_stream(utf8, prompt);
    check(utf8_stream.tokens == utf8_plain.second, "UTF-8 stream token mismatch");
    check(utf8_stream.tokens == llama_tokens({3 + 0xc3, 3 + 0xa9, 2}), "UTF-8 fixture produced unexpected tokens");
    check(utf8_stream.content == "\xc3\xa9", "UTF-8 display chunks were lost");
    check(json::parse(utf8_stream.packets[0].metadata).at("content") == "", "partial UTF-8 exposed as display text");
    utf8["n_predict"] = 1;
    check(fixture.binary_stream(utf8, prompt).tokens == llama_tokens{3 + 0xc3}, "stream lost incomplete UTF-8 at token limit");

    json quantum = options;
    quantum["n_predict"] = 11;
    quantum["splice"] = {{"token_after", 3}};
    quantum["grammar"] = "root ::= \"abcdef\"";
    const auto short_plain = fixture.binary(quantum, prompt);
    const auto short_stream = fixture.binary_stream(quantum, prompt);
    check(short_plain.second == llama_tokens({3 + 'a', 3 + 'b', 3 + 'c'}) && short_stream.tokens == short_plain.second,
        "quantum did not stop at the requested committed token count");
    check(short_stream.content == "abc" && short_plain.first.at("splice_boundary") == "quantum" &&
        short_stream.metadata.at("splice_boundary") == "quantum", "quantum boundary missing");

    quantum["n_predict"] = 9;
    quantum["ignore_eos"] = false;
    quantum["splice"]["token_after"] = 1;
    for (const std::string & text : {std::string("\xc3\xa9"), std::string("\xe2\x82\xac"), std::string("\xf0\x9f\x90\xba")}) {
        quantum["grammar"] = "root ::= " + json(text + "x").dump();
        const auto complete = fixture.binary_stream(quantum, prompt);
        check(complete.content == text && complete.tokens.size() == text.size(), "quantum split a UTF-8 character");
        check(complete.metadata.at("splice_boundary") == "quantum", "completed UTF-8 lacks quantum marker");
        check(json::parse(complete.packets.front().metadata).at("content") == "", "quantum exposed a partial UTF-8 character");
    }
    quantum["n_predict"] = 1;
    const auto incomplete = fixture.binary_stream(quantum, prompt, true);
    check(incomplete.metadata.at("type") == "error", "quantum hard limit silently accepted incomplete UTF-8");
    check(incomplete.content.empty(), "quantum hard limit exposed incomplete UTF-8 text");
    check(fixture.call(routes.post_completions_tokens, server_token_wire::encode(quantum.dump(), prompt)).first == 500,
        "nonstream quantum hard limit silently accepted incomplete UTF-8");

    quantum["n_predict"] = 16;
    quantum["splice"] = {{"sentence_after", 1}, {"token_after", 3}};
    quantum["grammar"] = "root ::= \"A. Next\"";
    const auto clean_sentence = fixture.binary_stream(quantum, prompt);
    check(clean_sentence.content == "A. " && clean_sentence.metadata.at("splice_boundary") == "sentence",
        "quantum took priority over a clean sentence");
    quantum["splice"]["sentence_after"] = 4096;
    quantum["grammar"] = "root ::= \"A\\n\\nNext\"";
    const auto clean_paragraph = fixture.binary_stream(quantum, prompt);
    check(clean_paragraph.content == "A\n\n" && clean_paragraph.metadata.at("splice_boundary") == "paragraph",
        "quantum took priority over a clean paragraph");

    quantum["splice"] = {{"sentence_after", 1}, {"token_after", 1}};
    quantum["grammar"] = "root ::= \"\\nNext\"";
    const auto continued_paragraph = fixture.binary_stream(quantum, {1, 3 + 'A', 3 + '\n'});
    check(continued_paragraph.content == "\n" && continued_paragraph.metadata.at("splice_boundary") == "paragraph",
        "quantum lost a paragraph separator across requests");
    quantum["grammar"] = "root ::= \" Next\"";
    const auto continued_sentence = fixture.binary_stream(quantum, {1, 3 + 'A', 3 + '.'});
    check(continued_sentence.content == " " && continued_sentence.metadata.at("splice_boundary") == "sentence",
        "quantum lost a sentence boundary across requests");
    quantum["grammar"] = "root ::= \"\\nNext\"";
    const auto empty_paragraph = fixture.binary_stream(quantum, {1, 3 + '\n'});
    check(empty_paragraph.metadata.at("splice_boundary") == "quantum", "quantum treated an empty prefix as a paragraph");

    llama_tokens labeled_prompt = {1};
    for (unsigned char byte : std::string("\n[A]: ")) {
        labeled_prompt.push_back(3 + byte);
    }
    const std::string labeled_text = "\n\nNow I have a point to explain.";
    std::string displayed;
    size_t fragments = 0;
    while (displayed.size() < labeled_text.size()) {
        const auto remaining = labeled_text.substr(displayed.size());
        const size_t count = std::min<size_t>(8, remaining.size());
        quantum["n_predict"] = count + 8;
        quantum["splice"] = {{"sentence_after", 1}, {"token_after", count}};
        quantum["grammar"] = "root ::= " + json(remaining).dump();
        const auto fragment = fixture.binary_stream(quantum, labeled_prompt);
        check(!fragment.content.empty() && remaining.compare(0, fragment.content.size(), fragment.content) == 0,
            "labeled quantum omitted or changed streamed text");
        llama_tokens expected;
        for (unsigned char byte : fragment.content) {
            expected.push_back(3 + byte);
        }
        check(fragment.tokens == expected, "labeled quantum text and raw token IDs diverged");
        if (!fragments) {
            // The scanner sees the coordinator's label as content. The coordinator must distinguish silent fragments.
            check(fragment.content == "\n\n" && fragment.metadata.at("splice_boundary") == "paragraph",
                "label-seeded whitespace no longer reproduces the early paragraph boundary");
        }
        displayed += fragment.content;
        labeled_prompt.insert(labeled_prompt.end(), fragment.tokens.begin(), fragment.tokens.end());
        ++fragments;
    }
    check(displayed == labeled_text && fragments >= 3, "labeled multi-quantum display lost content");

    crossthink_speech_guard speech;
    json guarded = options;
    guarded["n_predict"] = 10;
    guarded["splice"] = {{"token_after", 2}};
    guarded["grammar_lazy"] = false;
    guarded["backend_sampling"] = false;
    guarded["repeat_last_n"] = 64;
    guarded["presence_penalty"] = 10000;
    guarded["logit_bias"] = json::array({json::array({3 + '[', 1000}), json::array({3 + 'A', 900}),
        json::array({3 + 'B', 800}), json::array({3 + 'C', 700})});
    check(fixture.binary_stream(guarded, {1}).content == "[A", "unconstrained label fixture did not reproduce an echo");
    guarded["grammar"] = speech.grammar();
    check(fixture.binary_stream(guarded, {1}).content == "[C", "speech grammar allowed a reserved label prefix");

    guarded["presence_penalty"] = 0;
    guarded["n_predict"] = 9;
    guarded["splice"]["token_after"] = 1;
    const auto opening = fixture.binary_stream(guarded, {1});
    check(opening.content == "[", "speech grammar rejected an undecided bracket prefix");
    speech.feed(opening.content);
    guarded["grammar"] = speech.grammar();
    guarded["logit_bias"] = json::array({json::array({3 + 'A', 1000}), json::array({3 + 'B', 900}),
        json::array({3 + 'C', 800})});
    check(fixture.binary_stream(guarded, {1, 3 + '['}).content == "C",
        "speech grammar lost the reserved prefix across quantum requests");

    speech.reset();
    speech.feed("Inline math ");
    guarded["grammar"] = speech.grammar();
    guarded["n_predict"] = 11;
    guarded["splice"]["token_after"] = 3;
    guarded["presence_penalty"] = 10000;
    guarded["logit_bias"] = json::array({json::array({3 + '[', 1000}), json::array({3 + 'A', 900}),
        json::array({3 + ']', 800})});
    check(fixture.binary_stream(guarded, {1}).content == "[A]", "speech grammar rejected inline mathematical brackets");

    speech.feed("\n \t");
    guarded["grammar"] = speech.grammar();
    guarded["n_predict"] = 10;
    guarded["splice"]["token_after"] = 2;
    guarded["logit_bias"] = json::array({json::array({3 + '[', 1000}), json::array({3 + 'A', 900}),
        json::array({3 + 'B', 800}), json::array({3 + 'C', 700})});
    check(fixture.binary_stream(guarded, {1}).content == "[C", "newline did not restore the reserved-label guard");
    speech.feed("ordinary text\r");
    guarded["grammar"] = speech.grammar();
    check(fixture.binary_stream(guarded, {1}).content == "[C", "bare carriage return did not restore the reserved-label guard");
    guarded["n_predict"] = 9;
    guarded["splice"]["token_after"] = 1;
    guarded["logit_bias"] = json::array({json::array({3 + 0xc3, 1000}), json::array({3 + 0xa9, 900})});
    check(fixture.binary_stream(guarded, {1}).content == "\xc3\xa9", "speech grammar rejected multibyte Unicode");

    json splice = options;
    splice["n_predict"] = 96;
    splice["ignore_eos"] = false;
    splice["splice"] = {{"sentence_after", 4096}};
    const llama_tokens splice_prompt = {1};
    const std::vector<std::pair<std::string, std::string>> paragraphs = {
        {"First.\n\nSecond.", "First.\n\n"},
        {"First.\r\n \t\r\nSecond.", "First.\r\n \t\r\n"},
        {"```cpp\ncode\n\nmore\n```\n\nSecond.", "```cpp\ncode\n\nmore\n```\n\n"},
        {"\n\nFirst.\n\nSecond.", "\n\nFirst.\n\n"},
    };
    for (const auto & item : paragraphs) {
        splice["grammar"] = "root ::= " + json(item.first).dump();
        const auto plain = fixture.binary(splice, splice_prompt);
        const auto stream = fixture.binary_stream(splice, splice_prompt);
        check(plain.first.at("content") == item.second && stream.content == item.second, "splice did not preserve paragraph separator: " + json(item.first).dump());
        check(plain.second == stream.tokens && stream.tokens.size() == item.second.size(), "splice raw IDs lost or duplicated");
        check(stream.metadata.at("splice_boundary") == "paragraph" && stream.metadata.at("stop_type") == "limit", "wrong paragraph stop marker");
    }
    splice["splice"]["sentence_after"] = 1;
    splice["grammar"] = "root ::= \"First.)\\\" Next.\"";
    const auto sentence = fixture.binary_stream(splice, splice_prompt);
    check(sentence.content == "First.)\" " && sentence.metadata.at("splice_boundary") == "sentence", "sentence fallback failed");

    splice["splice"]["sentence_after"] = 4096;
    splice["cache_prompt"] = true;
    const std::string code_prefix = "```cpp\ncode\n\n";
    splice["n_predict"] = code_prefix.size();
    splice["grammar"] = "root ::= " + json(code_prefix + "rest").dump();
    const auto cut = fixture.binary_stream(splice, splice_prompt);
    check(cut.content == code_prefix && cut.metadata.at("splice_boundary") == "limit", "hard cut inside fence failed");
    llama_tokens continued = splice_prompt;
    continued.insert(continued.end(), cut.tokens.begin(), cut.tokens.end());
    splice["n_predict"] = 64;
    splice["grammar"] = "root ::= \"rest\\n\\ncode\\n```\\n\\nNext\"";
    const auto continued_code = fixture.binary_stream(splice, continued);
    check(continued_code.content == "rest\n\ncode\n```\n\n" && continued_code.metadata.at("splice_boundary") == "paragraph", "continued request lost fence state");

    // A splice can stop inside an accepted MTP batch. Its cached continuation must match a fresh evaluation.
    continued.insert(continued.end(), continued_code.tokens.begin(), continued_code.tokens.end());
    json continuation = options;
    continuation["cache_prompt"] = true;
    const auto reused = fixture.binary(continuation, continued);
    continuation["cache_prompt"] = false;
    check(reused.second == fixture.binary(continuation, continued).second, "splice poisoned cached continuation");

    for (const auto & option : std::vector<json>{
            {{"stream", 1}}, {{"n", 2}}, {{"n_cmpl", 2}}, {{"return_tokens", false}},
            {{"n_probs", 1}}, {{"logprobs", 1}}, {{"response_fields", json::array({"content"})}},
            {{"prompt", json::array({1})}}, {{"n_predict", -1}}, {{"n_predict", 0}},
            {{"splice", true}}, {{"splice", json::object()}}, {{"splice", {{"sentence_after", 0}}}},
            {{"splice", {{"sentence_after", -1}}}}, {{"splice", {{"sentence_after", 4097}}}},
            {{"splice", {{"sentence_after", 1.5}}}}, {{"splice", {{"sentence_after", 1}, {"unknown", true}}}},
            {{"splice", {{"token_after", 0}}}}, {{"splice", {{"token_after", -1}}}},
            {{"splice", {{"token_after", 4097}}}}, {{"splice", {{"token_after", UINT64_MAX}}}},
            {{"splice", {{"token_after", 1.5}}}}, {{"splice", {{"token_after", true}}}},
            {{"splice", {{"token_after", "1"}}}}, {{"splice", {{"token_after", 1}, {"unknown", true}}}},
            {{"splice", {{"sentence_after", 1}}}, {"stop", json::array({"x"})}},
            {{"splice", {{"sentence_after", 1}}}, {"n_indent", 1}},
            {{"splice", {{"sentence_after", 1}}}, {"t_max_predict_ms", 1}}}) {
        json invalid = options;
        for (const auto & field : option.items()) {
            invalid[field.key()] = field.value();
        }
        const auto result = fixture.call(routes.post_completions_tokens, server_token_wire::encode(invalid.dump(), prompt));
        check(result.first == 400, "invalid option accepted: " + option.dump());
    }
    std::vector<std::string> invalid = {
        "LLMTOK01", server_token_wire::encode("[1]", prompt), server_token_wire::encode("{", prompt),
        server_token_wire::encode(options.dump(), {}), server_token_wire::encode(options.dump(), {259}),
        server_token_wire::encode(options.dump(), llama_tokens(info.at("context_size").get<size_t>(), 1)),
    };
    for (const auto & body : invalid) {
        check(fixture.call(routes.post_completions_tokens, body).first == 400, "invalid request accepted");
    }
    std::printf("binary completion routes (MTP=%d): passed\n", int(mtp));
}

static void test_tool_routes(const char * model, bool mtp) {
    route_fixture fixture(model, mtp, true);
    auto & routes = fixture.routes;
    const auto info = json::parse(fixture.call(routes.get_tokens_info, "").second);
    check(info.at("n_vocab") == 263 && info.at("tool_parse") == true, "wrong tool fixture or missing tool capability");
    check(info.at("thought_commands") == true && info.at("thought_command_lines") == true,
        "line-anchored thought command capability missing");
    auto tokenize = [&](const std::string & text) {
        const auto result = fixture.call(routes.post_tokenize,
            json{{"content", text}, {"add_special", false}, {"parse_special", true}}.dump());
        check(result.first == 200, "tokenizer route failed");
        return json::parse(result.second).at("tokens").get<llama_tokens>();
    };
    const auto close = tokenize("</think>");
    check(close.size() == 1, "tool fixture has no single-token think close");
    const auto eog = tokenize("<|im_end|>");
    check(eog.size() == 1, "tool fixture has no single-token ChatML EOG");

    const std::string literal = "</think><|im_end|><ct:peek/>";
    const auto literal_response = fixture.call(routes.post_tokenize,
        json{{"content", literal}, {"add_special", false}, {"parse_special", false}}.dump());
    check(literal_response.first == 200, "literal tokenizer route failed");
    const auto literal_tokens = json::parse(literal_response.second).at("tokens").get<llama_tokens>();
    for (auto token : literal_tokens) {
        check(token >= 3 && token < 259, "literal injected thought became a control token");
    }
    const auto literal_decoded = fixture.call(routes.post_detokenize, json{{"tokens", literal_tokens}}.dump());
    // This byte-only SentencePiece fixture adds an escaped dummy space before the literal bytes.
    check(literal_decoded.first == 200 && json::parse(literal_decoded.second).at("content") == "\xe2\x96\x81" + literal,
        "literal injected thought failed to round trip: " + literal_decoded.second);

    const llama_tokens thinking_prompt = {1, 259};
    json thoughts = {
        {"n_predict", 128}, {"seed", 42}, {"temperature", 0}, {"cache_prompt", true},
        {"splice", {{"thought_commands", true}, {"stop_on_think_close", true}}},
    };
    const std::vector<std::pair<std::string, std::string>> thought_cases = {
        {"First.\n\n<ct:peek/>", "peek"}, {"<ct:inbox/>", "inbox"},
        {"<ct:send>hello\n\xe6\x80\x9d <ct:peek/></ct:send>", "send"},
    };
    for (const auto & item : thought_cases) {
        thoughts["grammar"] = "root ::= " + json(item.first + "unseen suffix").dump();
        const auto plain_command = fixture.binary(thoughts, thinking_prompt);
        const auto streamed_command = fixture.binary_stream(thoughts, thinking_prompt);
        check(plain_command.second == streamed_command.tokens && streamed_command.content == item.first,
            "thought boundary stopped early or exposed post-command tokens");
        check(streamed_command.metadata.at("splice_boundary") == "thought_command" &&
            streamed_command.metadata.at("splice_reason") == "thought_command" &&
            streamed_command.metadata.at("thought_command") == item.second, "wrong thought command metadata");
        if (item.second == "send") {
            check(streamed_command.metadata.at("thought_payload") == "hello\n\xe6\x80\x9d <ct:peek/>", "wire thought payload changed");
        }
        auto continued = thinking_prompt;
        continued.insert(continued.end(), streamed_command.tokens.begin(), streamed_command.tokens.end());
        const auto injection = tokenize("\n[received thought]\n");
        continued.insert(continued.end(), injection.begin(), injection.end());
        auto next = thoughts;
        next["grammar"] = "root ::= \"reply<ct:inbox/>\"";
        const auto cached = fixture.binary(next, continued);
        next["cache_prompt"] = false;
        check(cached.second == fixture.binary(next, continued).second,
            "thought stop poisoned cached continuation after injection");
    }

    for (const std::string & text : {std::string("Check inbox with <ct:inbox/>."), std::string("  <ct:peek/>"),
            std::string("```text\n<ct:inbox/>\n```"), std::string("~~~text\r\n<ct:peek/>\r\n~~~")}) {
        thoughts["grammar"] = "root ::= " + json(text).dump();
        const auto quoted = fixture.binary_stream(thoughts, thinking_prompt);
        check(quoted.content == text && quoted.metadata.at("splice_boundary") != "thought_command",
            "inline or fenced thought example executed on the server");
    }
    for (const std::string & head : {std::string("Check inbox with "), std::string("```text\n"),
            std::string("``")}) {
        thoughts["grammar"] = "root ::= " + json(head).dump();
        thoughts["n_predict"] = head.size();
        const auto first = fixture.binary_stream(thoughts, thinking_prompt);
        check(first.content == head && first.metadata.contains("thought_state"), "thought position checkpoint missing");
        auto prompt = thinking_prompt;
        prompt.insert(prompt.end(), first.tokens.begin(), first.tokens.end());
        thoughts["splice"]["thought_state"] = first.metadata.at("thought_state");
        const auto tail = (head == "``" ? std::string("`text\n") : std::string()) + "<ct:inbox/>";
        thoughts["grammar"] = "root ::= " + json(tail).dump();
        thoughts["n_predict"] = 128;
        const auto continued = fixture.binary_stream(thoughts, prompt);
        check(continued.content == tail && continued.metadata.at("splice_boundary") != "thought_command",
            "request boundary armed an inline/fenced command");
        thoughts["splice"].erase("thought_state");
    }
    thoughts["grammar"] = "root ::= \"<ct:peek/>tail\"";
    const auto outside = fixture.binary_stream(thoughts, tokenize("<think>done</think>"));
    check(outside.content == "<ct:peek/>tail" && outside.metadata.at("splice_boundary") != "thought_command",
        "answer content executed a thought command");
    thoughts["grammar"] = "root ::= \"ordinary answer\"";
    auto imported = thinking_prompt;
    imported.insert(imported.end(), literal_tokens.begin(), literal_tokens.end());
    const auto inert = fixture.binary_stream(thoughts, imported);
    check(inert.content == "ordinary answer" && inert.metadata.at("splice_boundary") != "thought_command",
        "imported prefix executed a thought command");

    thoughts["n_predict"] = 6;
    thoughts["grammar"] = "root ::= \"<ct:peek/>\"";
    const auto partial = fixture.binary_stream(thoughts, thinking_prompt);
    check(partial.content == "<ct:pe" && partial.metadata.at("thought_prefix") == "<ct:pe",
        "limited request lost partial command");
    auto resumed_prompt = thinking_prompt;
    resumed_prompt.insert(resumed_prompt.end(), partial.tokens.begin(), partial.tokens.end());
    thoughts["n_predict"] = 32;
    thoughts["grammar"] = "root ::= \"ek/>tail\"";
    thoughts["splice"]["thought_prefix"] = partial.metadata.at("thought_prefix");
    thoughts["splice"]["thought_state"] = partial.metadata.at("thought_state");
    const auto resumed_command = fixture.binary_stream(thoughts, resumed_prompt);
    check(resumed_command.content == "ek/>" && resumed_command.metadata.at("thought_command") == "peek",
        "partial command did not resume across requests");
    for (const json & bad_prefix : {json("<ct:peek/>"), json("ordinary prose"), json(std::string(16417, 'x')), json(1)}) {
        thoughts["splice"]["thought_prefix"] = bad_prefix;
        check(fixture.call(routes.post_completions_tokens, server_token_wire::encode(thoughts.dump(), resumed_prompt)).first == 400,
            "invalid thought_prefix accepted");
    }
    thoughts["splice"].erase("thought_prefix");
    for (const json & bad_state : {json(false), json{{"unknown", true}}, json{{"line_start", 1}},
            json{{"fence", "x"}}, json{{"fence", "`"}, {"fence_size", 2}}, json{{"line_indent", 4}},
            json{{"fence_size", -1}}, json{{"line_marker_size", 1}},
            json{{"line_start", true}, {"line_marker", "`"}, {"line_marker_size", 1}}}) {
        thoughts["splice"]["thought_state"] = bad_state;
        check(fixture.call(routes.post_completions_tokens, server_token_wire::encode(thoughts.dump(), resumed_prompt)).first == 400,
            "invalid thought_state accepted");
    }
    thoughts["splice"]["thought_state"] = json::object();
    thoughts["splice"]["thought_commands"] = false;
    check(fixture.call(routes.post_completions_tokens, server_token_wire::encode(thoughts.dump(), resumed_prompt)).first == 400,
        "thought_state accepted without thought_commands");
    json options = {
        {"n_predict", 16}, {"seed", 42}, {"temperature", 0}, {"cache_prompt", false},
        {"preserved_tokens", json::array({"</think>"})}, {"grammar", "root ::= \"</think>\""},
        {"logit_bias", json::array({json::array({close.front(), 1000})})},
        {"splice", {{"sentence_after", 4096}, {"token_after", 1}, {"stop_on_think_close", true}}},
    };
    const llama_tokens prompt = {1, 259};
    const auto plain = fixture.binary(options, prompt);
    const auto streamed = fixture.binary_stream(options, prompt);
    check(plain.second == close && streamed.tokens == close, "think closure lost or included following raw tokens");
    check(plain.first.at("splice_boundary") == "tool" && streamed.metadata.at("splice_boundary") == "tool",
        "think closure did not report tool boundary");
    check(streamed.metadata.at("stop_type") == "limit", "think closure has wrong stop type");

    for (const char * prefix : {"", "[", "ordinary text"}) {
        crossthink_speech_guard speech;
        speech.feed(prefix);
        auto guarded = options;
        guarded["grammar"] = speech.grammar();
        guarded["grammar_lazy"] = false;
        guarded["backend_sampling"] = false;
        const auto ended = fixture.binary_stream(guarded, prompt);
        check(ended.tokens == close && ended.metadata.at("splice_boundary") == "tool",
            "speech grammar blocked native reasoning closure");
    }

    options["splice"]["stop_on_think_close"] = false;
    options["splice"].erase("token_after");
    options["n_predict"] = 2;
    options.erase("grammar");
    const auto disabled = fixture.binary_stream(options, prompt);
    check(disabled.tokens == llama_tokens({close.front(), close.front()}), "ordinary splice unexpectedly stopped at think close");
    check(disabled.metadata.at("splice_boundary") == "limit", "ordinary splice reported a tool boundary");
    options["splice"]["stop_on_think_close"] = "yes";
    check(fixture.call(routes.post_completions_tokens, server_token_wire::encode(options.dump(), prompt)).first == 400,
        "invalid think-close option accepted");

    const json clean = {
        {"n_predict", 16}, {"seed", 42}, {"temperature", 0}, {"cache_prompt", false},
        {"grammar", "root ::= \"First\\n\\nNext\""}, {"splice", {{"sentence_after", 4096}, {"token_after", 12}}},
    };
    for (const char * marker : {"<|im_start|>", "<think>", "</think>"}) {
        const auto history = tokenize(std::string("```cpp\nold fenced text\n") + marker + "new thought\n");
        const auto fresh = fixture.binary_stream(clean, history);
        check(fresh.content == "First\n\n" && fresh.metadata.at("splice_boundary") == "paragraph",
            "scanner prefix optimization retained a fence before the latest reset marker");
    }

    const auto output = tokenize("</think>\n\n<tool_call>\n<function=calculator>\n<parameter=expression>\n1+1\n</parameter>\n</function>\n</tool_call>");
    json body = {
        {"messages", json::array({{{"role", "user"}, {"content", "Calculate 1 + 1"}}})},
        {"tools", json::array({{{"type", "function"}, {"function", {
            {"name", "calculator"}, {"description", "Evaluate an expression"},
            {"parameters", {{"type", "object"}, {"properties", {{"expression", {{"type", "string"}}}}},
                {"required", json::array({"expression"})}}},
        }}}})},
        {"add_generation_prompt", true}, {"chat_template_kwargs", {{"enable_thinking", true}}},
        {"parse_output", output},
    };
    const auto parsed = fixture.call(routes.post_apply_template, body.dump());
    check(parsed.first == 200, "tool parser route failed");
    const auto message = json::parse(parsed.second).at("message");
    check(message.at("tool_calls").size() == 1, "native route did not parse tool call");
    const auto & function = message.at("tool_calls").front().at("function");
    check(function.at("name") == "calculator" && json::parse(function.at("arguments").get<std::string>()).at("expression") == "1+1",
        "native route changed tool name or arguments");
    for (llama_token token : {2, eog.front()}) {
        body["parse_output"] = output;
        body["parse_output"].push_back(token);
        const auto ended = fixture.call(routes.post_apply_template, body.dump());
        check(ended.first == 200 && json::parse(ended.second).at("message") == message, "parser did not trim final EOG");
    }
    const auto native_tools = body["tools"];
    for (const bool with_tools : {false, true}) {
        body["tools"] = with_tools ? native_tools : json::array();
        for (const std::string & content : {std::string("Answer **42**."),
                std::string("<ct:send>My plan</ct:send>"), std::string("<ct:inbox/>")}) {
            // Model output uses the byte tokens directly. Tokenizing this fixture's text instead
            // escapes each space to literal SentencePiece U+2581 bytes, changing the test answer.
            auto answer_tokens = close;
            for (const unsigned char byte : "\n\n" + content) { answer_tokens.push_back(3 + byte); }
            answer_tokens.push_back(eog.front());
            body["parse_output"] = answer_tokens;
            const auto answer = fixture.call(routes.post_apply_template, body.dump());
            check(answer.first == 200, "native answer parser route failed");
            const auto decoded = json::parse(answer.second).at("message");
            check(decoded.at("content") == content && decoded.value("tool_calls", json::array()).empty(),
                "native answer text was lost or treated as a thought/native tool command: expected " +
                    json(content).dump() + ", got " + decoded.dump());
        }
    }
    body["tools"] = native_tools;
    for (const std::string & prefix : {std::string(), std::string("My plan is ready.\n")}) {
        body["parse_output"] = tokenize("</think>\n\n" + prefix +
            "<tool_call>\n<function=ct:send>\n<parameter=message>\nMy plan\n</parameter>\n</function>\n</tool_call>");
        body["parse_output"].push_back(eog.front());
        bool rejected = false;
        try { rejected = fixture.call(routes.post_apply_template, body.dump()).first != 200; }
        catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "native parser route silently discarded an unknown tool call");
    }
    for (const json & invalid : std::vector<json>{
            "invalid", json::array({-1}), json::array({263}), json::array({1.5}),
            json::array({UINT64_MAX}), json::array({2, 3}), json(llama_tokens(513, 3))}) {
        body["parse_output"] = invalid;
        bool rejected = false;
        try {
            rejected = fixture.call(routes.post_apply_template, body.dump()).first == 400;
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "invalid parse_output accepted");
    }
    std::printf("native tool routes (MTP=%d): passed\n", int(mtp));
}

int main(int argc, char ** argv) {
    try {
        check(argc <= 3, "usage: test-server-token-wire [tiny-qwen35-mtp.gguf [tiny-qwen35-tools.gguf]]");
        test_codec();
        test_splice_scanner();
        test_thought_scanner();
        test_speech_progress();
        if (argc >= 2) {
            llama_backend_init();
            test_routes(argv[1], false);
            test_routes(argv[1], true);
            if (argc == 3) {
                test_tool_routes(argv[2], false);
                test_tool_routes(argv[2], true);
            }
            llama_backend_free();
        }
        std::puts("token wire tests: passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "token wire tests: %s\n", error.what());
        return 1;
    }
}
