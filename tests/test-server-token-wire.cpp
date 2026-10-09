// Without arguments, test the codec. Pass gen-tiny-qwen35-mtp.py's GGUF to test routes without sockets.
#include "../tools/server/server-context.h"
#include "../tools/server/server-token-wire.h"

#include <cstdio>
#include <functional>
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

    route_fixture(const char * model, bool mtp) : routes(params, context) {
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
        if (mtp) {
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            params.speculative.draft.n_max = 2;
            params.speculative.draft.backend_sampling = false;
        }
        check(context.load_model(params), "fixture load failed");
        routes.update_meta(context);
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
        const auto packet = server_token_wire::decode(result.second, 259);
        return {json::parse(packet.metadata), packet.tokens};
    }

    streamed_result binary_stream(json options, const llama_tokens & prompt) {
        options["stream"] = true;
        const std::function<bool()> stop = [] { return false; };
        const server_http_req request = {{}, {}, {}, {}, server_token_wire::encode(options.dump(), prompt), {}, stop};
        auto response = routes.post_completions_tokens(request);
        check(response && response->status == 200 && response->is_stream(), "binary stream request failed");
        check(response->content_type == "application/octet-stream", "wrong stream content type");
        server_token_wire::stream_decoder parser(259, options.at("n_predict").get<size_t>());
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
                    check(type == "done" && p.tokens.empty(), "invalid terminal frame: " + p.metadata);
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

    for (const auto & option : std::vector<json>{
            {{"stream", 1}}, {{"n", 2}}, {{"n_cmpl", 2}}, {{"return_tokens", false}},
            {{"n_probs", 1}}, {{"logprobs", 1}}, {{"response_fields", json::array({"content"})}},
            {{"prompt", json::array({1})}}, {{"n_predict", -1}}, {{"n_predict", 0}}}) {
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

int main(int argc, char ** argv) {
    try {
        check(argc <= 2, "usage: test-server-token-wire [tiny-qwen35-mtp.gguf]");
        test_codec();
        if (argc == 2) {
            llama_backend_init();
            test_routes(argv[1], false);
            test_routes(argv[1], true);
            llama_backend_free();
        }
        std::puts("token wire tests: passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "token wire tests: %s\n", error.what());
        return 1;
    }
}
