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
}

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
    }

    for (const auto & option : std::vector<json>{
            {{"stream", true}}, {{"n", 2}}, {{"n_cmpl", 2}}, {{"return_tokens", false}},
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
