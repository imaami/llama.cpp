#include "ggml.h"
#include "llama.h"
#include "sampling.h"
#include "../src/llama-context.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static llama_token sample_dist(llama_sampler * sampler, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < (llama_token) logits.size(); ++token_id) {
        cur.push_back({ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0);
    GGML_ASSERT((size_t) cur_p.selected < cur_p.size);
    return cur_p.data[cur_p.selected].id;
}

static void test_dist_singleton_rng() {
    llama_sampler * singleton = llama_sampler_init_dist(4242);
    llama_sampler * control   = llama_sampler_init_dist(4242);

    sample_dist(singleton, { 0.0f });
    sample_dist(control,   { 0.0f, 0.0f });

    const std::vector<float> logits(256, 0.0f);
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(sample_dist(singleton, logits) == sample_dist(control, logits));
    }

    llama_sampler_free(singleton);
    llama_sampler_free(control);
}

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties((int32_t) probs.size(), (int32_t) last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

static void test_backend_sampled_id_fallback(llama_model * model) {
    auto * backend = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(backend, llama_sampler_init_greedy());
    llama_sampler_seq_config config = { 0, backend };
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    cparams.samplers = &config;
    cparams.n_samplers = 1;
    auto * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx);

    common_batch batch(ctx);
    batch.add(0, 0, 0, true);
    GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    llama_synchronize(ctx);
    const llama_token expected = llama_get_sampled_token_ith(ctx, 0);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(expected >= 0 && expected < n_vocab);
    auto * sampled = ctx->get_sampled_tokens();
    GGML_ASSERT(sampled);

    common_params_sampling params;
    params.temp = 0.0f;
    params.backend_sampling = true;
    auto * sampler = common_sampler_init(model, params);
    GGML_ASSERT(sampler);
    for (llama_token invalid : { LLAMA_TOKEN_NULL, llama_token(-2), llama_token(n_vocab), llama_token(INT_MAX) }) {
        sampled[0] = invalid;
        GGML_ASSERT(common_sampler_sample(sampler, ctx, 0) == expected);
    }
    sampled[0] = (expected + 1) % n_vocab;
    GGML_ASSERT(common_sampler_sample(sampler, ctx, 0) == sampled[0]);

    common_sampler_free(sampler);
    llama_free(ctx);
    llama_sampler_free(backend);
    printf("Invalid backend token ids fall back to CPU sampling\n");
}

static int test_common_top_k(const char * model_path) {
    if (!std::ifstream(model_path).good()) {
        fprintf(stderr, "Model fixture not found: %s\n", model_path);
        return 77;
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    GGML_ASSERT(model);

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx);

    common_batch batch(ctx);
    batch.add(0, 0, 0, true);
    GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    float * logits = llama_get_logits_ith(ctx, 0);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(n_vocab > 128);

    std::mt19937 rng(1234);
    for (const int k : { 1, 2, 3, 20, 64, 127, 128, 129, n_vocab, n_vocab + 1 }) {
        for (int mode = 0; mode < 8; ++mode) {
            common_params_sampling params;
            params.seed = 4321;
            params.top_k = k;
            params.temp = 1.0f;
            params.top_p = 1.0f;
            params.min_p = 0.0f;
            params.n_probs = 128;
            if (mode == 4) { params.penalty_repeat = 1.2f; }
            if (mode == 5) { params.dry_multiplier = 1.0f; }
            if (mode == 6) { params.top_n_sigma = 1.0f; }
            if (mode == 7) { params.logit_bias.push_back({ 7, 2.0f }); }

            auto control_params = params;
            // Disabled XTC consumes no RNG and prevents selection before the sampler chain.
            control_params.samplers.insert(control_params.samplers.begin(), COMMON_SAMPLER_TYPE_XTC);
            common_sampler * fast = common_sampler_init(model, params);
            common_sampler * control = common_sampler_init(model, control_params);
            GGML_ASSERT(fast && control);

            for (int step = 0; step < 32; ++step) {
                for (int i = 0; i < n_vocab; ++i) {
                    logits[i] = mode == 0 ? 0.0f : float(int(rng() % 33) - 16);
                    if (mode == 1 && i % 7 == 0) { logits[i] = -INFINITY; }
                    if (mode == 2 && i % 7 == 0) { logits[i] = -0.0f; }
                    if (mode == 3) { logits[i] = float(rng()) / float(rng.max()); }
                }

                const llama_token actual = common_sampler_sample(fast, ctx, 0);
                const llama_token expected = common_sampler_sample(control, ctx, 0);
                GGML_ASSERT(actual == expected);
                const auto * a = common_sampler_get_candidates(fast, false);
                const auto * b = common_sampler_get_candidates(control, false);
                GGML_ASSERT(a->size == b->size && a->selected == b->selected && a->sorted == b->sorted);
                for (size_t i = 0; i < a->size; ++i) {
                    GGML_ASSERT(a->data[i].id == b->data[i].id);
                    GGML_ASSERT(std::memcmp(&a->data[i].logit, &b->data[i].logit, sizeof(float)) == 0);
                    GGML_ASSERT(std::memcmp(&a->data[i].p, &b->data[i].p, sizeof(float)) == 0);
                }
                common_sampler_accept(fast, actual, true);
                common_sampler_accept(control, expected, true);
                if (step == 8) {
                    auto * cloned = common_sampler_clone(fast);
                    common_sampler_free(fast);
                    fast = cloned;
                }
                if (step == 16) {
                    auto * copied = common_sampler_init(model, params);
                    common_sampler_copy(fast, copied);
                    common_sampler_free(fast);
                    fast = copied;
                }
            }
            common_sampler_free(fast);
            common_sampler_free(control);
        }
    }

    // The required token is outside the initial top-k shortlist.
    for (int i = 0; i < n_vocab; ++i) {
        logits[i] = float(i);
    }
    llama_token token_x = LLAMA_TOKEN_NULL;
    for (llama_token i = 0; i < n_vocab; ++i) {
        if (common_token_to_piece(llama_model_get_vocab(model), i) == "x") {
            token_x = i;
            break;
        }
    }
    GGML_ASSERT(token_x >= 0 && token_x < n_vocab - 20);
    for (bool grammar_first : { false, true }) {
        common_params_sampling params;
        params.top_k = 20;
        params.seed = 1234;
        params.grammar = { COMMON_GRAMMAR_TYPE_USER, "root ::= \"x\"+" };
        auto * sampler = common_sampler_init(model, params);
        for (int step = 0; step < 8; ++step) {
            const auto token = common_sampler_sample(sampler, ctx, 0, grammar_first);
            GGML_ASSERT(token == token_x);
            common_sampler_accept(sampler, token, true);
        }
        common_sampler_free(sampler);
    }
    {
        common_params_sampling params;
        params.top_k = 20;
        params.reasoning_budget_start = { 1 };
        params.reasoning_budget_end = { { 2 } };
        params.reasoning_budget_forced = { 7, 2 };
        params.reasoning_budget_tokens = 16;
        auto * sampler = common_sampler_init(model, params);
        common_sampler_accept(sampler, 1, true);
        GGML_ASSERT(common_sampler_reasoning_budget_force(sampler));
        GGML_ASSERT(common_sampler_sample(sampler, ctx, 0) == 7);
        common_sampler_free(sampler);
    }

    llama_free(ctx);
    test_backend_sampled_id_fallback(model);
    llama_model_free(model);
    llama_backend_free();
    printf("Common top-k candidates, probabilities and tokens match\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2) {
        return test_common_top_k(argv[1]);
    }

    ggml_time_init();

    test_dist_singleton_rng();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    printf("OK\n");

    test_perf();

    return 0;
}
