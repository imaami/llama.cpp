// Integration coverage with tests/gen-tiny-qwen35-mtp.py's seeded fixture.
#include "common.h"
#include "speculative.h"
#include "sampling.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory-hybrid.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void test_sampling(llama_model * model, llama_context * ctx) {
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    float * logits = llama_get_logits_ith(ctx, -1);
    check(logits != nullptr, "missing logits");
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (int i = 0; i < nv; ++i) {
            logits[i] = pattern == 0 ? 0.0f : pattern == 1 ? (float) (i % 17) : std::sin((float) i);
        }
        for (int k : {1, 20, 128}) {
            for (float temp : {0.0f, 1.0f}) {
                common_params_sampling fast;
                fast.top_k = k;
                fast.temp = temp;
                fast.seed = 123;
                fast.backend_sampling = false;
                common_params_sampling reference = fast;
                // A zero bias disables the shortcut without changing the distribution.
                reference.logit_bias.push_back({0, 0.0f});
                common_sampler_ptr a(common_sampler_init(model, fast));
                common_sampler_ptr b(common_sampler_init(model, reference));
                for (int j = 0; j < 16; ++j) {
                    const auto ta = common_sampler_sample(a.get(), ctx, -1);
                    const auto tb = common_sampler_sample(b.get(), ctx, -1);
                    check(ta == tb, "top-k shortcut changed sampled token");
                    common_sampler_accept(a.get(), ta, true);
                    common_sampler_accept(b.get(), tb, true);
                }
            }
        }
    }
    // The only grammar-valid token lies outside the initial top-k shortlist.
    for (int i = 0; i < nv; ++i) {
        logits[i] = (float) i;
    }
    for (bool grammar_first : {false, true}) {
        common_params_sampling p;
        p.top_k = 20;
        p.seed = 123;
        p.backend_sampling = false;
        p.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= \"x\"+"};
        common_sampler_ptr sampler(common_sampler_init(model, p));
        for (int j = 0; j < 8; ++j) {
            const auto token = common_sampler_sample(sampler.get(), ctx, -1, grammar_first);
            check(token == 3 + 'x', "top-k shortlist lost the grammar-valid token");
            common_sampler_accept(sampler.get(), token, true);
        }
    }
}

static void test_window(llama_model * model, int window, bool unified) {
    common_params p;
    p.n_ctx = 4096;
    p.n_batch = 32;
    p.n_ubatch = 32;
    p.n_parallel = 2;
    p.kv_unified = unified;
    p.n_gpu_layers = 0;
    p.cpuparams.n_threads = 2;
    p.cpuparams_batch.n_threads = 2;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    p.speculative.draft.n_max = 2;
    p.speculative.draft.n_window = window;
    p.speculative.draft.backend_sampling = false;
    auto cp = common_context_params_to_llama(p);
    llama_context_ptr target(llama_init_from_model(model, cp));
    check(target != nullptr, "target context init failed");
    auto * target_mem = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
    check(target_mem != nullptr, "fixture must have hybrid memory");
    auto draft_params = common_base_params_to_speculative(p);
    common_speculative_init_result init(draft_params, model, target.get());
    auto * draft = init.context();
    check(draft != nullptr, "MTP context init failed");
    if (window == 16) {
        check(llama_n_ctx(draft) < llama_n_ctx(target.get()), "draft cache was not capped");
    } else {
        check(llama_n_ctx(draft) == llama_n_ctx(target.get()), "full/INT_MAX window changed capacity");
    }
    p.speculative.draft.ctx_tgt = target.get();
    p.speculative.draft.ctx_dft = draft;
    common_speculative_ptr spec(common_speculative_init(p.speculative, 2));
    check(spec != nullptr, "MTP driver init failed");
    check(common_speculative_get_types(spec.get()) == p.speculative.types, "MTP driver missing");
    llama_tokens prompts[2];
    auto feed = [&](llama_seq_id seq, int count) {
        common_batch batch(target.get());
        for (int j = 0; j < count; ++j) {
            const int pos = (int) prompts[seq].size();
            const llama_token token = 3 + pos % 256;
            batch.add(token, pos, seq, true);
            prompts[seq].push_back(token);
        }
        check(llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target decode failed");
        check(common_speculative_process(spec.get(), batch), "MTP catch-up failed");
    };
    // Leave an inactive slot at a shallow position while the other passes its draft capacity.
    feed(1, 8);
    const auto inactive_min = llama_memory_seq_pos_min(llama_get_memory(draft), 1);
    const auto inactive_max = llama_memory_seq_pos_max(llama_get_memory(draft), 1);
    for (int i = 0; i < 40; ++i) {
        feed(0, 32);
    }
    check(llama_memory_seq_pos_min(llama_get_memory(draft), 1) == inactive_min &&
          llama_memory_seq_pos_max(llama_get_memory(draft), 1) == inactive_max, "inactive slot was trimmed");
    // A one-token batch takes the deferred path; begin must flush it before drafting.
    feed(0, 1);
    common_speculative_begin(spec.get(), 0, prompts[0]);
    const auto min = llama_memory_seq_pos_min(llama_get_memory(draft), 0);
    check(window == 16 ? min >= 1280 - window : min == 0, "wrong retained draft prefix");
    check(llama_memory_seq_pos_max(llama_get_memory(draft), 0) == 1280, "deferred catch-up missing");
    check(target_mem->get_mem_attn()->seq_pos_min(0) == 0, "target history was trimmed");

    llama_tokens proposed;
    auto & dp = common_speculative_get_draft_params(spec.get(), 0);
    dp.drafting = true;
    dp.n_max = 1;
    dp.pos0 = (llama_pos) prompts[0].size();
    dp.id_last = 3 + dp.pos0 % 256;
    dp.prompt = &prompts[0];
    dp.result = &proposed;
    common_speculative_draft(spec.get());
    check(proposed.size() == 1, "MTP ignored the per-call draft cap");
    common_speculative_accept(spec.get(), 0, 0);
    check(llama_memory_seq_rm(llama_get_memory(draft), 0, dp.pos0, -1), "draft rollback failed");
    check(target_mem->get_mem_attn()->seq_pos_min(0) == 0, "draft changed target memory");
    // This deferred row leads straight into the next draft and shares its first decode.
    feed(0, 1);
    proposed.clear();
    dp.drafting = true;
    dp.pos0 = (llama_pos) prompts[0].size();
    dp.id_last = 3 + dp.pos0 % 256;
    common_speculative_draft(spec.get());
    check(proposed.size() == 1, "draft with deferred catch-up failed");
    check(llama_memory_seq_pos_max(llama_get_memory(draft), 0) == dp.pos0, "batched catch-up missing");
    check(target_mem->get_mem_attn()->seq_pos_min(0) == 0, "batched draft changed target memory");
    // Resume the shallow slot after the deep slot has wrapped its draft cache repeatedly.
    feed(1, 8);
    check(llama_memory_seq_pos_max(llama_get_memory(draft), 1) == 15, "shallow slot resume failed");
    if (window == 0 && !unified) {
        test_sampling(model, target.get());
    }
    printf("window=%d unified=%d: passed\n", window, (int) unified);
}

static void test_embedded_catchup(llama_model * model, bool unified) {
    common_params p;
    p.n_ctx = 512;
    p.n_batch = 8;
    p.n_ubatch = 8;
    p.n_parallel = 2;
    p.kv_unified = unified;
    p.n_gpu_layers = 0;
    p.cpuparams.n_threads = 2;
    p.cpuparams_batch.n_threads = 2;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    p.speculative.draft.n_max = 1;
    p.speculative.draft.backend_sampling = false;
    llama_context_ptr target(llama_init_from_model(model, common_context_params_to_llama(p)));
    check(target != nullptr, "embedded target context init failed");
    auto draft_params = common_base_params_to_speculative(p);
    common_speculative_init_result init(draft_params, model, target.get());
    common_speculative_init_result reference_init(draft_params, model, target.get());
    auto * draft = init.context();
    auto * reference = reference_init.context();
    check(draft && reference, "embedded draft context init failed");
    p.speculative.draft.ctx_tgt = target.get();
    p.speculative.draft.ctx_dft = draft;
    common_speculative_ptr spec(common_speculative_init(p.speculative, 2));
    check(spec != nullptr, "embedded MTP driver init failed");

    const size_t n_embd = llama_model_n_embd_inp(model);
    const size_t n_state = llama_model_n_embd_out(model);
    std::vector<float> carry(n_state, 0.0f);
    llama_pos pos = 0;
    const auto feed = [&](std::initializer_list<bool> embedded) {
        common_batch input(target.get());
        std::vector<float> rows(embedded.size()*n_embd);
        size_t row = 0;
        for (bool is_embd : embedded) {
            if (is_embd) {
                for (size_t k = 0; k < n_embd; ++k) {
                    rows[row*n_embd + k] = 0.1f*std::sin((float) (17*pos + k));
                }
                // Unequal spatial coordinates catch loss of the full M-RoPE position.
                const llama_pos positions[] = { pos, pos + 3, pos + 7, 0 };
                input.add_embd({ rows.data() + row*n_embd, 1, n_embd }, positions, 0, true);
            } else {
                input.add(3 + pos, pos, 0, true);
            }
            ++pos;
            ++row;
        }
        check(llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, input.get()) == 0,
                "embedded target decode failed");
        check(common_speculative_process(spec.get(), input), "embedded MTP catch-up failed");

        // Independent eager reference: send each original input row with the previous
        // target hidden state, without using the speculative driver's batching code.
        for (int i = 0; i < input.size(); ++i) {
            const auto & t = input.tokens[i];
            common_batch one(reference);
            const int32_t idx = t.id != LLAMA_TOKEN_NULL
                ? one.add(t.id, t.pos[0], t.seq_id, false)
                : one.add_embd(t.embd, t.pos.data(), t.seq_id, false);
            one.set_embd_state(idx, { carry.data(), 1, n_state });
            check(llama_process(reference, LLAMA_PROCESS_TYPE_DECODE, one.get()) == 0,
                    "embedded eager reference failed");
            const float * h = llama_get_embeddings_nextn_ith(target.get(), i);
            check(h != nullptr, "embedded target hidden state missing");
            std::copy(h, h + n_state, carry.begin());
        }
    };

    feed({ false });
    check(llama_memory_seq_pos_max(llama_get_memory(draft), 0) == -1, "text row was not deferred");
    feed({ true, true });
    check(llama_memory_seq_pos_max(llama_get_memory(draft), 0) == 2, "embedded rows did not flush deferred text");
    feed({ false });
    check(llama_memory_seq_pos_max(llama_get_memory(draft), 0) == 2, "second text row was not deferred");
    feed({ false, true });
    check(llama_memory_seq_pos_max(llama_get_memory(draft), 0) == 5, "token-first mixed rows were deferred");
    feed({ true, false });
    check(llama_memory_seq_pos_max(llama_get_memory(draft), 0) == 7, "embedding-first mixed rows were lost");

    for (auto * ctx : { draft, reference }) {
        common_batch probe(ctx);
        const int32_t idx = probe.add(3 + pos, pos, 0, true);
        probe.set_embd_state(idx, { carry.data(), 1, n_state });
        check(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, probe.get()) == 0, "embedded probe decode failed");
    }
    const float * actual = llama_get_logits_ith(draft, -1);
    const float * expected = llama_get_logits_ith(reference, -1);
    check(actual && expected, "embedded probe logits missing");
    for (int i = 0; i < llama_vocab_n_tokens(llama_model_get_vocab(model)); ++i) {
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]) &&
                std::abs(actual[i] - expected[i]) <= 1e-4f*(1.0f + std::abs(expected[i])),
                "embedded catch-up changed probe logits");
    }
    printf("embedded catch-up unified=%d: passed\n", (int) unified);
}

int main(int argc, char ** argv) {
    if (argc != 2 || !std::ifstream(argv[1]).good()) {
        fprintf(stderr, "Generate fixture: python3 tests/gen-tiny-qwen35-mtp.py <model.gguf>\n");
        return 77;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    check(model != nullptr, "fixture load failed");
    for (bool unified : { false, true }) {
        test_window(model.get(), 0, unified);
        test_window(model.get(), 16, unified);
        test_window(model.get(), INT_MAX, unified);
        test_embedded_catchup(model.get(), unified);
    }
    return 0;
}
