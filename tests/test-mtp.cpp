// Integration coverage with tests/gen_tiny_qwen35_mtp.py's seeded fixture.
#include "common.h"
#include "speculative.h"
#include "sampling.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
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

static void test_draft_state_output(llama_model * model, int n_max, bool backend_sampling) {
    common_params p;
    p.n_ctx = 512;
    p.n_batch = 8;
    p.n_ubatch = 8;
    p.n_parallel = 1;
    p.n_gpu_layers = 0;
    p.cpuparams.n_threads = 2;
    p.cpuparams_batch.n_threads = 2;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    p.speculative.draft.n_max = n_max;
    p.speculative.draft.backend_sampling = backend_sampling;
    llama_context_ptr target(llama_init_from_model(model, common_context_params_to_llama(p)));
    check(target != nullptr, "state output target init failed");
    auto draft_params = common_base_params_to_speculative(p);
    common_speculative_init_result candidate_init(draft_params, model, target.get());
    common_speculative_init_result reference_init(draft_params, model, target.get());
    llama_context * contexts[] = { candidate_init.context(), reference_init.context() };
    common_speculative_ptr specs[2];
    for (int i = 0; i < 2; ++i) {
        check(contexts[i] != nullptr, "state output draft init failed");
        p.speculative.draft.ctx_tgt = target.get();
        p.speculative.draft.ctx_dft = contexts[i];
        specs[i].reset(common_speculative_init(p.speculative, 1));
        check(specs[i] != nullptr, "state output driver init failed");
    }
    // Retain the old readback in the reference without changing the draft limit.
    llama_set_embeddings_nextn(contexts[1], true, true);

    llama_tokens prompt;
    const auto feed = [&](const llama_tokens & tokens) {
        common_batch batch(target.get());
        for (llama_token token : tokens) {
            batch.add(token, (llama_pos) prompt.size(), 0, true);
            prompt.push_back(token);
        }
        check(llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                "state output target decode failed");
        for (auto & spec : specs) {
            check(common_speculative_process(spec.get(), batch), "state output catch-up failed");
        }
    };
    feed({ 3, 4, 5, 6, 7, 8, 9, 10 });
    for (auto & spec : specs) {
        common_speculative_begin(spec.get(), 0, prompt);
    }

    for (int round = 0; round < 8; ++round) {
        const int per_call_max = round % 2 + 1;
        const llama_token anchor = 3 + (llama_token) prompt.size();
        llama_tokens proposed[2];
        for (int i = 0; i < 2; ++i) {
            auto & dp = common_speculative_get_draft_params(specs[i].get(), 0);
            dp.drafting = true;
            dp.n_max = per_call_max;
            dp.pos0 = (llama_pos) prompt.size();
            dp.id_last = anchor;
            dp.prompt = &prompt;
            dp.result = &proposed[i];
            common_speculative_draft(specs[i].get());
        }
        check(proposed[0].size() == (size_t) std::min(n_max, per_call_max), "wrong state output draft length");
        check(proposed[0] == proposed[1], "state output changed draft tokens");
        check((llama_get_embeddings_nextn(contexts[0]) != nullptr) == (n_max != 1),
                "wrong draft hidden-state readback presence");
        check(llama_get_embeddings_nextn(contexts[1]) != nullptr, "reference hidden state missing");

        const uint32_t n_sampled = llama_get_sampled_logits_count_ith(contexts[0], -1);
        check(n_sampled == llama_get_sampled_logits_count_ith(contexts[1], -1), "state output changed logits count");
        check(!backend_sampling || n_sampled == 10, "state output backend top-k was not exercised");
        const uint32_t n_logits = n_sampled ? n_sampled : llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * actual = llama_get_logits_ith(contexts[0], -1);
        const float * expected = llama_get_logits_ith(contexts[1], -1);
        check(actual && expected, "state output logits missing");
        for (uint32_t i = 0; i < n_logits; ++i) {
            check(std::isfinite(actual[i]) && actual[i] == expected[i], "state output changed draft logits");
        }
        const uint32_t n_candidates = llama_get_sampled_candidates_count_ith(contexts[0], -1);
        check(n_candidates == llama_get_sampled_candidates_count_ith(contexts[1], -1), "state output changed candidate count");
        if (n_candidates) {
            const auto * actual_ids = llama_get_sampled_candidates_ith(contexts[0], -1);
            const auto * expected_ids = llama_get_sampled_candidates_ith(contexts[1], -1);
            check(actual_ids && expected_ids, "state output candidates missing");
            check(std::equal(actual_ids, actual_ids + n_candidates, expected_ids), "state output changed candidates");
        }

        // Alternate rejected and accepted drafts, including a two-token accepted prefix.
        const uint16_t n_accepted = (uint16_t) (round % (proposed[0].size() + 1));
        llama_tokens accepted = { anchor };
        accepted.insert(accepted.end(), proposed[0].begin(), proposed[0].begin() + n_accepted);
        for (int i = 0; i < 2; ++i) {
            check(llama_memory_seq_rm(llama_get_memory(contexts[i]), 0, (llama_pos) prompt.size(), -1),
                    "state output draft rollback failed");
        }
        feed(accepted);
        for (auto & spec : specs) {
            common_speculative_accept(spec.get(), 0, n_accepted);
        }
    }
    printf("draft state output n_max=%d backend_sampling=%d: passed\n", n_max, (int) backend_sampling);
}

int main(int argc, char ** argv) {
    if (argc != 2 || !std::ifstream(argv[1]).good()) {
        fprintf(stderr, "Generate fixture: python3 tests/gen_tiny_qwen35_mtp.py <model.gguf>\n");
        return 77;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    check(model != nullptr, "fixture load failed");
    for (bool unified : { false, true }) {
        test_embedded_catchup(model.get(), unified);
    }
    for (int n_max : { 1, 2 }) {
        for (bool backend_sampling : { false, true }) {
            test_draft_state_output(model.get(), n_max, backend_sampling);
        }
    }
    return 0;
}
