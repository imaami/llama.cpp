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
    }
    return 0;
}
