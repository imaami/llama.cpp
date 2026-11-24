// Compare greedy decoding with and without periodic partial rollback.
// Each verification batch contains the accepted anchor followed by hypothetical draft tokens.
// Discarding those drafts must preserve the reference greedy token stream on every sequence.
#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_token greedy(llama_context * ctx, const llama_vocab * vocab) {
    const int     n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits  = llama_get_logits_ith(ctx, -1);

    llama_token best   = 0;
    float       best_v = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best   = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt    = "Explain how a jet engine works, step by step.";
    int         n_predict = 128;
    int         stride    = 16;  // roll back every `stride` accepted tokens
    int         rewind    = 4;   // how many tokens to discard and re-generate
    int         n_rs_seq  = 8;   // maximum number of tokens to discard

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            stride = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            rewind = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-k") && i + 1 < argc) {
            n_rs_seq = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            prompt = argv[i];
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-n N] [-s stride] [-r rewind] [-k n_rs_seq]\n", argv[0]);
        return 1;
    }

    if (rewind <= 0 || stride <= 0 || n_predict <= 0 || n_rs_seq < rewind) {
        fprintf(stderr, "rewind, stride and length must be positive, and -k must cover -r\n");
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 99;
    llama_model * model        = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int                n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> ptoks(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), ptoks.data(), ptoks.size(), true, true);

    int n_rollbacks = 0;

    auto run = [&](bool with_rollback, std::vector<llama_token> & out) -> bool {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx                = n_prompt + n_predict + 64;
        cparams.n_batch              = n_prompt + 64;
        cparams.n_rs_seq             = n_rs_seq;
        llama_context * ctx          = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "failed to create context\n");
            return false;
        }
        if ((uint64_t) n_rs_seq + 1 >= llama_n_ubatch(ctx) || llama_n_rs_seq(ctx) < (uint32_t) rewind) {
            fprintf(stderr, "rollback depth must fit the model, batch and micro-batch\n");
            llama_free(ctx);
            return false;
        }
        llama_memory_t mem = llama_get_memory(ctx);

        common_batch batch = common_batch_get_one(ctx, ptoks);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            fprintf(stderr, "decode(prompt) failed\n");
            return false;
        }
        int n_pos = n_prompt;

        int since = 0;
        while ((int) out.size() < n_predict) {
            llama_token tok = greedy(ctx, vocab);
            out.push_back(tok);

            ++since;
            const bool verify = with_rollback && since >= stride && (int) out.size() + rewind < n_predict;
            batch.clear();
            batch.add(tok, n_pos++, 0, true);
            if (verify) {
                // Rollback is bounded by the last verification micro-batch. Only the
                // accepted anchor produces logits; they remain valid after trimming.
                for (int r = 0; r < rewind; ++r) {
                    batch.add(tok, n_pos + r, 0, false);
                }
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                fprintf(stderr, "decode failed\n");
                return false;
            }
            if (verify) {
                if (!llama_memory_seq_rm(mem, 0, n_pos, -1)) {
                    fprintf(stderr, "ROLLBACK REFUSED: keep=%d rewind=%d\n", n_pos, rewind);
                    return false;
                }
                ++n_rollbacks;
                since = 0;
            }
        }
        llama_free(ctx);
        return true;
    };

    bool ok = true;

    std::vector<llama_token> ref, rb;
    if (!run(false, ref)) {
        return 1;
    }
    if (!run(true, rb)) {
        return 1;
    }

    size_t n        = ref.size() < rb.size() ? ref.size() : rb.size();
    size_t mismatch = 0;
    int    first    = -1;
    for (size_t i = 0; i < n; i++) {
        if (ref[i] != rb[i]) {
            if (first < 0) {
                first = (int) i;
            }
            mismatch++;
        }
    }

    printf("\n=== rs-rollback ===\n");
    printf("tokens compared : %zu\n", n);
    printf("mismatches      : %zu\n", mismatch);
    printf("rollbacks done  : %d\n", n_rollbacks);
    // a run that never entered the rollback branch compared two identical
    // straight-through generations and proved nothing, so it is not a pass
    if (n_rollbacks == 0) {
        printf("RESULT: FAIL (no rollback was attempted: need n_predict > stride + rewind, "
               "got n_predict=%d stride=%d rewind=%d)\n", n_predict, stride, rewind);
        ok = false;
    } else if (mismatch) {
        printf("first mismatch  : index %d (ref=%d rb=%d)\n", first, ref[first], rb[first]);
        printf("RESULT: FAIL\n");
        ok = false;
    } else {
        printf("RESULT: PASS (rollback reproduces the straight-through sequence)\n");
    }

    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
