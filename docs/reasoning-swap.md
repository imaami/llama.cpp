# Whole-reasoning token swap experiment

For continuous exchange and a browser console, see [Crossthink](crossthink.md). The one-shot driver below remains available.

This prototype runs two independent model instances, collects a complete reasoning block from each, and exchanges those blocks before generating their answers. The imported block occupies the recipient's own assistant reasoning prefix; no peer attribution or extra user message is inserted. It is an experiment in conditioning, with no claim that it improves reasoning quality.

The branch starts at `imaami/llama.cpp` `bonsai`, commit `82a0e12a0`. The existing Vulkan and MTP changes are retained. The prototype changes server transport and adds a small native driver; it does not change inference kernels or the speculative decoder.

## Exact sequence

1. Each server applies its own chat template to the supplied question with thinking enabled. The driver checks the open `<think>` prefix and the single-token `</think>` delimiter.
2. A and B generate independently, in parallel, using different sampling seeds. Each stops at `</think>`.
3. The driver requires a complete reasoning block from both. A token-limit stop, end-of-generation stop, or truncated context aborts the swap.
4. A receives `A_prompt + B_reasoning`, and B receives `B_prompt + A_reasoning`. Each imported block includes its original closing `</think>` token. The recipient's own generated reasoning is discarded from the submitted prompt.
5. Both generate their final answers in parallel.

The server treats each phase as an ordinary completion request. Normal prefix-cache matching invalidates the changed suffix and evaluates the imported IDs, including the MTP prompt processing path. There is no copying of KV tensors between models and no mutation of a live slot from an I/O thread.

The first version performs one such cycle per invocation. It does not cross-feed partially generated thoughts, recursively exchange answers, or start the recipient's prefill before the sender has finished. Token transport avoids detokenization/re-tokenization of exchanged reasoning. The recipient still has to run inference over the imported tokens; binary transport does not remove that cost.

## Build and run

Build with the same Vulkan configuration as the regular Bonsai server, adding the driver target:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build --target llama-server llama-reasoning-swap -j
```

The driver is built on Unix platforms when `LLAMA_BUILD_SERVER` is enabled. It uses the repository's existing C++ HTTP and JSON libraries; there is no Python runtime in the experiment itself.

Find the actual device names using `build/bin/llama-server --list-devices`. The example assumes `Vulkan0` and `Vulkan1`; check which is the 9070 XT on your machine. Use the identical GGUF for both processes initially.

Create a private directory for the Unix sockets:

```bash
swap_dir=$(mktemp -d /tmp/llama-reasoning-swap.XXXXXXXX)
printf '%s\n' "$swap_dir"
```

Start these in separate terminals, using that directory's printed path in both. Retain any other working Bonsai launch options you need. Each process must fit its own model and context on its chosen GPU.

```bash
build/bin/llama-server -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf \
    --device Vulkan0 --split-mode none --gpu-layers all \
    --ctx-size 262144 --parallel 1 --flash-attn on --jinja \
    --spec-type draft-mtp --spec-draft-n-max 1 \
    --host 127.0.0.1 --port 8080 --socket /tmp/llama-reasoning-swap.XXXXXXXX/a.sock
```

```bash
build/bin/llama-server -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf \
    --device Vulkan1 --split-mode none --gpu-layers all \
    --ctx-size 262144 --parallel 1 --flash-attn on --jinja \
    --spec-type draft-mtp --spec-draft-n-max 1 \
    --host 127.0.0.1 --port 8081 --socket /tmp/llama-reasoning-swap.XXXXXXXX/b.sock
```

Then run:

```bash
build/bin/llama-reasoning-swap \
    --socket-a "$swap_dir/a.sock" --socket-b "$swap_dir/b.sock" \
    --reasoning-tokens 8192 --answer-tokens 2048 \
    --prompt 'Find a binary de Bruijn sequence of order 5 and explain how to verify it.'
```

The token counts are upper bounds. If either reasoning block does not close within its bound, the driver stops with an error; increase the bound and retry. The server requires the prompt plus the full requested generation budget to fit in the slot, so this endpoint never deliberately shifts away earlier context.

Avoid server-wide reverse prompts or other output constraints for the initial experiment. The raw completion phases use their own stop marker and do not go through the chat reasoning parser or its budget policy. The prototype expects the Qwen-style `<think>` / `</think>` convention and deliberately rejects unsupported template layouts. It is not a general adapter for every reasoning model.

## Binary API

Unix-domain HTTP transport already exists in this branch. Both endpoints also follow the server's ordinary routing and authentication rules; the sample driver uses local Unix sockets in a private directory.

`GET /tokens/info` returns protocol version, vocabulary size, a token-ID mapping fingerprint, end-of-generation IDs, and slot context size. The mapping check detects accidental incompatible vocabularies; its FNV64 fingerprint is not a cryptographic identity or a weights checksum. Matching token IDs does not prove that two different models interpret the same prompt equally. These endpoints are available in single-model server mode, not router mode. The driver does not yet have an API-key option.

`POST /completion/tokens` accepts `application/octet-stream` and returns the same framing:

| Offset | Encoding | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | ASCII `LLMTOK01` |
| 8 | little-endian uint32 | JSON metadata byte count |
| 12 | little-endian uint32 | Token count |
| 16 | UTF-8 bytes | JSON metadata object, at most 65536 bytes |
| following metadata | little-endian uint32 array | Exact token IDs |

Request metadata holds completion controls such as `n_predict`, `seed`, `temperature`, `stop`, and `preserved_tokens`. The prompt is exclusively the packed token payload. Requests must use a positive finite generation budget and one completion. This driver uses nonstreaming output; the endpoint also supports the binary streaming mode described in [Crossthink](crossthink.md#binary-stream). Invalid framing, token IDs, unsupported modes, and requests that cannot fit in context are rejected.

For reasoning, the driver supplies `stop: ["</think>"]`, `preserved_tokens: ["</think>"]`, and `return_content: false`. Preserving the special token makes it visible to the existing stop-string check. The returned token payload includes the closing delimiter, although displayed completion text conventionally excludes the stop string.

Response metadata includes stop type, stopping word, counts, cache use, truncation state, timings, and optionally display text. If display text would exceed the metadata limit, the server omits it and sets `content_omitted: true`; the packed token output remains intact. The driver reports this as an error and recommends a smaller answer budget. Reasoning is exchanged exclusively as packed IDs. Ordinary JSON endpoints are used for the initial template/tokenizer setup. HTTP errors retain the normal JSON error format. The one-shot driver uses whole-response transport.

## Evaluation

Compare the same questions with ordinary independent runs and with swapped reasoning. Keep seeds, sampling settings, model files, and budgets recorded. Different seeds encourage divergent traces but do not guarantee independent ideas, and identical models may reinforce the same mistake. Judge correctness and useful diversity as well as latency.

The relevant performance measurements are reasoning time on each GPU, imported-prompt evaluation time, and answer time. The slower reasoner's completion is a barrier in this version. The continuous driver instead streams committed tokens and imports them at generation quantum boundaries; see [Crossthink](crossthink.md).

## Validation of this draft

The Release CPU build passed for `llama-server`, `llama-reasoning-swap`, and `test-server-token-wire`. The driver also passed a C++17 syntax check with strict compiler warnings.

The in-process server-route tests passed with the seeded tiny Qwen3.5 fixture, both without speculation and with `draft-mtp`. They checked exact generated token IDs against the ordinary JSON completion route, replacement of a warmed cached suffix against fresh evaluation, preservation of text and special-control stop tokens, and rejection of malformed frames, invalid IDs, incompatible options, and context overflow. The MTP test explicitly checked that drafting occurred. These tests call the actual request handlers and inference loop without a network listener.

```bash
python3 tests/gen-tiny-qwen35-mtp.py /tmp/tiny-qwen35-mtp.gguf
cmake -S . -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target test-server-token-wire -j
build/bin/test-server-token-wire /tmp/tiny-qwen35-mtp.gguf
```

The integration script below additionally exercises Unix-domain HTTP and the native driver's complete cross-swap against deterministic mock peers. The build environment denied Unix-socket creation (`EPERM`), so these socket tests could not run there. They are included for local execution:

```bash
python3 tests/test-reasoning-swap.py \
    --server build/bin/llama-server --driver build/bin/llama-reasoning-swap \
    --model /tmp/tiny-qwen35-mtp.gguf
```

No actual Bonsai weights or Radeon hardware were available. The tests establish CPU route/codec behavior with the fixture; they do not establish real-model reasoning quality, dual-GPU operation, or performance improvement.
