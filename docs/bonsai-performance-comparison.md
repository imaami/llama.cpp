# Controlled Bonsai server comparison

`scripts/server_perf_compare.py` measures an already-running local server. It uses one warmup and three measured requests by default. Every request uses the same prompt token IDs and seed, disables prompt reuse, ignores EOS, clears stop words, and requests exactly 2048 output tokens. Sampling otherwise inherits the server settings. The tool does not start, stop, or reconfigure a server.

Use a dedicated server with your current model, launch arguments and environment. Keep `--parallel 1`, add `--metrics`, and leave the slots endpoint enabled. Close other clients. Save the startup log for each build: it identifies the actual Vulkan device, driver, enabled features, cache allocation and launch settings that the HTTP API cannot report. Record any `GGML_VK_*`, `GGML_GDN_*`, `VK_*` and `RADV_*` overrides used to launch it. The tool rejects busy endpoints and counter mismatches, but cannot prove that there was no other traffic.

Put a representative prompt in `prompt.txt`. This is a raw completion prompt, so include its rendered chat template tokens if comparing a chat workload. The tool tokenizes it with special tokens enabled and sends the resulting IDs. It does not apply a chat template or add conversation history.

With the first build running:

```bash
python3 scripts/server_perf_compare.py run \
    --prompt-file prompt.txt --label before --output before.json
```

Stop it yourself, start the second build with the same arguments and environment, then run:

```bash
python3 scripts/server_perf_compare.py run \
    --prompt-file prompt.txt --label after --output after.json
python3 scripts/server_perf_compare.py compare before.json after.json
```

Then restart the first build and measure `before-again.json`. Comparing A/B/A helps separate a build difference from changing operating conditions. Keep the same driver, model bytes, power settings and other GPU workload. Use separate build/install directories so each server loads its matching libraries; its source checkout alone does not identify an installed binary. Optional repeated `--artifact PATH` arguments hash binaries or libraries you specify, but do not verify that the server loaded those files.

For a long-context comparison, use a longer prompt file with the same commands and output length. Every sample starts from an uncached prompt; comparing this with an unrelated ongoing conversation does not hold the decode workload constant. `--tokens`, `--seed`, `--repeats`, and `--warmup` are configurable, but use the same values for each build.

The table reports median and range of server generation throughput, draft acceptance and accepted length per verification, and whether all measured token sequences match. Different output hashes or acceptance mean the work performed changed; the throughput difference alone does not isolate a kernel regression. The JSON also preserves per-sample prompt/decode timings and verification counts. It contains prompt/output hashes, model path, effective generation settings and supplied artifact paths, but no prompt text, completion text or token arrays.

The comparison rejects different prompt tokens, request options, context sizes or effective sampling settings. Build identifiers and model paths may differ. Verify the same GGUF bytes yourself. These records do not capture every server flag, notably all draft-cache and device settings, so keep the startup logs too.

A short response, context truncation, cached prompt, HTTP error or inconsistent counters aborts the run without writing a result file. A valid multibyte completion can occasionally exceed the requested token limit; that sample is also rejected for a fixed-length comparison. Very large Prometheus counters can be rounded by the server; if a counter mismatch occurs on an otherwise dedicated server, restart it before measuring. Do not treat a failed run as a throughput result.
