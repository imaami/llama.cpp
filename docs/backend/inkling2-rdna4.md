# Inkling 2 Mini on RX 9070 XT with Vulkan

The `bonsai2-inkling-rdna4-strata` branch builds one Vulkan backend and one set of executables for Ternary Bonsai 2 and Inkling 2 Mini. The same server supports Bonsai's embedded Qwen3.5 MTP and Inkling's separate recurrent DSpark GGUF.

```sh
cmake -S . -B build-inkling -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-inkling -j --target llama-server llama-bench llama-fit-params llama-quantize
build-inkling/bin/llama-server --list-devices
```

Set `-DLLAMA_BUILD_UI=ON` at configure time to embed this branch’s chat UI, including its optional parallel MCP calls.

Select the device ID reported for the 9070 XT. On a dual-GPU system, pass it explicitly to both the target and the drafter. The examples below use `Vulkan0`.

## Model and launch

The original model measurements used Unsloth's mainline-compatible UD-IQ3_XXS. Its loop visits reuse 12 physical layers three times. Eight of those layers have IQ3_XXS gate/up weights and four have IQ2_S. All twelve use NVFP4 down projections.

```sh
src=https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/8078d149286eec7193795413b9e0643da6d11fa1/Mainline_Support
curl -fL --retry 3 -o inkling-iq3.gguf "$src/Inkling-2-Mini-UD-IQ3_XXS.gguf_file"
dev=Vulkan0
build-inkling/bin/llama-server -m inkling-iq3.gguf -dev "$dev" -ngl all \
    -fa on -np 1 -c 131072 -b 512 -ub 512 -ctk q8_0 -ctv q8_0 --jinja
```

For DSpark, download the draft paired with this Inkling GGUF.

```sh
curl -fL --retry 3 -o inkling-dspark.gguf "$src/dspark-Inkling-2-Mini-Q8_0.gguf_file"
```

Add `-md inkling-dspark.gguf --spec-type draft-dspark --spec-draft-device "$dev" --spec-draft-ngl all --spec-draft-n-max 3` to the server command. Use `--spec-draft-sampling probabilistic` for rejection sampling at positive temperature. Measure accepted tokens and end-to-end generation time on the same prompts before selecting a draft length. The tables below cover the text target alone.

For a Bonsai 2 GGUF that includes MTP weights, use the same server binary with the Bonsai model as the target. Set `--spec-draft-window` to a positive number to limit the draft's KV history independently of the target context.

```sh
build-inkling/bin/llama-server -m "$bonsai_mtp" -dev "$dev" -ngl all \
    -fa on -np 1 -c 131072 --spec-type draft-mtp --spec-draft-n-max 1
```

Bonsai uses upstream recurrent replay by default. On a single sequence, Qwen3.5 with CPU or Vulkan can use `GGML_GDN_STATE_SNAPSHOTS=1` to keep recent states on device for rollback. This trades cache memory for less replay work during verification.

## Ternary experts

Inkling's 24 routed gate/up matrices have 2,304 weights per row. Its 24 down projections have 1,152. Both row widths are divisible by 128, so both fit Bonsai's 28-byte PTQ1_0 block. The unrotated Inkling conversion fits a signed ternary scale directly to each block. With an importance matrix the fit uses the input-channel importance of each expert.

The experimental GGUF fits all 24.46 billion routed weights to PTQ1_0 and copies the other 656 tensors from mainline UD-IQ3_XXS, reducing the file from 10.86 GiB to 6.255 GiB. On four 128-token WikiText-2 chunks (252 scored next tokens), perplexity rose from 12.12 with UD-IQ3_XXS to 20.42 with ternary experts, mean KL divergence was 0.711 nats and top-token agreement was 66.27%. The gfx1201 planner reserved 8.80 GiB at 131,072 tokens and 11.31 GiB at 262,144 for the ternary text target with compact SWA, Q8_0 K/V, one sequence and a 512-token microbatch.

In a software Vulkan smoke test, a 96 MiB on-demand expert cache kept both PTQ1_0 expert products on Vulkan with zero CPU expert share. The greedy output matched CPU.

For a different mix, the following command quantizes the other eligible tensors to Q8_0 using the BF16 GGUF and matching imatrix from ggml-org.

```sh
build-inkling/bin/llama-quantize --imatrix imatrix.gguf \
    --tensor-type 'ffn_(gate_up|down)_exps[.]weight=ptq1_0' \
    Inkling-2-Mini-BF16.gguf Inkling-2-Mini-PTQ1-experts-Q8_0.gguf Q8_0
```

[BF16 GGUF and imatrix](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9)

## What fits

These reservations were recorded for the original Inkling changes based on upstream `4af4bb6d`, using the released GGUF tensor headers and a 16 GiB RADV gfx1201 device profile. The models use one sequence, one requested output, compact SWA, full layer offload, FA enabled and batch/microbatch 512. Device totals include model, context and compute. Allow additional memory for the display, driver, projector and draft as applicable.

| Target | Context | K/V | Stock device MiB | Patched device MiB |
| --- | ---: | --- | ---: | ---: |
| UD-IQ3_XXS | 32,768 | F16 | 15,507 | 12,418 |
| UD-IQ3_XXS | 131,072 | Q8_0 | 16,687 | 13,763 |
| UD-Q3_K_XL | 32,768 | F16 | 17,330 | 14,240 |
| UD-Q3_K_XL | 131,072 | Q8_0 | 18,509 | 15,585 |

At 131,072 with UD-IQ3_XXS and Q8_0, GPU compute scratch fell from 3,513 to 589 MiB and host compute scratch from 12,729 to 398 MiB. All 48 attention visits use indexed bias. At 262,144 the same patched target reserves 16,323 MiB. Reducing `-ub` to 128 brings it to 15,571 MiB. Using `-ctv q4_0` as well brings it to 14,522 MiB. These are capacity estimates. Cache precision affects the arithmetic.

On the earlier combined build, the gfx1201 memory planner reserved 11,815 MiB of device memory at 32,768 tokens and 13,735 MiB at 131,072 for the UD-IQ3_XXS text target with Q8_0 K/V, compact SWA, one output, and `-b 512 -ub 512`. The 131,072-token plan includes 561 MiB of GPU compute buffers and uses another 398 MiB of host compute buffers.

## Measure the card

Build the combined branch’s upstream base `3090e45a8675ebdefbc722f80609c75e5fa259ad` with the same compiler, Vulkan SDK and Release options. Set `baseline` and `patched` to the resulting build directories. This measures prefill and decode at several occupied-context depths separately.

```sh
for build in "$baseline" "$patched"; do
    "$build/bin/llama-bench" -m inkling-iq3.gguf -dev "$dev" -ngl 999 -fa on \
        -ctk q8_0 -ctv q8_0 -b 512 -ub 512 \
        -p 512 -n 128 -d 0,8192,32768 -r 5 -o json > "$build/inkling-iq3.json"
done
```

`llama-bench` measures the target directly. For DSpark, compare two `llama-server` runs with identical prompts, cache settings and sampling. The Bonsai short-convolution tiles can be isolated in the patched build with `GGML_VK_SSM_CONV_SMALL=0` or `1`.

## Vulkan dispatches

On the released UD-IQ3_XXS, two single-token target evaluations issued 9,824 Vulkan dispatches with the first combined branch (`f939dc67`) and 7,920 on the earlier Strata integration (`65351d70`, based on upstream `ba8521a4`). The trace used llvmpipe with the dense layers on Vulkan and the experts on CPU. Both builds produced the same greedy completion. The reduction averaged 952 dispatches per token.

| Change | Dispatches saved per token |
| --- | ---: |
| Fuse the four loop controllers' gate arithmetic, across 96 updates | 480 |
| Fuse static short convolution, residual addition and scaling | 352 |
| Use contiguous single-token stream views in place | 120 |

Multi-token prefill also benefits from bulk filling relative-position indices outside Inkling's learned extent. With 32,768 cached positions and 512 queries, preparing the indices took 3.03 ms versus 13.95 ms in a host microbenchmark. The result array and the device transfer still have the same size.

## Source work

Software Vulkan tests exercised the fused and unfused Inkling graphs, single and multi-token batches, F16 and Q8_0 KV, and multi-stream inputs. The matvecs and fused shaders also compiled to gfx1201 machine code. End-to-end throughput on a physical RX 9070 XT is unmeasured.

The Vulkan path combines the Bonsai short-convolution tiles, IQ3_XXS matvec changes, multi-column K/I-quant unpacking, and compact-bias Flash Attention. The Inkling graph shares work across loop passes and convolution taps. The Vulkan fusions perform the Q/K/relative L2 clipping and soft-capped SiTU directly.

- [Bonsai RDNA branch](https://github.com/imaami/llama.cpp/tree/optimize-bonsai2-vulkan-rdna)
- [IQ3_XXS matvec](https://github.com/ggml-org/llama.cpp/pull/30904) and [multi-column quantized matvec](https://github.com/ggml-org/llama.cpp/pull/31022)
- [Compact FA contract](https://github.com/ggml-org/llama.cpp/pull/32740) and [Vulkan implementation](https://github.com/ggml-org/llama.cpp/pull/32898)
- [Koosh0610's Inkling graph and Metal work](https://github.com/Koosh0610/llama.cpp-inkling) and [rafasmour's Inkling optimizations](https://github.com/rafasmour/llama.cpp-RDNA3/pull/132)
- [DSpark support](https://github.com/ggml-org/llama.cpp/pull/32218)
- [Bob's Strata fork and Maya/Metal parity results](https://github.com/robertkarl/UnbiasedStrata/tree/9f75192287c2abf88c77da3b4b6f54ccee01db83/bench/inkling)
- [Strata's Inkling HIP kernels](https://github.com/Niko1221/Strata/tree/d46f5f268f8ed797e83a5c63a69bffbd7a581cf6/engine/src/inkling2/hip) and [the llama.cpp HIP comparison](https://github.com/kr4ckhe4d/local-llm-benchmarks/blob/main/inkling-strata-vs-llama.md)
- [vLLM's loop controllers](https://vllm.ai/blog/2026-11-20-inkling2-mini), [Colibri's Vulkan inference](https://github.com/JustVugg/colibri) and [SaloQT's gfx1201-specific implementation](https://github.com/SaloQT/rdna4-amd-infer)
