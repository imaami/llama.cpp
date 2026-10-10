# Bonsai refresh and short Vulkan convolution tiles, 2026-10-10

This pass starts from published `imaami/bonsai` at `82a0e12a05ee6a680685d9abd797af153ec921de`, preserved as `bonsai-before-refresh-20261010`. The user reported approximately 80 tok/s with that build; this is the comparison baseline, not a measurement made here. Nothing was pushed.

## Upstream and Prism

Rebased all 222 fork commits onto freshly fetched `ggml-org/master` at `781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4`, 11 commits beyond the previous `8e2d31e0e` base. Range-diff accounts for every commit: 219 have identical patches and three have integration/context changes.

The main integration is [upstream #30257](https://github.com/ggml-org/llama.cpp/pull/30257), which separates MTP hidden states from token embeddings. Deferred catch-up, the combined catch-up/draft decode, and chained heads now use the state input. Ordinary draft and Eagle3 inputs keep their existing embedding API. The SYCL conflict retains both upstream MXFP4 and fork PQ2/PTQ1 cases. CUDA automerges preserve the ternary paths while incorporating upstream's attention barrier and dispatch fixes.

Two correctness fixes accompany the MTP integration:

- The new batch allocator now packs hidden states using each token's recorded state offset. Upstream copied the append-order array, which attaches the wrong state when setters run out of token order.
- Catch-up checks every row for embedded input before deferring it. `common_batch::has_embd()` only describes the first row. Embedded inputs use eager, homogeneous runs because MTP contexts reject mixed token/embedding batches before splitting. This preserves full spatial positions and hidden-state pairing; ordinary text still uses deferred catch-up.

Fetched 89 Prism branch heads. Its `prism` branch is at `668256445feafa00dfe5f194bac1ea37ae2c74e5`. Imported the new [#340](https://github.com/PrismML-Eng/llama.cpp/pull/340) follow-up, `66d12f18816bffd41740d0b43e9addbc9bc3fe65`, restricting the one-row PTQ1 CUDA GEMV schedule to sm_86 after a reported Hopper regression. This does not affect Vulkan.

`hadamard-folded-runtime` moved to `884fe849571157eb859cf53ce9debba747196e64`, with six commits based on upstream `79e2e74e`. The new work reorganizes loader helpers, metadata enums, model state and transform memoization. Bonsai already has equivalent manifest conversion and Qwen folding, together with its LoRA activation fix and borrowed DFlash embedding transforms. Replacing those paths would lose fork fixes; no new folding mathematics or missing model support was found to import.

Other reviewed changes were left out:

- Prism #107 duplicates the broader CUDA MoE weighted reduction already inherited from upstream #25952. Its exact-rounding expectations also differ from that implementation.
- Prism #339 changes SYCL release packaging.
- Upstream Vulkan #30280 requires token tiles of at least 512, so it does not cover this workload's ubatch 128 or two-token verification.
- Upstream RMS #29882 is already represented by the previous pass's broader implementation. Cooperative-matrix GDN #30207 has no new kernel code and still needs newer driver capabilities.

## RDNA4 convolution scheduling

SSM convolution and its SiLU and bias-plus-SiLU fusions previously used a `32 x 16` channel/token workgroup for every token count. At one token, 15 of the 16 token rows return without computing; at two tokens, 14 do.

RDNA4 now uses `64 x 1` for one token and `64 x 2` for two tokens. All other token counts retain `32 x 16`. The existing shader, descriptors, arithmetic and strides are unchanged. A 64-channel tile retains twice as many workgroups as the considered 128-channel alternative.

The change defaults on only for RDNA4. `GGML_VK_SSM_CONV_SMALL=0` selects the old schedule in the same binary; `=1` enables the candidate on other devices for validation. At Bonsai's 10,240 convolution channels, one sequence launches 10,240 rather than 163,840 threads for one token, or 20,480 rather than 163,840 for two. This does not imply a 16x or 8x speedup: inactive waves can exit early, and GPU occupancy and memory behavior still need measurement.

Offline Mesa 25.2.8 ACO compilation for gfx1201 used the production SSM settings: four storage buffers, 44 bytes of push constants, robust buffer access, and no requested/full subgroup override. All default variants use wave64, 12 VGPRs, no spills and no shared memory.

| Fusion | Old instructions | One-token instructions | Two-token instructions |
| --- | ---: | ---: | ---: |
| Plain convolution | 73 | 70 | 73 |
| Convolution and SiLU | 79 | 76 | 79 |
| Convolution, bias and SiLU | 85 | 81 | 85 |

These are compiler statistics, not device timings. The earlier PTQ integer-dot, packed attention, Hadamard/Q8 reuse, GDN and RMS optimizations remain in place.

All 45 offline specializations compiled: three fusion modes, five tile shapes, and default/explicit-32/explicit-64 wave requests. Explicit wave32 uses 24 VGPRs for both old and new tiles, also without spills or shared memory.

## Validation

The integrated Release build passed at `1407deac575eff395f5979188e85188626ee09a0`, including `llama-server`, `llama-bench`, the CPU and Vulkan backends, and the selected tests. Both binaries report that revision; no empty object files remain.

All 20 selected runtime CTests passed, covering DSpark metadata, sampling, reasoning, grammar/chat parsing, batch allocation, DFly fusion, quantization, PTQ/PQ regressions, GGUF handling and MTP. The expanded MTP integration test passes all six window/cache combinations (0, 16 and INT_MAX, each with unified cache off/on) plus both embedded-input cases. The tiny MTP model and eight DSpark fixture variants were generated locally; none of these tests skipped.

The candidate convolution path and legacy control each passed 174/174 software-Vulkan cases against the CPU reference, with zero unsupported or failed cases: 348 successful checks. The cases include all three fusion modes, one/two-token decoding, the three-token fallback boundary, long token tiles, multiple sequences, channel tails, Bonsai width, offset views and generic convolution lengths 3 and 9. There are 39 new cases per mode; the rest were existing coverage. Numerical tolerances, NaN/Inf checks and guard checks use the normal backend test harness.

The final offline results use the exact production `ssm_conv_f32.spv` emitted by this build, targeting Vulkan 1.2. The module passed `spirv-val`; all 45 production-module specializations match the compiler statistics above.

The toolchain is GCC 13.3, CMake 4.4.4, shaderc 2026.4 from Vulkan SDK 1.4.363.0, and Mesa 25.2.8 with LLVM 20.1.2. All seven Vulkan shader feature probes pass. UI asset downloads and OpenSSL are disabled only for this validation build.

The hidden-state regression's negative control, restoring the old append-order copy, fails 72 assertions. The fixed batch allocator suite passes 59 tests and 557 assertions. The MTP regression also fails against the old catch-up implementation, specifically when text-first mixed input is incorrectly deferred. The corrected implementation passes both unified-cache settings and matches all 259 probe logits against an independent per-row eager reference, including full M-RoPE positions and deferred-to-embedded transitions.

There is no Radeon device, CUDA toolkit or real 27B GGUF in this environment. Software Vulkan and offline AMD compilation establish neither model throughput nor a native CUDA build.

## Import and compare

The bundle contains `bonsai` and `bonsai-before-refresh-20261010`, and requires upstream history through `781dbc5ac`. Import it into a separate local branch:

```bash
git fetch https://github.com/ggml-org/llama.cpp.git master
git fetch /path/to/bonsai-refresh-20261010.bundle \
    bonsai:bonsai-refresh-20261010 \
    bonsai-before-refresh-20261010:bonsai-refresh-baseline-20261010
git switch bonsai-refresh-20261010
```

Compare the normal startup against `GGML_VK_SSM_CONV_SMALL=0` with the same binary, model, driver, context, prompt and flags. Keep the existing RMS setting fixed. Repeat the first configuration afterward and record MTP acceptance alongside throughput. See [the controlled comparison instructions](bonsai-performance-comparison.md) and `scripts/server_perf_compare.py`.

To isolate the rebase and MTP integration from the new convolution scheduling, compare the saved baseline against the refreshed branch with `GGML_VK_SSM_CONV_SMALL=0`, using separate build directories.
