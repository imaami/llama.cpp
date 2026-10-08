# Bonsai refresh, 2026-10-08

## Refs and scope

- Starting `imaami/bonsai`: `69f5acb3743d1b78299a474bd6eb543148284683`.
- Old upstream base: `a657f7e981ff8764d2ccce1e74ec3c7e9bf2cfa2`.
- Fresh `ggml-org/master`: `c35b66744f13cb0dcc476af063e112122eee9355`.
- Reviewed `PrismML-Eng/prism`: `4fbda1292562cc1f23904d5867fb79603737547a`.
- Reviewed `PrismML-Eng/hadamard-folded-runtime`: `3bfabd2861455dce72c9d1a3e4acf72c8ec79d7e`. Its sole parent is exactly the fresh upstream base above.
- Code and tests in this refresh end at `9ab600a3e5a29a5c59dae6064b7d786cce8ec08d`; the following commit records these notes.
- Original local branch preserved as `bonsai-before-refresh-20261008-1435`. No remote branch was pushed.

The original fork series had 133 commits. Range-diff shows 125 unchanged patches, seven adapted patches, and one omitted patch: the old Prism #216 CUDA GDN layout, superseded by upstream's new implementation. The refreshed series then adds 54 integration/fix commits. All 56 non-draft open Prism PRs returned in the review snapshot were assessed below; this is not a claim that every experimental PR can safely be combined.

## Hadamard and conflict resolutions

The standalone Hadamard branch mostly consolidates runtime support already in Bonsai. Its shared `build_hadamard_input` helper is integrated in `7f6102f79`. Existing DSpark borrowed-tensor rotations are retained. LoRA still receives the original activation basis, while folded base weights receive the transformed activation. The branch does not introduce a new Vulkan kernel.

The CUDA GDN resolution keeps upstream's segmented-lane layout and adaptive CTA warp count, with Prism raw gates, GB10 gate precomputation, row-indexed states and rollback snapshots adapted around it. The old whole-column-per-lane layout was not restored.

Other material resolutions:

- Carry upstream's new normalization launch dimensions through both GB10 RMS paths; preserve large-grid fallback guards.
- Preserve the newer upstream MoE matrix-vector batch dispatch while adding the PTQ1 multi-column bias/SwiGLU paths.
- Keep upstream flash-attention launch geometry when reducing the eight-column KV tile.
- Keep CPU BF16 conversion and the sampler RNG state introduced upstream.
- Separate Metal's new PTQ1 full-column function constant from upstream's split-dispatch constant; include both in the pipeline key.
- Preserve SYCL subgroup classification and the newer memory-pool organization.
- Guard NVIDIA-only L2 prefetch assembly from HIP/MUSA; use thread-local optional CUDA timing counters.
- Require successful recurrent checkpoint restores and retain Bonsai's single-sequence guard for plain in-place GDN rows.

New merged Prism #334 (DFlash1 prompt lookup, `50aea5999`) was also brought in. The current ARM NEON/i8mm #290 was already present before this refresh.

## PR audit

PR heads are pinned to the versions reviewed. “Included” means source integration, not performance verification on that backend.

| Prism PR | Reviewed head | Disposition |
| --- | --- | --- |
| [#107](https://github.com/PrismML-Eng/llama.cpp/pull/107) | [d3cbcde9f152](https://github.com/PrismML-Eng/llama.cpp/commit/d3cbcde9f1523eade29ff4630b2929d9355f474e) | Not imported: overlaps upstream's newer MoE weighted-reduction implementation. |
| [#116](https://github.com/PrismML-Eng/llama.cpp/pull/116) | [d90dab7fceac](https://github.com/PrismML-Eng/llama.cpp/commit/d90dab7fceacedb26923cddd74c163c1785ddd47) | Old feature branch not imported wholesale; many paths were replaced by newer work. Its tiny-model generator was adapted for the new MTP regression fixture. |
| [#189](https://github.com/PrismML-Eng/llama.cpp/pull/189) | [19836568c4f1](https://github.com/PrismML-Eng/llama.cpp/commit/19836568c4f1b504d8ee579b7ab7fcf4edbd8f78) | Old short-query workaround not ported over the current native quantized-KV dispatch without GPU validation. |
| [#190](https://github.com/PrismML-Eng/llama.cpp/pull/190) | [fd42a0b77692](https://github.com/PrismML-Eng/llama.cpp/commit/fd42a0b7769290a05e2f2972d57a52a6ec0bdd17) | Included. |
| [#207](https://github.com/PrismML-Eng/llama.cpp/pull/207) | [c875ea463d60](https://github.com/PrismML-Eng/llama.cpp/commit/c875ea463d607e4c2d2f5743cca2a8813610520f) | Not imported: broader multi-sequence recurrent-memory and lazy-allocation rewrite needs independent validation; existing rollback behavior retained. |
| [#208](https://github.com/PrismML-Eng/llama.cpp/pull/208) | [3ed7c4340d11](https://github.com/PrismML-Eng/llama.cpp/commit/3ed7c4340d114c4e05c3aa5b7035b578e1bb7f8e) | Per-call MTP draft cap is covered by the adapted #320. The recurrent snapshot-budget rewrite and rollback-plane changes were not imported. |
| [#209](https://github.com/PrismML-Eng/llama.cpp/pull/209) | [37fe65676fb3](https://github.com/PrismML-Eng/llama.cpp/commit/37fe65676fb32a9020da1c69cfa61c7726483846) | Included. |
| [#218](https://github.com/PrismML-Eng/llama.cpp/pull/218) | [285542d98d37](https://github.com/PrismML-Eng/llama.cpp/commit/285542d98d37d0f07f491cd206aefa31f1848f33) | Existing Bonsai #221 integration already contains the relevant small-batch PTQ1 work. |
| [#225](https://github.com/PrismML-Eng/llama.cpp/pull/225) | [239413b3ec1f](https://github.com/PrismML-Eng/llama.cpp/commit/239413b3ec1f4e99309780ae20bd7024e1e3d361) | Included. |
| [#226](https://github.com/PrismML-Eng/llama.cpp/pull/226) | [8535294f2f3f](https://github.com/PrismML-Eng/llama.cpp/commit/8535294f2f3f6bb9637775f9b0a14603e67125fd) | Not imported: unconditional generic supports_op fallback bypasses type support checks. Applied the narrower #190 fixes. |
| [#227](https://github.com/PrismML-Eng/llama.cpp/pull/227) | [7ea821280e4d](https://github.com/PrismML-Eng/llama.cpp/commit/7ea821280e4d3d504426109bad286a7ad4b35aad) | Already present in the starting Bonsai branch. |
| [#232](https://github.com/PrismML-Eng/llama.cpp/pull/232) | [bbee00689b0f](https://github.com/PrismML-Eng/llama.cpp/commit/bbee00689b0f6dedb56de726473110742e74bdd4) | Partially adapted to the new upstream parser: accept four newline variants, require closing tags, fix AC grammar termination; parser and grammar tests pass. Optional/missing closing-tag behavior omitted. |
| [#234](https://github.com/PrismML-Eng/llama.cpp/pull/234) | [d1578a48fe30](https://github.com/PrismML-Eng/llama.cpp/commit/d1578a48fe307a484b88dbc27b939d7cc2f63abf) | Already covered by upstream empty-schema support. |
| [#243](https://github.com/PrismML-Eng/llama.cpp/pull/243) | [1d9ef0c7e992](https://github.com/PrismML-Eng/llama.cpp/commit/1d9ef0c7e992796f96e841c6c492483c48a5af26) | Old combined integration branch not imported wholesale; retain current dispatch and rollback implementations. |
| [#244](https://github.com/PrismML-Eng/llama.cpp/pull/244) | [ca2f9fa8096b](https://github.com/PrismML-Eng/llama.cpp/commit/ca2f9fa8096b4fa78ac320ba1cb9ee796768be49) | Upstream token-aware parsing and AST sanitization retained; the old raw-text retry patch was not transplanted. |
| [#246](https://github.com/PrismML-Eng/llama.cpp/pull/246) | [15f868b7464a](https://github.com/PrismML-Eng/llama.cpp/commit/15f868b7464a62c11d7f18fa15ae496841d2e626) | Included. |
| [#249](https://github.com/PrismML-Eng/llama.cpp/pull/249) | [6d1d921ddd97](https://github.com/PrismML-Eng/llama.cpp/commit/6d1d921ddd97e715a98d499738e967e8ff85215e) | Included. |
| [#250](https://github.com/PrismML-Eng/llama.cpp/pull/250) | [876577f7a5e7](https://github.com/PrismML-Eng/llama.cpp/commit/876577f7a5e7cc4fc56d8e41bd8e21a3dcd99802) | Already present in the starting Bonsai branch. |
| [#251](https://github.com/PrismML-Eng/llama.cpp/pull/251) | [306d5012aaeb](https://github.com/PrismML-Eng/llama.cpp/commit/306d5012aaeb0b65a4de02a3009ac207a4b62094) | Not imported: NUMA placement cache uses persistent pointer state and needs allocation-lifetime/hardware validation. |
| [#252](https://github.com/PrismML-Eng/llama.cpp/pull/252) | [3958846e287c](https://github.com/PrismML-Eng/llama.cpp/commit/3958846e287c552e2f37d46225bab50fe443ffc8) | Already present in the starting Bonsai branch (Vulkan PTQ1 dequant mat-vec). |
| [#253](https://github.com/PrismML-Eng/llama.cpp/pull/253) | [29cddbe0294b](https://github.com/PrismML-Eng/llama.cpp/commit/29cddbe0294bbdacab1f3f3a2a707125aafefdb2) | Not imported: introduces the separate PQ1 format, not an improvement to the requested PTQ1 model. |
| [#274](https://github.com/PrismML-Eng/llama.cpp/pull/274) | [924a8ef58f91](https://github.com/PrismML-Eng/llama.cpp/commit/924a8ef58f919b8144933e5a7a945f0c66a84d85) | Already present in the starting Bonsai branch (Vulkan Intel XE1 path). |
| [#275](https://github.com/PrismML-Eng/llama.cpp/pull/275) | [0ffb0c76bd5b](https://github.com/PrismML-Eng/llama.cpp/commit/0ffb0c76bd5b1d24d2fa5767dc19d34d2ceb1887) | Included. |
| [#276](https://github.com/PrismML-Eng/llama.cpp/pull/276) | [1ad814a9e2c6](https://github.com/PrismML-Eng/llama.cpp/commit/1ad814a9e2c65a1953f5be225b201625466c7fb2) | Included. |
| [#277](https://github.com/PrismML-Eng/llama.cpp/pull/277) | [9f69b287a400](https://github.com/PrismML-Eng/llama.cpp/commit/9f69b287a400e2d7ded75fe312df12b56c17f479) | Included. |
| [#279](https://github.com/PrismML-Eng/llama.cpp/pull/279) | [89bde0ce40af](https://github.com/PrismML-Eng/llama.cpp/commit/89bde0ce40af66e72157d78bd66489ffaf9f9c6c) | Superseded by the broader #293 Metal PQ2 verification path included here. |
| [#280](https://github.com/PrismML-Eng/llama.cpp/pull/280) | [96c3886e118a](https://github.com/PrismML-Eng/llama.cpp/commit/96c3886e118a00567c438d7ee71cdb1bb7b16947) | Included. |
| [#281](https://github.com/PrismML-Eng/llama.cpp/pull/281) | [ff6b2a06ce91](https://github.com/PrismML-Eng/llama.cpp/commit/ff6b2a06ce91fb3e3df7d6639f3f1ef6ea170ebd) | Already present in the starting Bonsai branch (Vulkan PQ2/BC-250 path). |
| [#289](https://github.com/PrismML-Eng/llama.cpp/pull/289) | [6371010c12e3](https://github.com/PrismML-Eng/llama.cpp/commit/6371010c12e39f5d2b1ded15c021338327639cff) | Not imported: experimental GDN verification tape; reported end-to-end slowdown. |
| [#293](https://github.com/PrismML-Eng/llama.cpp/pull/293) | [42f871fabed4](https://github.com/PrismML-Eng/llama.cpp/commit/42f871fabed4f2df1a6d0f9eb607f2507d2dd4f2) | Included. |
| [#295](https://github.com/PrismML-Eng/llama.cpp/pull/295) | [4cdb6989d948](https://github.com/PrismML-Eng/llama.cpp/commit/4cdb6989d94823f7b4d7a2c31c7eafbdaf67eb28) | Already present in the starting Bonsai branch. |
| [#296](https://github.com/PrismML-Eng/llama.cpp/pull/296) | [e33251768124](https://github.com/PrismML-Eng/llama.cpp/commit/e33251768124bd49d845c723f53681203b51235b) | Included. |
| [#300](https://github.com/PrismML-Eng/llama.cpp/pull/300) | [fd83fcacc901](https://github.com/PrismML-Eng/llama.cpp/commit/fd83fcacc901d7783bc06db59b5ef239d8952068) | Not imported wholesale: mixes obsolete CUDA/CCCL workarounds with an unknown-tensor fallback that hides errors. |
| [#306](https://github.com/PrismML-Eng/llama.cpp/pull/306) | [2183eafa8c25](https://github.com/PrismML-Eng/llama.cpp/commit/2183eafa8c25cd041283b42b6929b35997bfc252) | Included. |
| [#307](https://github.com/PrismML-Eng/llama.cpp/pull/307) | [d01570f8a44d](https://github.com/PrismML-Eng/llama.cpp/commit/d01570f8a44d03239eb755ba6d1150845d591175) | Included. |
| [#308](https://github.com/PrismML-Eng/llama.cpp/pull/308) | [755422313aeb](https://github.com/PrismML-Eng/llama.cpp/commit/755422313aeb20a79755597ee1d972e83981b7ef) | Adapted: reduce the 8-column, head-size-256 KV tile to 32 while retaining upstream's 128-thread/2-occupancy geometry. |
| [#309](https://github.com/PrismML-Eng/llama.cpp/pull/309) | [97aec1b10907](https://github.com/PrismML-Eng/llama.cpp/commit/97aec1b10907a5b83201d40c8d42850bb3ee9736) | Included. |
| [#310](https://github.com/PrismML-Eng/llama.cpp/pull/310) | [791eb6198222](https://github.com/PrismML-Eng/llama.cpp/commit/791eb6198222dd7d0b7550f08103ff739d3d9c05) | Included. |
| [#311](https://github.com/PrismML-Eng/llama.cpp/pull/311) | [ef33b5d10b19](https://github.com/PrismML-Eng/llama.cpp/commit/ef33b5d10b19d77361c2e630b0f75985b3d3202e) | Included. |
| [#312](https://github.com/PrismML-Eng/llama.cpp/pull/312) | [ca0c09bca3ca](https://github.com/PrismML-Eng/llama.cpp/commit/ca0c09bca3ca943f8f2ae5537d96908fb124dac1) | Included. |
| [#313](https://github.com/PrismML-Eng/llama.cpp/pull/313) | [a51ea62e79ca](https://github.com/PrismML-Eng/llama.cpp/commit/a51ea62e79cae2e4b0c5a95278187d834c7b2f6e) | Included. |
| [#314](https://github.com/PrismML-Eng/llama.cpp/pull/314) | [e697877d9c5e](https://github.com/PrismML-Eng/llama.cpp/commit/e697877d9c5e4db5f312f41625b082e3374b1015) | Included. |
| [#316](https://github.com/PrismML-Eng/llama.cpp/pull/316) | [5768085b580f](https://github.com/PrismML-Eng/llama.cpp/commit/5768085b580f860cb226e0a23f42cd5f39fb473f) | Not imported: blanket Hadamard-metadata requirement would reject otherwise supported PQ2/PTQ1 files. |
| [#317](https://github.com/PrismML-Eng/llama.cpp/pull/317) | [4105522bdbb3](https://github.com/PrismML-Eng/llama.cpp/commit/4105522bdbb38afabb887538edecc9f3ce21d23c) | Already in the freshly fetched upstream. |
| [#319](https://github.com/PrismML-Eng/llama.cpp/pull/319) | [6fa6f2bb5d68](https://github.com/PrismML-Eng/llama.cpp/commit/6fa6f2bb5d686df0c11ac568aecb9e7486d74719) | Not imported: experimental CUDA VMM tiered-KV architecture; no Vulkan path, no suitable hardware validation here. |
| [#320](https://github.com/PrismML-Eng/llama.cpp/pull/320) | [971dcaddc72a](https://github.com/PrismML-Eng/llama.cpp/commit/971dcaddc72a951abbee6fd5e6f27e86f6b9ec14) | Adapted window and per-call MTP cap. Added per-sequence eviction, deferred catch-up coverage, overflow-safe allocation and tests. Omitted tiered-KV tail controls and unsafe depth-only shrinking. |
| [#321](https://github.com/PrismML-Eng/llama.cpp/pull/321) | [c38acb748dbf](https://github.com/PrismML-Eng/llama.cpp/commit/c38acb748dbfb705fb39634c9d098ef26b22d6fc) | Included. |
| [#322](https://github.com/PrismML-Eng/llama.cpp/pull/322) | [487212042603](https://github.com/PrismML-Eng/llama.cpp/commit/487212042603ec84d1cdd5ad408bb7dadb612714) | Included. |
| [#323](https://github.com/PrismML-Eng/llama.cpp/pull/323) | [819054fee35e](https://github.com/PrismML-Eng/llama.cpp/commit/819054fee35e22fcece7e39283f40665c0bf9f2f) | Included. |
| [#325](https://github.com/PrismML-Eng/llama.cpp/pull/325) | [34a932b6df8f](https://github.com/PrismML-Eng/llama.cpp/commit/34a932b6df8fdbeb3d8ef722fc422d1669024e78) | Included. |
| [#326](https://github.com/PrismML-Eng/llama.cpp/pull/326) | [5de20c94894c](https://github.com/PrismML-Eng/llama.cpp/commit/5de20c94894c77d485c81a70654de52e26c70963) | Included. |
| [#327](https://github.com/PrismML-Eng/llama.cpp/pull/327) | [7d5e51b962e1](https://github.com/PrismML-Eng/llama.cpp/commit/7d5e51b962e189c9dfb9afe8a15f8de82843357d) | Included through #328, including its small-batch/register-decode follow-up. |
| [#328](https://github.com/PrismML-Eng/llama.cpp/pull/328) | [b61a07146b66](https://github.com/PrismML-Eng/llama.cpp/commit/b61a07146b6624122c3352d4a389acc111952642) | Included. |
| [#330](https://github.com/PrismML-Eng/llama.cpp/pull/330) | [066e3e36a387](https://github.com/PrismML-Eng/llama.cpp/commit/066e3e36a387908d40dc9baf9efe3d8aa528eb44) | Included. |
| [#333](https://github.com/PrismML-Eng/llama.cpp/pull/333) | [958a5498e9ba](https://github.com/PrismML-Eng/llama.cpp/commit/958a5498e9bac5e1977f7c77d5df15d57e0c0571) | Not imported: empty union handling would silently turn unsatisfiable/invalid constraints into unconstrained acceptance. |
| [#335](https://github.com/PrismML-Eng/llama.cpp/pull/335) | [519f22456b0b](https://github.com/PrismML-Eng/llama.cpp/commit/519f22456b0be4dcb1ec1c4010887ae972bb0467) | Included. |

Draft PRs #94 and #297 were not imported as completed improvements.

## Validation

On Linux x86-64, GCC 13, Release CPU build:

- Full configured build passed, including server, CLI, examples and tests.
- 22 selected CTest tests passed with no skips: Unicode, sampling, reasoning budget, grammar/parser/schema suites, chat, batch allocation, MTP catch-up, the new MTP window test, quantization, PTQ1 element mapping, PQ2 row shapes, GGUF, DSpark metadata and allocation.
- 52/52 Hadamard backend cases passed.
- 45/45 selected GDN backend cases passed.
- 97/97 selected GET_ROWS/CONCAT backend cases passed.
- All eight generated DSpark log-SNR metadata variants behaved as expected.
- New tiny Qwen3.5/MTP fixture passed with window 0, 16 and INT_MAX, each with split and unified KV caches and two slots. It covers cache wrap, inactive-slot preservation, deferred flushing and batched catch-up, draft limits and target-history preservation.
- The fixture also checks 288 seeded top-k samples against the unshortened sampler path, plus grammar-first and grammar-retry paths whose valid token lies outside the shortlist.
- Qwen tool parsing/grammar tests cover all four closing-tag newline variants with adjacent string arguments.
- `git diff --check` passed.

The additional `test-arg-parser` run passed its local argument/environment checks, then failed its external GET to `http://ggml.ai/`. It is not counted as a passing test.

CUDA, HIP, Vulkan, SYCL and Metal were not compiled or executed: this environment has no corresponding GPU/toolchain setup. The GPU conflict resolutions received source review, but still require builds and device testing. The real 27B GGUF was not available, so no model-quality, acceptance-rate or GPU throughput claim is made.

Useful reproduction commands, after configuring a build with tests:

```sh
python3 tests/gen-tiny-qwen35-mtp.py build/tests/tiny-qwen35-mtp.gguf
python3 tests/gen-tiny-dspark.py --suite build/tests/dspark-logsnr-variants
ctest --test-dir build --output-on-failure -R '^(test-mtp-window|test-mtp-catchup-batch|test-chat|test-dspark-logsnr-meta)$'
build/bin/test-backend-ops -b CPU -o MUL_MAT_HADAMARD
build/bin/test-backend-ops -b CPU -o GATED_DELTA_NET -p 'head_count=[48],head_size=(16|32|64|128),'
build/bin/test-backend-ops -b CPU -o GET_ROWS,CONCAT -p 'type=(f16|f32|bf16|ptq1_0|pq2_0),'
```

The fixture generator needs NumPy and the normal gguf-py dependencies. Without a generated model, the MTP window test explicitly skips with code 77.

## Vulkan tuning to try

First compare the working command with one added option:

```sh
--spec-draft-window 16384
```

This opt-in Qwen3.5 MTP setting retains 16,384 prior draft positions plus the active batch and allocation headroom. It reduces draft-cache capacity when possible. The target keeps its full context and absolute positions. The default, 0, retains full draft history.

The expected benefit is less long-context draft attention work and draft KV memory. Shortening the draft's history can change its acceptance rate, so compare total generated tokens per second over a fixed output length at the same prompt depth. Try 32,768 if 16,384 loses too much acceptance.

Separately compare `--spec-draft-n-max 1` with `2`; greater acceptance length is useful only if it outweighs extra drafting and verification. Keep the existing working KV/sampling settings while making each comparison. CUDA-only tuning flags do not affect the Vulkan backend.

## Bundle import

The accompanying incremental Git bundle requires the upstream history through `c35b66744f13cb0dcc476af063e112122eee9355`. Fetch that upstream history first if necessary, then import to a new local branch:

```sh
git fetch /path/to/bonsai-refresh-20261008.bundle bonsai:bonsai-refreshed
git switch bonsai-refreshed
```

The bundle contains the complete refreshed fork series above that upstream base. Keep the old Bonsai branch available when comparing on the GPU.
