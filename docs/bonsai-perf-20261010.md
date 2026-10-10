# Bonsai regression investigation and Vulkan/MTP work, 2026-10-10

This pass starts from `7a4f6fadf7e42981ad5c42a1133171ff35c5673e`, saved as `bonsai-before-perf-20261010`. The comparison baseline is `82a0e12a05ee6a680685d9abd797af153ec921de`, saved as `bonsai-before-refresh-20261010`. The user reported about 80 tok/s with the baseline and about 75 tok/s after the refresh, with either setting of `GGML_VK_SSM_CONV_SMALL`. Those are user observations, not measurements made here. No further rebase or push was performed.

## Regression investigation

80 to 75 tok/s is a 6.25% throughput loss, or 6.67% more time per output token: 12.5 to 13.333 ms. The source comparison has not established a cause for that gap.

- No Vulkan shader files changed between those two revisions. The AMD-relevant Vulkan host change is the short SSM convolution schedule; `GGML_VK_SSM_CONV_SMALL=0` restores its old geometry. The other Vulkan change only affects NVIDIA.
- Attention, ternary matmul, Hadamard, GDN, RMS, graph submission and barrier code are unchanged. The CPU conversion changes are s390x-only.
- MTP input handling changed to a separate hidden-state channel. The ordinary text algorithm, graph arithmetic and number of draft decodes are unchanged. A same-host CPU microbenchmark of batch assembly, state packing and microbatch allocation at width 5120 measured roughly 4-13 microseconds for one to three rows on both revisions. This is not Radeon inference timing; it bounds the suspected extra packing cost far below the approximately 833 microseconds per output token in the reported gap.
- The JSON vendor update runs serialization in the HTTP response thread, after the decode thread queues a typed result. A representative 180-byte streaming chunk takes roughly one microsecond with either header in a same-compiler microbenchmark. This is not a plausible explanation for the full difference.

For a one-token draft, lower acceptance produces fewer output tokens per verification round. At unchanged round cost, acceptance falling from 0.86 to about 0.744 would produce the entire 80-to-75 difference. That is an illustration, not an assertion about these runs. Context length and the accepted/rejected pattern can also change the work inside each round.

`scripts/server_perf_compare.py compare` now reports median generation milliseconds per verification round from its existing saved records. This includes drafting, catch-up and host overhead. It is not a GPU kernel timer. Use the same prompt, seed, output length, context, model bytes and startup settings, and compare acceptance and output hashes alongside throughput. See [the comparison procedure](bonsai-performance-comparison.md).

## New changes

### Skip unused MTP draft-state readback

For a configured `--spec-draft-n-max 1`, the draft context no longer copies its final hidden state to the CPU. This row is only an input to a subsequent draft token; the next verification round obtains its state from the target. The accessor also moves below the global and per-call stopping checks. At width 5120, this removes a 20 KiB draft readback per round, its Vulkan copy/barrier commands and an unnecessary synchronization accessor. Sampling still has its own required synchronization. The target readback remains.

This is automatic for one-token MTP. Multi-token drafting retains hidden-state output. The paired regression test compares a context with the new behavior against one explicitly retaining the old readback: eight rounds each for global limits 1/2 and backend sampling off/on, per-call limits 1/2, accepted/rejected prefixes, exact draft tokens, logits and candidate IDs. The hidden-state buffer is absent only in the one-token candidate.

This waste existed before the reported regression; removing it is an optimization, not an identified regression fix.

### Fuse GDN normalization and scalar scaling

Qwen's joint Q/K normalization is implemented as `RMS_NORM(eps/n)` followed by `SCALE(1/sqrt(n))`. The Vulkan backend now folds that scalar epilogue into RMS, removing one dispatch and intermediate read/write per recurrent layer. It preserves the original normalization and multiplication order rather than replacing the expression with an algebraically equivalent L2 reduction. A helper isolates the rounding constraint to the epilogue so it does not disable contraction in the existing RMS reduction. The original signed-zero bias bits are retained.

The fusion requires F32 tensors, contiguous input rows, contiguous outputs, zero bias and the existing use-count/output/alias checks. A norm output needed elsewhere and the existing ADD-partial reuse path decline this fusion. Other RMS shader variants retain their previous preprocessed source. It defaults on for RDNA4; `GGML_VK_RMS_SCALE=0` disables it in the same binary, and `=1` enables it on another device for validation.

The graph optimizer also preserves eligible RMS/SCALE pairs and keeps their inputs alive through the fused output. Model tracing caught that the original scheduler otherwise pulled independent matrix multiplications between these nodes, preventing the new fusion. The protection is conditional on the same device, shape and dataflow eligibility; the mutable ADD-partial check stays in execution selection.

### Optional eight-row ternary matvec

`GGML_VK_PTQ1_ROWS=8` makes the existing PTQ1 integer-dot shader compute eight output rows per workgroup for one- and two-token inputs on RDNA4. The default remains four. Other column counts, quantizations and devices retain their existing schedules. Both the workgroup dispatch divisor and the shader row specialization use the same value; residual-add fusion uses the same pipeline.

This reuses each activation load across more weight rows without changing the arithmetic within each output row. Offline gfx1201 compilation indicated fewer instructions per output row, but some wave64 variants require more registers. It is an experiment requiring device measurements, not a claimed throughput gain. Do not combine it with other experimental settings for its first A/B comparison.

The wave64 register allocation changes from 60 to 72 for one token and 84 to 96 for two, reducing the compiler's estimated occupancy. Wave32 allocation remains 96 for both token counts. Compiler inverse-throughput estimates per row improve by about 10-11%, but are not device timings. The patch leaves subgroup selection unchanged.

## Validation

All 15 existing RMS SPIR-V modules remain byte-identical. Both new modules pass Vulkan 1.2 SPIR-V validation. All 24 offline gfx1201 specializations compile: legacy workgroup 512 and subgroup workgroups 128/256/512, default/32/64 wave requests, and baseline/fused epilogues. The fusion leaves VGPR and shared-memory allocations unchanged, with zero spills. The `NoContraction` annotations are confined to epilogue multiplication.

A direct software-Vulkan comparison produces byte-identical results for the old unfused binary, the new binary with fusion disabled, and the fused path: 2,139,264 finite F32 values across 60 cases. These cover widths 63/128/129/512/1025, one/two tokens, two sequences, contiguous and offset/strided views, positive/negative/zero scales, and both signs of zero bias. The performance logger confirms all 60 fused dispatches. This executes with Lavapipe's subgroup size 8; it is not a Radeon timing or numerical measurement.

The eight-row PTQ path passes 171/171 supported software-Vulkan cases; two existing strided cases remain unsupported. A separate deterministic four-row/eight-row comparison passes 52/52 CPU-reference cases in each mode, with all 52 tensors (49,460 bytes) bit-identical between modes. Coverage includes one/two-token inputs, row boundaries 7/8/9, larger tails, broadcasts and residual fusions. The isolated test backend enables Lavapipe's supported integer-dot path and bypasses only the RDNA4 gate; these validation overrides are not in the delivered code. The normal Lavapipe fallback would not test this path.

An initial scratch capture using artificially identical initialization across input tensors exceeded the CPU quantization tolerance in one case equally for both schedules, while still matching between them. The final capture uses independently seeded tensors, and both it and the normal broad suite pass. The comparison does not claim to eliminate the existing Q8 activation approximation.

All eight production-config PTQ specializations (four/eight rows, one/two tokens, wave32/64) compile for gfx1201 with robust buffer access and full subgroups enabled, without spills or scratch use.

The integrated RMS backend suite passes 27/27 cases in each of three configurations: fusion with subgroup reduction, fusion disabled with subgroup reduction, and fusion with the legacy reduction. That is 81 CPU-reference checks, including the new eligibility/fallback fixtures.

After the scheduler fix, the final backend again passes all 27 RMS cases. A real tiny-Qwen35 graph with 16 prompt tokens and four single-token decode steps now selects all 15 expected RMS/SCALE fusions under normal scheduling: three prompt dispatches and twelve decode dispatches. All 1,295 finite output logits are byte-identical to the earlier unfused control, and the corresponding 15 standalone SCALE launches disappear. This checks actual model construction, scheduling, allocation and recurrent-state execution, rather than just a hand-built operator graph.

All four focused runtime CTests pass: sampling, batch allocation, MTP catch-up batching and the expanded MTP window/integration test. The latter passes all six existing window/cache combinations, both embedded-input cases and all four new state-output configurations; none skip. An unmodified MTP negative control fails the new readback-presence assertion.

The Release build includes `llama-server`, `llama-bench`, both backends and the focused test binaries. The initial integrated checks ran at `a7f3ba508`; the subsequent scheduler fix is `3dde65196`. A malformed local linker output and one missing executable bit were repaired before runtime validation; binaries were linked from the compiled objects using the CMake-generated commands. No source change was needed for that repair. The final scheduler pass rebuilds the Vulkan host object and build information before relinking the same targets.

The final server reports `3dde65196`. Later commits only record these notes.

There is no physical Radeon device or real 27B model in this environment. Software Vulkan, the tiny Qwen35 fixture and offline AMD compilation establish correctness evidence, not a measured tok/s gain or resolution of the reported regression.

## Further approaches

A larger possible step is keeping target-to-draft hidden states on the GPU. This needs a persistent device buffer, explicit producer/consumer synchronization and correct carry-row lifetime through acceptance, rollback and graph reuse. The target graph's tensor cannot simply be borrowed: its arena may be reused before deferred catch-up consumes it. An evaluation callback would introduce graph splits and synchronization. No such bridge is included in this patch.

Another source-level candidate is allowing the existing SwiGLU/Hadamard fusion to cross the zero-offset contiguous reshape in the recurrent output path. Its present matcher expects an adjacent GLU-to-sign-multiply chain. This could remove another dispatch without new mathematics, but needs actual graph matching and alias validation; it is not included here.

Two-token drafting is another hardware experiment: at conditional acceptance 0.8, its illustrative expected output is 2.44 rather than 1.8 tokens per round. It wins only if the extra draft step and three-token target verification cost less than that approximately 36% output increase. Real conditional acceptance and context-dependent costs must be measured; this pass keeps the one-token configuration.

## Import and compare

The bundle includes this pass plus the complete refresh, and both earlier comparison branches. It requires upstream history through `781dbc5ac`:

```bash
git fetch https://github.com/ggml-org/llama.cpp.git master
git fetch /path/to/bonsai-perf-20261010.bundle \
    bonsai:bonsai-perf-20261010 \
    bonsai-before-perf-20261010:bonsai-perf-before-20261010 \
    bonsai-before-refresh-20261010:bonsai-perf-80-baseline-20261010
git switch bonsai-perf-20261010
```

Build into a separate directory and use the same server arguments as before. First measure the new defaults, with the same SSM setting used for the previous measurement. Then try only `GGML_VK_PTQ1_ROWS=8`; unset it to restore the default. `GGML_VK_RMS_SCALE=0` separately disables just the new normalization fusion, leaving the MTP readback optimization enabled.

To answer the regression question, compare `bonsai-perf-80-baseline-20261010` against `bonsai-perf-before-20261010`, setting `GGML_VK_SSM_CONV_SMALL=0` for the latter. This isolates the last refresh from the new work in this pass. Use the fixed-request [comparison procedure](bonsai-performance-comparison.md), include the original build again afterward, and preserve the startup logs. The `compare` command accepts all three saved JSON files at once.
