# Bonsai Vulkan performance follow-up, 2026-10-09

This pass starts at `084d0f14fbb616ec424ceb21ee76df062be3fb2a`, preserved as `bonsai-before-perf-20261009-0841`. It does not rebase or fetch upstream again.

## Suspected refresh regression

The user observed approximately 67 tok/s before the refresh and approximately 62 tok/s afterward. The available evidence does not establish the cause of that difference.

Comparing `da90ef9776ef41720bb33916de96ca860f9f2a15` with `084d0f14f` shows only four changed SYCL source files and the refresh notes. Vulkan, core inference, model graphs, sampling, server and build-system sources are identical. Both Vulkan source trees have Git tree ID `5ec92121837c5a0f6bc4a0c28ee55ae5d17e8176`.

The preserved local build also provides an artifact comparison:

- All 160 embedded Vulkan shader objects, two Vulkan support objects and the generated shader header match byte-for-byte.
- `libllama`, `libggml-cpu` and `libggml` match byte-for-byte.
- `libggml-base` differs only in the nine-character commit string and 20-byte GNU build ID. Its executable code is unchanged.
- The preserved Vulkan host object has a deliberate test-only integer-dot override and tracing, so whole-library equality is not claimed for that library. The underlying production host source is unchanged.

This rules out a Vulkan source change or shader-regeneration difference in the builds inspected here. It does not verify the user's installed executable, loaded libraries, compiler, driver, environment, context, GPU state or MTP acceptance. The comparison below records acceptance separately from throughput; a throughput change with a different accepted-token sequence does not isolate kernel performance.

## Kernel changes

The PTQ matvec now retains activation columns while decoding one output row at a time. This shortens the lifetime of ternary decode state and integer accumulators. It preserves weight reuse across columns, the existing quantizer and per-output FMA order. Partial row tiles stop at the valid row count. Single-column decoding, dispatch geometry and residual epilogues are unchanged.

Packed attention uses its existing two-token invariant to replace dynamic quotient/remainder operations with comparisons and subtraction. The cooperative-matrix kernel loads each token's mask once for a four-row group, including groups that cross the GQA6 boundary. Odd-token tails and padded rows retain their guards.

Grouped split-K attention also selects one lane to write each normalization summary after the reductions. Previously, every lane wrote the same values. This applies to the scalar and cooperative-matrix paths; split-K scheduling itself is unchanged.

Offline Mesa ACO results are compiler evidence, not measured Radeon throughput:

| Production PTQ specialization | Before VGPR allocation | After VGPR allocation |
| --- | ---: | ---: |
| gfx1201, N2, wave32 | 120 | 96 |
| gfx1201, N2, wave64 | 108 | 84 |
| gfx1201, N3, wave32/64 | 144 | 120 |
| Navi21, N2, wave32 | 112 | 96 |
| Navi21, N2, wave64 | 128 | 96 |
| Navi21, N3, wave32 | 144 | 128 |
| Navi21, N3, wave64 | 168 | 128 |

Across N1 through N8 on both targets and wave sizes, single-column results are unchanged. Multi-column compiler inverse-throughput estimates improve by approximately 15-24%, with no increased register allocation or spills. Those estimates cannot be converted into an end-to-end tok/s prediction.

The packed cooperative-matrix attention candidate reduces static instructions by approximately 5% on gfx1201, with unchanged VGPR, shared-memory and occupancy allocations. The unpacked specialization's executable hash and compiler statistics are unchanged by the indexing/mask change.

| Packed attention, including summary-store change | Before instructions | After instructions |
| --- | ---: | ---: |
| gfx1201, wave32 | 8515 | 8129 |
| gfx1201, wave64 | 5182 | 4896 |

No GDN scheduling change is included. Cooperative Q/K loading made wave64 shuffles too expensive; moving Q loads later lowered register allocation but worsened estimated latency. Assigning four contiguous state values per lane reduced memory instructions but had conflicting compiler estimates and changed reduction grouping. These require device profiling before adoption.

## Validation

The Release Vulkan build passed for `llama-server`, `llama-bench` and `test-backend-ops`, using the existing GCC 13 and shaderc 2026.5-dev toolchain.

- The actual PTQ integer-dot path passed 164 software-Vulkan cases: 64 production shapes, 40 small-K/tail/broadcast cases and 60 residual cases. Two pre-existing strided cases remain unsupported and are not counted as passes. A test-only host override permits lavapipe's supported but non-accelerated integer-dot feature; it is absent from the delivered source.
- A deterministic scratch test harness compared 36 captured output tensors, totaling 41,808 F32 bytes, against the previous shader. Every byte matched. Coverage includes N1/2/3/5/7/8, K1024/5120, odd row tails, in-place and offset residuals, and strided fallback. Both shaders also passed the CPU-reference checks.
- ACO compiled all 64 baseline/candidate combinations across N1-8, wave32/64 and gfx1201/Navi21. Independent compilation from the production shader generator reproduced the results. N1's executable hash is unchanged.
- Independent attention review checked the two-token invariant, padded rows, masks and summary-store lane ownership. Enumeration matched the previous mask selection for 632 vector cases across GQA ratios 2-32. Two added backend cases cover GQA4 and GQA7 odd-token boundaries.
- All 15 attention cases passed on normal software Vulkan with packing enabled and disabled. All 15 also passed in the final integer-dot overlay, including GQA4/7, causal masks, odd tails, offset/permuted views, guarded fallbacks and split-K.
- All 22 changed-family production SPIR-V modules passed validation for Vulkan 1.3: 10 scalar attention, three cooperative-matrix1 attention, three cooperative-matrix2 attention and six PTQ variants. The integrated PTQ module matches the runtime-tested candidate byte-for-byte.
- The comparison script passed six requests against the existing tiny synthetic Qwen3.5 MTP model and HTTP fixtures covering invalid lengths, cached/truncated prompts, token-count mismatches, invalid timings, HTTP errors, counter interference and incompatible settings. Python compilation and whitespace checks passed; the optional `ty` checker was unavailable.

There is no Radeon device or real 27B GGUF in this environment. Software Vulkan cannot execute the cooperative-matrix path. Its shader compilation, offline device compilation and indexing checks do not substitute for cooperative-matrix runtime validation or end-to-end model timing.

## Reproduce the throughput comparison

Follow [Controlled Bonsai server comparison](bonsai-performance-comparison.md). Use the same model bytes, startup flags, environment, prompt, seed and output length for each build, and retain the startup logs. The script sends repeated uncached requests to a dedicated local server, records prompt/decode timings and MTP acceptance, and checks output-token hashes. It does not start or stop the server.

For the original regression, compare `da90ef977` against `084d0f14f`. For this optimization pass, compare `084d0f14f` against the updated branch. Repeat the first build afterward (A/B/A), using separate build/install directories so the executable and libraries match.

## Import

The bundle contains the updated branch, its pre-pass backup and the pre-refresh 67 tok/s baseline. It requires upstream history through `3d65c90d04d337e88f2b1f7f0061f40a5324e662`. Import into new branch names:

```bash
git fetch /path/to/bonsai-vulkan-perf-20261009.bundle \
    bonsai:bonsai-vulkan-perf-20261009 \
    bonsai-before-perf-20261009-0841:bonsai-perf-baseline-20261009 \
    bonsai-before-refresh-20261009-0558:bonsai-67tps-baseline-20261009
git switch bonsai-vulkan-perf-20261009
```

Nothing was pushed.

Functional commits:

- `da6c36a56`: packed attention indexing and mask loads.
- `f78021223`: single-writer grouped split-K summaries.
- `960d60f90`: additional attention boundary fixtures.
- `0b81d4f71`: PTQ multi-column register-state reduction.
- `c49eff31c`: controlled server comparison script and instructions.
