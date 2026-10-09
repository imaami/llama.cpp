# Bonsai refresh and RDNA4 RMS reductions, 2026-10-09

This pass starts from published `imaami/bonsai` at
`417d84e1f7435700717a395ca316d42440dc1b60`, preserved as
`bonsai-before-refresh-20261009-evening`. The checkout and its earlier notes were
recovered from GitHub. Nothing was pushed.

## Fresh upstream and Prism audit

The branch is rebased onto freshly fetched `ggml-org/master` at
`8e2d31e0eb658d0382d3106a61369dff81f0c080`, 22 commits beyond the previous
`3d65c90d04d337e88f2b1f7f0061f40a5324e662` base. All 215 fork commits remain:
207 replay with identical patches and eight have integration/context adjustments.

The relevant upstream changes include:

- [RMS dispatch overflow fix](https://github.com/ggml-org/llama.cpp/pull/30145):
  clamp dispatch dimensions and loop over excess channels/samples in the shader.
- [CUDA MMQ precision API](https://github.com/ggml-org/llama.cpp/pull/30168):
  ternary loaders and template instantiations now pass `GGML_PREC_Q8` explicitly.
- [Static backend sampling graphs](https://github.com/ggml-org/llama.cpp/pull/30223):
  preserve the fixed per-sequence output/node budget together with DSpark padding
  and its greedy shortcut. This affects backend sampling, not every decode path.
- [Embedding graph order](https://github.com/ggml-org/llama.cpp/pull/30160):
  retain the early token lookup nodes while keeping Hadamard inverse/signs before
  LoRA, padding and scaling. Borrowed DFlash embeddings retain their transform.
- [Chat API refactor](https://github.com/ggml-org/llama.cpp/pull/30210):
  reasoning policy now reads session-owned end tags. The refactor also avoids a
  PEG-arena copy during output parsing and caches template analysis. No measured
  throughput improvement is claimed for these CPU-side changes.

The server's upstream default port is now 9931. An explicit `--port 8080`, as in
the user's command, still selects 8080.

Prism was inspected through `2a42998c560cbda1c225de8b9bd678573ffa5c4a`, including
87 branch heads and 51 open pull requests. Its new merged CUDA FA-quant and Ada
PQ2 multi-column patches were already represented in Bonsai. The
`hadamard-folded-runtime` head remains
`3bfabd2861455dce72c9d1a3e4acf72c8ec79d7e`; grammar-clone binary search was also
already included.

Two new Prism proposals do not warrant an import for this task:

- [#337](https://github.com/PrismML-Eng/llama.cpp/pull/337) guards a quantized-KV
  overwrite in the earlier #189 path, which this branch never imported. Its
  guard does not make that original path a suitable new optimization here.
- [#338](https://github.com/PrismML-Eng/llama.cpp/pull/338) adds opt-in periodic
  prompt checkpoints. These can help replay edited long prompts, but add prefill
  copies and RAM use rather than improving decode. Active checkpoint memory is
  not bounded by `--cache-ram` in the proposal.

## RDNA4 optimization

Adapted the subgroup RMS reduction idea from
[upstream #29882](https://github.com/ggml-org/llama.cpp/pull/29882) for RDNA4:

- Use 128-thread workgroups for rows up to 128 columns, 256 for rows up to 256,
  and 512 otherwise.
- Reduce inside subgroups, write one partial per subgroup, and combine those
  partials after one workgroup barrier. This removes most shared-memory reduction
  barriers from the old tree reduction.
- Derive the subgroup count from the specialized workgroup size and the fixed
  default subgroup size. A first version using the runtime subgroup-count builtin
  generated extra code; that version is superseded. Full subgroups are required,
  varying sizes are disabled, and no explicit subgroup-size override is needed.
- Cover plain RMS, weight multiplication, residual/post-multiplication,
  F32/F16 set-rows and supported RoPE fusions. Preserve the upstream overflow
  loops and their shared-memory reuse barrier. Partial-sum RMS kernels are
  unchanged.

The new path defaults on for capable RDNA4 devices. Set
`GGML_VK_RMS_SUBGROUPS=0` to compare against the legacy path using the same binary;
`=1` requests the new path on other capable devices. This changes floating-point
summation order and is not expected to be bit-identical.

The legacy shader keeps its original literal 512-thread workgroup declaration;
only the subgroup variant uses a specialized block size. This avoids changing
the compiler's treatment of the fallback and other GPU architectures.

The existing PTQ integer-dot matvec, packed two-token attention, signed Hadamard,
GDN state handling and Q8 activation reuse optimizations are retained.

Other reviewed upstream proposals were left out: small-N GEMM split-K (#30146)
does not target the main PTQ one/two-token path or the user's ubatch 128; chunked
cooperative-matrix GDN (#30207) is still preliminary and requires newer AMD
driver features; asynchronous input upload (#30169) did not establish a gain on
the relevant Linux/ReBAR configuration. None is a justified blanket switch for
this workload.

## Validation

The integrated Release build passed for `llama-server`, `llama-bench` and the
selected test executables at `792dc98c1`. Both binaries report that source
revision. A zero-byte intermediate Vulkan object was regenerated before the
successful final build; no empty object files remain.

The subgroup path passed 95/95 focused software-Vulkan cases with no unsupported
or failed cases: 27 plain/view/in-place RMS, 29 residual/post-multiply, 34
set-rows/RoPE cases including F32/F16 and normal/NEOX modes, and five existing
ADD-to-RMS partial-sum cases. Coverage includes both plain overflow axes and
fused residual/RoPE overflow. Output tensors were checked against the CPU
reference using the test suite's numerical tolerances, NaN/Inf checks and guard
checks; this is not a bit-identity claim.

All 20 built CPU/runtime CTests passed, including the new chat-session policy
cases, sampling/reasoning/grammar, quantization and PTQ/PQ regressions. The tiny
fixtures exercised all eight DSpark metadata variants and six MTP window
combinations (0/16/INT_MAX, unified cache on/off). Five explicit multi-output
backend-sampling tests also passed on CPU with `GGML_SCHED_DEBUG_REALLOC=1`.
There were no fixture skips. Two registered but unbuilt tests (chat-template and
arg-parser) are excluded from these counts.

The legacy RMS control passed the same 95/95 cases, also with no unsupported or
failed cases: 190 successful numerical checks across both modes.

All 15 emitted RMS SPIR-V modules passed validation, including the six new
subgroup modules and six corresponding legacy modules. Disassembly confirms that
the final subgroup shaders no longer use the runtime `NumSubgroups` builtin.

Offline Mesa ACO compilation for gfx1201 used the actual production settings:
default subgroup size 64, no explicit size override, and full subgroups required
only for the candidate. The baseline is the rebased shader, including upstream's
overflow fix, before the subgroup optimization. Both versions compile to wave64,
with 48 VGPRs and no spills in each case below. These are whole-kernel compiler
statistics; the row width selects the corresponding workgroup specialization.

| Kernel / row width | Workgroup before / after | Instructions before / after | Shared memory before / after |
| --- | ---: | ---: | ---: |
| RMS with weight, 128 | 512 / 128 | 10,747 / 10,093 | 2,048 / 512 B |
| RMS with weight, 256 | 512 / 256 | 10,747 / 10,153 | 2,048 / 1,024 B |
| RMS with weight, 5120 | 512 / 512 | 10,747 / 10,364 | 2,048 / 2,048 B |
| RMS with weight and RoPE, 256 | 512 / 256 | 11,527 / 11,110 | 6,144 / 3,072 B |

The disabled plain and RoPE variants reproduce the rebased baseline's SPIR-V
and AMD assembly byte-for-byte. An initial compiler comparison accidentally used
the pre-refresh shader without the overflow loops; the table above corrects that
baseline. Against the correct baseline, compiler latency estimates also improve
by approximately 3–5%, but there is no measured device timing in this pass.

Explicit wave32 compilation increases allocated VGPRs from 48 to 96, with no
spills and no decrease in the compiler's reported same-wave capacity bound.
That remains a risk to measure on a driver using wave32; the inspected default
wave64 configuration does not have this allocation increase.

The validation toolchain is GCC 13, CMake 4.4.4, shaderc 2026.5-dev at
`ba3e587`, and Mesa 25.2.8 software Vulkan. The Release build includes the CPU
and Vulkan backends, server, benchmark and focused test executables. Prebuilt UI
asset downloads and OpenSSL are disabled for this validation build.

There is no Radeon device, CUDA toolkit or real 27B GGUF in this environment.
CUDA changes have static API/instantiation review, not a CUDA compile. Software
Vulkan and offline gfx1201 compilation do not establish a tok/s improvement or
replace model-level checks on the 9070 XT.
Lavapipe's subgroup size is eight, so it executes the general second-stage
reduction. The at-most-four-subgroups branch used by narrow RDNA4 rows was
reviewed and compiled for gfx1201, but was not executed on local hardware.

## Compare on the 9070 XT

Use the same model, driver, compiler, startup flags and fresh-context prompt for
both runs. Start one run normally and the other with
`GGML_VK_RMS_SUBGROUPS=0` before `llama-server`. Repeat the first configuration
afterward. Record MTP acceptance alongside throughput; reduction-order changes
can alter sampled tokens, so token-sequence differences need to be distinguished
from kernel speed.

The existing [controlled comparison instructions](bonsai-performance-comparison.md)
and `scripts/server_perf_compare.py` provide the request and timing checks.
To isolate the rebase itself, compare the saved pre-refresh branch against the
refreshed branch with `GGML_VK_RMS_SUBGROUPS=0`, using separate build directories.

## Import

The bundle contains `bonsai` and `bonsai-before-refresh-20261009-evening` and
requires upstream history through `8e2d31e0e`. Import into a new branch name:

```bash
git fetch https://github.com/ggml-org/llama.cpp.git master
git fetch /path/to/bonsai-refresh-20261009-evening.bundle \
    bonsai:bonsai-refresh-20261009-evening \
    bonsai-before-refresh-20261009-evening:bonsai-refresh-baseline-20261009-evening
git switch bonsai-refresh-20261009-evening
```

This keeps the existing local `bonsai` ref available until the refreshed branch
has been compared on the user's hardware.
