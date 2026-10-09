# Bonsai Vulkan/AMD optimization pass, 2026-10-09

This pass starts from `881e60e252daced895f381765b17509d7571cf37`, preserved as
`bonsai-before-vulkan-20261009`. It adds kernel and graph improvements to the
previously refreshed branch; it does not perform another upstream rebase.
The upstream base remains `71ad0590f4808b6202f9213d166913858c73b1bc`.
Kernel changes end at `cb4ae3946`; the following commit records these notes.

## Default paths

| Change | Eligibility and effect |
| --- | --- |
| SwiGLU + signed Hadamard | Contiguous split F32 SwiGLU feeds signs and FWHT in one dispatch. Preserves intermediate rounding and rejects observable/aliased intermediates. |
| FWHT + Q8 activation cache | Subgroup FWHT can also emit the existing packed Q8_1 cache for the immediate compatible PTQ matvec. Removes the separate quantizer dispatch and its F32 read. Retains F32 output for other consumers. |
| Residual epilogue | Dense PTQ matvecs with 2-8 columns can absorb one or two same-shape residual adds. Broadcasts, strided residuals, larger matrices and input/output overlap fall back. |
| Raw GDN gates | Applies beta sigmoid and decay softplus inside the recurrent shader, removing the separate gate operations on the supported Qwen35 path. |
| Indexed recurrent state | Reads the selected cache row directly. Existing snapshot writes, SET_ROWS updates and rollback layout remain intact. Model selection retains its single-sequence and relocation guards. |
| Shared verification attention | On RDNA4 coopmat1, packs two GQA6 queries with q8_0 K / q4_0 V into the existing 16-row tile. Shares K/V loading and decoding between tokens, preserving each token's mask and output position. |

The FWHT cache uses the same 144-byte Q8_1 record per 128 values, including
four independently rounded half-precision scales and stored sums. It does
not adopt CUDA's alternative sum representation. Scratch reuse follows the
existing one-entry cache. Shared-memory FWHT, F16 input, incompatible views,
intervening dispatches and ineligible consumers retain the old quantizer.
Tests include 1024-wide Hadamard blocks as well as smaller generic blocks;
the ternary quantization block size of 128 is a separate quantity.

The attention default excludes sinks, ALiBi, softcap and sparse attention.
Single-token attention is unchanged. `GGML_VK_FA_GQA_PACK=0` disables packing;
`=1` allows compatible experimental scalar and wider verification cases.

## Experimental paths

Both options are disabled by default. Test them independently before combining.

| Environment variable | Candidate |
| --- | --- |
| `GGML_VK_PTQ1_PACKED_DOT=1` | Packed-trit integer dots for single-column PTQ decode and one-column remainders. Preserves integer sums, quantization and floating-point accumulation. Fewer source dot calls and lower live-register demand do not by themselves establish higher throughput. |
| `GGML_VK_PTQ1_MMQ=1` | RDNA4 integer cooperative-matrix PTQ prefill for dense matrices with at least 16 columns. Changes activation precision from F16 to Q8_1; requires model-output/perplexity evaluation as well as timing. |

These are pipeline selections, not dynamic per-element branches. Leaving the
variables unset or setting them to `0` retains the existing respective paths.
The MMQ decoder retains all four activation scales and stored-sum bias terms
per ternary block. Its shared-memory accounting includes those metadata arrays.

## Validation

Release CPU and Vulkan builds passed, including `llama-server`, the Vulkan
benchmark and backend test executable. Shader compilation used shaderc
2026.5-dev (`ba3e587dbc13d423c713e964ac08e094731a034d`). All 34 final SPIR-V
modules for the changed shader families passed validation, including
cooperative-matrix variants.

- CPU: 22 selected CTests passed, including MTP, grammar, quantization and
  PTQ mapping checks. All 59 GDN fixtures passed.
- Mesa 25.2.8 lavapipe / LLVM 20.1.2: 60 residual cases, 59 GDN cases, 77
  supported Hadamard/producer-consumer cases, and 12 prefill fallback cases
  passed against CPU. Five large Hadamard cases remain unsupported and are
  not counted as passes. The 14 SwiGLU cases also passed with fusion disabled.
- Attention: all 13 new cases passed with packing enabled and disabled.
  All 13 also passed through the software integer-dot scalar path, covering
  per-token causal masks, split-K, odd token tails, offset/permuted views,
  padded masks, asymmetric head dimensions and guarded fallbacks.
- An isolated test-only backend enabled lavapipe's supported but
  non-accelerated integer-dot feature. No such override is in the delivered
  source. This exercised the actual integer kernels: all 60 residual cases,
  12 FWHT cache cases with fusion on and off, and 55 supported packed-dot
  matrix cases passed. One existing strided matrix case remains unsupported.
  Twelve default-decoder control cases also passed.
- Dispatch traces confirm eligible FWHT sidecars remove one standalone
  quantizer. Shared consumers reuse the cache; an intervening scratch user
  forces regeneration before the original activation is reused.
- A direct Vulkan shader harness checked 180 adversarial cases and
  14,256,000 Q8 output bytes against both existing quantizers. Every byte
  matched, retained F32 outputs matched bitwise, and output guards remained
  intact. This caught and fixed a real half-integer-boundary rounding
  difference caused by extra `precise` qualifiers. Runtime used subgroup8;
  block1024 was tested by direct specialization, while the normal lavapipe
  backend selects its shared-memory fallback at that size.
- A temporary Vulkan adaptation of `test-mtp-window` passed all six
  configurations, including wrapped draft caches, inactive-slot preservation,
  rollback, deferred catch-up and resumed shallow slots. Instrumentation
  observed 792 Vulkan GDN nodes with raw gates and indexed state reads active.
- The experimental decoder passed exhaustive packed-byte/signed-byte checks
  and five million full-word comparisons. The MMQ decoder matched CPU across
  1,376,256 decoded values; 4,096 dot products matched an independent reference
  within float rounding.

Offline Mesa ACO compilation for gfx1201 passed the production specializations
without spills or scratch. FWHT sidecars retained the baseline 96-VGPR
allocation at block1024/wave32. Packed attention retained the baseline
register/LDS allocation on wave32 and wave64. Packed single-column dots reduced
live-register demand, but compiler throughput estimates were mixed, supporting
their opt-in status. Integer prefill compiled all three tile sizes. These are
compiler-resource checks, not cooperative-matrix runtime or timing results.

No Radeon device or real 27B GGUF is available in this environment. Software
Vulkan execution, arithmetic checks and offline gfx1201 compilation cannot
establish end-to-end speed or model quality. No tokens/second improvement is
claimed from these checks.

## Radeon comparison

Build `bonsai-before-vulkan-20261009` and `bonsai` with the same compiler and
options in separate directories. Keep the model, server flags, request,
context depth and output length identical; compare prompt processing and
generation separately after warm-up. Include fresh and long contexts, and
record speculative acceptance and mean accepted length.

Run focused correctness checks first, using the path to your build:

```sh
build/bin/test-backend-ops -b Vulkan0 -o MUL_MAT_ADD -p 'type_a=ptq1_0'
build/bin/test-backend-ops -b Vulkan0 -o MUL_MAT_HADAMARD,FWHT_PTQ1
build/bin/test-backend-ops -b Vulkan0 -o GATED_DELTA_NET,GATED_DELTA_NET_ROWS_UPDATE
GGML_VK_FA_GQA_PACK=1 build/bin/test-backend-ops -b Vulkan0 \
  -o FLASH_ATTN_EXT -p 'causal=1'
GGML_VK_PTQ1_PACKED_DOT=1 build/bin/test-backend-ops -b Vulkan0 \
  -o MUL_MAT -p 'type_a=ptq1_0,type_b=f32,m=67,n=(1|2|3|4|7),'
GGML_VK_PTQ1_MMQ=1 build/bin/test-backend-ops -b Vulkan0 \
  -o MUL_MAT -p 'type_a=ptq1_0,type_b=f32,m=(33|65),n=(15|16|17|31|32|33),k=(128|384),'
```

For diagnosis, `GGML_VK_PERF_LOGGER=1` shows operations and fused dispatches.
Turn it off for throughput measurements. `GGML_VK_DISABLE_FUSION=1` disables
graph fusion, including the FWHT sidecar, but also affects unrelated existing
fusions. `GGML_GDN_RAW_GATES_DISABLE=1` and `GGML_GDN_STATE_GATHER=1` restore
the prior model-level GDN gate and gathered-state paths. The baseline branch
is the cleanest whole-change comparison.

## Import

The bundle includes `bonsai` and `bonsai-before-vulkan-20261009`, and requires
upstream history through `71ad0590f4808b6202f9213d166913858c73b1bc`.
Import into new local branch names to preserve your current checkout:

```sh
git fetch /path/to/bonsai-vulkan-amdgpu-20261009.bundle \
  bonsai:bonsai-vulkan-amdgpu \
  bonsai-before-vulkan-20261009:bonsai-vulkan-amdgpu-baseline
git switch bonsai-vulkan-amdgpu
```

No remote branch was pushed.

## Functional commits

- `4434b6a76`: PTQ residual epilogues; `fd57460a8` retains Hadamard-hint fallback.
- `b1839fbfe`: split SwiGLU/signed-FWHT fusion.
- `dbdb2c6dc`: native raw GDN gates.
- `0f8fa4b16`: indexed recurrent-state reads.
- `febbdf25a`: shared verification attention.
- `e356d0739`: opt-in integer PTQ prefill.
- `470856a12`: opt-in packed single-column PTQ dots.
- `a56682b35`: FWHT Q8 sidecar; `cb4ae3946` matches existing quantizer arithmetic.
