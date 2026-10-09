# Bonsai refresh, 2026-10-09

## Refs

- Starting local and freshly fetched `imaami/bonsai`: `da90ef9776ef41720bb33916de96ca860f9f2a15`.
- Backup: `bonsai-before-refresh-20261009-0558`.
- Previous upstream: `71ad0590f4808b6202f9213d166913858c73b1bc`.
- Fresh `ggml-org/master`: `3d65c90d04d337e88f2b1f7f0061f40a5324e662`.
- Rebased code tip: `11fa16d27`; the following commit records this refresh.

All 208 fork commits are retained. The rebase preserves the complete Vulkan
directory, CUDA/MUSA FWHT source, backend tests and ternary SYCL XMX files
byte-for-byte. The only source changes relative to the starting branch are
upstream's four SYCL files: 133 additions and 84 deletions.

## Upstream and conflicts

Two upstream commits arrived since the preceding refresh:

- `de7fa0a3c`: MUSA FWHT fallback, already covered by Bonsai.
- `3d65c90d0`: SYCL Q5_K reordered-layout MMVQ and fused GLU.

The MUSA change conflicted with the older block-per-row FWHT refactor and the
later fork fallback patch. Both the legacy shared-memory and current block
launch paths keep their 8192-wide MUSA guards. The final FWHT file matches
the previous Bonsai version exactly.

The SYCL update applies without manual conflict resolution. Its Q5_K
activation-reuse traits and BMG fused-reorder eligibility remain separate
from the fork's PTQ1/PQ2 standard MMVQ and XMX paths. Independent source
review found no interaction issue; no SYCL or MUSA toolchain was available.

## Prism review

Fresh API review covered 53 open PRs, including 51 non-drafts, and 88 branches.
Only the `prism` branch advanced. `hadamard-folded-runtime` remains at
`3bfabd2861455dce72c9d1a3e4acf72c8ec79d7e`.

The six new main-branch commits merge previously reviewed patches:

| PR | Merge commit | Disposition |
| --- | --- | --- |
| [#252](https://github.com/PrismML-Eng/llama.cpp/pull/252) | `c53526a094cd22fde83f36ac71330dc03c5fbc16` | Dedicated Vulkan PTQ matvec already present. |
| [#189](https://github.com/PrismML-Eng/llama.cpp/pull/189) | `0d1825501d97f937efa987225de1fa668bd42819` | Retain the previous deferral; see below. |
| [#225](https://github.com/PrismML-Eng/llama.cpp/pull/225) | `5e9365f05e56c7d28ff596c874dcffa3da785b15` | Metal five-row PTQ matvec already present. |
| [#313](https://github.com/PrismML-Eng/llama.cpp/pull/313) | `527ea4d9f2aee165bb55f7057406f66b2de0488a` | Top-k shortlist behavior already integrated with regression coverage. |
| [#312](https://github.com/PrismML-Eng/llama.cpp/pull/312) | `bf197c43d06d7e513dc956459043fa081c4d91a6` | Prompt-checkpoint buffer reuse already present. |
| [#311](https://github.com/PrismML-Eng/llama.cpp/pull/311) | `56f21fb10d42024e58f8e8cc27b2e3e71c9b942f` | Used-cell-range removal already present; this is the current Prism head. |

The changed [#320](https://github.com/PrismML-Eng/llama.cpp/pull/320) head is
`93fcb95a3e8d37c40f514f3e6cb54dd701434f83`. Its per-sequence window trimming
and removal of tiered-tail options are already covered by Bonsai's existing
implementation, including inactive-sequence and deferred-catch-up handling.

No additional Prism patch is imported. In particular, applying #189 verbatim
would drop the native `type_KV` template argument, potentially invoking F16
conversion without corresponding buffer reservation. It also forces a
64-column variant on actual Turing, where the kernel rejects tiles wider than
32 columns. Two independent source reviews confirmed these incompatibilities
with the current fork. A safe CUDA adaptation requires its own reproduction,
allocation/dispatch changes and device validation.

## Validation

The fresh Release Vulkan build passed for `llama-server`, `llama-bench` and
`test-backend-ops`, using GCC 13 and the existing shaderc 2026.5-dev toolchain.
The server enumerated the Vulkan device successfully.

- All 22 selected CTests passed with no skips: MTP window/catch-up, DSpark
  metadata, grammar/schema/chat, sampling, quantization, PTQ element mapping,
  PQ2 row shapes, GGUF and allocation regressions.
- Mesa 25.2.8 lavapipe / LLVM 20.1.2 passed 238 backend cases: 2 exhaustive
  PTQ decodes, 65 Hadamard, 12 FWHT/PTQ producer-consumer, 102 residual-add,
  52 GDN and 5 state-update/rollback cases.
- All 13 packed-attention cases passed with `GGML_VK_FA_GQA_PACK=1`.
- Five large Hadamard cases remain unsupported on this software device;
  they are not counted as passes.
- Upstream ancestry, range-diff, exact final-tree comparisons and
  `git diff --check` passed. An independent review confirmed that every added
  or removed source line relative to the prior tip matches the upstream
  SYCL update.

This refresh used the normal software Vulkan backend, without the previous
test-only integer-dot override. Integer-dot and cooperative-matrix execution
were not retested here; their sources are unchanged. The preceding
[optimization validation](bonsai-vulkan-amdgpu-20261009.md) records those
checks and their limitations. Radeon performance, SYCL and MUSA device
execution were not measured in this environment.

## Import

The bundle includes the refreshed branch and its pre-rebase backup. It
requires upstream history through `3d65c90d04d337e88f2b1f7f0061f40a5324e662`.
Fetch upstream if necessary, then import into new local branch names:

```sh
git fetch /path/to/bonsai-refresh-20261009.bundle \
  bonsai:bonsai-refreshed-20261009 \
  bonsai-before-refresh-20261009-0558:bonsai-backup-20261009
git switch bonsai-refreshed-20261009
```

No remote branch was pushed.
