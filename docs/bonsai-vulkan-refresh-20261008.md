# Bonsai refresh and Vulkan kernels, 2026-10-08

## Rebase and Prism refresh

- Starting `imaami/bonsai`: `f619989c3ec0767a34a86389c9774e8810cd011f`.
- Previous upstream: `c35b66744f13cb0dcc476af063e112122eee9355`.
- Fresh `ggml-org/master`: `71ad0590f4808b6202f9213d166913858c73b1bc`.
- Original branch preserved as `bonsai-before-refresh-20261008-1910`.
- Kernel and test changes end at `53338c29c0d2bb80172d511407fcd88152c05f41`;
  the following commit records these notes.
- Baseline for the Vulkan changes: `bonsai-vulkan-baseline-20261008`,
  `8ec354e96752192d8b7a95d735449c2bb3d51079`. This includes the rebase, Prism
  refresh and new tests, but predates both Vulkan optimizations.

All 187 existing fork commits were retained. Range-diff shows 186 unchanged
patches and one adapted patch: the GB10 CUDA integration. Upstream's MMQ fix
changed tile selection and input padding. Dense, MoE and fused/shared-Q8 paths
now use the same launch selection logic. Shared quantized input buffers reserve
enough padding for both normal and fallback tile geometries. The upstream
top-k selection and DFlash head-sharing fixes are retained.

The new Prism review covered 59 open PRs, 57 non-draft. Relative to the preceding
review, #336's grammar clone lookup is imported, with a test fix that checks
pointer ownership through equality rather than ordering unrelated pointers.
The updated #320 MTP draft-limit behavior is already covered by Bonsai's bounded
draft-window implementation. The new `fwht-musa-smem` branch
(`b6f1bb50da9e7c1be0d2433b685a8113968e967d`) supplies the 8192-wide MUSA FWHT
fallback; both block and shared-memory dispatch sites are guarded here.

`PrismML-Eng/prism` remains at `4fbda1292562cc1f23904d5867fb79603737547a`;
`hadamard-folded-runtime` remains at `3bfabd2861455dce72c9d1a3e4acf72c8ec79d7e`.
The earlier [PR audit](bonsai-refresh-20261008.md) records the existing
integration decisions. No remote branch was pushed.

## Vulkan changes

### Packed PTQ1 decoding

The generic PTQ1 decoder replaces repeated base-3 recurrence steps with exact
modular multiplication. Two- and four-element helpers share 32-bit packed loads
and decode separate bytes in 16-bit lanes. Matrix tiles and standalone
dequantization use paired four-element decodes; row gathers use paired decodes.
Unaligned element indices retain scalar fallbacks. The 28-byte block layout,
last-eight-element mapping, and scale-before-conversion order are unchanged.

This targets prompt processing, gathers and dequantization. The existing
specialized PTQ1 matrix-vector kernels and dispatch thresholds are unchanged.

### Signed Hadamard fusion

The F32 `MUL(signs) -> RESHAPE -> Hadamard MUL_MAT` chain can execute in a single
FWHT shader. Both subgroup and shared-memory implementations load signs directly.
For each eligible transform this removes one dispatch and the intermediate
activation write/read: `8 * width * tokens` bytes. Signs wrap at the original
activation row width, including 40 blocks of 128 in the 5120-wide regression
case. This test shape does not establish the Hadamard block size of a GGUF.

The shader preserves `(input * sign) * normalization` rounding order. Fusion
requires contiguous F32 inputs, whole transform blocks per row, compatible
shapes, and a supported pipeline. Live intermediate outputs, additional
consumers, in-place multiplication and overlapping buffers retain the existing
unfused path. No new command-line option is required.

## Validation

Release builds used GCC 13 on x86-64. The configured CPU build passed, including
server and CLI. Vulkan builds of `llama-server`, `llama-bench` and
`test-backend-ops` passed; the server successfully enumerated the Vulkan device.
Vulkan shader compilation used shaderc 2026.5-dev
(`ba3e587dbc13d423c713e964ac08e094731a034d`), with all seven compiler feature
checks enabled, including cooperative matrices and integer dot products.

- CPU: 22/22 selected CTests passed with zero skips, including the grammar,
  MTP, DSpark, quantization and PTQ element-mapping checks from the previous
  refresh. All 62 Hadamard backend cases and both exhaustive PTQ decode cases
  passed.
- Software Vulkan (Mesa 25.2.8 lavapipe, LLVM 20.1.2): 57 supported Hadamard
  cases passed. Five 8192/16384-wide cases were unsupported on this device;
  they are not counted as passes. Both exhaustive packed-byte gather cases
  passed with zero error, together with eight other row-gather cases and 26
  matrix cases covering small batches, broadcasts, indexed matrices,
  5120-wide Bonsai inputs, larger prefill tiles and partial output rows.
- Dispatch traces confirmed a single `FWHT_SIGNED` dispatch for eligible
  block-128 / width-5120 chains, versus two dispatches in the baseline. All
  five fallback cases retained separate operations. The ten new cases also
  passed with Vulkan fusion explicitly disabled.
- SPIR-V validation passed for 24 PTQ/FWHT modules and 38 generic matrix
  modules containing the PTQ decoder, including cooperative-matrix variants.
  Exhaustive host arithmetic checks independently confirmed packed-lane
  decoding against the sequential recurrence.

The standalone PTQ dequantization fallback compiled and passed SPIR-V and
decoder arithmetic checks; lavapipe's matrix dispatch did not select that
fallback. Cooperative-matrix and integer-dot shaders compiled, but lavapipe
does not expose the corresponding runtime features. CUDA/MUSA changes received
source review, not compilation or device testing.

No Radeon GPU or real 27B GGUF was available here. These checks establish
operation correctness and dispatch reduction, not an end-to-end tokens/second
gain or model-quality result. Benchmark both branches on the target card.

## Comparing on Radeon

Use the baseline branch above and `bonsai` in separate build directories, with
the same compiler and CMake options. Compare the same GGUF, prompt, output
length, context depth and server flags. Warm up each build, repeat runs, and
record prompt and generation throughput separately. A fresh-context result and
a long-context result are both useful for the MTP workload.

Focused correctness checks with a build that enables tests:

```sh
build/bin/test-backend-ops -b Vulkan0 -o PTQ1_DECODE
build/bin/test-backend-ops -b Vulkan0 -o MUL_MAT_HADAMARD
build/bin/test-backend-ops -b Vulkan0 -o MUL_MAT -p 'type_a=ptq1_0,.*m=7,'
build/bin/test-backend-ops -b Vulkan0 -o MUL_MAT \
  -p 'type_a=ptq1_0,type_b=f32,m=(64|96|129|160),n=(64|128),k=(256|384|512),bs=\[1,1\],nr=\[1,1\]'
```

For a dispatch trace of the signed transform:

```sh
GGML_VK_PERF_LOGGER=1 build/bin/test-backend-ops -b Vulkan0 \
  -o MUL_MAT_HADAMARD \
  -p 'blk=128,width=5120,n_tokens=7,type_x=f32,swiglu=0,options=0'
```

`GGML_VK_DISABLE_FUSION=1` is useful for a correctness cross-check, but disables
other Vulkan fusions too. Use the baseline branch to isolate these changes in
end-to-end timing comparisons.

## Bundle import

The incremental bundle requires upstream history through
`71ad0590f4808b6202f9213d166913858c73b1bc`. Fetch upstream first if necessary,
then import into new local branch names:

```sh
git fetch /path/to/bonsai-vulkan-refresh-20261008.bundle \
  bonsai:bonsai-vulkan-refreshed \
  bonsai-vulkan-baseline-20261008:bonsai-vulkan-baseline
git switch bonsai-vulkan-refreshed
```
