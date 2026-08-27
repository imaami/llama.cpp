# Ternary GGUF format compatibility

The tensor type and block layout determine compatibility. Older Prism model releases used `Q2_0` for a different layout, so the filename alone is not sufficient.

| Format | GGML type ID | Values per block | Compatibility |
| --- | --- | --- | --- |
| `Q2_0` | 42 | 64 | The current `Q2_0` layout. Older model filenames may identify it as `Q2_0_g64` or `Q2_g64`. |
| `PQ2_0` | 142 | 128 | The group-128 layout under its distinct type ID. Requires `PQ2_0` support in the selected backend. |
| Legacy Prism `Q2_0` | 42 | 128 | Incompatible with the current type-42 layout. Use a replacement `PQ2_0` or group-64 `Q2_0` GGUF. |

A legacy group-128 file cannot be made compatible by renaming it. The loader checks tensor offsets and reports a legacy-layout hint when an offset mismatch matches the old layout. Obtain the correctly encoded model instead of treating the legacy file as current `Q2_0`.

Some older `*-Q2_0.gguf` files use the legacy layout; newer files with that suffix can use the current group-64 layout. Check the model release's format description. `*-PQ2_0.gguf` identifies the separate group-128 format.

Use executables and backend libraries from the same build. Mixing libraries from incompatible forks or revisions can introduce ABI and format mismatches.
