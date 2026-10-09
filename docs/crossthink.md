# Continuous crossthink

`llama-crossthink` runs two reasoning streams and feeds each model's newly generated token IDs into the other's open reasoning context. By default, both models seek paragraph boundaries before exchanging reasoning; the faster model waits for its peer. It also serves a browser console for watching both streams, sending messages, requesting answers, pausing, and resetting. This is a native C++ tool using the repository's HTTP/JSON dependencies.

The server's new `--socket PATH` adds a Unix HTTP listener while keeping `--host` and `--port` available for the ordinary web UI and API. `LLAMA_ARG_SOCKET` is the corresponding environment variable. Paths need not end in `.sock`; the old `--host *.sock` convention still works. The Unix listener uses plain HTTP even when TCP uses TLS. On clean shutdown, the server removes only the filesystem socket it bound, if its inode still matches; an existing path is never removed at startup.

## Build and launch

From the `bonsai-crossthink` branch, retain your working Vulkan build configuration:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build --target llama-server llama-crossthink -j
```

Use the same GGUF for both peers initially. Keep your current model, GPU, context, and MTP settings, adding these listener options to the respective server commands:

```bash
# Server A, on the first GPU:
--host 127.0.0.1 --port 8080 --socket /tmp/crossthink.XXXXXXXX/a.sock

# Server B, on the second GPU:
--host 127.0.0.1 --port 8081 --socket /tmp/crossthink.XXXXXXXX/b.sock
```

Create the directory with `mktemp -d /tmp/crossthink.XXXXXXXX` and substitute its printed path in all commands. Both servers need `--jinja` and a thinking-enabled Qwen-style chat template ending in an open `<think>` block, with single-token `<think>` and `</think>` delimiters. The prototype rejects incompatible token mappings or unsupported template layouts. Use single-model server mode; router mode does not expose the binary endpoints.

Once both servers are ready:

```bash
build/bin/llama-crossthink \
    --socket-a /tmp/crossthink.XXXXXXXX/a.sock \
    --socket-b /tmp/crossthink.XXXXXXXX/b.sock \
    --port 8090 --max-segment-tokens 512 --sentence-after 256 \
    --answer-tokens 1024
```

Open **http://127.0.0.1:8090/** and send a message to start. Alternatively supply `--prompt 'Your question'` or `--file prompt.txt` (`--file -` reads stdin). `--seed` defaults to 42 and `--temperature` to 1.0. If the model servers require the same bearer key, pass `--api-key KEY` to the coordinator. The console defaults to loopback and has no authentication of its own.

The ordinary model web UIs remain at **http://127.0.0.1:8080/** and **http://127.0.0.1:8081/**. Those chats are separate from the coordinated conversation; use port 8090 to address the pair. Other requests can displace a server's prompt cache or delay crossthink. Each segment resubmits the authoritative token sequence, so cache displacement does not silently change the conversation.

## Interaction and scheduling

| Control | Behavior |
| --- | --- |
| Send message | Finish the current segments, import pending peer reasoning, close the current reasoning blocks, and append a real user turn to both histories. Both resume thinking. |
| Pause / Resume | Pause when the current segments finish; preserve both token histories and pending imports. Resume continues from those histories. |
| Answer now | Finish the current segments, drain imports, close both reasoning blocks, and generate two answers. Answers are not fed back as peer reasoning. A subsequent message starts the next thinking turn. |
| New session | Cancel current requests, discard both histories and queued imports, and return to an empty session. Late packets from the old session are ignored. |

The HTTP control API mirrors these controls: `POST /message` with `{"text":"..."}`, and `POST /pause`, `/resume`, `/answer`, or `/reset` with `{}`. Requests use `Content-Type: application/json`. `GET /state` returns status and counters; `GET /events?after=ID` streams SSE trace events with resumable IDs. The console retains a bounded trace; reconnecting after old events expire produces an explicit gap notice.

Each server streams committed IDs immediately, including verified MTP output, so the console remains live while a paragraph is being written. In the default `--splice-mode paragraph`, the recipient buffers that segment until both models finish their current segments. Both then import the other's complete segment and continue. The faster model waits at this rendezvous rather than advancing ahead of its peer. Imported IDs are never retransmitted, and reasoning is never retokenized in transit. Display text is separate metadata.

The boundary detector examines the raw bytes of committed tokens. It recognizes blank lines, including `\n\n`, CRLF, and blank lines containing spaces or tabs, across token boundaries. Blank lines and sentence endings inside fenced code do not stop a segment. Each request scans its prompt to recover the fence state, including after a forced cut inside code. A valid boundary must follow some non-whitespace output generated in that request.

Splicing happens only at the end of a whole token. If a token contains a paragraph separator followed by the beginning of another paragraph, generation continues until a later suitable token boundary. The separator is preserved in the exchanged token IDs. This avoids the text trimming performed by ordinary stop strings, which cannot remove only part of a raw token.

| Option | Default | Behavior |
| --- | --- | --- |
| `--splice-mode paragraph` | `paragraph` | Wait for both peers and exchange completed segments together. |
| `--sentence-after N` | `256` | After this many generated tokens, permit a sentence ending as a fallback when no paragraph boundary has appeared. |
| `--max-segment-tokens N` | `512` | Hard generation ceiling per segment. If no clean boundary appears, exchange at this limit and report a forced cut. |
| `--chunk-tokens N` | `512` | Compatibility alias for `--max-segment-tokens`. |
| `--splice-mode fixed` | | Restore independent loops, importing available peer tokens before each fixed-size segment. |

The token ceiling and sentence threshold each accept `1..4096`. A sentence threshold above the ceiling permits only paragraph boundaries before a forced cut; at the ceiling, a sentence ending can still qualify. Sentence detection is a punctuation heuristic, not a language parser; it cannot reliably distinguish every abbreviation or other use of punctuation. The hard ceiling also applies inside fenced code, so it bounds waiting time at the cost of occasional incomplete sentences or code blocks. The console shows completed exchanges, peers waiting at the rendezvous, each peer's last boundary, and forced-cut counts. A notice identifies forced cuts.

To reproduce the original 32-token experiment, use `--splice-mode fixed --chunk-tokens 32`. Fixed mode retains the original asynchronous behavior: one model may import the other's unfinished segment, with bounded queues applying backpressure to the faster peer. The sentence threshold has no effect in fixed mode. Existing launch scripts with only `--chunk-tokens 32` now select paragraph rendezvous with a 32-token ceiling; remove that option or increase it to benefit from longer boundaries.

This is a continuous exchange within one evolving conversation, implemented as completion requests. It does not mutate a slot's KV cache mid-decode. Longer segments reduce the frequency of prompt processing and imports, but also delay user controls. Each request constructs a sampler afresh, with a deterministic per-peer sequence of seeds; results are not equivalent to one uninterrupted sampling call. Keep server-wide reverse prompts and other output constraints disabled for this experiment.

During continuous thinking the coordinator suppresses reasoning/chat terminators and end-of-generation tokens. Thinking runs until a user control, an error, or the context reserve stops it. Near context capacity it pauses, preserving room for pending imports and final answers. It never silently deletes older context; request answers or reset. Larger answer budgets reserve more context. The browser's token counters expose generated, imported, and queued IDs separately.

Unlike the original [whole-block swap](reasoning-swap.md), continuous crossthink retains local reasoning and interleaves incoming peer reasoning at segment boundaries. No peer label or extra user message is inserted for these imports. Only an actual user message creates a user turn. Paragraph boundaries reduce syntactic disruption but do not reconcile contradictory assumptions or variable names. This experiment makes no claim of improved accuracy or inference speed.

## Binary stream

`POST /completion/tokens` keeps the existing `LLMTOK01` request envelope and packed little-endian token IDs. With metadata `"stream": true`, its response body concatenates envelopes:

* `{"type":"tokens", "content":"optional display fragment"}` plus committed token IDs.
* `{"type":"done", ...}` plus an empty token payload, with stop reason, counts, cache/truncation flags, and timings.
* `{"type":"error", "error": ...}` plus an empty token payload if an error occurs after response headers.

HTTP chunk boundaries are independent of envelope boundaries. The incremental decoder handles split headers, metadata, token words, and multiple envelopes per chunk. It checks lengths, vocabulary bounds, and truncation. Raw IDs are emitted exactly once even when their text forms an incomplete UTF-8 fragment or a stop marker. Rejected speculative drafts are never emitted. Errors before streaming starts keep the ordinary HTTP error status. Nonstreaming responses and `llama-reasoning-swap` remain supported. `/tokens/info` advertises `"stream": true`.

Binary completion requests may also set `"splice": {"sentence_after": 256}`. `n_predict` remains the hard segment ceiling. The server stops after a clean token boundary, and the terminal result adds `"splice_boundary": "paragraph"`, `"sentence"`, or `"limit"`. `stop_type` remains `"limit"`; use `splice_boundary` to distinguish a clean early stop from a forced cut. `/tokens/info` advertises `"splice": true`. Paragraph mode requires both servers to support this capability; rebuild both servers as well as the coordinator when upgrading.

## Validation

Release CPU builds passed for the server, both native drivers, and the affected test binaries. The in-process binary endpoint tests passed with a seeded tiny Qwen3.5 fixture, both without speculation and with MTP enabled. They cover paragraph and sentence stops, preserved separators, CRLF and whitespace-only blank lines, fence continuation after a forced cut, cached continuation versus fresh evaluation, and malformed splice options, in addition to the earlier token/framing tests. Scanner tests cover arbitrary piece fragmentation and a single piece containing a separator followed by another paragraph's text.

The random fixture rejected every speculative draft, so these runs exercise MTP verification/rejection and cached continuation but do not validate a splice inside an accepted multi-token batch. Actual Bonsai/MTP GPU operation remains untested here.

Coordinator tests use deterministic gated transports to check both modes: paired completion before import, no re-echo, waiting and forced-cut counters, pause/resume with one waiting peer, user messages and answers, stale-packet reset, and rejection of invalid completion records. Tests also verify that a pending user intervention cannot import partial data after a peer fails. Extracted browser JavaScript passed a syntax check.
```bash
cmake -S . -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target test-crossthink test-server-token-wire -j
build/bin/test-crossthink
python3 tests/gen-tiny-qwen35-mtp.py /tmp/tiny-qwen35-mtp.gguf
build/bin/test-server-token-wire /tmp/tiny-qwen35-mtp.gguf
```

The argument parser's local checks, including the new socket flag cases, passed before its unrelated URL-download test failed on network access. This build environment denied socket creation (`EPERM`), so actual listener/HTTP integration and browser interaction could not be run here. No Bonsai weights or Radeon devices were available; dual-GPU behavior and performance remain to be tested on your machine.
