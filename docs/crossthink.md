# Crossthink with a telepathic link

`llama-crossthink` runs two agents that can choose to share a labeled reasoning channel. Both receive a built-in `think_with_telepathic_link` tool; either can enable or disable the link. It starts off, with private reasoning. While linked, both see the same ordered stream of contributions, including explicit speaker and interruption labels. Tool calls, tool results, and final answers stay separate. A browser console shows the shared channel and the agents' private activity, and provides messages, link controls, pause/resume, answers, and reset.

The earlier covert paragraph/fixed exchange remains available with `--legacy-splice` for comparisons. This is a native C++ tool using the repository's HTTP/JSON dependencies.

The server's new `--socket PATH` adds a Unix HTTP listener while keeping `--host` and `--port` available for the ordinary web UI and API. `LLAMA_ARG_SOCKET` is the corresponding environment variable. Paths need not end in `.sock`; the old `--host *.sock` convention still works. The Unix listener uses plain HTTP even when TCP uses TLS. On clean shutdown, the server removes only the filesystem socket it bound, if its inode still matches; an existing path is never removed at startup.

## Build and launch

From the `reasoning-swap` branch, retain your working Vulkan build configuration:

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
    --port 8090 --link-quantum 64 --private-quantum 256 --link-wait-tokens 512 \
    --answer-tokens 1024
```

Open **http://127.0.0.1:8090/** and send a message to start. Alternatively supply `--prompt 'Your question'` or `--file prompt.txt` (`--file -` reads stdin). `--seed` defaults to 42 and `--temperature` to 1.0. If the model servers require the same bearer key, pass `--api-key KEY` to the coordinator. The console defaults to loopback and has no authentication of its own.

The ordinary model web UIs remain at **http://127.0.0.1:8080/** and **http://127.0.0.1:8081/**. Those chats are separate from the coordinated conversation; use port 8090 to address the pair. Other requests can displace a server's prompt cache or delay crossthink. Each segment resubmits the authoritative token sequence, so cache displacement does not silently change the conversation.

## Thinking with the link

Both agents receive their own identity and the built-in tool schema through the native chat template. The tool is available without `--mcp-config`. Its arguments are:

```json
{
  "enabled": true,
  "yield_until": "paragraph"
}
```

`enabled` is required and controls the link for both agents. Set it to `false` to return to private reasoning. `yield_until` is optional: `fragment`, `sentence`, or `paragraph` lets the other agent continue to that boundary before the caller takes another turn. Calling with `enabled: true` while already linked can therefore yield the floor without changing the link state. Without a voluntary yield, the coordinator alternates short speaking quanta. A yield has a token ceiling, so a model that never completes a paragraph cannot monopolize the channel indefinitely.

Every linked contribution carries a coordinator-supplied speaker label in both models' contexts and in the shared console. A speaker change before a sentence or paragraph boundary is marked as an interruption. The other agent can then continue the same thought, challenge it, or introduce another one. Labels identify the actual source regardless of what names occur in generated prose. During linked reasoning, an eager sampling grammar reserves line-leading `[A` and `[B` prefixes for the coordinator, so a model cannot autocomplete another speaker label. The constraint follows partial prefixes across requests from the same speaker. Inline brackets remain available for ordinary reasoning. Native tool turns, private reasoning, and final answers do not use this constraint.

Sharing uses short, serialized completion requests. Only one agent contributes at a time, and its committed token IDs enter both histories before the other generates. This gives both agents the same causal order: concurrent requests would each sample from a stale view of what the other is saying. Their complete histories still differ because identities, private reasoning, and tools are private. The shared subsequence is identical; this is not a shared KV cache.

The default shared quantum is 64 tokens; a sentence or paragraph boundary with model-generated text can end the speaking turn earlier. Tokens stream to the browser as the server commits them, and the next speaker gets them when that quantum ends. A quantum can extend to finish an incomplete UTF-8 character. It does not wait for a paragraph. Fragments containing only whitespace, ASCII punctuation, or common Unicode dashes/bullets keep the same speaker instead of inserting another label and handing over an empty turn. After 64 generated tokens without substantive speech, the session pauses with an explicit error. No generated text or token IDs are silently removed. A voluntary yield still has its configured token ceiling. Reducing the quantum makes interruptions more immediate but adds more completion requests, label tokens, and prompt/cache work; increasing it gives longer uninterrupted turns. Speech remains interleaved, not simultaneous, and there is no claim of improved throughput or accuracy.

| Option | Default | Behavior |
| --- | --- | --- |
| `--link-quantum N` | `64` | Shared reasoning request quantum, `1..4096` generated tokens. A character may require a small extension. |
| `--private-quantum N` | `256` | Private reasoning request quantum, `1..4096` generated tokens; independent of shared handoffs. |
| `--link-wait-tokens N` | `512` | Voluntary yield ceiling, `1..65536` generated tokens, with up to eight extra tokens allowed to finish a UTF-8 character. |
| `--max-segment-tokens N` | `512` | Legacy exchange ceiling, `1..4096`; does not change explicit telepathy quanta. |
| `--answer-tokens N` | `1024` | Final answer budget, `1..65536`. |
| `--legacy-splice` | Off | Restore the earlier covert exchange and omit the built-in link tool. |

With the link off, both agents use independent private requests of up to 256 tokens by default. Private requests do not stop at sentence boundaries; a paragraph or native reasoning closure can still end them earlier. This reduces repeated prompt evaluation and sampler setup. Output still streams token by token. Larger private quanta can delay manual controls or a pending link change until the current safe boundary.

A model can close its reasoning naturally to call a tool or answer. When one gives its final answer, the shared link ends. Its partner continues reasoning privately until its own answer; it is not forced to answer prematurely. The console keeps the answers in separate panels. **Answer now** explicitly asks both to finish instead.

The link begins off after reset. To encourage deliberate collaboration, ask, for example: "Use the telepathic link to discuss competing approaches. Yield until the other finishes a paragraph when useful, and answer separately when ready." Models can still decide not to enable the link; the console's **Enable link** and **Disable link** controls let you change it directly.

## MCP tools over HTTP

The coordinator can give both models tools from existing MCP servers. Configure it with the same HTTP endpoint URLs and static authentication headers that you would use in the ordinary web UI. The coordinator connects directly to those servers; browser CORS settings do not apply. Each model gets its own MCP sessions and the same discovered tool catalog.

Create `mcp.json`, substituting the URLs of your running calculator and compiler MCP servers:

```json
{
  "mcpServers": {
    "calculator": {
      "url": "http://127.0.0.1:8000/mcp"
    },
    "compiler": {
      "url": "http://127.0.0.1:8001/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_MCP_SERVER_TOKEN"
      }
    }
  }
}
```

Remove `headers` if the endpoint needs no authentication. These example URLs do not start a service: the calculator/compiler tools must already be provided by external MCP servers. The telepathic link is built in and needs no MCP server. Calculator, compiler, and shell tools must come from the configured servers.

```bash
build/bin/llama-crossthink \
    -a /tmp/crossthink.XXXXXXXX/a.sock \
    -b /tmp/crossthink.XXXXXXXX/b.sock \
    --mcp-config mcp.json \
    --mcp-timeout 60 --tool-turn-tokens 2048 --max-tool-rounds 8
```

Both llama-server instances and the coordinator must be rebuilt for this update. Use `--jinja` on the model servers, as above. Tool calling uses the model's chat template and native output parser, including Qwen3.5's XML tool-call format; it does not rely on the model printing a separate crossthink command syntax. In legacy mode, MCP requires `--splice-mode paragraph`.

| Option | Default | Behavior |
| --- | --- | --- |
| `--mcp-config FILE` | None | Read an `mcpServers` JSON object with HTTP/HTTPS `url` and optional string-valued `headers`. File size is limited to 1 MiB. |
| `--mcp-timeout N` | `60` | HTTP timeout in seconds, `1..600`, for MCP requests. |
| `--tool-turn-tokens N` | `2048` | Private native tool/answer turn budget, `1..16384`; the effective ceiling is the larger of this value and `--answer-tokens`. |
| `--max-tool-rounds N` | `8` | Consecutive native tool round limit, `1..64`; a nonempty reasoning fragment resets the counter. In legacy mode, the paragraph rendezvous resets it. |

This client supports MCP Streamable HTTP, including JSON and SSE responses to POST requests, session initialization, tool discovery, and tool calls. HTTPS requires a build with TLS support, normally `-DLLAMA_OPENSSL=ON`. The older HTTP+SSE transport, stdio commands, OAuth login, MCP resources/prompts, and server-initiated sampling are not supported. `--api-key` remains the bearer key for the model servers; MCP authentication is configured separately in each endpoint's `headers`.

Configured tools are called automatically when either model requests them. Calls run with the permissions of the selected external MCP service. The console shows the number of available tools and each peer's call count/status; expand **Tool activity** to inspect arguments and results. A prompt such as "Use the calculator to check the arithmetic" or "Compile a small C program to verify this" can encourage a call, but the model decides whether to use a tool.

Calls and their results stay in the calling model's conversation, with native assistant/tool roles. Shared speaking pauses during a private tool turn. When it completes, the agents can continue sharing reasoning about what they learned. Calls are never copied to the other agent or executed twice by the coordinator. Reset cancels current work and clears the displayed tool trace along with the conversation; it cannot undo effects of an external tool call that already ran.

## Interaction and scheduling

| Control | Behavior |
| --- | --- |
| Send message | Finish current work, close the current reasoning blocks, and append a real user turn to both histories. Both resume thinking. |
| Enable / Disable link | Change the shared link after current work finishes; preserve whether the session was running or paused. Available in explicit telepathy mode. |
| Pause / Resume | Pause after current work; retain histories and resume from them. |
| Answer now | End sharing, close both reasoning blocks, and generate separate answers. A subsequent message starts the next thinking turn. |
| New session | Cancel current requests, discard both histories, disable the link, and clear the displayed conversation. Late packets from the old session are ignored. |

The HTTP control API mirrors these controls: `POST /message` with `{"text":"..."}`, and `POST /link_on`, `/link_off`, `/pause`, `/resume`, `/answer`, or `/reset` with `{}`. Requests use `Content-Type: application/json`. `GET /state` returns status and counters, including `telepathy`, `link_enabled`, `link_speaker`, and `link_wait`; `GET /events?after=ID` streams SSE trace events with resumable IDs. A `shared` event contains `peer` and `text`, with `label: true` for coordinator labels. Append that text as received to display the model-visible shared channel. Private `token` events and separate `answer` events must not be added to it. The console retains a bounded trace; reconnecting after old events expire produces an explicit gap notice.

Requests resubmit each agent's authoritative token sequence. Imported generated IDs are never retokenized or retransmitted; coordinator labels are tokenized separately and inserted into both histories. Each completion creates a fresh sampler with deterministic per-agent seeds, so results differ from one uninterrupted sampling call. Keep server-wide reverse prompts and other output constraints disabled. Near context capacity the session pauses and preserves room for final answers; it never silently deletes older context. Larger answer and tool budgets reserve more context.

## Legacy covert exchange

Use `--legacy-splice` to retain the earlier experiment: both models generate independently and receive each other's reasoning without source labels or knowledge of the exchange. Paragraph mode is the default within this legacy mode. Both reach a segment boundary, then import each other's complete segments together. The faster model waits at this rendezvous. The console shows each model's generated text once; imports are not repeated in the display.

| Option | Default | Behavior in legacy mode |
| --- | --- | --- |
| `--splice-mode paragraph` | `paragraph` | Wait for both peers and exchange complete segments together. |
| `--sentence-after N` | `256` | Permit sentence endings after this many generated tokens, `1..4096`. |
| `--max-segment-tokens N` | `512` | Hard ceiling, `1..4096`; report a forced cut if no clean boundary appears. |
| `--chunk-tokens N` | `512` | Alias for `--max-segment-tokens`. |
| `--splice-mode fixed` | | Independent loops importing available peer tokens before each fixed-size segment. |

Paragraph detection recognizes blank lines, including `\n\n`, CRLF, and whitespace-only blank lines across token boundaries. Blank lines or punctuation inside fenced code do not end a segment. A separator must end at a whole token boundary; if a token also contains text from the next paragraph, generation continues. Sentence detection is a punctuation heuristic. The hard ceiling applies inside code too. A sentence threshold above the ceiling allows only paragraph boundaries before a forced cut.

To reproduce the original 32-token experiment, use `--legacy-splice --splice-mode fixed --chunk-tokens 32`. Fixed mode permits imports from an unfinished peer segment, with bounded queues applying backpressure; its sentence threshold has no effect. MCP tools require paragraph mode. In legacy MCP mode, the calling model's unfinished pre-tool paragraph stays private and the other model waits at the rendezvous. A natural final answer triggers the other model's final answer. Without MCP, legacy continuous thinking suppresses reasoning/chat terminators and EOG until a user control requests an answer. Unlike the original [whole-block swap](reasoning-swap.md), this mode retains each model's own reasoning and interleaves peer imports.

## Binary stream

`POST /completion/tokens` keeps the existing `LLMTOK01` request envelope and packed little-endian token IDs. With metadata `"stream": true`, its response body concatenates envelopes:

* `{"type":"tokens", "content":"optional display fragment"}` plus committed token IDs.
* `{"type":"done", ...}` plus an empty token payload, with stop reason, counts, cache/truncation flags, and timings.
* `{"type":"error", "error": ...}` plus an empty token payload if an error occurs after response headers.

HTTP chunk boundaries are independent of envelope boundaries. The incremental decoder handles split headers, metadata, token words, and multiple envelopes per chunk. It checks lengths, vocabulary bounds, and truncation. Raw IDs are emitted exactly once even when their text forms an incomplete UTF-8 fragment or a stop marker. Rejected speculative drafts are never emitted. Errors before streaming starts keep the ordinary HTTP error status. Nonstreaming responses and `llama-reasoning-swap` remain supported. `/tokens/info` advertises `"stream": true`.

Binary completion requests may also set `"splice": {"sentence_after": 256}`. `n_predict` remains the hard segment ceiling. The server stops after a clean token boundary, and the terminal result adds `"splice_boundary": "paragraph"`, `"sentence"`, or `"limit"`. `stop_type` remains `"limit"`; use `splice_boundary` to distinguish a clean early stop from a forced cut. `/tokens/info` advertises `"splice": true`. Paragraph mode requires both servers to support this capability; rebuild both servers as well as the coordinator when upgrading.

Explicit telepathy additionally requires the `"splice_quantum": true` capability. Set `splice.token_after` to a token count in `1..4096` to stop as soon as that count is reached and the output ends on a complete UTF-8 character. A clean paragraph or permitted sentence boundary can end the request earlier; otherwise the result reports `"splice_boundary": "quantum"`. `n_predict` remains the hard ceiling, and hitting it with an incomplete UTF-8 character produces an error. The coordinator reserves eight extra tokens for character completion and uses `sentence_after: 1` for telepathic quanta. Both servers and the coordinator must be rebuilt together.

When native tools are enabled, `splice.stop_on_think_close` also stops at the single `</think>` token and reports `"splice_boundary": "tool"`. That token is retained in the calling model's raw tape. The coordinator then generates a private assistant tail through EOG. `/apply-template` accepts optional `parse_output` token IDs and returns a parsed `message` alongside its ordinary `prompt`, using the same native chat parser as chat completions. A terminal EOG is removed before parsing; tokens following EOG are rejected. `/tokens/info` advertises `"tool_parse": true` for this extension.

## Validation

The `reasoning-swap` branch was rebased onto `imaami/llama.cpp`'s `bonsai` commit `5ada589b7811232c44d55dbf210329d98414fb4d`. All four preceding prototype commits replayed without conflicts or patch changes. Release CPU builds passed for `llama-server`, `llama-crossthink`, and the four test binaries below.

Coordinator tests cover default-off private reasoning, built-in tool discovery without MCP, identical ordering of the shared transcript, speaker/interruption labels, enable/disable barriers, bounded fragment/sentence/paragraph yields, private MCP calls and results, and independent natural final answers. They also cover pause/reset cancellation, repeated deliberate yields, tool-loop limits, malformed completion records, and user controls racing with the last final answer. Whitespace, punctuation-only, and temporarily empty UTF-8 display fragments are tested separately from speech: empty turns do not alternate labels, and sustained output without speech pauses explicitly. The label constraint is tested across request boundaries, with ordinary inline brackets, Unicode text, and native reasoning closure. Earlier fixed/paragraph exchange tests remain enabled.

The in-process binary endpoint tests passed with seeded tiny Qwen3.5 fixtures, both without speculation and with MTP enabled. They cover token framing, cached continuation, paragraph/sentence stops, UTF-8 completion across two-, three-, and four-byte characters, incomplete-character errors at the hard ceiling, boundary priority, cross-request separators, code fences and prefix resets, native reasoning closure, and tool output parsing. The random fixtures rejected every speculative draft, so these runs do not validate a splice inside an accepted multi-token batch.

Native template tests passed for Qwen3.5 XML and Qwen JSON tool parsing, including boolean link arguments, identity preservation, and append-only result suffixes that reopen thinking. Fake HTTP MCP tests passed for initialization, sessions, JSON/SSE responses, pagination, protocol limits, cancellation, and ambiguous calls without retry. CLI bounds checks, extracted browser JavaScript syntax, and mocked DOM checks for the shared channel, controls, tool logs, separate answers, and reset also passed.

Run the binaries from the repository root for the template files:

```bash
cmake -S . -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target test-crossthink test-crossthink-mcp test-crossthink-tool-template test-server-token-wire -j
build/bin/test-crossthink
build/bin/test-crossthink-mcp
build/bin/test-crossthink-tool-template
python3 tests/gen-tiny-qwen35-mtp.py /tmp/tiny-qwen35-mtp.gguf
python3 tests/gen-tiny-qwen35-mtp.py --tool-tokens /tmp/tiny-qwen35-tools.gguf
build/bin/test-server-token-wire /tmp/tiny-qwen35-mtp.gguf /tmp/tiny-qwen35-tools.gguf
```

Unix-domain socket creation was denied (`EPERM`) in this build environment, so live coordinator-to-model HTTP integration could not be exercised. Live MCP services, TLS, and an actual browser session were not tested. No Bonsai weights or Radeon devices were available; dual-GPU behavior, accepted MTP batches, and conversational latency still need testing on your machine.
