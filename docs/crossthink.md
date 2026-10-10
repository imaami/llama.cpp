# Crossthink with parallel thought commands

`llama-crossthink` runs two independent agents in parallel. Each agent can read its partner's reasoning, send it a message, and collect incoming messages by emitting short commands inside its own reasoning. Nothing is injected unsolicited, and neither agent has to enable a shared channel. The browser console shows separate reasoning and answers, plus a messaging view for sent and collected thoughts.

The earlier covert paragraph/fixed exchange remains available with `--legacy-splice` for comparisons. The former serialized `think_with_telepathic_link` mode has been replaced by thought commands. This is a native C++ coordinator using the repository's HTTP/JSON dependencies.

The server's new `--socket PATH` adds a Unix HTTP listener while keeping `--host` and `--port` available for the ordinary web UI and API. `LLAMA_ARG_SOCKET` is the corresponding environment variable. Paths need not end in `.sock`; the old `--host *.sock` convention still works. The Unix listener uses plain HTTP even when TCP uses TLS. On clean shutdown, the server removes only the filesystem socket it bound, if its inode still matches; an existing path is never removed at startup.

## Build and launch

Rebuild **both `llama-server` and `llama-crossthink`** from the `reasoning-swap` branch; the server needs the `thought_commands` and `thought_command_lines` capabilities. Retain your working Vulkan build configuration:

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
    --port 8090 --answer-tokens 1024
```

Open **http://127.0.0.1:8090/** and send a message to start. Alternatively supply `--prompt 'Your question'` or `--file prompt.txt` (`--file -` reads stdin). `--seed` defaults to 42 and `--temperature` to 1.0. If the model servers require the same bearer key, pass `--api-key KEY` to the coordinator. The console defaults to loopback and has no authentication of its own.

The ordinary model web UIs remain at **http://127.0.0.1:8080/** and **http://127.0.0.1:8081/**. Those chats are separate from the coordinated conversation; use port 8090 to address the pair. Other requests can displace a server's prompt cache or delay crossthink. Each continuation resubmits the authoritative token sequence, so cache displacement does not silently change the conversation.

## Commands inside reasoning

Both agents receive a default system prompt explaining their identity (A or B), their partner, and these three commands. They are available without an MCP configuration, native tool-call syntax, or an activation step. The model emits the literal command while its `<think>` block is open:

| Command | Effect |
| --- | --- |
| `<ct:peek/>` | Read the partner's latest reasoning up to the instant the command is handled. |
| `<ct:send>message</ct:send>` | Queue the enclosed text in the partner's inbox. |
| `<ct:inbox/>` | Collect messages waiting in the caller's inbox. |

These commands only execute during reasoning, with the opening tag at **column zero of a new line**, outside a fenced code block. The start of a reasoning turn also counts as a line start. Inline mentions, indented tags, bullet/quote-prefixed tags, and examples in backtick or tilde fences are inert. They are also ordinary text in a final answer, native tool call, or tool result. The exact lowercase syntax matters. Emit one command at a time and wait for its marked result before continuing. A thought command does not close `</think>` or start a native assistant/tool turn. For example, an agent can reason as follows:

```text
I should check whether B already found the missing assumption.
<ct:peek/>
```

The coordinator appends the requested result to that agent's open reasoning, inside explicit markers, then the agent continues thinking. Only complete generated commands trigger an action; input prompts and imported thought text do not execute commands.

### Reading thoughts

The first `<ct:peek/>` in a partner's current or most recent reasoning receives that reasoning from its beginning. Further peeks continue from the previous cut, returning only newly available text. When the partner begins another reasoning turn, the next peek starts at the beginning of that new turn. A new user turn or a continuation after a native tool result starts a new reasoning capture. If no text has appeared since the last peek, the result is an empty capture with a zero byte count.

A snapshot contains committed reasoning exactly as it stood when captured, up to the latest complete UTF-8 character. It can end mid-sentence or mid-word: the reader does not wait for a paragraph, sentence, or the partner's final answer. The capture contains only the partner's own generated reasoning, including any literal thought commands it wrote. Imported captures, inbox results, native tool calls/results, and final answers are excluded, so reading thoughts does not recursively copy earlier imports. Opening and closing markers distinguish the captured content from the reader's own thoughts. Peeking is passive; the partner keeps generating and does not need to approve it. Captures appear only in the receiver's reasoning panel, not in the messaging view.

A result uses this structure; `turn`, `offset`, and `bytes` identify the source turn and byte range, and `active` reports whether that reasoning is still open:

```text
<ct:result command="peek">
<ct:thought source="B" turn="1" offset="0" bytes="..." active="true">verbatim captured text</ct:thought>
</ct:result>
```

The payload stays contiguous and unescaped. The byte count distinguishes payload text from the coordinator's closing marker even if the captured reasoning itself contains marker-like text. A capture that cannot fit is reported as an error; its read cursor does not advance.

### Sending and collecting thoughts

`<ct:send>...</ct:send>` queues a deliberate message for the other agent. It does not interrupt the recipient or insert anything into its live reasoning. Keep each message below 16 KiB. The sender receives an acknowledgment and continues. The recipient uses `<ct:inbox/>` to receive the queued messages together, with source markers, inside its own reasoning. Each delivered message uses a `<ct:message id="..." from="B" bytes="...">...</ct:message>` wrapper inside the inbox result. Reading an empty inbox also returns immediately. A mailbox holds at most 256 messages and 1 MiB of message text. A full mailbox or a result that cannot fit in the caller's remaining context produces an explicit error. Only successfully delivered inbox messages are removed; arrivals after the collection snapshot remain queued. The messaging view shows sent messages and their collection status so you can distinguish queued communication from text the recipient has actually received.

The system prompt explains that reasoning commands are part of this experiment, despite the usual separation between reasoning and native tool calls. It asks each agent to send a brief planning message at the start of a new user task, then keep working without waiting for a reply. It encourages concise updates and inbox/peek checks at useful milestones, and explicitly says to continue after an empty inbox result. These are instructions to the models; the coordinator does not manufacture commands or force communication. The agents still decide when to peek, send, collect, call MCP tools, and answer.

### Parallel execution

Both agents normally have a long-running, streaming completion in flight at the same time. There is no speaker token, rendezvous, shared-floor lock, periodic quantum cut, or automatic import in this mode. A thought command briefly ends **only the caller's** completion at the command boundary. The coordinator appends the result to its token history and starts a cached continuation of the same open reasoning block. The partner continues generating throughout.

This is append-and-resume through the existing completion API, not in-place insertion into an actively decoding server slot. Command continuations still incur HTTP, prompt-suffix evaluation, and sampler setup work. Their cost depends on the amount of imported text and prompt-cache reuse. Independent requests allow both GPUs to work concurrently; this does not guarantee a throughput or reasoning-quality improvement.

A model can close its reasoning naturally to call an MCP tool or answer. Its partner continues independently, including while that native tool call runs. Each model's final answer appears in its own panel. **Answer now** asks both to finish.

| Option | Default | Behavior |
| --- | --- | --- |
| `--answer-tokens N` | `1024` | Final answer budget, `1..65536`. |
| `--link-quantum N` | Compatibility only | Accepted but ignored; thought commands have no shared generation quantum. |
| `--private-quantum N` | Compatibility only | Accepted but ignored; independent reasoning no longer restarts periodically. |
| `--link-wait-tokens N` | Compatibility only | Accepted but ignored; no shared floor or voluntary yield exists. |
| `--max-segment-tokens N` | `512` | Legacy exchange ceiling, `1..4096`; does not limit thought-command reasoning. |
| `--legacy-splice` | Off | Restore the earlier covert exchange; thought commands are unavailable. |

Old launch commands can retain the compatibility flags, but removing them makes the active configuration clearer. Start a new session after upgrading; earlier label-filled or shared-channel contexts describe a different protocol.

## Browser console

Open the coordinator's port, normally **http://127.0.0.1:8090/**. The composer targets **Both**, **A**, or **B**. The individual panes show each agent's reasoning, imported thought captures, answer, and native tool activity. The third pane uses alternating message bubbles for deliberate messages between the agents; it is no longer a line-by-line shared reasoning transcript.

**Follow** is an explicit checkbox. Scrolling up does not turn it off; disable it yourself to read older output without automatic following. Each output pane retains its own setting.

Model output uses the upstream web UI's Markdown pipeline: GitHub-flavored Markdown, fenced code with syntax highlighting and raw copy, KaTeX math, Mermaid diagrams, and sandboxed HTML previews. This rendering applies to reasoning, answers, thought messages, and native tool activity. Rendered HTML and SVG are sanitized. Rendering assets are bundled with the console for offline use; there is no required CDN connection. These are presentation features: the model receives the original text, not the rendered HTML.

Reasoning turns and imported thought results render as separate Markdown documents. An interrupted turn ending in an unclosed code fence therefore cannot turn the next user turn or thought result into code. Reasoning resumed after a native tool result also starts a new document, and native tool records are separated in the same way. Copy raw preserves the original text without adding artificial closing fences or UI dividers.

Normal C++ builds use the checked-in generated UI header and do not require Node.js. To modify the UI sources, run `npm ci` followed by `npm run build` in `tools/crossthink/ui`, then rebuild `llama-crossthink`.

Each agent has a generated-token count and a tokens-per-second meter over a rolling five-second window. Those per-agent counts include its generated reasoning, answers, and native tool-call text. The messaging pane shows A + B **reasoning** token totals and the sum of their reasoning rates. Counts accumulate until **New session**. Imported context tokens are reported separately and are not counted as newly generated tokens. The rates measure observed stream delivery over wall-clock time, including idle periods, rather than the model server's decode-only benchmark rate.

Per-agent status also shows the current phase, tokens received for the current request, time since that request started, and time since its last token. After reasoning closes, native answer/tool-call text is buffered until the native turn finishes and can be parsed; the reasoning pane can be quiet while the agent is still generating. The status distinguishes this from waiting for the server, generating reasoning, handling a thought result, or running an MCP call. Waiting for the first token includes queueing and prompt processing; the coordinator cannot distinguish those server-side phases.

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

Remove `headers` if the endpoint needs no authentication. These example URLs do not start a service: the calculator/compiler tools must already be provided by external MCP servers. The three thought commands are built in and need no MCP server. Calculator, compiler, and shell tools must come from the configured servers.

```bash
build/bin/llama-crossthink \
    -a /tmp/crossthink.XXXXXXXX/a.sock \
    -b /tmp/crossthink.XXXXXXXX/b.sock \
    --mcp-config mcp.json \
    --mcp-timeout 60 --tool-turn-tokens 2048 --max-tool-rounds 8
```

Both llama-server instances and the coordinator must be rebuilt for this update. Use `--jinja` on the model servers, as above. MCP tool calling uses the model's chat template and native output parser, including Qwen3.5's XML tool-call format. These native tool calls are separate from the three reasoning-only thought commands. In legacy mode, MCP requires `--splice-mode paragraph`.

| Option | Default | Behavior |
| --- | --- | --- |
| `--mcp-config FILE` | None | Read an `mcpServers` JSON object with HTTP/HTTPS `url` and optional string-valued `headers`. File size is limited to 1 MiB. |
| `--mcp-timeout N` | `60` | HTTP timeout in seconds, `1..600`, for MCP requests. |
| `--tool-turn-tokens N` | `2048` | Private native tool/answer turn budget, `1..16384`; the effective ceiling is the larger of this value and `--answer-tokens`. |
| `--max-tool-rounds N` | `8` | Consecutive native tool round limit, `1..64`; generated reasoning resets the counter. In legacy mode, the paragraph rendezvous resets it. |

This client supports MCP Streamable HTTP, including JSON and SSE responses to POST requests, session initialization, tool discovery, and tool calls. HTTPS requires a build with TLS support, normally `-DLLAMA_OPENSSL=ON`. The older HTTP+SSE transport, stdio commands, OAuth login, MCP resources/prompts, and server-initiated sampling are not supported. `--api-key` remains the bearer key for the model servers; MCP authentication is configured separately in each endpoint's `headers`.

Configured tools are called automatically when either model requests them. Calls run with the permissions of the selected external MCP service. The console shows the number of available tools and each peer's call count/status; expand **Tool activity** to inspect arguments and results. A prompt such as "Use the calculator to check the arithmetic" or "Compile a small C program to verify this" can encourage a call, but the model decides whether to use a tool.

Calls and their results stay in the calling model's conversation, with native assistant/tool roles. The partner continues its own reasoning while a native tool turn runs. The caller can later send a summary or let the partner inspect its reasoning about the result. Native calls and tool results themselves are not copied to the other agent or executed twice by the coordinator. Reset cancels current work and clears the displayed tool trace along with the conversation; it cannot undo effects of an external tool call that already ran.

## Interaction and scheduling

| Control | Behavior |
| --- | --- |
| Send to Both / A / B | Cancel the selected agents' current requests, append a real user turn to their histories, and resume their reasoning. A message to one agent leaves the other generating. |
| Pause / Resume | Stop reasoning generation and retain histories; native tool/answer turns finish before pausing, so tool results are not lost or replayed. Resume continues from the retained histories. |
| Answer now | Close both open reasoning blocks and generate separate answers. A subsequent message starts another thinking turn. |
| New session | Cancel current requests, discard both histories, clear thought cursors and inboxes, and clear the displayed conversation. Late packets from the old session are ignored. |

The HTTP control API mirrors these controls: `POST /message` with `{"text":"...","target":"both"}`, `"target":"A"`, or `"target":"B"`; an omitted target means both. `POST /pause`, `/resume`, `/answer`, or `/reset` accepts `{}`. Requests use `Content-Type: application/json`. The obsolete `/link_on` and `/link_off` controls are gone. `GET /state` returns status and counters; `GET /events?after=ID` streams SSE trace events with resumable IDs. The console retains a bounded trace; reconnecting after old events expire produces an explicit gap notice.

In the default mode, state reports `splice_mode: "independent"` and `thought_tools: 3`; `tools` counts only external native tools. Per-peer fields include `generated`, `imported`, `thinking_tokens`, `tokens_per_second`, `thinking_tokens_per_second`, `mailbox`, and `thought_turn`. Top-level `thinking_tokens` and rate fields sum both agents.

Per-peer `phase` reports `waiting_reasoning`, `reasoning`, `waiting_native`, `native_output`, `parsing_native`, `thought_result`, `mcp`, or `tool_result` while active, and `idle`, `ready`, `paused`, `done`, or `error` otherwise. Legacy answer streaming uses `answer`. `request_generated` resets for each completion request. `request_elapsed_seconds` is null before the first request, and `last_token_age_seconds` is null until the current request delivers tokens. These ages are wall-clock diagnostics, not server decode timings.

A `thought_result` event contains the complete marked insertion for the named peer's reasoning display. A separate `thought_capture` event identifies the source and captured byte range; do not append it again or the display will duplicate the capture. A `thought_message` event has a stable `message_id`, `from`, `to`, `text`, and a status of `sent` or `received`. Update the matching message bubble when its status changes. Peeks never create message bubbles.

Requests resubmit each agent's authoritative token sequence. Each resumed completion creates a fresh sampler with deterministic per-agent seeds, so results can differ from one uninterrupted sampling call. Keep server-wide reverse prompts and other output constraints disabled. Near context capacity, the affected agent uses its reserved space for a separate final answer while its partner continues. Legacy mode pauses the session at its context limit. Neither mode silently deletes older context. Larger answer and native tool budgets reserve more context.

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

The earlier quantum extension remains available to binary API clients: `splice.token_after` in `1..4096` stops when that count is reached and the output ends on a complete UTF-8 character. A clean paragraph or permitted sentence boundary can stop earlier; otherwise the result reports `"splice_boundary": "quantum"`. `n_predict` remains the hard ceiling, and hitting it with an incomplete UTF-8 character produces an error. `/tokens/info` advertises `"splice_quantum": true`. The default parallel coordinator no longer uses quantum stops.

Parallel thought commands require `"thought_commands": true` and `"thought_command_lines": true` from `/tokens/info`. A request sets `"splice": {"thought_commands": true, "stop_on_think_close": true}` without a quantum or sentence threshold. Complete commands at column zero outside fenced blocks in newly generated reasoning end the caller's request with `"splice_boundary": "thought_command"`, `"thought_command": "peek"`, `"inbox"`, or `"send"`, and `"thought_payload"` for a send. `splice_reason` aliases the boundary. An oversized send body returns `thought_error`; the body limit is 16384 bytes, with a 32768-byte JSON-escaped limit to keep the binary response metadata bounded. In this mode, ordinary paragraph breaks do not stop generation.

Stops occur at the token containing the command's closing marker. Tokens are indivisible: if that token also contains a suffix after the marker, the suffix stays in the raw history, before the inserted result. All committed token IDs remain intact. The optional bounded `thought_prefix` carries an incomplete, previously generated command across a pause or request limit; ordinary submitted prefixes and imported captures are not scanned for commands. The accompanying `thought_state` checkpoint carries line and fence state from before that prefix. A continuing client must pass both values back under `splice`, including `thought_state` when the prefix is empty, so a pause inside prose or a fence does not turn a later example into a command. The coordinator uses the same scanner locally to retain state when cancellation prevents a terminal response. Native reasoning closure remains `"splice_boundary": "tool"`. Rebuild both model servers and the coordinator when upgrading; an older server cannot support the new default mode.

When native tools are enabled, `splice.stop_on_think_close` also stops at the single `</think>` token and reports `"splice_boundary": "tool"`. That token is retained in the calling model's raw tape. The coordinator then generates a private assistant tail through EOG. `/apply-template` accepts optional `parse_output` token IDs and returns a parsed `message` alongside its ordinary `prompt`, using the same native chat parser as chat completions. A terminal EOG is removed before parsing; tokens following EOG are rejected. `/tokens/info` advertises `"tool_parse": true` for this extension.

## Validation

The `reasoning-swap` branch is based on `imaami/llama.cpp`'s `bonsai` commit `5ada589b7811232c44d55dbf210329d98414fb4d`. Release CPU builds passed for llama-server and llama-crossthink. The coordinator suite covers concurrent reasoning and MCP, incremental raw-token captures, opt-in inbox delivery, targeted user messages, independent answers, metrics, and pause/reset races. Additional regressions cover inert inline/fenced commands, line/fence state across requests, exactly-once deliberate commands after resuming, and empty-inbox continuation through reasoning and buffered native output. Binary route tests exercise MTP off and on, including split and resumed commands, inert imports and code examples, literal delimiters, bounded payloads, malformed checkpoints, and cached continuation. Native chat-template and HTTP MCP suites also passed.

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

Real Chromium tests passed against a mocked HTTP/SSE coordinator for formatting in every output pane, literal thought commands, sanitization, diagrams, stable streamed content, Follow behavior, targeted messages, inbox status, and rolling metrics. Regressions also reproduce an interrupted C fence followed by a new user turn, thought import, or native tool continuation; verify faithful raw copying and bounded clipping; and check waiting/native-output status indicators. Desktop and mobile layouts were inspected. Run these checks with `npm test` in `tools/crossthink/ui` after installing its dependencies and Chromium.

CPU fixtures do not establish Bonsai conversational behavior or dual-GPU throughput. The random MTP fixtures rejected their drafts, so accepted multi-token speculative batches remain unvalidated. Unix socket creation is denied in this test environment; live coordinator-to-model-server integration, real MCP services, TLS, and dual-GPU command latency still need verification on the intended setup.
