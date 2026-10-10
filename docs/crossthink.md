# Crossthink with parallel thought tools

`llama-crossthink` runs two independent agents in parallel. Four built-in native tools let each agent send thought mail, collect its inbox, read its partner's reasoning, and request its attention. Tool requests use the model's normal tool-call format; their marked results appear inside the caller's next open reasoning block. Mail waits until collected. Only an explicit ping notifies the recipient. The browser console shows separate reasoning and answers, plus a messaging view for sent and collected thoughts.

The earlier covert paragraph/fixed exchange remains available with `--legacy-splice` for comparisons. `--legacy-thought-commands` additionally enables the former inline-command protocol in independent mode. The serialized `think_with_telepathic_link` mode is no longer used. This is a native C++ coordinator using the repository's HTTP/JSON dependencies.

The server's new `--socket PATH` adds a Unix HTTP listener while keeping `--host` and `--port` available for the ordinary web UI and API. `LLAMA_ARG_SOCKET` is the corresponding environment variable. Paths need not end in `.sock`; the old `--host *.sock` convention still works. The Unix listener uses plain HTTP even when TCP uses TLS. On clean shutdown, the server removes only the filesystem socket it bound, if its inode still matches; an existing path is never removed at startup.

## Build and launch

Build **both `llama-server` and `llama-crossthink`** from the `reasoning-swap` branch for the binary streaming, native parser, and reasoning-boundary extensions. Updating an installation that already has those server extensions requires rebuilding and restarting the coordinator; the legacy inline protocol additionally requires the server's `thought_commands` and `thought_command_lines` capabilities. Retain your working Vulkan build configuration:

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

After an update, restart the rebuilt coordinator and reload any open console tab. The bundled UI is served with caching disabled, but an already-open page keeps its previous JavaScript until reloaded.

The ordinary model web UIs remain at **http://127.0.0.1:8080/** and **http://127.0.0.1:8081/**. Those chats are separate from the coordinated conversation; use port 8090 to address the pair. Other requests can displace a server's prompt cache or delay crossthink. Each continuation resubmits the authoritative token sequence, so cache displacement does not silently change the conversation.

## Native thought tools

Both agents receive a default system prompt explaining their identity (A or B), their partner, and the four built-in tools. They require no MCP configuration or activation step. The same native tool catalog contains any configured external MCP tools; a collision with a built-in name is rejected at startup.

| Tool | Arguments | Effect |
| --- | --- | --- |
| `send_thought` | `{"message":"Your update"}` | Queue the message in the partner's inbox without interrupting it. |
| `check_inbox` | `{}` | Collect messages waiting in the caller's inbox immediately. |
| `read_thoughts` | `{}` | Read the partner's current or most recent reasoning up to the instant the request is handled. |
| `ping_peer` | `{}` | Send a fixed attention notice to the partner, carrying no message payload. |

The model closes its reasoning and makes a normal native tool call, using its chat template's usual wrappers. It does not write command tags into its reasoning. The coordinator supplies a minimal native tool response such as `Result follows in reasoning.`, opens the next reasoning block through the template, and inserts the actual result with explicit start/end markers. This gives the model the normal outgoing tool-call path while keeping received thoughts inside reasoning.

Plain text such as `<ct:send>...</ct:send>`, whether in reasoning, a final answer, or an imported result, is inert by default. Merely describing a tool call does not execute it. The model must emit a valid native call to a listed tool. A batch of calls is handled in its listed order; the coordinator does not ask the partner to pause or alternate turns.

### Reading thoughts

The first `read_thoughts` call receives the partner's available reasoning from the beginning of its current task. Further reads continue from the previous cut, returning only newly available text. Native tool continuations preserve that capture and its read cursor; a real user message starts a new capture for its recipients. If no new text has appeared, the result is an empty capture with a zero byte count.

A snapshot contains committed reasoning exactly as it stood when captured, up to the latest complete UTF-8 character. It can end mid-sentence or mid-word: the reader does not wait for a paragraph, sentence, or the partner's final answer. The capture contains only the partner's own generated reasoning. Imported captures, inbox results, native tool calls/results, and final answers are excluded, so reading thoughts does not recursively copy earlier imports. Reading is passive; the partner keeps generating and does not need to approve it. Captures appear only in the receiver's reasoning panel, not in the messaging view.

Results retain the existing internal marker names. `turn`, `offset`, and `bytes` identify the source capture and byte range, and `active` reports whether that reasoning is still open:

```text
<ct:result command="peek">
<ct:thought source="B" turn="1" offset="0" bytes="..." active="true">verbatim captured text</ct:thought>
</ct:result>
```

These are coordinator-produced result markers, not commands for the model to generate. The payload stays contiguous and unescaped. The byte count distinguishes payload text from the closing marker even if the captured reasoning contains marker-like text. A capture that cannot fit is reported as an error; its read cursor does not advance.

### Sending and collecting thoughts

`send_thought` queues a deliberate message for the other agent. It does not interrupt the recipient, wake a finished agent, or insert anything into its live reasoning. Messages are limited to 16 KiB. The sender receives an acknowledgment and continues. `check_inbox` receives queued messages together, with source markers, inside the caller's reasoning. Each delivered message uses a `<ct:message id="..." from="B" bytes="...">...</ct:message>` wrapper inside the inbox result.

An empty inbox returns immediately; there is no wait for the partner to reply. A mailbox holds at most 256 messages and 1 MiB of message text. A full mailbox or a result that cannot fit produces an explicit error. Only successfully delivered inbox messages are removed; arrivals after the collection snapshot remain queued. Inbox consumption and thought-read cursors commit when the result tokens enter reasoning, so a later failure in a mixed native/MCP batch cannot discard unread data. The messaging view shows both the sent message and its collection status.

The system prompt asks each agent to send a brief planning message at the start of a new user task, then keep working without waiting for a reply. It encourages concise updates, inbox checks at useful milestones, and thought reads when useful. These are instructions to the models; the coordinator does not manufacture calls or force communication. A send succeeds only when its coordinator acknowledgment appears.

### Requesting attention

`ping_peer` sends a fixed, content-free attention notice. Put the actual plan or question in `send_thought`; ping only asks the recipient to attend to its partner. It does not drain the inbox or deliver the queued mail automatically.

If the recipient is actively reasoning, the coordinator cancels only that reasoning completion, appends a marked attention notice inside its open reasoning, and resumes from its retained token history. Native answer/tool-call generation, MCP execution, and result preparation finish before a queued notice is delivered. The sender keeps working throughout. This is a one-shot notification, not a conversation barrier or a request for the recipient to yield control.

Pending pings coalesce. A two-second cooldown suppresses repeated notifications without making either agent wait. A ping can wake a finished recipient with a fixed notification turn; its earlier answer remains visible. Paused sessions retain pending pings until resumed. **Answer now** clears them and prevents renewed reasoning. A real user message clears pending pings for its selected recipients, and **New session** clears both.

### Parallel execution

Both agents normally have a long-running streaming completion in flight at the same time. There is no speaker token, rendezvous, shared-floor lock, or periodic quantum cut. A native call ends only the caller's reasoning block. After the call, the coordinator appends the template continuation and marked result, then starts its next cached completion. The partner continues generating unless explicitly pinged.

This is append-and-resume through the existing completion API, not in-place insertion into an actively decoding server slot. Tool continuations and pings still incur HTTP, prompt-suffix evaluation, and sampler setup work. Their cost depends on the amount of imported text and prompt-cache reuse. Independent requests allow both GPUs to work concurrently; this does not guarantee a throughput or reasoning-quality improvement.

Each model's final answer appears in its own panel. Its partner can continue independently. **Answer now** asks both to finish.

| Option | Default | Behavior |
| --- | --- | --- |
| `--answer-tokens N` | `1024` | Final answer budget, `1..65536`. |
| `--protocol-retries N` | `-1` | Corrective feedback retries for known native-format rejections per agent per real user message: `-1` has no fixed limit, `0` disables feedback retries, and positive values cap them. Executed tools are never automatically repeated. |
| `--legacy-thought-commands` | Off | Also enable the former inline commands and their compatibility normalization. Native thought tools remain available. |
| `--strict-thought-protocol` | Off | Deprecated compatibility switch; disables normalization only when legacy thought commands are enabled. |
| `--link-quantum N` | Compatibility only | Accepted but ignored; there is no shared generation quantum. |
| `--private-quantum N` | Compatibility only | Accepted but ignored; independent reasoning no longer restarts periodically. |
| `--link-wait-tokens N` | Compatibility only | Accepted but ignored; no shared floor or voluntary yield exists. |
| `--max-segment-tokens N` | `512` | Legacy exchange ceiling, `1..4096`; does not limit independent reasoning. |
| `--legacy-splice` | Off | Restore the earlier covert exchange; native thought tools are unavailable. |

Start a new session after upgrading. Existing contexts describe the old protocol and can continue steering the models toward obsolete command syntax.

## Browser console

Open the coordinator's port, normally **http://127.0.0.1:8090/**. The composer targets **Both**, **A**, or **B**. The individual panes show each agent's reasoning, imported thought captures, answer, and native tool activity. The third pane uses alternating message bubbles for deliberate messages between the agents; it is no longer a line-by-line shared reasoning transcript.

**Follow** is an explicit checkbox. Scrolling up does not turn it off; disable it yourself to read older output without automatic following. Each output pane retains its own setting.

Model output uses the upstream web UI's Markdown pipeline: GitHub-flavored Markdown, fenced code with syntax highlighting and raw copy, KaTeX math, Mermaid diagrams, and sandboxed HTML previews. This rendering applies to reasoning, answers, thought messages, and native tool activity. Rendered HTML and SVG are sanitized. Rendering assets are bundled with the console for offline use; there is no required CDN connection. These are presentation features: the model receives the original text, not the rendered HTML.

Reasoning turns and imported thought results render as separate Markdown documents. An interrupted turn ending in an unclosed code fence therefore cannot turn the next user turn or thought result into code. Reasoning resumed after a native tool result also starts a new document, and native tool records are separated in the same way. Copy raw preserves the original text without adding artificial closing fences or UI dividers.

Normal C++ builds use the checked-in generated UI header and do not require Node.js. To modify the UI sources, run `npm ci` followed by `npm run build` in `tools/crossthink/ui`, then rebuild `llama-crossthink`.

Each agent has a generated-token count and a tokens-per-second meter over a rolling five-second window. Those per-agent counts include its generated reasoning, answers, and native tool-call text. The messaging pane shows A + B **reasoning** token totals and the sum of their reasoning rates. Counts accumulate until **New session**. Imported context tokens are reported separately and are not counted as newly generated tokens. The rates measure observed stream delivery over wall-clock time, including idle periods, rather than the model server's decode-only benchmark rate.

Per-agent status also shows the current phase, tokens received for the current request, time since that request started, and time since its last token. After reasoning closes, native answer/tool-call text is buffered until the native turn finishes and can be parsed; the reasoning pane can be quiet while the agent is still generating. The status distinguishes this from waiting for the server, generating reasoning, handling a thought result, or running an MCP call. Waiting for the first token includes queueing and prompt processing; the coordinator cannot distinguish those server-side phases.

Expand **Native response** to inspect or copy the text generated after reasoning closed, before native parsing. This includes malformed calls that cannot be parsed. Parsed `reasoning_content` is displayed separately in the individual reasoning pane, but commands found there are not executed retroactively. A native turn with no non-whitespace answer and no tool calls is reported as **Finished without an answer**. That agent stops while its partner continues. Once both finish, the session is **Incomplete** if either failed to produce an answer. Send a follow-up message to the affected agent to continue.

A known native parser format rejection receives marked **Coordinator feedback** and a new open reasoning block. Nothing from that rejected response executes; the model must retry using the listed native tool names and its normal tool-call format. `--protocol-retries -1` removes the fixed retry cutoff. Set `0` to disable feedback retries, or a positive value to cap them per agent per real user message. Native tool turns do not replenish a finite retry budget; a targeted user message resets only its recipient's budget.

Retry-budget exhaustion, unsupported template continuations, insufficient context, or unrecoverable native-call errors stop only the affected agent as **Finished without an answer**. Its partner continues. Unlimited format retries remain constrained by available context and user controls. Generic HTTP/transport failures and already-executed tool calls are never automatically retried: a failed connection can occur after a side effect has happened.

The Qwen3-coder parser used by Qwen3.5 requires complete consumption of a native response. Previously its optional tool-call branch could accept zero calls and silently discard an unknown or malformed `<tool_call>`, even one following valid prose or a valid call. Rejected responses now retain their generated text for inspection, and no call from the rejected response executes by default.

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

Remove `headers` if the endpoint needs no authentication. These example URLs do not start a service: the calculator/compiler tools must already be provided by external MCP servers. The four thought tools are built in and need no MCP server. Calculator, compiler, and shell tools must come from the configured servers.

```bash
build/bin/llama-crossthink \
    -a /tmp/crossthink.XXXXXXXX/a.sock \
    -b /tmp/crossthink.XXXXXXXX/b.sock \
    --mcp-config mcp.json \
    --mcp-timeout 60 --tool-turn-tokens 2048 --max-tool-rounds 8
```

Both llama-server instances and the coordinator must be rebuilt for this update. Use `--jinja` on the model servers, as above. MCP tool calling uses the model's chat template and native output parser, including Qwen3.5's XML tool-call format. The four built-in thought tools use the same native call syntax as MCP tools. Their actual results are inserted into reasoning; ordinary MCP results retain their normal tool-role representation. In legacy mode, MCP requires `--splice-mode paragraph`.

| Option | Default | Behavior |
| --- | --- | --- |
| `--mcp-config FILE` | None | Read an `mcpServers` JSON object with HTTP/HTTPS `url` and optional string-valued `headers`. File size is limited to 1 MiB. |
| `--mcp-timeout N` | `60` | HTTP timeout in seconds, `1..600`, for MCP requests. |
| `--tool-turn-tokens N` | `2048` | Private native tool/answer turn budget, `1..16384`; the effective ceiling is the larger of this value and `--answer-tokens`. |
| `--max-tool-rounds N` | `8` | Consecutive native tool round limit, `1..64`; generated reasoning resets the counter. In legacy mode, the paragraph rendezvous resets it. |

This client supports MCP Streamable HTTP, including JSON and SSE responses to POST requests, session initialization, tool discovery, and tool calls. HTTPS requires a build with TLS support, normally `-DLLAMA_OPENSSL=ON`. The older HTTP+SSE transport, stdio commands, OAuth login, MCP resources/prompts, and server-initiated sampling are not supported. `--api-key` remains the bearer key for the model servers; MCP authentication is configured separately in each endpoint's `headers`.

Configured tools are called automatically when either model requests them. Calls run with the permissions of the selected external MCP service. The console shows the number of available tools and each peer's call count/status; expand **Tool activity** to inspect arguments and results. A prompt such as "Use the calculator to check the arithmetic" or "Compile a small C program to verify this" can encourage a call, but the model decides whether to use a tool.

External calls and their results stay in the calling model's conversation, with native assistant/tool roles. The partner continues its own reasoning while a native tool turn runs. The caller can later send a summary or let the partner inspect its reasoning about the result. Native calls and tool results themselves are not copied to the other agent or executed twice by the coordinator. Reset cancels current work and clears the displayed tool trace along with the conversation; it cannot undo effects of an external tool call that already ran.

## Interaction and scheduling

| Control | Behavior |
| --- | --- |
| Send to Both / A / B | Cancel the selected agents' current requests, append a real user turn to their histories, and resume their reasoning. A message to one agent leaves the other generating. |
| Pause / Resume | Stop reasoning generation and retain histories; native tool/answer turns finish before pausing, so tool results are not lost or replayed. Resume continues from the retained histories. |
| Answer now | Close both open reasoning blocks and generate separate answers. A subsequent message starts another thinking turn. |
| New session | Cancel current requests, discard both histories, clear thought cursors and inboxes, and clear the displayed conversation. Late packets from the old session are ignored. |

The HTTP control API mirrors these controls: `POST /message` with `{"text":"...","target":"both"}`, `"target":"A"`, or `"target":"B"`; an omitted target means both. `POST /pause`, `/resume`, `/answer`, or `/reset` accepts `{}`. Requests use `Content-Type: application/json`. The obsolete `/link_on` and `/link_off` controls are gone. `GET /state` returns status and counters; `GET /events?after=ID` streams SSE trace events with resumable IDs. The console retains a bounded trace; reconnecting after old events expire produces an explicit gap notice.

In the default mode, state reports `splice_mode: "independent"` and `thought_tools: 4`; `tools` counts only external MCP tools. Per-peer fields include `generated`, `imported`, `thinking_tokens`, `tokens_per_second`, `thinking_tokens_per_second`, `mailbox`, `thought_turn`, `attention_pending`, and `attention_received`. `attention_received` counts delivered notices; pending or suppressed requests do not increment it. Top-level `thinking_tokens` and rate fields sum both agents.

Per-peer `phase` reports `waiting_reasoning`, `reasoning`, `waiting_native`, `native_output`, `parsing_native`, `repairing_protocol`, `recovering_protocol`, `thought_tool`, `thought_result`, `attention`, `mcp`, or `tool_result` while active, and `idle`, `ready`, `paused`, `done`, or `error` otherwise. Legacy answer streaming uses `answer`. `request_generated` resets for each completion request. `request_elapsed_seconds` is null before the first request, and `last_token_age_seconds` is null until the current request delivers tokens. These ages are wall-clock diagnostics, not server decode timings.

`empty_response: true` and `phase: "empty_response"` identify a stopped native turn that did not produce a usable answer or tool call. `answer_done` means that peer's generation has ended; use `empty_response` to distinguish failure from a completed answer. The aggregate terminal mode is `incomplete` rather than `answered` when any participating peer has this condition. A `native_output` event carries the raw tail in `text`; `native_reasoning` carries reasoning returned by the native parser. Neither event executes thought tools or represents a delivered thought message.

`protocol_retries` counts committed feedback retries for that peer's current user message. A `protocol_feedback` event carries `peer`, `text`, `attempt`, and `limit`; `limit: -1` means no fixed retry cap. It is coordinator feedback, not a tool execution or thought message. With `--legacy-thought-commands`, `protocol_repairs` counts successful command normalizations for the session, and resets only with **New session**. A separate `protocol_repair` event carries `peer`, `command`, and `text` for the coordinator's normalization notice. The operation's actual result and delivery still use the normal `thought_result` and `thought_message` events.

A `thought_result` event contains the complete marked insertion for the named peer's reasoning display. A separate `thought_capture` event identifies the source and captured byte range; do not append it again or the display will duplicate the capture. A `thought_message` event has a stable `message_id`, `from`, `to`, `text`, and a status of `sent` or `received`. Update the matching message bubble when its status changes. Thought reads never create message bubbles. An `attention` event carries `from`, `to`, and `status` (`queued`, `coalesced`, or `delivered`) for the messaging view. An `attention_result` event carries the recipient `peer` and exact marked `text` inserted into its reasoning. Attention notifications are separate from inbox mail and do not change a message's sent/received status.

Requests resubmit each agent's authoritative token sequence. Each resumed completion creates a fresh sampler with deterministic per-agent seeds, so results can differ from one uninterrupted sampling call. Keep server-wide reverse prompts and other output constraints disabled. Near context capacity, the affected agent uses its reserved space for a separate final answer while its partner continues. Legacy mode pauses the session at its context limit. Neither mode silently deletes older context. Larger answer and native tool budgets reserve more context.

## Comparing the agents

Identical GGUF files do not make A and B identical sampling runs. Their system prompts name different agents, and the coordinator seeds each request with `base_seed + peer_index + 2 * sequence`, modulo `UINT32_MAX`. With the default seed, A starts at 42 and B at 43; their request counters advance independently as they issue commands and continue. Reset starts those same sequences again, so a repeated B-specific mistake can follow its identity and seed even on identical hardware.

The coordinator overrides temperature and seed but leaves sampling options such as `top_k`, `top_p`, `min_p`, and penalties to each model server's defaults. Compare both servers' `/props` responses under `default_generation_settings.params` when investigating asymmetric behavior. The tokenizer-compatibility check does not verify identical model weights, templates, or sampling defaults. To distinguish a role/seed effect from an endpoint effect, start fresh sessions after swapping `--socket-a` and `--socket-b`. Behavior following an endpoint still needs its full configuration checked before attributing it to GPU architecture.

## Legacy inline thought commands

`--legacy-thought-commands` adds the earlier inline protocol to independent mode for comparisons. It is off by default and is separate from `--legacy-splice`. Only this compatibility mode scans generated reasoning for these literal commands:

| Command | Native equivalent |
| --- | --- |
| `<ct:peek/>` | `read_thoughts({})` |
| `<ct:send>message</ct:send>` | `send_thought({"message":"message"})` |
| `<ct:inbox/>` | `check_inbox({})` |

The opening tag must be at column zero of a new line, outside backtick or tilde code fences. Inline mentions, indented or quoted examples, and tags in imported captures, user messages, or results do not execute. A complete generated command briefly ends only its caller's completion; its marked result is appended before cached reasoning resumes.

Legacy compatibility normalization also recognizes a response consisting entirely of one complete thought operation in literal syntax or a supported Qwen-style wrapper. It excludes prose, quoted/fenced examples, multiple or mixed calls, external calls, and missing send payloads. `--strict-thought-protocol` disables this normalization while leaving the inline reasoning scanner active. Send limits are 16 KiB of payload and 32 KiB after JSON escaping.

Normalization preserves the original native response, appends a coordinator notice, and opens a new reasoning block. The pending operation executes once through the legacy handler before decoding continues. Successful normalization does not consume feedback retries. **Pause** retains the pending operation; **Answer now** discards it. A new user message clears it for the selected recipients, and **New session** clears both. Completed operations are not undone.

Existing scanner and normalization regression tests retain this mode explicitly. The default native-tool path does not reinterpret plain final-answer tags as calls.

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

The optional legacy inline protocol requires `"thought_commands": true` and `"thought_command_lines": true` from `/tokens/info`. A request sets `"splice": {"thought_commands": true, "stop_on_think_close": true}` without a quantum or sentence threshold. Complete commands at column zero outside fenced blocks in newly generated reasoning end the caller's request with `"splice_boundary": "thought_command"`, `"thought_command": "peek"`, `"inbox"`, or `"send"`, and `"thought_payload"` for a send. `splice_reason` aliases the boundary. An oversized send body returns `thought_error`; the body limit is 16384 bytes, with a 32768-byte JSON-escaped limit to keep the binary response metadata bounded. In this mode, ordinary paragraph breaks do not stop generation.

Stops occur at the token containing the command's closing marker. Tokens are indivisible: if that token also contains a suffix after the marker, the suffix stays in the raw history, before the inserted result. All committed token IDs remain intact. The optional bounded `thought_prefix` carries an incomplete, previously generated command across a pause or request limit; ordinary submitted prefixes and imported captures are not scanned for commands. The accompanying `thought_state` checkpoint carries line and fence state from before that prefix. A continuing client must pass both values back under `splice`, including `thought_state` when the prefix is empty, so a pause inside prose or a fence does not turn a later example into a command. The legacy inline coordinator uses the same scanner locally to retain state when cancellation prevents a terminal response. The default native-tool mode leaves `thought_commands` disabled and uses only `stop_on_think_close`; literal tags cannot stop its reasoning stream. Native reasoning closure remains `"splice_boundary": "tool"`.

When native tools are enabled, `splice.stop_on_think_close` also stops at the single `</think>` token and reports `"splice_boundary": "tool"`. That token is retained in the calling model's raw tape. The coordinator then generates a private assistant tail through EOG. `/apply-template` accepts optional `parse_output` token IDs and returns a parsed `message` alongside its ordinary `prompt`, using the same native chat parser as chat completions. A terminal EOG is removed before parsing; tokens following EOG are rejected. `/tokens/info` advertises `"tool_parse": true` for this extension.

## Validation

The `reasoning-swap` branch is based on `imaami/llama.cpp`'s `bonsai` commit `5ada589b7811232c44d55dbf210329d98414fb4d`. The coordinator suite covers independent reasoning, incremental captures, inbox delivery, targeted messages, MCP, separate answers, metrics, and control races. The former inline scanner and normalization tests explicitly enable `--legacy-thought-commands`; those tests do not establish behavior of the new default native tools.

Native-tool regressions cover all four built-ins without MCP, reserved-name collisions, inert command-looking reasoning/final text, capture continuity through calls, result insertion after an open reasoning suffix, exactly-once side effects, native/MCP call ordering, and retained unread data after failed mixed batches. Ping regressions cover active reasoning, deferred native/MCP work, finished peers, pause/resume, coalescing/cooldown, and Answer now/reset/targeted-message precedence. Release CPU builds, the coordinator/protocol/template/MCP suites, CLI help, and Chromium UI regressions passed for this change.

Run the binaries from the repository root for the template files:

```bash
cmake -S . -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target test-crossthink test-crossthink-protocol test-crossthink-mcp test-crossthink-tool-template test-server-token-wire -j
build/bin/test-crossthink
build/bin/test-crossthink-protocol
build/bin/test-crossthink-mcp
build/bin/test-crossthink-tool-template
python3 tests/gen-tiny-qwen35-mtp.py /tmp/tiny-qwen35-mtp.gguf
python3 tests/gen-tiny-qwen35-mtp.py --tool-tokens /tmp/tiny-qwen35-tools.gguf
build/bin/test-server-token-wire /tmp/tiny-qwen35-mtp.gguf /tmp/tiny-qwen35-tools.gguf
```

The Chromium suite runs against a mocked HTTP/SSE coordinator for formatting in every output pane, literal thought commands, sanitization, diagrams, stable streamed content, Follow behavior, targeted messages, inbox status, and rolling metrics. Regressions also reproduce an interrupted C fence followed by a new user turn, thought import, or native tool continuation; verify faithful raw copying and bounded clipping; and check waiting/native-output status indicators. Run these checks with `npm test` in `tools/crossthink/ui` after installing its dependencies and Chromium.

CPU fixtures do not establish Bonsai conversational behavior or dual-GPU throughput. The random MTP fixtures rejected their drafts, so accepted multi-token speculative batches remain unvalidated. Unix socket creation is denied in this test environment; live coordinator-to-model-server integration, real MCP services, TLS, and dual-GPU command latency still need verification on the intended setup.
