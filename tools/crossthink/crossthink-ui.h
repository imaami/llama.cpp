#pragma once

static constexpr char crossthink_ui[] = R"html(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>llama.cpp crossthink</title>
<style>
    :root { color-scheme: dark; font: 15px/1.5 system-ui, sans-serif; background: #101216; color: #e7e9ef; }
    * { box-sizing: border-box; }
    body { margin: 0 auto; max-width: 1600px; padding: 22px 24px 30px; }
    button, textarea { font: inherit; }
    button { cursor: pointer; border: 1px solid #424853; background: #242933; color: #eef0f5; border-radius: 6px; padding: 7px 13px; }
    button:hover:not(:disabled) { border-color: #9ca9c0; background: #303744; }
    button:disabled { opacity: .4; cursor: default; }
    button:focus-visible, textarea:focus-visible, input:focus-visible { outline: 2px solid #91c8ff; outline-offset: 3px; }
    .primary { background: #c3dfaa; color: #14240a; border-color: #c3dfaa; font-weight: 650; }
    .primary:hover:not(:disabled) { background: #d2eabe; color: #14240a; }
    header, .toolbar, .panel-head, .send-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    header { flex-wrap: wrap; margin-bottom: 6px; }
    h1 { font-size: 23px; letter-spacing: -.5px; margin: 0; }
    h2 { font-size: 16px; margin: 0; }
    h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #b3bbcb; margin: 0; }
    p { margin: 7px 0; }
    .muted { color: #a3acbc; font-size: 13px; }
    .connection { font-size: 12px; color: #e7b77d; }
    .connection.online { color: #bbd6a5; }
    .toolbar { flex-wrap: wrap; margin: 18px 0; }
    .actions { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
    .mode { min-width: 130px; text-transform: capitalize; }
    .status-dot { display: inline-block; height: 7px; width: 7px; border-radius: 50%; background: #9aa4b4; margin-right: 8px; }
    .mode[data-mode="thinking"] .status-dot { background: #c3dfaa; }
    .mode[data-mode="answering"] .status-dot { background: #8dbbff; }
    .mode[data-mode="error"] .status-dot { background: #ff9992; }
    .notice { padding: 10px 12px; border: 1px solid #67523b; border-radius: 6px; background: #2b231c; color: #f5d3a8; margin: 10px 0; font-size: 13px; }
    .notice.error { background: #301f23; border-color: #775059; color: #ffc0c0; }
    [hidden] { display: none !important; }
    .peers { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .peer { min-width: 0; background: #171b22; border: 1px solid #343b48; border-radius: 9px; overflow: hidden; }
    .peer-header { padding: 13px 16px; border-bottom: 1px solid #343b48; }
    .peer-a h2 { color: #b7d5ff; }
    .peer-b h2 { color: #d4baff; }
    .stats { display: flex; gap: 6px 14px; flex-wrap: wrap; margin-top: 7px; font: 11px/1.6 ui-monospace, monospace; color: #aeb8ca; }
    .panel-head { padding: 12px 16px 6px; }
    .follow { font-size: 12px; color: #aab4c5; user-select: none; white-space: nowrap; }
    .follow input { accent-color: #b7d5ff; vertical-align: middle; margin-right: 5px; }
    .output { margin: 0; padding: 8px 16px 16px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; scrollbar-gutter: stable; }
    .thought { height: min(43vh, 480px); min-height: 230px; color: #c0c8d7; font: 13px/1.65 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .answer-wrap { background: #1c232c; border-top: 1px solid #343b48; }
    .answer { min-height: 130px; max-height: 320px; font: 15px/1.6 system-ui, sans-serif; }
    .tools-wrap { border-top: 1px solid #343b48; }
    .tools-wrap summary { cursor: pointer; padding: 10px 16px; font-size: 13px; color: #b3bbcb; }
    .tools-wrap summary span { margin-left: 8px; color: #a3acbc; }
    .tools { max-height: 300px; font: 12px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .output.empty::before { content: attr(data-empty); color: #788399; font: 13px/1.6 system-ui, sans-serif; }
    .clipped { font-size: 11px; color: #c7b089; padding: 0 16px; }
    .messages { max-height: 170px; overflow-y: auto; margin: 16px 0 8px; }
    .user-message { margin: 8px 0; padding: 9px 12px; border-left: 2px solid #748a9f; background: #171c24; font-size: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }
    .message-label { color: #96a8bd; font-size: 11px; margin-bottom: 3px; }
    form { margin-top: 15px; }
    textarea { width: 100%; min-height: 87px; resize: vertical; border: 1px solid #4d596c; border-radius: 7px; background: #171c24; color: #f1f4f9; padding: 12px; }
    textarea::placeholder { color: #818c9e; }
    .send-row { margin-top: 9px; }
    .send-row .muted { max-width: 760px; }
    footer { margin-top: 17px; font-size: 12px; color: #8793a6; }
    @media (max-width: 760px) {
        body { padding: 16px 12px 24px; }
        .peers { grid-template-columns: 1fr; }
        .thought { height: 270px; }
        .send-row { align-items: flex-start; }
        .send-row button { white-space: nowrap; }
        .toolbar { gap: 10px; }
        .mode { min-width: auto; }
    }
</style>
</head>
<body>
<header>
    <h1>llama.cpp <span style="font-weight:400;color:#aeb9cd">/ crossthink</span></h1>
    <span id="connection" class="connection" role="status">Connecting...</span>
</header>
<p class="muted">Two live reasoning streams, exchanging their newly generated reasoning.</p>
<div class="toolbar">
    <div id="mode" class="mode"><span class="status-dot"></span><span id="mode-text">Loading...</span></div>
    <div class="actions">
        <button type="button" id="pause" disabled>Pause</button>
        <button type="button" id="resume" disabled>Resume</button>
        <button type="button" id="answer" disabled>Answer now</button>
        <button type="button" id="reset" disabled title="Clear both models' shared session and displayed history">New session</button>
    </div>
</div>
<p id="splice-status" class="muted" role="status"></p>
<p id="tools-status" class="muted" role="status"></p>
<div id="error" class="notice error" role="alert" hidden></div>
<div id="warning" class="notice" role="status" hidden></div>
<main class="peers">
    <section class="peer peer-a" aria-label="Model A">
        <div class="peer-header">
            <h2>Model A</h2>
            <div class="stats" id="stats-A"><span>Waiting for status</span></div>
        </div>
        <div class="panel-head"><h3>Live reasoning</h3><label class="follow"><input id="follow-A" type="checkbox" checked>Follow</label></div>
        <div class="clipped" id="clipped-A" hidden>Older displayed text was trimmed. Model context is unchanged.</div>
        <pre id="thought-A" class="output thought empty" data-empty="Send a message to begin."></pre>
        <div class="answer-wrap">
            <div class="panel-head"><h3>Answer</h3></div>
            <div id="answer-A" class="output answer empty" data-empty="Choose Answer now to finish reasoning and generate answers."></div>
        </div>
        <details class="tools-wrap">
            <summary>Tool activity <span id="tool-status-A">No calls</span></summary>
            <div id="tools-clipped-A" class="clipped" hidden>Older displayed tool activity was trimmed.</div>
            <pre id="tools-A" class="output tools empty" data-empty="Tool calls and results appear here."></pre>
        </details>
    </section>
    <section class="peer peer-b" aria-label="Model B">
        <div class="peer-header">
            <h2>Model B</h2>
            <div class="stats" id="stats-B"><span>Waiting for status</span></div>
        </div>
        <div class="panel-head"><h3>Live reasoning</h3><label class="follow"><input id="follow-B" type="checkbox" checked>Follow</label></div>
        <div class="clipped" id="clipped-B" hidden>Older displayed text was trimmed. Model context is unchanged.</div>
        <pre id="thought-B" class="output thought empty" data-empty="Send a message to begin."></pre>
        <div class="answer-wrap">
            <div class="panel-head"><h3>Answer</h3></div>
            <div id="answer-B" class="output answer empty" data-empty="Choose Answer now to finish reasoning and generate answers."></div>
        </div>
        <details class="tools-wrap">
            <summary>Tool activity <span id="tool-status-B">No calls</span></summary>
            <div id="tools-clipped-B" class="clipped" hidden>Older displayed tool activity was trimmed.</div>
            <pre id="tools-B" class="output tools empty" data-empty="Tool calls and results appear here."></pre>
        </details>
    </section>
</main>
<div id="messages" class="messages" aria-label="Your messages"></div>
<form id="message-form">
    <label for="message" class="muted">Message both models</label>
    <textarea id="message" name="message" placeholder="Ask a question, add a constraint, or redirect the discussion." required></textarea>
    <div class="send-row">
        <span class="muted">Messages and pause requests take effect after the current segments finish. Ctrl+Enter or Cmd+Enter sends.</span>
        <button id="send" class="primary" type="submit" disabled>Send to both</button>
    </div>
</form>
<footer>This page controls one shared crossthink session. The ordinary llama-server web interfaces remain separate conversations. Each reasoning pane shows only that model's own output; imported tokens are not repeated.</footer>
<script>
(() => {
    'use strict';
    const byId = id => document.getElementById(id);
    const peers = ['A', 'B'];
    const textLimit = 512 * 1024;
    const outputs = {};
    let state = null;
    let source = null;
    let lastEvent = 0;
    let connected = false;
    let pending = false;
    let retry = 500;
    let reconnectTimer = null;

    function notice(id, message) {
        byId(id).textContent = message || '';
        byId(id).hidden = !message;
    }

    function clearOutput(key) {
        const output = outputs[key];
        output.node.data = '';
        output.element.classList.add('empty');
        if (key.startsWith('thought-')) {
            byId('clipped-' + key.slice(-1)).hidden = true;
        }
    }

    function appendOutput(key, text) {
        const output = outputs[key];
        if (!output || typeof text !== 'string' || !text) {
            return;
        }
        const follow = output.follow ? output.follow.checked :
            output.element.scrollHeight - output.element.scrollTop - output.element.clientHeight < 50;
        output.node.appendData(text);
        output.element.classList.remove('empty');
        if (output.node.length > textLimit) {
            output.node.deleteData(0, output.node.length - textLimit);
            if (output.follow) {
                byId('clipped-' + key.slice(-1)).hidden = false;
            }
        }
        if (follow) {
            output.element.scrollTop = output.element.scrollHeight;
        }
    }

    for (const peer of peers) {
        for (const kind of ['thought', 'answer']) {
            const key = kind + '-' + peer;
            const element = byId(key);
            const node = document.createTextNode('');
            element.appendChild(node);
            outputs[key] = { element, node, follow: kind === 'thought' ? byId('follow-' + peer) : null };
        }
        const output = outputs['thought-' + peer];
        output.follow.addEventListener('change', () => {
            if (output.follow.checked) {
                output.element.scrollTop = output.element.scrollHeight;
            }
        });
        output.element.addEventListener('scroll', () => {
            if (output.element.scrollHeight - output.element.scrollTop - output.element.clientHeight > 70) {
                output.follow.checked = false;
            }
        }, { passive: true });
    }

    function updateControls() {
        const available = !!state && connected && !pending && !state.busy;
        const mode = state ? state.mode : '';
        byId('pause').disabled = !available || mode !== 'thinking';
        byId('resume').disabled = !available || mode !== 'paused' || !!state.error;
        byId('answer').disabled = !available || !['thinking', 'paused'].includes(mode);
        byId('reset').disabled = !available;
        byId('send').disabled = !available || mode === 'answering' || !byId('message').value.trim();
    }

    function applyState(next) {
        if (!next || typeof next !== 'object') {
            return;
        }
        state = next;
        byId('mode').dataset.mode = state.mode;
        byId('mode-text').textContent = (state.mode || 'Unknown') + (state.busy ? ' / applying request' : '');
        const number = value => Number.isFinite(value) ? value.toLocaleString() : '?';
        const paragraph = state.splice_mode === 'paragraph';
        byId('splice-status').textContent = paragraph ?
            'Paragraph rendezvous / ' + number(state.exchanges) + ' exchanges' : 'Fixed intervals';
        const hasTools = Number(state.tools) > 0;
        byId('tools-status').textContent = hasTools ?
            'MCP / ' + number(state.tools) + ' tools available to each model' : 'MCP tools not configured';
        for (const peer of state.peers || []) {
            if (!peers.includes(peer.name)) {
                continue;
            }
            const stats = byId('stats-' + peer.name);
            byId('tool-status-' + peer.name).textContent = number(peer.tool_calls || 0) + ' calls' +
                (peer.tool_status ? ' / ' + peer.tool_status : '');
            byId('answer-' + peer.name).dataset.empty = hasTools ?
                'Answers appear when a model finishes or you choose Answer now.' :
                'Choose Answer now to finish reasoning and generate answers.';
            stats.replaceChildren();
            const details = [
                'Context ' + number(peer.tokens) + ' / ' + number(peer.context_size),
                'Queued ' + number(peer.queued),
                'Generated ' + number(peer.generated),
                'Imported ' + number(peer.imported),
            ];
            if (paragraph) {
                const boundary = { paragraph: 'paragraph', sentence: 'sentence', limit: 'token limit' }[peer.boundary];
                if (peer.waiting) { details.push('Waiting for peer'); }
                if (boundary) { details.push('Last boundary: ' + boundary); }
                details.push('Forced cuts ' + number(peer.forced_splices));
            }
            for (const text of details) {
                const item = document.createElement('span');
                item.textContent = text;
                stats.appendChild(item);
            }
        }
        if (state.error) {
            notice('error', state.error);
        }
        updateControls();
    }

    function resetDisplay() {
        for (const key of Object.keys(outputs)) {
            clearOutput(key);
        }
        for (const peer of peers) {
            byId('tools-' + peer).textContent = '';
            byId('tools-' + peer).classList.add('empty');
            byId('tools-clipped-' + peer).hidden = true;
            byId('tool-status-' + peer).textContent = 'No calls';
        }
        byId('messages').replaceChildren();
        notice('error', '');
        notice('warning', '');
    }

    function showUser(text) {
        const message = document.createElement('div');
        message.className = 'user-message';
        const label = document.createElement('div');
        label.className = 'message-label';
        label.textContent = 'YOU';
        message.appendChild(label);
        message.appendChild(document.createTextNode(text || ''));
        const messages = byId('messages');
        messages.appendChild(message);
        while (messages.children.length > 100) {
            messages.firstChild.remove();
        }
        messages.scrollTop = messages.scrollHeight;
        for (const peer of peers) {
            clearOutput('answer-' + peer);
            if (outputs['thought-' + peer].node.length) {
                appendOutput('thought-' + peer, '\n\n--- New message ---\n\n');
            }
        }
    }

    function showTool(peer, record) {
        const output = byId('tools-' + peer);
        const follow = output.scrollHeight - output.scrollTop - output.clientHeight < 50;
        const result = record.type === 'tool_result';
        const value = result ? record.result : record.arguments;
        const header = (result ? 'RESULT ' : 'CALL ') + (record.name || 'tool') +
            (record.call_id ? ' [' + record.call_id + ']' : '');
        let text = output.textContent + header + '\n' + (JSON.stringify(value, null, 2) || 'null') + '\n\n';
        if (text.length > textLimit) {
            text = text.slice(-textLimit);
            byId('tools-clipped-' + peer).hidden = false;
        }
        output.textContent = text;
        output.classList.remove('empty');
        if (follow) {
            output.scrollTop = output.scrollHeight;
        }
    }

    function handleEvent(event) {
        let record;
        try {
            record = JSON.parse(event.data);
            if (!record || typeof record !== 'object') {
                throw new Error('Invalid event');
            }
        } catch (_) {
            notice('error', 'Received an invalid event from the controller.');
            return;
        }
        const id = Number(record.id || event.lastEventId);
        if (Number.isSafeInteger(id) && id > 0) {
            if (id <= lastEvent) {
                return;
            }
            lastEvent = id;
        }
        const peer = peers.includes(record.peer) ? record.peer : null;
        switch (record.type) {
            case 'token':
                if (peer) { appendOutput('thought-' + peer, record.text); }
                break;
            case 'answer_start':
                if (peer) { clearOutput('answer-' + peer); }
                break;
            case 'answer':
                if (peer) { appendOutput('answer-' + peer, record.text); }
                break;
            case 'tool_call':
            case 'tool_result':
                if (peer) { showTool(peer, record); }
                break;
            case 'state': applyState(record.state); break;
            case 'user': showUser(record.text); break;
            case 'reset': resetDisplay(); break;
            case 'error': notice('error', record.message || record.text || 'The controller reported an error.'); break;
            case 'notice': notice('warning', record.message || ''); break;
            case 'gap':
                notice('warning', 'Some earlier events are no longer available. The displayed reasoning may have gaps; the models retain their current context.');
                break;
        }
    }

    async function refreshState() {
        const response = await fetch('/state', { cache: 'no-store' });
        if (!response.ok) {
            throw new Error('Status request failed (HTTP ' + response.status + ').');
        }
        const next = await response.json();
        if (Number.isSafeInteger(next.last_event_id) && next.last_event_id < lastEvent) {
            lastEvent = 0;
            resetDisplay();
        }
        applyState(next);
    }

    function connectionStatus(online, text) {
        connected = online;
        byId('connection').textContent = text;
        byId('connection').classList.toggle('online', online);
        updateControls();
    }

    async function connect() {
        try {
            await refreshState();
        } catch (error) {
            notice('error', error.message);
        }
        source = new EventSource('/events?after=' + lastEvent);
        source.onmessage = handleEvent;
        source.onopen = () => {
            retry = 500;
            connectionStatus(true, 'Connected');
        };
        source.onerror = () => {
            source.close();
            source = null;
            connectionStatus(false, 'Disconnected / reconnecting...');
            reconnectTimer = window.setTimeout(connect, retry);
            retry = Math.min(retry * 2, 10000);
        };
    }

    async function command(path, body = {}) {
        if (pending) {
            return false;
        }
        pending = true;
        notice('error', '');
        updateControls();
        try {
            const response = await fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const text = await response.text();
            let result;
            try { result = JSON.parse(text); } catch (_) { result = null; }
            if (!response.ok) {
                const error = result && result.error;
                const detail = typeof error === 'string' ? error : error && error.message;
                throw new Error(detail || (result && result.message) || 'Request failed (HTTP ' + response.status + ').');
            }
            try {
                await refreshState();
            } catch (error) {
                notice('error', 'Request accepted, but status refresh failed: ' + error.message);
            }
            return true;
        } catch (error) {
            notice('error', error.message);
            return false;
        } finally {
            pending = false;
            updateControls();
        }
    }

    for (const action of ['pause', 'resume', 'answer']) {
        byId(action).addEventListener('click', () => command('/' + action));
    }
    byId('reset').addEventListener('click', () => {
        if (window.confirm('Start a new session? This clears both models\' context and the displayed conversation.')) {
            command('/reset');
        }
    });
    byId('message').addEventListener('input', updateControls);
    byId('message').addEventListener('keydown', event => {
        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !byId('send').disabled) {
            event.preventDefault();
            byId('message-form').requestSubmit();
        }
    });
    byId('message-form').addEventListener('submit', async event => {
        event.preventDefault();
        const text = byId('message').value.trim();
        if (!text || byId('send').disabled) {
            return;
        }
        if (await command('/message', { text })) {
            if (byId('message').value.trim() === text) {
                byId('message').value = '';
            }
            byId('message').focus();
            updateControls();
        }
    });
    window.addEventListener('beforeunload', () => {
        window.clearTimeout(reconnectTimer);
        if (source) { source.close(); }
    });
    connect();
})();
</script>
</body>
</html>)html";
