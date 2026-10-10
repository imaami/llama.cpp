import { RichOutput, copyText } from './render.js';

const byId = id => document.getElementById(id);
const peers = ['A', 'B'];
const textLimit = 512 * 1024;
const segmentLimit = 512;
const outputs = {};
const bubbles = new Map();
let state = null;
let source = null;
let lastEvent = 0;
let connected = false;
let pending = false;
let retry = 500;
let reconnectTimer = null;
let statePoll = null;
let refreshing = false;
let stateSerial = 0;

function notice(id, message) {
    byId(id).textContent = message || '';
    byId(id).hidden = !message;
}

function follow(element, checkbox) {
    if (checkbox.checked) element.scrollTop = element.scrollHeight;
}

function initOutput(key, followId, clippedId = null, render = text => text) {
    const element = byId(key);
    const checkbox = byId(followId);
    const output = { element, raw: '', segments: [], current: null, nextLabel: null,
        follow: checkbox, clipped: clippedId ? byId(clippedId) : null,
        render, empty: element.dataset.empty };
    outputs[key] = output;
    // Scroll/layout changes NEVER change this explicit preference.
    checkbox.addEventListener('change', () => follow(element, checkbox));
    element.addEventListener('load', () => follow(element, checkbox), true);
}
for (const peer of peers) {
    initOutput('thought-' + peer, 'follow-' + peer, 'clipped-' + peer);
    initOutput('answer-' + peer, 'follow-answer-' + peer);
    initOutput('tools-' + peer, 'follow-tools-' + peer, 'tools-clipped-' + peer);
    initOutput('native-' + peer, 'follow-native-' + peer, 'native-clipped-' + peer,
        text => fence(text, 'text'));
}
byId('follow-messages').addEventListener('change', () => follow(byId('thread'), byId('follow-messages')));

function clearOutput(key) {
    const output = outputs[key];
    output.raw = '';
    for (const segment of output.segments) segment.renderer.clear();
    output.segments = [];
    output.current = null;
    output.nextLabel = null;
    output.element.replaceChildren();
    output.element.classList.add('empty');
    output.element.dataset.empty = output.empty;
    if (output.clipped) output.clipped.hidden = true;
}

function outputBoundary(key, label = null) {
    const output = outputs[key];
    output.current = null;
    output.nextLabel = output.raw ? label : null;
}

function appendOutput(key, text, isolated = false, label = null) {
    const output = outputs[key];
    if (!output || typeof text !== 'string' || (!text && !label)) return;
    if (isolated) output.current = null;
    if (label) output.nextLabel = label;
    if (!output.current) {
        const element = document.createElement('section');
        element.className = 'output-segment' + (isolated ? ' isolated-output' : '');
        if (output.nextLabel) {
            const label = document.createElement('div');
            label.className = 'output-boundary';
            label.textContent = output.nextLabel;
            element.append(label);
        }
        const body = document.createElement('div');
        body.className = 'segment-body';
        element.append(body);
        output.element.append(element);
        output.current = { element, raw: '', renderer: new RichOutput(body,
            () => follow(output.element, output.follow)) };
        output.segments.push(output.current);
        output.nextLabel = null;
    }
    output.current.raw += text;
    output.raw += text;
    output.element.classList.remove('empty');
    let excess = Math.max(0, output.raw.length - textLimit);
    let removed = 0;
    while (excess || output.segments.length > segmentLimit) {
        const first = output.segments[0];
        if (first.raw.length <= excess || output.segments.length > segmentLimit) {
            removed += first.raw.length;
            excess = Math.max(0, excess - first.raw.length);
            first.renderer.clear();
            first.element.remove();
            output.segments.shift();
        } else {
            first.raw = first.raw.slice(excess);
            removed += excess;
            excess = 0;
            first.renderer.update(output.render(first.raw));
        }
    }
    if (removed) {
        output.raw = output.raw.slice(removed);
        if (output.clipped) output.clipped.hidden = false;
    }
    output.current.renderer.update(output.render(output.current.raw));
    // Each received result is a separate Markdown document. Neither side of
    // an import may inherit an unfinished code fence from the other.
    if (isolated) output.current = null;
}

function updateControls() {
    const available = !!state && connected && !pending && !state.busy;
    const mode = state?.mode || '';
    byId('pause').disabled = !available || mode !== 'thinking';
    byId('resume').disabled = !available || mode !== 'paused' || !!state.error;
    byId('answer').disabled = !available || !['thinking', 'paused'].includes(mode);
    byId('reset').disabled = !available;
    byId('send').disabled = !available || (mode === 'answering' && byId('target').value === 'both') || !byId('message').value.trim();
    byId('send').textContent = byId('target').value === 'both' ? 'Send to both' : 'Send to ' + byId('target').value;
}

const number = value => Number.isFinite(value) ? value.toLocaleString() : '0';
const rate = value => Number.isFinite(value) ? value.toFixed(1) : '0.0';
const phaseLabels = {
    idle: 'Idle', ready: 'Ready', paused: 'Paused', done: 'Finished', error: 'Error', answer: 'Answering',
    empty_response: 'Finished without an answer',
    waiting_reasoning: 'Waiting for server (reasoning)', reasoning: 'Thinking',
    waiting_native: 'Waiting for server (answer/tool call)',
    native_output: 'Generating answer/tool call (buffered)', parsing_native: 'Parsing answer/tool call',
    thought_result: 'Applying thought result', mcp: 'Waiting for MCP tool', tool_result: 'Applying tool result'
};
function applyState(next) {
    if (!next || typeof next !== 'object') return;
    state = next;
    byId('mode').dataset.mode = state.mode;
    byId('mode-text').textContent = (state.mode === 'incomplete' ? 'Incomplete' : state.mode || 'Unknown') +
        (state.busy ? ' / applying request' : '');
    const legacy = state.telepathy === false;
    for (const option of byId('target').options) option.disabled = legacy && option.value !== 'both';
    if (legacy) byId('target').value = 'both';
    byId('description').textContent = legacy ? 'Two reasoning streams with legacy covert peer imports.' :
        'Two independent minds. Read thoughts, send a message, check the inbox.';
    byId('tools-status').textContent = (legacy ? 'Legacy splice mode' : 'Reasoning commands: read thoughts · send thought · check inbox') +
        (state.tools ? ' / ' + number(state.tools) + ' native tools' : '');
    let total = 0;
    let speed = 0;
    for (const peer of state.peers || []) {
        if (!peers.includes(peer.name)) continue;
        byId('count-' + peer.name).textContent = number(peer.generated);
        byId('speed-' + peer.name).textContent = rate(peer.tokens_per_second);
        byId('tool-status-' + peer.name).textContent = number(peer.tool_calls || 0) + ' calls' +
            (peer.tool_status ? ' / ' + peer.tool_status : '');
        const answer = outputs['answer-' + peer.name];
        answer.element.dataset.empty = peer.phase === 'empty_response' ?
            'This model finished without producing an answer. Inspect Native response below.' : answer.empty;
        const stats = byId('stats-' + peer.name);
        stats.replaceChildren();
        if (peer.phase) {
            const phase = document.createElement('span');
            phase.className = 'peer-phase';
            phase.dataset.phase = peer.phase;
            phase.textContent = phaseLabels[peer.phase] || peer.phase;
            stats.append(phase);
        }
        for (const text of [
            ...(Number.isFinite(peer.request_elapsed_seconds) ? [
                'Request ' + number(peer.request_generated) + ' tokens / ' + rate(peer.request_elapsed_seconds) + ' s',
                ...(Number.isFinite(peer.last_token_age_seconds) ? ['Last token ' + rate(peer.last_token_age_seconds) + ' s ago'] : [])
            ] : []),
            'Context ' + number(peer.tokens) + ' / ' + number(peer.context_size),
            'Thought ' + number(peer.thinking_tokens),
            'Imported ' + number(peer.imported),
            ...(peer.queued ? ['Queued ' + number(peer.queued)] : [])
        ]) {
            const item = document.createElement('span');
            item.textContent = text;
            stats.append(item);
        }
        total += peer.thinking_tokens || 0;
        speed += peer.thinking_tokens_per_second || 0;
    }
    byId('count-total').textContent = number(state.thinking_tokens ?? total);
    byId('speed-total').textContent = rate(state.thinking_tokens_per_second ?? speed);
    if (state.error) notice('error', state.error);
    updateControls();
}

function resetDisplay() {
    for (const key of Object.keys(outputs)) clearOutput(key);
    for (const peer of peers) byId('tool-status-' + peer).textContent = 'No calls';
    byId('messages').replaceChildren();
    byId('thread').replaceChildren();
    byId('thread').classList.add('empty');
    bubbles.clear();
    notice('error', '');
    notice('warning', '');
}

function showUser(record) {
    const message = document.createElement('div');
    message.className = 'user-message';
    const label = document.createElement('div');
    label.className = 'message-label';
    const target = peers.includes(record.target) ? record.target : 'both';
    label.textContent = 'YOU → ' + (target === 'both' ? 'BOTH' : target);
    message.append(label, document.createTextNode(record.text || ''));
    const messages = byId('messages');
    messages.append(message);
    while (messages.children.length > 100) messages.firstChild.remove();
    messages.scrollTop = messages.scrollHeight;
    for (const peer of target === 'both' ? peers : [target]) {
        clearOutput('answer-' + peer);
        outputBoundary('thought-' + peer, 'New user message');
    }
}

function showThought(record) {
    if (!peers.includes(record.from) || !peers.includes(record.to)) return;
    const id = String(record.message_id);
    let bubble = bubbles.get(id);
    if (!bubble) {
        const element = document.createElement('article');
        element.className = 'bubble from-' + record.from;
        const meta = document.createElement('div');
        meta.className = 'bubble-meta';
        const sender = document.createElement('span');
        sender.textContent = record.from + ' → ' + record.to;
        const status = document.createElement('span');
        status.className = 'bubble-state';
        meta.append(sender, status);
        const body = document.createElement('div');
        body.className = 'bubble-body';
        element.append(body, meta);
        byId('thread').append(element);
        byId('thread').classList.remove('empty');
        bubble = { element, status, text: null, received: false,
            renderer: new RichOutput(body, () => follow(byId('thread'), byId('follow-messages'))) };
        bubbles.set(id, bubble);
        while (bubbles.size > 300) {
            const first = bubbles.keys().next().value;
            bubbles.get(first).element.remove();
            bubbles.delete(first);
        }
    }
    if (typeof record.text === 'string' && record.text !== bubble.text) {
        bubble.text = record.text;
        bubble.renderer.update(record.text);
    }
    bubble.received ||= record.status === 'received';
    bubble.status.textContent = bubble.received ? 'Received' : 'Sent · waiting in inbox';
    bubble.status.classList.toggle('received', bubble.received);
    follow(byId('thread'), byId('follow-messages'));
}

function fence(text, language) {
    const marks = String.fromCharCode(96);
    const runs = text.match(/\x60+/g) || [];
    const delimiter = marks.repeat(Math.max(3, ...runs.map(run => run.length + 1)));
    return '\n' + delimiter + language + '\n' + text + '\n' + delimiter + '\n';
}

function showTool(peer, record) {
    const result = record.type === 'tool_result';
    const value = result ? record.result : record.arguments;
    const header = (result ? 'RESULT ' : 'CALL ') + (record.name || 'tool') +
        (record.call_id ? ' [' + record.call_id + ']' : '');
    // Headers are data too: a code fence prevents model-controlled markup.
    let text = fence(header, 'text');
    if (result && value && Array.isArray(value.content)) {
        for (const content of value.content) {
            if (content.type === 'text') text += '\n' + (content.text || '') + '\n';
            else text += fence(JSON.stringify(content, null, 2) || 'null', 'json');
        }
        if (value.isError) text += '\n**Tool reported an error.**\n';
    } else if (result && typeof value === 'string') {
        text += '\n' + value + '\n';
    } else {
        let parsed = value;
        if (!result && typeof value === 'string') {
            try { parsed = JSON.parse(value); } catch (_) {}
        }
        text += fence(JSON.stringify(parsed, null, 2) || 'null', 'json');
    }
    appendOutput('tools-' + peer, text + '\n---\n', true);
}

function handleEvent(event) {
    let record;
    try {
        record = JSON.parse(event.data);
        if (!record || typeof record !== 'object') throw new Error('Invalid event');
    } catch (_) {
        notice('error', 'Received an invalid event from the controller.');
        return;
    }
    const id = Number(record.id || event.lastEventId);
    if (Number.isSafeInteger(id) && id > 0) {
        if (id <= lastEvent) return;
        lastEvent = id;
    }
    const peer = peers.includes(record.peer) ? record.peer : null;
    switch (record.type) {
        case 'token':
            if (peer) appendOutput('thought-' + peer, record.text);
            break;
        case 'thought_result':
            if (peer) appendOutput('thought-' + peer, record.text, true);
            break;
        case 'native_output':
            if (peer) {
                appendOutput('native-' + peer, record.text, true,
                    record.text ? 'Native response' : 'Empty native response');
            }
            break;
        case 'native_reasoning':
            if (peer && record.text) {
                appendOutput('thought-' + peer, record.text, true, 'Reasoning returned by native parser');
            }
            break;
        case 'thought_message': showThought(record); break;
        // thought_capture is metadata; thought_result already has its verbatim
        // content and framing. Showing both would duplicate imported thoughts.
        case 'answer_start': if (peer) clearOutput('answer-' + peer); break;
        case 'answer': if (peer) appendOutput('answer-' + peer, record.text); break;
        case 'tool_call':
        case 'tool_result':
            if (peer) {
                showTool(peer, record);
                if (record.type === 'tool_result') outputBoundary('thought-' + peer, 'After tool result');
            }
            break;
        case 'state':
            // Replayed historical states must not roll current meters backwards.
            if (!state || (record.state?.last_event_id ?? id) >= (state.last_event_id ?? 0)) {
                stateSerial++;
                applyState(record.state);
            }
            break;
        case 'user': showUser(record); break;
        case 'reset': resetDisplay(); break;
        case 'error': notice('error', record.message || record.text || 'Controller error.'); break;
        case 'notice': notice('warning', record.message || ''); break;
        case 'gap': notice('warning', 'Earlier events were trimmed. Displayed history has gaps; model context is unchanged.'); break;
    }
}

async function refreshState() {
    if (refreshing) return;
    refreshing = true;
    const serial = stateSerial;
    const cursorAtStart = lastEvent;
    try {
        const response = await fetch('/state', { cache: 'no-store' });
        if (!response.ok) throw new Error('Status request failed (HTTP ' + response.status + ').');
        const next = await response.json();
        if (Number.isSafeInteger(next.last_event_id) && next.last_event_id < cursorAtStart) {
            lastEvent = 0;
            resetDisplay();
        }
        // A newer SSE state may have arrived while HTTP was in flight.
        if (serial === stateSerial || (next.last_event_id ?? 0) >= (state?.last_event_id ?? 0)) applyState(next);
    } finally { refreshing = false; }
}

function connectionStatus(online, text) {
    connected = online;
    byId('connection').textContent = text;
    byId('connection').classList.toggle('online', online);
    if (!online) for (const id of ['speed-A', 'speed-B', 'speed-total']) byId(id).textContent = '—';
    updateControls();
}

async function connect() {
    try { await refreshState(); } catch (error) { notice('error', error.message); }
    source = new EventSource('/events?after=' + lastEvent);
    source.onmessage = handleEvent;
    source.onopen = () => { retry = 500; connectionStatus(true, 'Connected'); };
    source.onerror = () => {
        source.close();
        source = null;
        connectionStatus(false, 'Disconnected / reconnecting...');
        reconnectTimer = window.setTimeout(connect, retry);
        retry = Math.min(retry * 2, 10000);
    };
}

async function command(path, body = {}) {
    if (pending) return false;
    pending = true;
    notice('error', '');
    updateControls();
    try {
        const response = await fetch(path, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
        });
        const text = await response.text();
        let result;
        try { result = JSON.parse(text); } catch (_) { result = null; }
        if (!response.ok) {
            const error = result?.error;
            throw new Error((typeof error === 'string' ? error : error?.message) ||
                result?.message || 'Request failed (HTTP ' + response.status + ').');
        }
        await refreshState();
        return true;
    } catch (error) {
        notice('error', error.message);
        return false;
    } finally { pending = false; updateControls(); }
}

for (const action of ['pause', 'resume', 'answer']) byId(action).addEventListener('click', () => command('/' + action));
byId('reset').addEventListener('click', () => {
    if (window.confirm('Start a new session? This clears both contexts and displayed history.')) command('/reset');
});
for (const button of document.querySelectorAll('[data-copy]')) {
    button.addEventListener('click', async () => {
        try {
            await copyText(outputs[button.dataset.copy].raw);
            button.textContent = 'Copied';
            setTimeout(() => { button.textContent = 'Copy raw'; }, 1200);
        } catch (error) { notice('error', error.message); }
    });
}
byId('message').addEventListener('input', updateControls);
byId('target').addEventListener('change', updateControls);
byId('message').addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !byId('send').disabled) {
        event.preventDefault();
        byId('message-form').requestSubmit();
    }
});
byId('message-form').addEventListener('submit', async event => {
    event.preventDefault();
    const text = byId('message').value.trim();
    const target = byId('target').value;
    if (!text || byId('send').disabled) return;
    if (await command('/message', { text, target })) {
        if (byId('message').value.trim() === text) byId('message').value = '';
        byId('message').focus();
        updateControls();
    }
});
// Generation can run for minutes without a boundary: metrics cannot depend on
// state events alone. Polls also decay rolling rates to zero while idle.
statePoll = window.setInterval(() => {
    if (connected) refreshState().catch(error => notice('error', error.message));
}, 1000);
window.addEventListener('beforeunload', () => {
    clearInterval(statePoll);
    clearTimeout(reconnectTimer);
    if (source) source.close();
});
connect();
