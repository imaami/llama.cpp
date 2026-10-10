import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = dirname(fileURLToPath(import.meta.url));
const html = await readFile(join(root, 'dist.html'));
const clients = new Set();
const posts = [];
let id = 0;
let polls = 0;
const state = { mode: 'thinking', busy: false, tools: 1, last_event_id: 0,
    peers: ['A', 'B'].map(name => ({ name, tokens: 3200, context_size: 32000,
        generated: 110, thinking_tokens: 100, imported: 14,
        tokens_per_second: 22, thinking_tokens_per_second: 20 })) };
function emit(record) {
    record.id = ++id;
    state.last_event_id = id;
    for (const client of clients) client.write('id: ' + id + '\ndata: ' + JSON.stringify(record) + '\n\n');
}
const server = createServer(async (req, res) => {
    if (req.url.startsWith('/events')) {
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
        res.write(': connected\n\n');
        clients.add(res);
        req.on('close', () => clients.delete(res));
    } else if (req.url === '/state') {
        polls++;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(state));
    } else if (req.method === 'POST') {
        let body = '';
        for await (const chunk of req) body += chunk;
        posts.push({ path: req.url, ...JSON.parse(body || '{}') });
        res.setHeader('Content-Type', 'application/json');
        res.end('{}');
    } else {
        res.setHeader('Content-Type', 'text/html');
        res.end(html);
    }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
try {
    const context = await browser.newContext({ viewport: { width: 1800, height: 1250 } });
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:' + server.address().port);
    await page.waitForFunction(() => document.getElementById('connection').textContent === 'Connected');
    assert.equal(await page.locator('#count-A').textContent(), '110');
    assert.equal(await page.locator('#count-total').textContent(), '200');
    assert.equal(await page.locator('#speed-total').textContent(), '40.0');
    emit({ type: 'token', peer: 'A', text: '# A considers the problem\n\nA **stable paragraph**, with $x^2$ and \\(y+1\\).\n\n' });
    emit({ type: 'token', peer: 'B', text: '## B checks independently\n\n| Value | Result |\n| --- | --- |\n| 2 + 2 | **4** |\n\n~~~cpp\nint answer = 42;\n~~~\n\n' });
    await page.waitForSelector('#thought-A .katex');
    await page.waitForSelector('#thought-B .hljs-type');
    await page.evaluate(() => { window.stable = document.querySelector('#thought-A .markdown-block'); });
    emit({ type: 'token', peer: 'A', text: 'Another streamed thought.\n\n<ct:peek/>\n\n' });
    await page.waitForFunction(() => document.getElementById('thought-A').textContent.includes('<ct:peek/>'));
    assert.equal(await page.evaluate(() => window.stable === document.querySelector('#thought-A .markdown-block')), true);
    emit({ type: 'thought_capture', peer: 'A', source: 'B', text: 'DO NOT DUPLICATE', turn: 1, offset: 0, end: 16 });
    emit({ type: 'thought_result', peer: 'A', command: 'peek', text: '\n[THOUGHT DUMP BEGIN]\n## Captured thought\n\n**Verbatim** fragment mid-wo\n[THOUGHT DUMP END]\n' });
    await page.waitForSelector('#thought-A h2');
    assert.equal(await page.locator('#thread .bubble').count(), 0);
    assert.equal(await page.locator('#thought-A').textContent().then(text => text.includes('DO NOT DUPLICATE')), false);
    emit({ type: 'thought_message', from: 'A', to: 'B', message_id: 1, text: 'Try **factoring** $x^2-1$.\n\n~~~python\nprint(6 * 7)\n~~~', status: 'sent' });
    emit({ type: 'thought_message', from: 'B', to: 'A', message_id: 2, text: 'I get **42**.', status: 'sent' });
    emit({ type: 'thought_message', from: 'A', to: 'B', message_id: 1, text: 'Try **factoring** $x^2-1$.\n\n~~~python\nprint(6 * 7)\n~~~', status: 'received' });
    await page.waitForSelector('#thread .katex');
    assert.equal(await page.locator('#thread .bubble').count(), 2);
    assert.equal(await page.locator('#thread .from-A .bubble-state').textContent(), 'Received');
    emit({ type: 'tool_call', peer: 'A', name: 'compiler', call_id: 'test', arguments: '{"code":"int main() {}"}' });
    emit({ type: 'tool_result', peer: 'A', name: 'compiler', call_id: 'test', result: { content: [{ type: 'text', text: '**Success**: $a=b$\n\n~~~text\nexit 0\n~~~' }] } });
    await page.locator('.peer-a .tools-wrap:not(.native-wrap)').evaluate(node => { node.open = true; });
    await page.waitForSelector('#tools-A .katex');
    emit({ type: 'answer', peer: 'A', text: '**Answer:** $42$\n\n~~~js\nconsole.log(42);\n~~~' });
    emit({ type: 'answer', peer: 'B', text: '## Done\n\nThe result is **42**.' });
    await page.waitForSelector('#answer-A .katex');
    await page.waitForSelector('#answer-B strong');
    const malicious = '<img src=x onerror="window.pwned=1"><script>window.pwned=2</script>\n\n[bad](javascript:alert(1))';
    emit({ type: 'token', peer: 'B', text: malicious + '\n\n~~~mermaid\nflowchart TD\n A[Think] --> B[Check inbox]\n~~~\n\n' });
    await page.waitForSelector('#thought-B .diagram');
    assert.equal(await page.evaluate(() => window.pwned), undefined);
    assert.equal(await page.locator('#thought-B a[href^="javascript:"]').count(), 0);
    assert.equal(await page.locator('#thought-B img[onerror]').count(), 0);
    assert.equal(await page.locator('#thought-B .diagram').evaluate(node => !!node.shadowRoot.querySelector('svg')), true);
    assert.equal(await page.locator('#thought-B .diagram').evaluate(node => node.shadowRoot.textContent.includes('Check inbox')), true);
    emit({ type: 'token', peer: 'A', text: '\n\n' + 'A longer line for scroll verification.\n\n'.repeat(55) });
    await page.waitForFunction(() => document.getElementById('thought-A').scrollHeight > 1500);
    await page.locator('#thought-A').evaluate(node => { node.scrollTop = 0; node.dispatchEvent(new Event('scroll')); });
    assert.equal(await page.locator('#follow-A').isChecked(), true, 'scroll must not disable Follow');
    await page.locator('#follow-A').uncheck();
    await page.locator('#thought-A').evaluate(node => { node.scrollTop = 100; });
    emit({ type: 'token', peer: 'A', text: '\n\nThis arrives while Follow is off.\n' });
    await page.waitForFunction(() => document.getElementById('thought-A').textContent.includes('Follow is off'));
    assert.equal(await page.locator('#thought-A').evaluate(node => node.scrollTop), 100);
    await page.locator('#follow-A').check();
    assert.equal(await page.locator('#thought-A').evaluate(node => node.scrollHeight - node.scrollTop - node.clientHeight < 5), true);
    // A quiet stream still needs rolling /state updates.
    state.peers[0].generated = 150;
    state.peers[0].tokens_per_second = 0;
    state.peers[0].thinking_tokens_per_second = 0;
    state.peers[1].tokens_per_second = 0;
    state.peers[1].thinking_tokens_per_second = 0;
    state.peers[0].phase = 'native_output';
    state.peers[0].request_generated = 81;
    state.peers[0].request_elapsed_seconds = 4.5;
    state.peers[0].last_token_age_seconds = 0.2;
    state.peers[1].phase = 'waiting_reasoning';
    state.peers[1].request_generated = 0;
    state.peers[1].request_elapsed_seconds = 3;
    state.peers[1].last_token_age_seconds = null;
    await page.waitForFunction(() => document.getElementById('count-A').textContent === '150');
    assert.equal(await page.locator('#speed-total').textContent(), '0.0');
    assert.equal(await page.locator('#stats-A .peer-phase').textContent(), 'Generating answer/tool call (buffered)');
    assert.match(await page.locator('#stats-A').textContent(), /Request 81 tokens \/ 4\.5 sLast token 0\.2 s ago/);
    assert.equal(await page.locator('#stats-B .peer-phase').textContent(), 'Waiting for server (reasoning)');
    assert.doesNotMatch(await page.locator('#stats-B').textContent(), /Last token/);
    assert.ok(polls > 1);
    emit({ type: 'user', target: 'A', text: 'Only you: double-check.' });
    await page.waitForFunction(() => document.getElementById('messages').textContent.includes('YOU → A'));
    assert.equal(await page.locator('#answer-B strong').textContent(), '42');
    await page.locator('#target').selectOption('B');
    await page.locator('#message').fill('A private instruction');
    await page.locator('#send').click();
    assert.deepEqual(posts.at(-1), { path: '/message', text: 'A private instruction', target: 'B' });

    // A new user turn must not inherit the previous turn's open C fence.
    emit({ type: 'reset' });
    const unfinished = 'Terminal size:\n\n```c\nstatic int get_size(int *cols, int *rows) {\n' +
        '    struct winsize ws;\n    *cols = (env';
    const otherThought = 'B keeps its **independent** turn.\n\n';
    emit({ type: 'token', peer: 'A', text: unfinished });
    emit({ type: 'token', peer: 'B', text: otherThought });
    await page.waitForFunction(() => document.querySelector('#thought-A pre')?.textContent.includes('*cols = (env'));
    await page.waitForSelector('#thought-B strong');
    await page.evaluate(() => { window.otherTurn = document.querySelector('#thought-B .output-segment'); });
    emit({ type: 'user', target: 'A', text: 'Send a thought message, then check your inbox.' });
    const newThought = 'I will send a **thought**, then check the inbox with <ct:inbox/>.\n';
    emit({ type: 'token', peer: 'A', text: newThought });
    const inbox = '\n<ct:result command="inbox">\nInbox empty.\n</ct:result>\n';
    emit({ type: 'thought_result', peer: 'A', command: 'inbox', text: inbox });
    await page.waitForFunction(() => document.querySelector('#thought-A .isolated-output')?.textContent.includes('Inbox empty.'));
    assert.equal(await page.locator('#thought-A .output-boundary').textContent(), 'New user message');
    assert.equal(await page.locator('#thought-A strong').textContent(), 'thought');
    assert.equal(await page.locator('#thought-A pre').count(), 1);
    assert.equal(await page.locator('#thought-A pre').textContent().then(text => text.includes('<ct:inbox/>')), false);
    assert.equal(await page.locator('#thought-A .isolated-output').textContent().then(text => text.includes('</ct:result>')), true);
    assert.equal(await page.evaluate(() => window.otherTurn === document.querySelector('#thought-B .output-segment')), true);
    assert.equal(await page.locator('#thought-B .output-boundary').count(), 0);

    // Imports and native tool records are independently rendered documents.
    const capture = '\n<ct:result command="peek">\n```cpp\nint unfinished =';
    emit({ type: 'thought_result', peer: 'A', command: 'peek', text: capture });
    const continuation = '\n## My own thoughts resume\n\n**Visible prose**, $x+1$, and <ct:inbox/>.\n';
    emit({ type: 'token', peer: 'A', text: continuation });
    await page.waitForSelector('#thought-A h2');
    assert.equal(await page.locator('#thought-A h2').textContent(), 'My own thoughts resume');
    assert.equal(await page.locator('#thought-A .output-segment:last-child pre').count(), 0);
    assert.equal(await page.locator('#thought-A .output-segment:last-child .katex').count(), 1);
    await page.locator('[data-copy="thought-A"]').click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), unfinished + newThought + inbox + capture + continuation,
        'Copy raw preserves received text exactly, without UI dividers or synthesized fence delimiters');
    emit({ type: 'token', peer: 'A', text: '\n```c\nint before_tool =' });
    emit({ type: 'tool_call', peer: 'A', name: 'compiler', arguments: '{"code":"int main() {}"}' });
    emit({ type: 'tool_result', peer: 'A', name: 'compiler', result: '```text\nunclosed diagnostics' });
    emit({ type: 'tool_result', peer: 'A', name: 'calculator', result: '**Separate result**: $42$' });
    await page.waitForSelector('#tools-A strong');
    assert.equal(await page.locator('#tools-A strong').textContent(), 'Separate result');
    assert.equal(await page.locator('#tools-A .output-segment').count(), 3);
    emit({ type: 'token', peer: 'A', text: '\n**Fresh thought** after native tools.\n' });
    await page.waitForFunction(() => document.querySelector('#thought-A .output-segment:last-child strong')?.textContent === 'Fresh thought');
    assert.equal(await page.locator('#thought-A .output-segment:last-child .output-boundary').textContent(), 'After tool result');
    assert.equal(await page.locator('#thought-A .output-segment:last-child pre').count(), 0);
    assert.equal(await page.evaluate(() => window.otherTurn === document.querySelector('#thought-B .output-segment')), true);
    assert.equal(await page.locator('#thought-B .output-boundary').count(), 0);

    // A completed native turn may contain reasoning but no user-facing answer.
    const nativeReasoning = '**Parsed reasoning** mentions <ct:inbox/>.\n\n```cpp\nint unfinished =';
    const nativeRaw = '<think>\n' + nativeReasoning + '\n</think>\n<ct:send>not executed</ct:send>';
    const nativeRawNext = '```xml\n<ct:peek/>';
    const postsBeforeNative = posts.length;
    emit({ type: 'native_output', peer: 'A', text: nativeRaw });
    emit({ type: 'native_reasoning', peer: 'A', text: nativeReasoning });
    emit({ type: 'native_output', peer: 'A', text: nativeRawNext });
    emit({ type: 'native_reasoning', peer: 'A', text: '**Separate native reasoning** with <ct:peek/>.' });
    emit({ type: 'native_output', peer: 'B', text: '' });
    emit({ type: 'answer', peer: 'B', text: '**Normal answer** remains visible.' });
    state.mode = 'incomplete';
    state.peers[0].phase = 'empty_response';
    state.peers[1].phase = 'done';
    emit({ type: 'state', state });
    await page.waitForFunction(() => document.getElementById('mode-text').textContent === 'Incomplete');
    assert.equal(await page.locator('#stats-A .peer-phase').textContent(), 'Finished without an answer');
    assert.equal(await page.locator('#answer-A').getAttribute('data-empty'),
        'This model finished without producing an answer. Inspect Native response below.');
    assert.equal(await page.locator('#answer-A .output-segment').count(), 0);
    assert.equal(await page.locator('#answer-B strong').textContent(), 'Normal answer');
    assert.equal(await page.locator('#tool-status-A').textContent(), '0 calls');
    assert.equal(await page.locator('#thread .bubble').count(), 0);
    assert.equal(posts.length, postsBeforeNative, 'displaying native tags must never execute commands');
    assert.equal(await page.locator('.peer-a .native-wrap').getAttribute('open'), null);
    await page.locator('.peer-a .native-wrap').evaluate(node => { node.open = true; });
    await page.waitForFunction(() => document.querySelectorAll('#native-A pre').length === 2);
    assert.equal(await page.locator('#native-A .output-segment').count(), 2);
    assert.equal(await page.locator('#native-A pre').first().textContent(), nativeRaw + '\n');
    assert.equal(await page.locator('#native-A pre').last().textContent(), nativeRawNext + '\n');
    await page.locator('[data-copy="native-A"]').click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), nativeRaw + nativeRawNext,
        'Native Copy raw preserves tags and unfinished fences without synthesized delimiters');
    await page.waitForFunction(() => document.querySelector('#thought-A .output-segment:last-child strong')?.textContent === 'Separate native reasoning');
    assert.equal(await page.locator('#thought-A .output-segment:last-child .output-boundary').textContent(),
        'Reasoning returned by native parser');
    assert.equal(await page.locator('#thought-A .output-segment:last-child pre').count(), 0);
    assert.equal(await page.locator('#thought-A .output-segment:last-child').textContent().then(text => text.includes('<ct:peek/>')), true);
    assert.equal(await page.locator('#native-B .output-boundary').textContent(), 'Empty native response');

    await page.screenshot({ path: join(root, 'test-desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 700, height: 950 });
    await page.screenshot({ path: join(root, 'test-mobile.png'), fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    // Clipping must bound both retained raw text and obsolete segment DOM.
    emit({ type: 'token', peer: 'A', text: '\n\n' + 'x'.repeat(512 * 1024) });
    await page.waitForFunction(() => !document.getElementById('clipped-A').hidden);
    assert.equal(await page.locator('#thought-A .output-segment').count(), 1);
    await page.locator('[data-copy="thought-A"]').click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), 'x'.repeat(512 * 1024));
    emit({ type: 'answer', peer: 'A', text: '**Recovered answer.**' });
    state.mode = 'answered';
    state.peers[0].phase = 'done';
    emit({ type: 'state', state });
    await page.waitForFunction(() => document.getElementById('mode-text').textContent === 'answered');
    await page.waitForSelector('#answer-A strong');
    assert.equal(await page.locator('#answer-A strong').textContent(), 'Recovered answer.');
    assert.doesNotMatch(await page.locator('#answer-A').getAttribute('data-empty'), /without producing an answer/);
    emit({ type: 'reset' });
    await page.waitForFunction(() => !document.querySelector('#native-A .output-segment'));
    assert.equal(await page.locator('#native-B .output-segment').count(), 0);
    assert.deepEqual(errors, []);
    console.log('UI browser regressions passed: rich output, sanitization, diagrams, streaming, follow, targeting, inbox, metrics, turn/result isolation, native response diagnostics, verbatim copy, clipping.');
} finally {
    await browser.close();
    for (const client of clients) client.end();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
}
