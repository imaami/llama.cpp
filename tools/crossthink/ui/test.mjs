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
    await page.locator('.peer-a details').evaluate(node => { node.open = true; });
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
    await page.waitForFunction(() => document.getElementById('count-A').textContent === '150');
    assert.equal(await page.locator('#speed-total').textContent(), '0.0');
    assert.ok(polls > 1);
    emit({ type: 'user', target: 'A', text: 'Only you: double-check.' });
    await page.waitForFunction(() => document.getElementById('messages').textContent.includes('YOU → A'));
    assert.equal(await page.locator('#answer-B strong').textContent(), '42');
    await page.locator('#target').selectOption('B');
    await page.locator('#message').fill('A private instruction');
    await page.locator('#send').click();
    assert.deepEqual(posts.at(-1), { path: '/message', text: 'A private instruction', target: 'B' });
    await page.screenshot({ path: join(root, 'test-desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 700, height: 950 });
    await page.screenshot({ path: join(root, 'test-mobile.png'), fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    assert.deepEqual(errors, []);
    console.log('UI browser regressions passed: rich output, sanitization, diagrams, streaming, follow, targeting, inbox, metrics.');
} finally {
    await browser.close();
    for (const client of clients) client.end();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
}
