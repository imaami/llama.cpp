// Same remark/rehype, GFM, KaTeX and highlight.js stack as tools/ui.
// Upstream preprocessing is imported directly, so fixes apply to both UIs.
import { remark } from 'remark';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkBreaks from 'remark-breaks';
import remarkRehype from 'remark-rehype';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import { all } from 'lowlight';
import { visit } from 'unist-util-visit';
import DOMPurify from 'dompurify';
import mermaid from 'mermaid';
import 'katex/contrib/mhchem';
import { preprocessLaTeX } from '../../ui/src/lib/utils/latex-protection';
import { remarkLiteralHtml } from '../../ui/src/lib/components/app/content/MarkdownContent/plugins/remark/literal-html';

// CommonMark sees <ct:peek/> as a URI autolink rather than an HTML tag.
// Keep our reserved reasoning commands visibly bracketed, including inline use.
function literalThoughtCommands() {
    return tree => visit(tree, 'link', (node, index, parent) => {
        if (parent && /^ct:/.test(node.url) && node.children.length === 1 &&
            node.children[0].type === 'text' && node.children[0].value === node.url) {
            parent.children[index] = { type: 'text', value: '<' + node.url + '>' };
        }
    });
}
const processor = remark().use(remarkGfm).use(remarkMath).use(remarkBreaks).use(literalThoughtCommands)
    .use(remarkLiteralHtml).use(remarkRehype).use(rehypeKatex, { trust: false })
    .use(rehypeHighlight, { languages: all, aliases: { xml: ['svelte', 'vue'] }, detect: false })
    .use(rehypeStringify);
mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'dark', htmlLabels: false, flowchart: { htmlLabels: false } });
let diagramId = 0;

// Disallow SMIL URL changes, mirroring upstream sanitize-svg.ts.
DOMPurify.addHook('uponSanitizeAttribute', (_, data) => {
    if (data.attrName === 'attributename' && /href$/i.test(data.attrValue.trim())) data.keepAttr = false;
});

export async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return;
    }
    // GPU machines are commonly reached over plain LAN HTTP.
    const area = document.createElement('textarea');
    area.value = text;
    area.style.cssText = 'position:fixed;top:-10000px;left:0';
    document.body.append(area);
    area.select();
    const copied = document.execCommand('copy');
    area.remove();
    if (!copied) throw new Error('Clipboard unavailable in this browser.');
}

function action(label, callback) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click', async () => {
        try { await callback(button); } catch (_) { button.textContent = 'Action failed'; }
    });
    return button;
}

function previewHTML(code) {
    const dialog = document.createElement('dialog');
    dialog.className = 'preview-dialog';
    const close = action('Close preview', () => dialog.close());
    const iframe = document.createElement('iframe');
    iframe.title = 'Sandboxed HTML preview';
    iframe.setAttribute('sandbox', '');
    iframe.srcdoc = '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:; font-src data:">' + code;
    dialog.append(close, iframe);
    dialog.addEventListener('close', () => dialog.remove());
    document.body.append(dialog);
    dialog.showModal();
}

async function decorate(block, node, source, afterLayout) {
    for (const link of block.querySelectorAll('a')) {
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
    }
    for (const table of block.querySelectorAll('table')) {
        const wrap = document.createElement('div');
        wrap.className = 'table-scroll';
        table.replaceWith(wrap);
        wrap.append(table);
    }
    for (const pre of block.querySelectorAll('pre')) {
        const code = pre.querySelector('code');
        if (!code) continue;
        const raw = node.type === 'code' ? node.value : code.textContent;
        const language = [...code.classList].find(c => c.startsWith('language-'))?.slice(9) || 'text';
        const wrapper = document.createElement('div');
        wrapper.className = 'code-wrapper';
        const header = document.createElement('div');
        header.className = 'code-header';
        const label = document.createElement('span');
        label.textContent = language;
        header.append(label, action('Copy', async button => {
            await copyText(raw);
            button.textContent = 'Copied';
            setTimeout(() => { button.textContent = 'Copy'; }, 1200);
        }));
        if (['html', 'htm'].includes(language)) header.append(action('Preview', () => previewHTML(raw)));
        pre.replaceWith(wrapper);
        wrapper.append(header, pre);
        const complete = node.type === 'code' && /(?:\n|^)\s*(\x60{3,}|~{3,})\s*$/.test(source);
        if (!complete) continue;
        if (language === 'mermaid' || language === 'svg' || (language === 'xml' && raw.trimStart().startsWith('<svg'))) {
            try {
                const svg = language === 'mermaid' ? (await mermaid.render('ct-diagram-' + (++diagramId), raw)).svg : raw;
                if (!block.isConnected) return;
                const clean = DOMPurify.sanitize(svg, {
                    USE_PROFILES: { svg: true, svgFilters: true },
                    FORBID_TAGS: ['foreignObject', 'animate', 'set', ...(language === 'mermaid' ? [] : ['style'])]
                });
                const view = document.createElement('div');
                view.className = 'diagram';
                // Shadow DOM stops generated IDs/classes affecting UI chrome.
                view.attachShadow({ mode: 'open' }).innerHTML = clean;
                pre.hidden = true;
                wrapper.append(view);
                header.append(action('Source', button => {
                    pre.hidden = !pre.hidden;
                    view.hidden = !pre.hidden;
                    button.textContent = pre.hidden ? 'Source' : 'Diagram';
                    afterLayout();
                }));
                afterLayout();
            } catch (_) {
                // Invalid diagrams remain highlighted source.
            }
        }
    }
    for (const image of block.querySelectorAll('img')) image.addEventListener('load', afterLayout, { once: true });
}

export class RichOutput {
    constructor(element, afterLayout = () => {}) {
        this.element = element;
        this.afterLayout = afterLayout;
        this.blocks = [];
        this.version = 0;
        this.pending = null;
        this.running = false;
        this.raw = '';
    }

    clear() {
        this.version++;
        this.pending = null;
        this.raw = '';
        this.blocks = [];
        this.element.replaceChildren();
    }

    update(text) {
        this.raw = text;
        this.pending = text;
        if (this.running) return;
        this.running = true;
        // Coalesce tokens and preserve stable block DOM, selection, code scroll.
        setTimeout(() => this.flush(), 100);
    }

    async flush() {
        try {
            while (this.pending !== null) {
                const raw = this.pending;
                this.pending = null;
                const version = this.version;
                const source = preprocessLaTeX(raw);
                const tree = processor.parse(source);
                const definitions = tree.children.filter(n => n.type === 'definition');
                const definitionKey = JSON.stringify(definitions);
                const next = [];
                for (const node of tree.children) {
                    const slice = source.slice(node.position.start.offset, node.position.end.offset);
                    const key = JSON.stringify(node) + definitionKey;
                    const old = this.blocks[next.length];
                    if (old?.key === key) { next.push(old); continue; }
                    const transformed = await processor.run({ type: 'root', children: [node, ...definitions] });
                    if (this.version !== version) break;
                    const block = document.createElement('div');
                    block.className = 'markdown-block';
                    block.innerHTML = DOMPurify.sanitize(processor.stringify(transformed), {
                        USE_PROFILES: { html: true, mathMl: true, svg: true },
                        FORBID_TAGS: ['style', 'iframe', 'form', 'object', 'embed', 'animate', 'set'],
                        ADD_ATTR: ['xmlns']
                    });
                    next.push({ key, block, node, slice, fresh: true });
                }
                if (this.version !== version) continue;
                for (let i = 0; i < next.length; i++) {
                    if (this.element.children[i] !== next[i].block) {
                        this.element.insertBefore(next[i].block, this.element.children[i] || null);
                    }
                }
                while (this.element.children.length > next.length) this.element.lastChild.remove();
                this.blocks = next;
                for (const entry of next) {
                    if (entry.fresh) {
                        entry.fresh = false;
                        void decorate(entry.block, entry.node, entry.slice, this.afterLayout);
                    }
                }
                this.afterLayout();
                if (this.pending !== null) await new Promise(resolve => requestAnimationFrame(resolve));
            }
        } catch (_) {
            // Renderer failures must not turn untrusted model text into HTML.
            this.element.textContent = this.raw;
        } finally {
            this.running = false;
        }
    }
}
