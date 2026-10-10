# Crossthink browser UI

The coordinator serves a self-contained, offline UI from the checked-in
crossthink-ui.generated.h. Normal C++ builds do not need Node, npm, a CDN,
or an installed upstream UI.

To edit the UI:

~~~sh
cd tools/crossthink/ui
npm ci
npm run build
~~~

Commit the source changes, generated header, lockfile, and regenerated
THIRD_PARTY_LICENSES.txt together. dist.html is a local build/test artifact.
The build uses pinned dependencies and bundles JavaScript, CSS and KaTeX WOFF2
fonts into the header. Mermaid's dynamically imported diagram types are bundled
too; the browser does not download rendering assets.

For browser regressions:

~~~sh
npx playwright install chromium
npm test
~~~

Tests use a local mock HTTP/SSE coordinator, checking rich rendering throughout,
model-output sanitization, visible command markers, stable streamed blocks,
explicit Follow controls, recipient selection, message delivery status, and
rolling metrics without generation boundaries.

## Upstream reuse

The renderer follows tools/ui's remark/rehype pipeline and dependency versions:
GFM tables/task lists/strikethrough, line breaks, KaTeX, all highlight.js languages,
Mermaid diagrams, SVG rendering, and code copy/HTML preview. It imports upstream
latex-protection.ts, remark/literal-html.ts, and their constants directly.
It preserves stable rendered block DOM while coalescing token updates.

Raw HTML in model text is displayed literally, all generated markdown HTML is
sanitized, and Mermaid runs in strict mode. SVG/diagram rendering is isolated in
shadow roots; HTML code previews use an opaque sandbox with scripts and network
requests blocked. Copy raw uses the original displayed source, independent of
rendered DOM. LAN HTTP clipboard fallback is supported.

licenses/remark-math.txt is the monorepo MIT license from
https://github.com/remarkjs/remark-math/blob/main/license, included because the
published remark-math/rehype-katex packages omit their license file. All other
bundled package licenses are collected directly from installed packages.

## UI semantics

Follow preferences change only on checkbox interaction, never on scrolling or
asynchronous markdown layout. The individual panels stay visible. Thought
messages are shown once with sent/received status; capture metadata is not shown
again because thought_result includes the full injected text in the recipient's
reasoning panel.

Per-agent meters use generated token counts and five-second rates from /state.
The center meter sums reasoning tokens/rates only. Imported tokens remain a
separate context statistic. Status is polled every second so rates update during
long continuous requests and decay when idle.
