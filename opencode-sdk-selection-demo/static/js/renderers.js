import * as React from 'https://esm.sh/react@19.2.0';
import { createRoot } from 'https://esm.sh/react-dom@19.2.0/client';
import { Streamdown } from 'https://esm.sh/streamdown@2.3.0?deps=react@19.2.0,react-dom@19.2.0';

import { esc, stringify } from './utils.js';

const THINKING_PLACEHOLDER = 'Thinking output will stream here.';
const RESPONSE_PLACEHOLDER = 'Assistant response will stream here.';
const expandedToolCalls = new Set();
const boundToolLists = new WeakSet();

export function createMarkdownRenderers({ thinkingEl, responseEl }) {
  const thinkingRoot = createRoot(thinkingEl);
  const responseRoot = createRoot(responseEl);

  function renderMarkdown(root, markdown, placeholder) {
    const text = (markdown || '').trim();
    if (!text) {
      root.render(React.createElement('span', { className: 'hint' }, placeholder));
      return;
    }

    root.render(
      React.createElement(
        Streamdown,
        { mode: 'streaming' },
        text,
      ),
    );
  }

  function renderThinking(text) {
    renderMarkdown(thinkingRoot, text, THINKING_PLACEHOLDER);
  }

  function renderResponse(text) {
    renderMarkdown(responseRoot, text, RESPONSE_PLACEHOLDER);
  }

  return { renderThinking, renderResponse };
}

function bindToolToggles(toolsListEl) {
  if (boundToolLists.has(toolsListEl)) {
    return;
  }

  toolsListEl.addEventListener('click', (event) => {
    const toggleEl = event.target.closest('[data-tool-toggle]');
    if (!toggleEl) return;

    const cardEl = toggleEl.closest('[data-call-id]');
    if (!cardEl) return;

    const callId = cardEl.getAttribute('data-call-id');
    if (!callId) return;

    const nextExpanded = !expandedToolCalls.has(callId);
    if (nextExpanded) {
      expandedToolCalls.add(callId);
    } else {
      expandedToolCalls.delete(callId);
    }

    cardEl.classList.toggle('collapsed', !nextExpanded);
    cardEl.classList.toggle('expanded', nextExpanded);
    toggleEl.setAttribute('aria-expanded', String(nextExpanded));
    toggleEl.textContent = nextExpanded ? 'Collapse' : 'Expand';
  });

  boundToolLists.add(toolsListEl);
}

export function renderToolsList({ toolsMap, toolsListEl, toolMetaEl }) {
  bindToolToggles(toolsListEl);

  const tools = Array.from(toolsMap.values());
  toolMetaEl.textContent = tools.length + (tools.length === 1 ? ' call' : ' calls');

  if (!tools.length) {
    toolsListEl.innerHTML = '<div class="hint">Tool executions will appear here with status, input, metadata, and output.</div>';
    return;
  }

  toolsListEl.innerHTML = tools.map((tool, index) => {
    const callId = String(tool.call_id || tool.callID || `${tool.tool || 'tool'}-${index}`);
    const expanded = expandedToolCalls.has(callId);
    const input = stringify(tool.input);
    const metadata = stringify(tool.metadata);
    const output = stringify(tool.output);
    const error = stringify(tool.error);

    return `
      <article class="tool-card ${expanded ? 'expanded' : 'collapsed'}" data-call-id="${esc(callId)}">
        <div class="tool-head">
          <div class="tool-name">${esc(tool.tool || 'unknown')}</div>
          <div class="tool-badges">
            ${tool.is_mcp ? '<span class="badge">mcp</span>' : ''}
            <span class="badge ${esc(tool.status || 'pending')}">${esc(tool.status || 'pending')}</span>
            <button type="button" class="tool-toggle" data-tool-toggle aria-expanded="${expanded ? 'true' : 'false'}">${expanded ? 'Collapse' : 'Expand'}</button>
          </div>
        </div>
        <div class="tool-body">
          <div class="tool-chunk">
            <div class="tool-chunk-title">Input</div>
            <pre>${esc(input || '{}')}</pre>
          </div>
          <div class="tool-chunk">
            <div class="tool-chunk-title">Metadata (Streaming)</div>
            <pre>${esc(metadata || '{}')}</pre>
          </div>
          <div class="tool-chunk">
            <div class="tool-chunk-title">Output</div>
            <pre>${esc(output || '(pending)')}</pre>
          </div>
          ${error ? `
            <div class="tool-chunk">
              <div class="tool-chunk-title">Error</div>
              <pre>${esc(error)}</pre>
            </div>
          ` : ''}
        </div>
      </article>
    `;
  }).join('');
}

export function renderTraceList({ trace, traceEl }) {
  if (!trace.length) {
    traceEl.innerHTML = '<div class="trace-item">Waiting for stream events...</div>';
    return;
  }
  traceEl.innerHTML = trace.map((line) => `<div class="trace-item">${esc(line)}</div>`).join('');
}
