import { state } from './state.js';
import { createUI } from './ui.js';
import { safeSendJson, isSocketOpen } from './socket.js';

const elements = {
  promptInput: document.getElementById('promptInput'),
  sessionSelect: document.getElementById('sessionSelect'),
  modelSelect: document.getElementById('modelSelect'),
  submitBtn: document.getElementById('submitBtn'),
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),
  thinkingEl: document.getElementById('thinkingStream'),
  responseEl: document.getElementById('responseStream'),
  toolsListEl: document.getElementById('toolsList'),
  traceListEl: document.getElementById('traceList'),
  toolMetaEl: document.getElementById('toolMeta'),
  responseMetaEl: document.getElementById('responseMeta'),
  thinkingAutoToggleEl: document.getElementById('thinkingAutoToggle'),
  responseAutoToggleEl: document.getElementById('responseAutoToggle'),
  toolsAutoToggleEl: document.getElementById('toolsAutoToggle'),
  traceAutoToggleEl: document.getElementById('traceAutoToggle'),
};

const ui = createUI(elements);
let promptLayoutFreezeRaf = 0;

function freezePromptHeightForTyping() {
  const promptPanelBody = document.querySelector('#promptPanel .panel-body');
  if (!promptPanelBody) {
    return;
  }

  if (promptLayoutFreezeRaf) {
    cancelAnimationFrame(promptLayoutFreezeRaf);
  }

  const fixed = Math.ceil(promptPanelBody.getBoundingClientRect().height);
  promptPanelBody.style.height = `${fixed}px`;
  promptPanelBody.style.minHeight = `${fixed}px`;
  promptPanelBody.style.maxHeight = `${fixed}px`;
}

function handleMessage(msg) {
  if (msg.type === 'ready') {
    ui.applyReadyPayload(msg);
    ui.pushTrace('ready payload received');
    return;
  }

  if (msg.type === 'turn_started') {
    ui.setBusy(true, 'Thinking');
    ui.setStatus('loading', 'Thinking...');
    ui.pushTrace(`turn started · session ${msg.session_id}`);
    if (msg.session_id) {
      state.currentSessionId = msg.session_id;
      ui.refreshSessions();
    }
    return;
  }

  if (msg.type === 'sessions_update') {
    state.sessions = msg.sessions || [];
    state.currentSessionId = msg.current_session_id || state.currentSessionId;
    ui.refreshSessions();
    return;
  }

  if (msg.type === 'phase') {
    if (msg.value === 'thinking') {
      ui.setBusy(true, 'Thinking');
      ui.setStatus('loading', 'Thinking...');
    } else if (msg.value === 'generating') {
      ui.setBusy(true, 'Generating');
      ui.setStatus('generating', 'Generating...');
    } else if (msg.value === 'done') {
      ui.setStatus('done', 'Done');
    } else if (msg.value === 'error') {
      ui.setStatus('error', 'Error');
    }
    ui.pushTrace(`phase -> ${msg.value}`);
    return;
  }

  if (msg.type === 'reasoning_update') {
    state.reasoningParts.set(msg.part_id, msg.text || '');
    ui.renderThinking();
    return;
  }

  if (msg.type === 'reasoning_delta') {
    const prev = state.reasoningParts.get(msg.part_id) || '';
    state.reasoningParts.set(msg.part_id, prev + (msg.delta || ''));
    ui.renderThinking();
    return;
  }

  if (msg.type === 'response_update') {
    state.responseParts.set(msg.part_id, msg.text || '');
    ui.renderResponse();
    return;
  }

  if (msg.type === 'response_delta') {
    const prev = state.responseParts.get(msg.part_id) || '';
    state.responseParts.set(msg.part_id, prev + (msg.delta || ''));
    ui.renderResponse();
    return;
  }

  if (msg.type === 'tool_update') {
    const existing = state.tools.get(msg.call_id) || {};
    state.tools.set(msg.call_id, {
      ...existing,
      ...msg,
    });
    ui.renderTools();
    ui.pushTrace(`tool ${msg.tool} -> ${msg.status}`);
    return;
  }

  if (msg.type === 'usage') {
    const tokens = msg.tokens || {};
    elements.responseMetaEl.textContent = `tokens in:${tokens.input || 0} out:${tokens.output || 0} reasoning:${tokens.reasoning || 0}`;
    return;
  }

  if (msg.type === 'assistant_summary') {
    const cost = typeof msg.cost === 'number' ? msg.cost.toFixed(5) : '0.00000';
    ui.pushTrace(`assistant summary · model ${msg.api_model_id} · cost ${cost}`);
    return;
  }

  if (msg.type === 'trace') {
    ui.pushTrace(msg.event_type || 'event');
    return;
  }

  if (msg.type === 'error') {
    ui.setStatus('error', msg.message || 'Error');
    ui.pushTrace(`error · ${msg.message || 'unknown'}`);
    return;
  }

  if (msg.type === 'turn_complete') {
    ui.setBusy(false, 'Submit');
    ui.pushTrace('turn complete');
  }
}

function connectSocket() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${location.host}/ws/stream`);
  state.socket = socket;

  socket.addEventListener('open', () => {
    ui.pushTrace('websocket connected');
  });

  socket.addEventListener('close', () => {
    ui.pushTrace('websocket disconnected');
    if (state.busy) {
      ui.setBusy(false, 'Submit');
      ui.setStatus('error', 'Connection lost');
    }
  });

  socket.addEventListener('message', (evt) => {
    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch {
      return;
    }
    handleMessage(msg);
  });
}

async function fetchModels() {
  const res = await fetch('/api/models');
  const payload = await res.json();
  ui.applyReadyPayload(payload);
}

function submitTurn() {
  if (state.busy) return;
  if (!isSocketOpen(state.socket)) {
    ui.setStatus('error', 'WebSocket is not connected');
    return;
  }

  const prompt = elements.promptInput.value.trim();
  const modelId = elements.modelSelect.value;
  if (!prompt) {
    ui.setStatus('error', 'Prompt cannot be empty');
    return;
  }
  if (!modelId) {
    ui.setStatus('error', 'Select a model first');
    return;
  }

  ui.resetTurn();
  ui.setBusy(true, 'Thinking');
  ui.setStatus('loading', 'Thinking...');
  ui.pushTrace('submit');

  const sent = safeSendJson(state.socket, {
    type: 'submit',
    prompt,
    session_id: elements.sessionSelect.value,
    model_id: modelId,
  });

  if (!sent) {
    ui.setBusy(false, 'Submit');
    ui.setStatus('error', 'Failed to send request');
  }
}

elements.submitBtn.addEventListener('click', submitTurn);

elements.promptInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    submitTurn();
  }
});

elements.promptInput.addEventListener('input', () => {
  if (promptLayoutFreezeRaf) {
    cancelAnimationFrame(promptLayoutFreezeRaf);
  }
  promptLayoutFreezeRaf = requestAnimationFrame(() => {
    promptLayoutFreezeRaf = 0;
    freezePromptHeightForTyping();
  });
});

(async function boot() {
  ui.setBusy(true, 'Loading');
  ui.setStatus('loading', 'Loading models...');
  ui.initAutoScroll();
  ui.bindAutoToggle('thinking');
  ui.bindAutoToggle('response');
  ui.bindAutoToggle('tools');
  ui.bindAutoToggle('trace');
  ui.renderTrace();
  ui.renderThinking();
  ui.renderResponse();
  freezePromptHeightForTyping();
  await fetchModels().catch(() => {
    ui.setBusy(false, 'Submit');
    ui.setStatus('error', 'Failed to load models');
  });
  connectSocket();
})();
