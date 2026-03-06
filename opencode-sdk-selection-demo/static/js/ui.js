import { state, resetTurnState } from './state.js';
import { renderSessionOptions } from './session.js';
import { createMarkdownRenderers, renderToolsList, renderTraceList } from './renderers.js';
import { esc } from './utils.js';

const STARTUP_FREEZE_MS = 300;

export function createUI(elements) {
  const {
    statusDot,
    statusText,
    submitBtn,
    sessionSelect,
    modelSelect,
    thinkingEl,
    responseEl,
    toolsListEl,
    traceListEl,
    toolMetaEl,
    responseMetaEl,
    thinkingAutoToggleEl,
    responseAutoToggleEl,
    toolsAutoToggleEl,
    traceAutoToggleEl,
  } = elements;

  const markdown = createMarkdownRenderers({ thinkingEl, responseEl });
  const startupFrozenUntil = performance.now() + STARTUP_FREEZE_MS;

  function createAutoScroller(containerEl) {
    const BOTTOM_THRESHOLD = 28;
    let enabled = false;
    let stickToBottom = true;
    let ignoreScrollEventsUntil = 0;
    let rafId = 0;

    function isNearBottom() {
      const remaining = containerEl.scrollHeight - containerEl.scrollTop - containerEl.clientHeight;
      return remaining <= BOTTOM_THRESHOLD;
    }

    function onUserIntent() {
      ignoreScrollEventsUntil = 0;
      if (!enabled) return;
      stickToBottom = isNearBottom();
    }

    function onScroll() {
      if (!enabled) {
        return;
      }
      if (performance.now() < ignoreScrollEventsUntil) {
        return;
      }
      stickToBottom = isNearBottom();
    }

    function schedule(force = false) {
      if (!force && performance.now() < startupFrozenUntil) {
        return;
      }

      if (!enabled) {
        return;
      }

      if (!force && !stickToBottom) {
        return;
      }

      if (rafId) {
        cancelAnimationFrame(rafId);
      }

      if (containerEl.scrollHeight <= containerEl.clientHeight + 1) {
        return;
      }

      rafId = requestAnimationFrame(() => {
        rafId = 0;
        ignoreScrollEventsUntil = performance.now() + 260;
        containerEl.scrollTo({
          top: containerEl.scrollHeight,
          behavior: 'smooth',
        });
      });
    }

    function reset() {
      stickToBottom = true;
      schedule(true);
    }

    function setEnabled(nextEnabled) {
      enabled = Boolean(nextEnabled);
      if (enabled) {
        stickToBottom = isNearBottom();
        schedule(true);
      }
    }

    containerEl.addEventListener('wheel', onUserIntent, { passive: true });
    containerEl.addEventListener('touchstart', onUserIntent, { passive: true });
    containerEl.addEventListener('mousedown', onUserIntent, { passive: true });
    containerEl.addEventListener('scroll', onScroll, { passive: true });

    return {
      schedule,
      reset,
      setEnabled,
    };
  }

  const autoScroll = {
    thinking: createAutoScroller(thinkingEl),
    response: createAutoScroller(responseEl),
    tools: createAutoScroller(toolsListEl),
    trace: createAutoScroller(traceListEl),
  };

  const autoToggleMap = {
    thinking: thinkingAutoToggleEl,
    response: responseAutoToggleEl,
    tools: toolsAutoToggleEl,
    trace: traceAutoToggleEl,
  };

  function updateAutoToggle(name, enabled) {
    const el = autoToggleMap[name];
    if (!el) return;
    el.setAttribute('aria-pressed', enabled ? 'true' : 'false');
    el.textContent = enabled ? 'Auto: On' : 'Auto: Off';
  }

  function setAutoScroll(name, enabled) {
    const scroller = autoScroll[name];
    if (!scroller) return;
    scroller.setEnabled(enabled);
    updateAutoToggle(name, enabled);
  }

  function initAutoScroll() {
    setAutoScroll('thinking', false);
    setAutoScroll('response', false);
    setAutoScroll('tools', true);
    setAutoScroll('trace', true);
  }

  function bindAutoToggle(name) {
    const el = autoToggleMap[name];
    if (!el) return;

    el.addEventListener('click', () => {
      const nextEnabled = el.getAttribute('aria-pressed') !== 'true';
      setAutoScroll(name, nextEnabled);
    });
  }

  function setStatus(stateName, text) {
    statusDot.className = 'dot ' + stateName;
    statusText.textContent = text;
  }

  function setBusy(busy, buttonLabel) {
    state.busy = busy;
    submitBtn.disabled = busy;
    sessionSelect.disabled = busy;
    modelSelect.disabled = busy;
    submitBtn.textContent = buttonLabel;
  }

  function renderThinking() {
    const text = Array.from(state.reasoningParts.values()).join('\n\n').trim();
    markdown.renderThinking(text);
    autoScroll.thinking.schedule();
  }

  function renderResponse() {
    const text = Array.from(state.responseParts.values()).join('\n\n').trim();
    markdown.renderResponse(text);
    autoScroll.response.schedule();
  }

  function renderTools() {
    renderToolsList({
      toolsMap: state.tools,
      toolsListEl,
      toolMetaEl,
    });
    autoScroll.tools.schedule();
  }

  function renderTrace() {
    renderTraceList({
      trace: state.trace,
      traceEl: traceListEl,
    });
    autoScroll.trace.schedule();
  }

  function pushTrace(label) {
    const stamp = new Date().toLocaleTimeString();
    state.trace.push(`${stamp} · ${label}`);
    if (state.trace.length > 240) state.trace.shift();
    renderTrace();
  }

  function resetTurn() {
    resetTurnState();
    autoScroll.thinking.reset();
    autoScroll.response.reset();
    autoScroll.tools.reset();
    autoScroll.trace.reset();
    renderThinking();
    renderResponse();
    renderTools();
    renderTrace();
    responseMetaEl.textContent = 'live';
  }

  function applyReadyPayload(payload) {
    state.models = payload.models || [];
    state.sessions = payload.sessions || [];
    state.currentSessionId = payload.current_session_id || null;
    state.defaultModel = payload.default_model || null;

    renderSessionOptions({ state, sessionSelect });

    if (!state.models.length) {
      modelSelect.innerHTML = '<option value="">No github-copilot models found</option>';
      setBusy(false, 'Submit');
      submitBtn.disabled = true;
      setStatus('error', 'No github-copilot model is available from OpenCode server.');
      return;
    }

    modelSelect.innerHTML = state.models.map((model) => {
      const label = model.name ? `${model.name} (${model.id})` : model.id;
      return `<option value="${esc(model.id)}">${esc(label)}</option>`;
    }).join('');

    const selected = state.defaultModel && state.models.some((m) => m.id === state.defaultModel)
      ? state.defaultModel
      : state.models[0].id;
    modelSelect.value = selected;
    setBusy(false, 'Submit');

    const diagnostics = payload.diagnostics || {};
    if (!diagnostics.github_copilot_auth_found) {
      setStatus('error', 'GitHub Copilot auth missing in local OpenCode auth.json');
    } else {
      setStatus('done', 'Ready');
    }
  }

  function refreshSessions() {
    renderSessionOptions({ state, sessionSelect });
  }

  return {
    setStatus,
    setBusy,
    resetTurn,
    renderThinking,
    renderResponse,
    renderTools,
    renderTrace,
    pushTrace,
    applyReadyPayload,
    refreshSessions,
    initAutoScroll,
    bindAutoToggle,
  };
}
