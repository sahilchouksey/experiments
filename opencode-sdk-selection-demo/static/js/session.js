import { esc } from './utils.js';

export function formatSessionLabel(session) {
  const title = (session.title || '').trim();
  const shortId = (session.id || '').slice(0, 12);
  return title ? `${title} (${shortId})` : (session.id || 'session');
}

export function renderSessionOptions({ state, sessionSelect }) {
  const sessions = state.sessions || [];
  const current = state.currentSessionId;
  const currentInList = Boolean(current && sessions.some((s) => s.id === current));

  const options = ['<option value="__new__">New Session</option>'];
  if (current && !currentInList) {
    const shortCurrent = current.slice(0, 12);
    options.push(`<option value="${esc(current)}">${esc(`Current (${shortCurrent})`)}</option>`);
  }

  for (const session of sessions) {
    const selected = current && current === session.id ? ' selected' : '';
    options.push(`<option value="${esc(session.id)}"${selected}>${esc(formatSessionLabel(session))}</option>`);
  }

  sessionSelect.innerHTML = options.join('');
  if (current && sessions.some((s) => s.id === current)) {
    sessionSelect.value = current;
    return;
  }
  sessionSelect.value = '__new__';
}
