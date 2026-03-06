export const state = {
  socket: null,
  busy: false,
  models: [],
  sessions: [],
  currentSessionId: null,
  defaultModel: null,
  reasoningParts: new Map(),
  responseParts: new Map(),
  tools: new Map(),
  trace: [],
};

export function resetTurnState() {
  state.reasoningParts = new Map();
  state.responseParts = new Map();
  state.tools = new Map();
  state.trace = [];
}
