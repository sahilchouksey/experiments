export function safeSendJson(socket, payload) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return false;
  }

  try {
    socket.send(JSON.stringify(payload));
    return true;
  } catch {
    return false;
  }
}

export function isSocketOpen(socket) {
  return Boolean(socket && socket.readyState === WebSocket.OPEN);
}
