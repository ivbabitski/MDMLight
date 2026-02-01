const KEY = "mdm_auth_v1";

export function loadAuth() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function getUserId() {
  return loadAuth()?.user_id ?? null;
}

export function getUserHeaders() {
  const userId = getUserId();
  return userId ? { "X-User-Id": String(userId) } : {};
}

export function saveAuth(auth) {
  localStorage.setItem(KEY, JSON.stringify(auth));
}

export function clearAuth() {
  localStorage.removeItem(KEY);
}
