/**
 * Zucchini — API Service
 *
 * Centralized API communication utility.
 * All backend calls go through this module.
 */

const API_BASE = '/api';
const TIMEOUT_MS = 60000; // 60 seconds — LLM responses can be slow

/**
 * Send a chat message to the backend.
 *
 * @param {string} message - The user's message text.
 * @param {Array<{role: string, content: string}>} history - Conversation history.
 * @returns {Promise<{response: string, language: string, metadata: object}>}
 */
export async function sendMessage(message, history = []) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Server error (${res.status})`
      );
    }

    return await res.json();
  } catch (err) {
    clearTimeout(timeoutId);

    if (err.name === 'AbortError') {
      throw new Error('Request timed out. The server may be busy — please try again.');
    }
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      throw new Error('Cannot connect to the server. Make sure the backend is running.');
    }
    throw err;
  }
}

/**
 * Check backend health status.
 * @returns {Promise<{status: string}>}
 */
export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error('Health check failed');
  }
  return await res.json();
}
