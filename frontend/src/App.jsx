import { useEffect, useState } from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import { sendMessage } from './services/api';

const THEME_KEY = 'zucchini-theme';

function getInitialTheme() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    /* ignore */
  }
  return 'light';
}

/**
 * App — main Zucchini application component.
 *
 * Manages conversation state, API communication,
 * and error handling.
 */
export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const handleSend = async (text) => {
    // Add user message to the conversation
    const userMessage = { role: 'user', content: text };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setLoading(true);
    setError(null);

    // Build conversation history for the API
    // (exclude the current message — it's sent as 'message')
    const history = messages.map(({ role, content }) => ({ role, content }));

    try {
      const data = await sendMessage(text, history);

      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        language: data.language,
      };
      setMessages([...updatedMessages, assistantMessage]);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="app-container">
      <Header
        onClear={handleClear}
        hasMessages={messages.length > 0}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)} title="Dismiss">×</button>
        </div>
      )}

      <ChatWindow messages={messages} loading={loading} />
      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}
