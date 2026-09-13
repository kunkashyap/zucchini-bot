import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';

/**
 * ChatWindow — scrollable message list with auto-scroll,
 * empty state, and loading indicator.
 */
export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive or loading changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="chat-window">
        <div className="empty-state">
          <div className="empty-state-icon">⌘</div>
          <p className="empty-state-title">Start a conversation</p>
          <p className="empty-state-hint">
            Ask questions in English, Hindi, or Hinglish.
            Zucchini will detect your language and respond accordingly.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-window">
      {messages.map((msg, index) => (
        <MessageBubble key={index} message={msg} />
      ))}

      {loading && (
        <div className="loading-indicator">
          <div className="loading-dot" />
          <div className="loading-dot" />
          <div className="loading-dot" />
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
