/**
 * MessageBubble — renders a single user or assistant message.
 *
 * Assistant messages display a small language badge showing
 * the detected language of the user's query.
 */
export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`message-row ${isUser ? 'message-row-user' : 'message-row-assistant'}`}>
      <div className={`message-bubble ${isUser ? 'message-user' : 'message-assistant'}`}>
        <div className="message-content">
          {message.content}
        </div>
        {!isUser && message.language && (
          <span className="language-badge">{message.language}</span>
        )}
      </div>
    </div>
  );
}
