import { Trash2 } from 'lucide-react';

/**
 * Header component — app branding and clear conversation button.
 */
export default function Header({ onClear, hasMessages }) {
  return (
    <header className="header">
      <div className="header-brand">
        <h1 className="header-title">Zucchini</h1>
        <p className="header-subtitle">Multilingual Intelligence Assistant</p>
      </div>

      {hasMessages && (
        <button
          className="clear-button"
          onClick={onClear}
          title="Clear conversation"
        >
          <Trash2 size={14} />
          <span>Clear</span>
        </button>
      )}
    </header>
  );
}
