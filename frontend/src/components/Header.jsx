import { Trash2, Sun, Moon } from 'lucide-react';

/**
 * Header component — app branding, theme toggle, and clear conversation.
 */
export default function Header({ onClear, hasMessages, theme, onToggleTheme }) {
  const isDark = theme === 'dark';

  return (
    <header className="header">
      <div className="header-brand">
        <h1 className="header-title">Zucchini</h1>
        <p className="header-subtitle">Multilingual Intelligence Assistant</p>
      </div>

      <div className="header-actions">
        <button
          className="theme-toggle"
          onClick={onToggleTheme}
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? <Sun size={16} /> : <Moon size={16} />}
        </button>

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
      </div>
    </header>
  );
}
