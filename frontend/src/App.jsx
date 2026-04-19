import { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

const API = "http://localhost:5000";

// ─── Helpers ────────────────────────────────────────────────────────────────
const riskMeta = {
  low:    { label: "Low Risk",    color: "#4caf50", bg: "#e8f5e9", icon: "🛡️" },
  medium: { label: "Medium Risk", color: "#f59e0b", bg: "#fffde7", icon: "⚠️" },
  high:   { label: "High Risk",   color: "#ef4444", bg: "#ffebee", icon: "🚨" },
};

function highlightKeywords(text, keywords) {
  if (!keywords || keywords.length === 0) return [{ type: "text", content: text }];
  const escaped = keywords.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp(`(${escaped.join("|")})`, "gi");
  const parts = text.split(regex);
  return parts.map((part, i) => ({
    type: keywords.some(k => k.toLowerCase() === part.toLowerCase()) ? "highlight" : "text",
    content: part,
    key: i,
  }));
}

function HighlightedText({ text, keywords }) {
  const parts = highlightKeywords(text, keywords);
  return (
    <span>
      {parts.map(p =>
        p.type === "highlight" ? (
          <mark key={p.key} className="spam-word">{p.content}</mark>
        ) : (
          <span key={p.key}>{p.content}</span>
        )
      )}
    </span>
  );
}

function ConfidenceBar({ score, prediction }) {
  const color = prediction === "spam"
    ? score > 70 ? "#ef4444" : score > 40 ? "#f59e0b" : "#4caf50"
    : "#4caf50";
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-track">
        <div
          className="conf-bar-fill"
          style={{ width: `${score}%`, background: color }}
        />
      </div>
      <span className="conf-bar-label" style={{ color }}>{score}%</span>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [emailText, setEmailText]     = useState("");
  const [result, setResult]           = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState("");
  const [history, setHistory]         = useState([]);
  const [stats, setStats]             = useState({ total: 0, spam: 0, ham: 0 });
  const [darkMode, setDarkMode]       = useState(false);
  const [activeTab, setActiveTab]     = useState("analyze"); // analyze | history | analytics
  const fileInputRef = useRef(null);

  // Load history + stats on mount
  useEffect(() => {
    fetchHistory();
    fetchStats();
  }, []);

  async function fetchHistory() {
    try {
      const r = await fetch(`${API}/history`);
      const d = await r.json();
      setHistory(d);
    } catch (_) {}
  }

  async function fetchStats() {
    try {
      const r = await fetch(`${API}/stats`);
      const d = await r.json();
      setStats(d);
    } catch (_) {}
  }

  async function analyzeEmail() {
    if (!emailText.trim()) {
      setError("Please enter some email text to analyze.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const resp = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: emailText }),
      });
      if (!resp.ok) throw new Error("Server error");
      const data = await resp.json();
      setResult(data);
      fetchHistory();
      fetchStats();
    } catch (e) {
      setError("Could not reach the backend. Make sure it's running on port 5000.");
    } finally {
      setLoading(false);
    }
  }

  function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => setEmailText(ev.target.result);
    reader.readAsText(file);
    e.target.value = "";
  }

  async function deleteRecord(id) {
    await fetch(`${API}/history/${id}`, { method: "DELETE" });
    fetchHistory();
    fetchStats();
  }

  const spamPct = stats.total ? Math.round((stats.spam / stats.total) * 100) : 0;

  return (
    <div className={`app-root${darkMode ? " dark" : ""}`}>
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="logo">
          <span className="logo-icon">🛡️</span>
          <span className="logo-text">SpamShield</span>
        </div>
        <nav className="sidebar-nav">
          {["analyze","history","analytics"].map(tab => (
            <button
              key={tab}
              className={`nav-btn${activeTab === tab ? " active" : ""}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === "analyze"   ? "🔍 Analyze"   :
               tab === "history"   ? "📋 History"   :
                                     "📊 Analytics"}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button className="theme-toggle" onClick={() => setDarkMode(d => !d)}>
            {darkMode ? "☀️ Light Mode" : "🌙 Dark Mode"}
          </button>
          <p className="sidebar-credit">AI-Powered Spam Detection</p>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="main-content">

        {/* ── ANALYZE TAB ── */}
        {activeTab === "analyze" && (
          <div className="tab-pane">
            <header className="page-header">
              <h1>Email Analyzer</h1>
              <p className="subtitle">Paste your email below and let AI detect spam instantly</p>
            </header>

            <div className="card input-card">
              <div className="card-header-row">
                <span className="card-title">✉️ Email Content</span>
                <button
                  className="upload-btn"
                  onClick={() => fileInputRef.current?.click()}
                >
                  📎 Upload .txt
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt"
                  style={{ display: "none" }}
                  onChange={handleFileUpload}
                />
              </div>
              <textarea
                className="email-input"
                placeholder="Paste email content here…&#10;&#10;e.g. Congratulations! You have been selected as a lucky winner. Click here to claim your prize now!"
                value={emailText}
                onChange={e => setEmailText(e.target.value)}
                rows={10}
              />
              <div className="input-footer">
                <span className="char-count">{emailText.length} characters</span>
                <div className="action-row">
                  <button className="clear-btn" onClick={() => { setEmailText(""); setResult(null); }}>
                    Clear
                  </button>
                  <button
                    className="analyze-btn"
                    onClick={analyzeEmail}
                    disabled={loading}
                  >
                    {loading ? (
                      <span className="spinner-wrap">
                        <span className="spinner" />Analyzing…
                      </span>
                    ) : "🔍 Analyze Email"}
                  </button>
                </div>
              </div>
              {error && <div className="error-banner">⚠️ {error}</div>}
            </div>

            {/* Result Card */}
            {result && (
              <div className={`card result-card ${result.prediction}`}>
                <div className="result-header">
                  <div className="verdict-badge" data-verdict={result.prediction}>
                    {result.prediction === "spam" ? "🚫 SPAM" : "✅ HAM"}
                  </div>
                  <div
                    className="risk-chip"
                    style={{
                      background: riskMeta[result.risk_level].bg,
                      color: riskMeta[result.risk_level].color,
                      border: `1.5px solid ${riskMeta[result.risk_level].color}`,
                    }}
                  >
                    {riskMeta[result.risk_level].icon} {riskMeta[result.risk_level].label}
                  </div>
                </div>

                <div className="result-scores">
                  <div className="score-block">
                    <span className="score-label">Confidence</span>
                    <ConfidenceBar score={result.confidence} prediction={result.prediction} />
                  </div>
                  <div className="score-block">
                    <span className="score-label">Spam Score</span>
                    <ConfidenceBar score={result.spam_score} prediction="spam" />
                  </div>
                </div>

                {result.keywords?.length > 0 && (
                  <div className="keywords-section">
                    <h4>🔑 Spam-Triggering Words</h4>
                    <div className="keyword-chips">
                      {result.keywords.map(kw => (
                        <span key={kw} className="keyword-chip">{kw}</span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="highlighted-section">
                  <h4>📝 Highlighted Text</h4>
                  <div className="highlighted-text">
                    <HighlightedText text={emailText} keywords={result.keywords} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── HISTORY TAB ── */}
        {activeTab === "history" && (
          <div className="tab-pane">
            <header className="page-header">
              <h1>Analysis History</h1>
              <p className="subtitle">{history.length} emails analyzed so far</p>
            </header>
            {history.length === 0 ? (
              <div className="empty-state">
                <span className="empty-icon">📭</span>
                <p>No emails analyzed yet. Go to the Analyze tab to get started!</p>
              </div>
            ) : (
              <div className="history-list">
                {history.map(h => (
                  <div key={h.id} className={`history-card ${h.prediction}`}>
                    <div className="history-top">
                      <span className={`history-badge ${h.prediction}`}>
                        {h.prediction === "spam" ? "🚫 SPAM" : "✅ HAM"}
                      </span>
                      <span className="history-conf">
                        {Math.round(h.confidence * 100)}% confidence
                      </span>
                      <span className="history-time">
                        {new Date(h.created_at).toLocaleString()}
                      </span>
                      <button
                        className="delete-btn"
                        onClick={() => deleteRecord(h.id)}
                        title="Delete"
                      >✕</button>
                    </div>
                    <p className="history-text">{h.email_text.slice(0, 160)}{h.email_text.length > 160 ? "…" : ""}</p>
                    {h.keywords?.length > 0 && (
                      <div className="history-keywords">
                        {h.keywords.slice(0, 5).map(kw => (
                          <span key={kw} className="mini-chip">{kw}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── ANALYTICS TAB ── */}
        {activeTab === "analytics" && (
          <div className="tab-pane">
            <header className="page-header">
              <h1>Analytics</h1>
              <p className="subtitle">Overview of all emails processed by SpamShield</p>
            </header>
            <div className="analytics-grid">
              <div className="stat-card total">
                <div className="stat-icon">📧</div>
                <div className="stat-num">{stats.total}</div>
                <div className="stat-lbl">Total Analyzed</div>
              </div>
              <div className="stat-card spam">
                <div className="stat-icon">🚫</div>
                <div className="stat-num">{stats.spam}</div>
                <div className="stat-lbl">Spam Detected</div>
              </div>
              <div className="stat-card ham">
                <div className="stat-icon">✅</div>
                <div className="stat-num">{stats.ham}</div>
                <div className="stat-lbl">Ham (Clean)</div>
              </div>
            </div>

            {stats.total > 0 && (
              <div className="card chart-card">
                <h3 className="card-title">📊 Spam vs Ham Breakdown</h3>
                <div className="donut-wrap">
                  <svg viewBox="0 0 200 200" className="donut-svg">
                    <DonutChart spam={stats.spam} ham={stats.ham} />
                  </svg>
                  <div className="donut-legend">
                    <div className="legend-row">
                      <span className="legend-dot spam" />
                      <span>Spam — {spamPct}%</span>
                    </div>
                    <div className="legend-row">
                      <span className="legend-dot ham" />
                      <span>Ham — {100 - spamPct}%</span>
                    </div>
                  </div>
                </div>
                <div className="progress-section">
                  <div className="progress-row">
                    <span>Spam Rate</span>
                    <div className="prog-track">
                      <div className="prog-fill spam" style={{ width: `${spamPct}%` }} />
                    </div>
                    <span>{spamPct}%</span>
                  </div>
                  <div className="progress-row">
                    <span>Ham Rate</span>
                    <div className="prog-track">
                      <div className="prog-fill ham" style={{ width: `${100 - spamPct}%` }} />
                    </div>
                    <span>{100 - spamPct}%</span>
                  </div>
                </div>
              </div>
            )}

            {stats.total === 0 && (
              <div className="empty-state">
                <span className="empty-icon">📊</span>
                <p>Analyze some emails first to see your statistics here!</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

// ─── Donut Chart SVG ────────────────────────────────────────────────────────
function DonutChart({ spam, ham }) {
  const total = spam + ham || 1;
  const r = 70, cx = 100, cy = 100;
  const circumference = 2 * Math.PI * r;
  const spamAngle = (spam / total) * 360;

  function describeArc(startDeg, endDeg) {
    const toRad = d => (d - 90) * (Math.PI / 180);
    const x1 = cx + r * Math.cos(toRad(startDeg));
    const y1 = cy + r * Math.sin(toRad(startDeg));
    const x2 = cx + r * Math.cos(toRad(endDeg));
    const y2 = cy + r * Math.sin(toRad(endDeg));
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`;
  }

  return (
    <>
      <circle cx={cx} cy={cy} r={r} fill="#f0f0f0" />
      {spam > 0 && (
        <path d={describeArc(0, spamAngle)} fill="#ef4444" opacity="0.85" />
      )}
      {ham > 0 && (
        <path d={describeArc(spamAngle, 360)} fill="#4caf50" opacity="0.85" />
      )}
      <circle cx={cx} cy={cy} r={42} fill="var(--card-bg)" />
      <text x={cx} y={cy - 6} textAnchor="middle" fontSize="22" fontWeight="bold" fill="var(--text)">
        {total}
      </text>
      <text x={cx} y={cy + 14} textAnchor="middle" fontSize="10" fill="var(--text-muted)">
        emails
      </text>
    </>
  );
}
