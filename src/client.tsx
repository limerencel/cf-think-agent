import { Component, Suspense, type ReactNode, useEffect, useRef, useState } from "react";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { getToolName, isToolUIPart } from "ai";
import type { UIMessage } from "ai";

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 24, fontFamily: "system-ui, sans-serif", color: "#ebebe6" }}>
          <h2>Something went wrong</h2>
          <pre style={{ whiteSpace: "pre-wrap", color: "#b36060" }}>{String(this.state.error)}</pre>
          <button onClick={() => location.reload()}>Reload</button>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ---------------- types & storage ---------------- */

type Convo = { id: string; title: string; ts: number };

const LS_KEY = "edgeagent.conversations";

function loadConvs(): Convo[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as Convo[]) : [];
  } catch {
    return [];
  }
}

function saveConvs(convos: Convo[]): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(convos));
  } catch {
    // storage may be disabled (private mode / webview) — in-memory only is fine
  }
}

function newId(): string {
  return "c" + Math.random().toString(36).slice(2, 10);
}

/* ---------------- icons ---------------- */

function MenuIcon() {
  return (
    <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
      <path d="M3 5.5h14M3 10h14M3 14.5h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true">
      <path
        d="M8 13V3.5M8 3.5 3.8 7.7M8 3.5l4.2 4.2"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

function XIcon() {
  return (
    <svg viewBox="0 0 14 14" width="12" height="12" aria-hidden="true">
      <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none" />
    </svg>
  );
}

/* ---------------- small pieces ---------------- */

function greeting(): string {
  const h = new Date().getHours();
  if (h < 5) return "Late night";
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

function textOf(message: UIMessage): string {
  return message.parts
    .filter((p) => p.type === "text")
    .map((p) => (p as { type: "text"; text: string }).text)
    .join("");
}

function ToolBits({ message }: { message: UIMessage }) {
  const tools = message.parts.filter(isToolUIPart);
  if (!tools.length) return null;
  return (
    <div className="tools">
      {tools.map((part, i) => {
        const name = getToolName(part);
        const state = "state" in part ? (part.state as string) : "";
        const cls = state === "output-available" ? "ok" : state === "output-error" ? "err" : "run";
        return (
          <span key={`${name}-${i}`} className={`chip ${cls}`}>
            {name.replace(/^gbrain_/, "gbrain · ")}
          </span>
        );
      })}
    </div>
  );
}

function Composer({
  draft,
  setDraft,
  onSubmit,
  busy,
}: {
  draft: string;
  setDraft: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 176) + "px";
  }, [draft]);

  return (
    <div className="composer">
      <textarea
        ref={ref}
        rows={1}
        value={draft}
        placeholder="How can I help you today?"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
      />
      <div className="composer-row">
        <span className="pill">deepseek-v4-flash · edge</span>
        <button type="button" className="send" onClick={onSubmit} disabled={busy || !draft.trim()} aria-label="Send">
          <SendIcon />
        </button>
      </div>
    </div>
  );
}

const HINTS = [
  "What infra pages do I have in GBrain?",
  "Write today's notes in the workspace",
  "Look up my Schwab records",
];

/* ---------------- one conversation ---------------- */

function Chat({
  convoId,
  onMenu,
  onFirstMessage,
}: {
  convoId: string;
  onMenu: () => void;
  onFirstMessage: (text: string) => void;
}) {
  const agent = useAgent({ agent: "Assistant", name: convoId });
  const { messages, sendMessage, status } = useAgentChat({ agent });
  const [draft, setDraft] = useState("");
  const busy = status === "submitted" || status === "streaming";
  const titled = useRef(false);
  const empty = messages.length === 0;

  const submit = () => {
    const text = draft.trim();
    if (!text || busy) return;
    if (!titled.current) {
      titled.current = true;
      onFirstMessage(text);
    }
    sendMessage({ text });
    setDraft("");
  };

  return (
    <div className={empty ? "page home" : "page chat"}>
      {!empty && (
        <header>
          <button type="button" className="ghost" onClick={onMenu} aria-label="Menu">
            <MenuIcon />
          </button>
          <span className="wordmark">edge agent</span>
          <span className="spacer" />
        </header>
      )}

      {empty ? (
        <div className="home-inner">
          <button type="button" className="ghost home-menu" onClick={onMenu} aria-label="Menu">
            <MenuIcon />
          </button>
          <h1>{greeting()}, Aki</h1>
          <div className="home-composer">
            <Composer draft={draft} setDraft={setDraft} onSubmit={submit} busy={busy} />
            <div className="hints">
              {HINTS.map((h) => (
                <button
                  key={h}
                  type="button"
                  className="hint"
                  onClick={() => {
                    if (!titled.current) {
                      titled.current = true;
                      onFirstMessage(h);
                    }
                    sendMessage({ text: h });
                  }}
                >
                  {h}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <>
          <main>
            {messages.map((m) => (
              <article key={m.id} className={m.role === "user" ? "from-user" : "from-agent"}>
                {m.role !== "user" && <ToolBits message={m} />}
                <div className="body">{textOf(m)}</div>
              </article>
            ))}
            {busy && <div className="thinking">Thinking…</div>}
          </main>
          <Composer draft={draft} setDraft={setDraft} onSubmit={submit} busy={busy} />
        </>
      )}
    </div>
  );
}

/* ---------------- shell with sidebar ---------------- */

export function App() {
  return (
    <ErrorBoundary>
      <AppInner />
    </ErrorBoundary>
  );
}

function AppInner() {
  const [convos, setConvos] = useState<Convo[]>(loadConvs);
  const [active, setActive] = useState<string>(() => loadConvs()[0]?.id ?? newId());
  const [drawer, setDrawer] = useState(false);

  useEffect(() => {
    saveConvs(convos);
  }, [convos]);

  const touch = (id: string, title?: string) => {
    setConvos((prev) => {
      const existing = prev.find((c) => c.id === id);
      const entry: Convo = {
        id,
        title: title ?? existing?.title ?? "New chat",
        ts: Date.now(),
      };
      return [entry, ...prev.filter((c) => c.id !== id)].slice(0, 30);
    });
  };

  const newChat = () => {
    setActive(newId());
    setDrawer(false);
  };

  return (
    <div className="shell">
      <aside className={drawer ? "open" : ""}>
        <div className="side-top">
          <span className="wordmark">edge agent</span>
        </div>
        <nav className="side-list">
          <button type="button" className="side-item new-chat" onClick={newChat}>
            <PlusIcon />
            New chat
          </button>
          {convos.length === 0 && <p className="side-empty">No conversations yet</p>}
          {convos.map((c) => (
            <div
              key={c.id}
              className={"side-item" + (c.id === active ? " active" : "")}
              onClick={() => {
                setActive(c.id);
                setDrawer(false);
              }}
            >
              <span className="side-title">{c.title}</span>
              <button
                type="button"
                className="side-del"
                aria-label="Delete"
                onClick={(e) => {
                  e.stopPropagation();
                  setConvos((prev) => {
                    const next = prev.filter((x) => x.id !== c.id);
                    if (c.id === active) setActive(next[0]?.id ?? newId());
                    return next;
                  });
                }}
              >
                <XIcon />
              </button>
            </div>
          ))}
        </nav>
        <div className="side-foot">
          <span>Cloudflare Edge</span>
        </div>
      </aside>

      {drawer && <div className="scrim" onClick={() => setDrawer(false)} />}

      <Suspense
        fallback={
          <div className="page home">
            <div className="home-inner">
              <h1>Connecting…</h1>
            </div>
          </div>
        }
      >
        <Chat
          key={active}
          convoId={active}
          onMenu={() => setDrawer(true)}
          onFirstMessage={(text) => touch(active, text.slice(0, 48))}
        />
      </Suspense>
    </div>
  );
}
