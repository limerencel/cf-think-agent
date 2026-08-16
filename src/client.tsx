import { Component, Suspense, type ReactElement, type ReactNode, useEffect, useRef, useState } from "react";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { getToolName, isToolUIPart } from "ai";
import type { UIMessage } from "ai";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

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

/** localStorage is a cache only; the cloud ConvoIndex DO is the source of truth. */
function loadLocalConvs(): Convo[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as Convo[]) : [];
  } catch {
    return [];
  }
}

function saveLocalConvs(convos: Convo[]): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(convos));
  } catch {
    // storage may be disabled (private mode / webview) — in-memory only is fine
  }
}

async function cloudListConvs(): Promise<Convo[]> {
  const res = await fetch("/api/convos", { headers: { accept: "application/json" } });
  if (!res.ok) throw new Error(`cloud list ${res.status}`);
  const data = (await res.json()) as { ok: boolean; convos: Convo[] };
  if (!data.ok) throw new Error("cloud list !ok");
  return data.convos;
}

async function cloudTouchConvs(id: string, title?: string): Promise<Convo[]> {
  const res = await fetch("/api/convos", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id, title }),
  });
  if (!res.ok) throw new Error(`cloud touch ${res.status}`);
  const data = (await res.json()) as { ok: boolean; convos: Convo[] };
  if (!data.ok) throw new Error("cloud touch !ok");
  return data.convos;
}

async function cloudRemoveConvo(id: string): Promise<Convo[]> {
  const res = await fetch("/api/convos/remove", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error(`cloud remove ${res.status}`);
  const data = (await res.json()) as { ok: boolean; convos: Convo[] };
  if (!data.ok) throw new Error("cloud remove !ok");
  return data.convos;
}

function newId(): string {
  return "c" + Math.random().toString(36).slice(2, 10);
}

/* ---------------- icons ---------------- */

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

function CopyIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
      <rect x="5" y="5" width="8" height="8" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M11 5V4a1.5 1.5 0 0 0-1.5-1.5H4A1.5 1.5 0 0 0 2.5 4v5.5A1.5 1.5 0 0 0 4 11h1"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
      />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
      <path d="M3 8.5 6.5 12 13 4.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SideCollapseIcon() {
  return (
    <svg viewBox="0 0 20 20" width="17" height="17" aria-hidden="true">
      <rect x="2.5" y="3.5" width="15" height="13" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 3.5v13" stroke="currentColor" strokeWidth="1.5" />
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

/* ---------------- markdown rendering ---------------- */

function CodeBlock({ children }: { children?: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // children is the <code className="language-x"> element produced by react-markdown
  const codeEl = children as ReactElement<{ className?: string }> | undefined;
  const lang = /language-([\w-]+)/.exec(codeEl?.props?.className ?? "")?.[1] ?? "text";

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const copy = async () => {
    const text = ref.current?.querySelector("code")?.textContent ?? "";
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setCopied(true);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="codeblock" ref={ref}>
      <div className="codeblock-head">
        <span className="codeblock-lang">{lang}</span>
        <button type="button" className="codeblock-copy" onClick={copy} aria-label="Copy code">
          {copied ? <CheckIcon /> : <CopyIcon />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  );
}

function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{ pre: CodeBlock }}
      >
        {text}
      </ReactMarkdown>
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
  onFirstMessage,
}: {
  convoId: string;
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
      {empty ? (
        <div className="home-inner">
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
                {m.role === "user" ? (
                  <div className="body">{textOf(m)}</div>
                ) : (
                  <Markdown text={textOf(m)} />
                )}
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
  const [convos, setConvos] = useState<Convo[]>(loadLocalConvs);
  const [active, setActive] = useState<string>(() => loadLocalConvs()[0]?.id ?? newId());
  // Desktop: sidebar starts expanded and pushes content. Mobile: starts hidden, overlays.
  const [sideOpen, setSideOpen] = useState<boolean>(
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 900px)").matches
  );
  const [cloudReady, setCloudReady] = useState(false);

  const closeSideOnMobile = () => {
    if (typeof window !== "undefined" && !window.matchMedia("(min-width: 900px)").matches) {
      setSideOpen(false);
    }
  };

  // On mount: pull the cloud list as the source of truth. If we have local
  // entries the cloud doesn't know yet (e.g. messages sent while storage was
  // broken), merge them in and push back up so nothing is lost.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cloud = await cloudListConvs();
        if (cancelled) return;
        const local = loadLocalConvs();
        const seen = new Set(cloud.map((c) => c.id));
        const orphans = local.filter((c) => !seen.has(c.id));
        const merged = [...cloud, ...orphans]
          .sort((a, b) => b.ts - a.ts)
          .slice(0, 30);
        setConvos(merged);
        saveLocalConvs(merged);
        setCloudReady(true);
        // If the current active id exists nowhere (fresh id from a broken
        // storage session) and the cloud has conversations, open the most
        // recent one so history is visible immediately.
        if (merged.length > 0) {
          const localIds = new Set(local.map((c) => c.id));
          if (!localIds.has(active) && !merged.some((c) => c.id === active)) {
            setActive(merged[0].id);
          }
        }
        // push orphans back up so the cloud list converges
        for (const o of orphans) {
          try {
            await cloudTouchConvs(o.id, o.title);
          } catch {
            /* best effort */
          }
        }
      } catch {
        // cloud unreachable — keep local cache as-is
        if (!cancelled) setCloudReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    saveLocalConvs(convos);
  }, [convos]);

  const applyCloud = (next: Convo[]) => {
    setConvos(next);
    saveLocalConvs(next);
  };

  const touch = (id: string, title?: string) => {
    const now = Date.now();
    setConvos((prev) => {
      const existing = prev.find((c) => c.id === id);
      const entry: Convo = {
        id,
        title: title ?? existing?.title ?? "New chat",
        ts: now,
      };
      return [entry, ...prev.filter((c) => c.id !== id)].slice(0, 30);
    });
    if (cloudReady) {
      cloudTouchConvs(id, title).then(applyCloud).catch(() => {});
    }
  };

  const removeConvo = (id: string) => {
    setConvos((prev) => {
      const next = prev.filter((x) => x.id !== id);
      if (id === active) setActive(next[0]?.id ?? newId());
      return next;
    });
    if (cloudReady) {
      cloudRemoveConvo(id).then(applyCloud).catch(() => {});
    }
  };

  const newChat = () => {
    setActive(newId());
    closeSideOnMobile();
  };

  return (
    <div className={"shell" + (sideOpen ? " side-open" : "")}>
      {!sideOpen && (
        <button
          type="button"
          className="ghost mobile-open"
          onClick={() => setSideOpen(true)}
          aria-label="Open sidebar"
        >
          <SideCollapseIcon />
        </button>
      )}
      {!sideOpen && (
        <div className="side-rail">
          <button
            type="button"
            className="ghost side-expand"
            onClick={() => setSideOpen(true)}
            aria-label="Expand sidebar"
          >
            <SideCollapseIcon />
          </button>
        </div>
      )}
      <aside className={sideOpen ? "open" : ""}>
        <div className="side-top">
          <span className="wordmark">edge agent</span>
          <button
            type="button"
            className="ghost side-collapse"
            onClick={() => setSideOpen(false)}
            aria-label="Collapse sidebar"
          >
            <SideCollapseIcon />
          </button>
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
                closeSideOnMobile();
              }}
            >
              <span className="side-title">{c.title}</span>
              <button
                type="button"
                className="side-del"
                aria-label="Delete"
                onClick={(e) => {
                  e.stopPropagation();
                  removeConvo(c.id);
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

      {sideOpen && <div className="tap-away" onClick={() => setSideOpen(false)} />}

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
          onFirstMessage={(text) => touch(active, text.slice(0, 48))}
        />
      </Suspense>
    </div>
  );
}
