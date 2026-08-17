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

export type ThemeMode = "system" | "light" | "dark";

export interface ProviderConfig {
  id: string;
  name: string;
  endpoint: string;
  apiKey: string;
  selectedModel: string;
  cachedModels: string[];
  isDefault?: boolean;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
}

const DEFAULT_PRESET: ProviderConfig = {
  id: "cf-default",
  name: "Cloudflare AI Gateway",
  endpoint: "",
  apiKey: "",
  selectedModel: "deepseek-v4-flash",
  cachedModels: ["deepseek-v4-flash"],
  isDefault: true,
};

const LS_KEY = "edgeagent.conversations";
const LS_THEME_KEY = "edgeagent.theme";
const LS_PROVIDERS_KEY = "edgeagent.providers";
const LS_ACTIVE_PROVIDER_KEY = "edgeagent.active_provider_id";

function loadLocalTheme(): ThemeMode {
  try {
    const raw = localStorage.getItem(LS_THEME_KEY);
    if (raw === "light" || raw === "dark" || raw === "system") return raw;
    return "system";
  } catch {
    return "system";
  }
}

function saveLocalTheme(theme: ThemeMode): void {
  try {
    localStorage.setItem(LS_THEME_KEY, theme);
  } catch {
    /* ignore */
  }
}

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
    // storage may be disabled
  }
}

function loadLocalProviders(): ProviderConfig[] {
  try {
    const raw = localStorage.getItem(LS_PROVIDERS_KEY);
    if (raw) {
      const list = JSON.parse(raw) as ProviderConfig[];
      if (!list.some((p) => p.id === "cf-default")) {
        return [DEFAULT_PRESET, ...list];
      }
      return list;
    }
    // Migration from old model_configs format
    const oldRaw = localStorage.getItem("edgeagent.model_configs");
    if (oldRaw) {
      const oldList = JSON.parse(oldRaw) as any[];
      const migrated: ProviderConfig[] = oldList.map((m) => ({
        id: m.id,
        name: m.name || "Custom Provider",
        endpoint: m.endpoint || "",
        apiKey: m.apiKey || "",
        selectedModel: m.modelId || "deepseek-chat",
        cachedModels: [m.modelId || "deepseek-chat"],
        isDefault: !!m.isDefault,
      }));
      if (!migrated.some((p) => p.id === "cf-default")) {
        migrated.unshift(DEFAULT_PRESET);
      }
      return migrated;
    }
    return [DEFAULT_PRESET];
  } catch {
    return [DEFAULT_PRESET];
  }
}

function saveLocalProviders(providers: ProviderConfig[]): void {
  try {
    localStorage.setItem(LS_PROVIDERS_KEY, JSON.stringify(providers));
  } catch {
    /* ignore */
  }
}

function loadActiveProviderId(providers: ProviderConfig[]): string {
  try {
    const raw = localStorage.getItem(LS_ACTIVE_PROVIDER_KEY) || localStorage.getItem("edgeagent.active_model_id");
    if (raw && providers.some((p) => p.id === raw)) return raw;
    const def = providers.find((p) => p.isDefault);
    return def?.id ?? providers[0]?.id ?? "cf-default";
  } catch {
    return "cf-default";
  }
}

function saveActiveProviderId(id: string): void {
  try {
    localStorage.setItem(LS_ACTIVE_PROVIDER_KEY, id);
  } catch {
    /* ignore */
  }
}

/* ---------------- Cloud API Sync helpers ---------------- */

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

async function cloudListProviders(): Promise<{ providers: ProviderConfig[]; activeId?: string }> {
  const res = await fetch("/api/providers", { headers: { accept: "application/json" } });
  if (!res.ok) throw new Error(`cloud list providers ${res.status}`);
  const data = (await res.json()) as { ok: boolean; providers: ProviderConfig[]; activeId?: string };
  if (!data.ok) throw new Error("cloud list providers !ok");
  return { providers: data.providers, activeId: data.activeId };
}

async function cloudSaveProvider(provider: ProviderConfig): Promise<ProviderConfig[]> {
  const res = await fetch("/api/providers", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!res.ok) throw new Error(`cloud save provider ${res.status}`);
  const data = (await res.json()) as { ok: boolean; providers: ProviderConfig[] };
  if (!data.ok) throw new Error("cloud save provider !ok");
  return data.providers;
}

async function cloudSaveAllProviders(providers: ProviderConfig[]): Promise<ProviderConfig[]> {
  const res = await fetch("/api/providers", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ providers }),
  });
  if (!res.ok) throw new Error(`cloud save all providers ${res.status}`);
  const data = (await res.json()) as { ok: boolean; providers: ProviderConfig[] };
  if (!data.ok) throw new Error("cloud save all providers !ok");
  return data.providers;
}

async function cloudRemoveProvider(id: string): Promise<ProviderConfig[]> {
  const res = await fetch("/api/providers/remove", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error(`cloud remove provider ${res.status}`);
  const data = (await res.json()) as { ok: boolean; providers: ProviderConfig[] };
  if (!data.ok) throw new Error("cloud remove provider !ok");
  return data.providers;
}

async function cloudSetActiveProvider(id: string): Promise<void> {
  const res = await fetch("/api/providers/active", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error(`cloud set active provider ${res.status}`);
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

function SettingsIcon() {
  return (
    <svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true">
      <path
        d="M10 12.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path
        d="M16.2 12.4a1.2 1.2 0 0 0 .24 1.34l.05.05a1.44 1.44 0 1 1-2.04 2.04l-.05-.05a1.2 1.2 0 0 0-1.34-.24 1.2 1.2 0 0 0-.73 1.1v.15a1.44 1.44 0 1 1-2.88 0v-.08a1.2 1.2 0 0 0-.77-1.1 1.2 1.2 0 0 0-1.34.24l-.05.05a1.44 1.44 0 1 1-2.04-2.04l.05-.05a1.2 1.2 0 0 0 .24-1.34 1.2 1.2 0 0 0-1.1-.73H4.44a1.44 1.44 0 0 1 0-2.88h.08a1.2 1.2 0 0 0 1.1-.77 1.2 1.2 0 0 0-.24-1.34l-.05-.05a1.44 1.44 0 1 1 2.04-2.04l.05.05a1.2 1.2 0 0 0 1.34.24h.07a1.2 1.2 0 0 0 .73-1.1V3.44a1.44 1.44 0 0 1 2.88 0v.08a1.2 1.2 0 0 0 .73 1.1 1.2 1.2 0 0 0 1.34-.24l.05-.05a1.44 1.44 0 1 1 2.04 2.04l-.05.05a1.2 1.2 0 0 0-.24 1.34v.07a1.2 1.2 0 0 0 1.1.73h.15a1.44 1.44 0 0 1 0 2.88h-.08a1.2 1.2 0 0 0-1.1.77z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChevronDownIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
      <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ChevronRightIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
      <path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path d="M1.5 8s2.5-4.5 6.5-4.5S14.5 8 14.5 8s-2.5 4.5-6.5 4.5S1.5 8 1.5 8z" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="8" cy="8" r="2" fill="none" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function EyeOffIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path d="M2 2l12 12M6.6 6.7a2 2 0 0 0 2.7 2.7M4 4.5C2.7 5.5 1.5 8 1.5 8s2.5 4.5 6.5 4.5c1.6 0 3-.7 4.1-1.6M8 3.5c4 0 6.5 4.5 6.5 4.5s-.8 1.5-2.2 2.7" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SparklesIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
      <path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3M3.4 3.4l2.1 2.1M10.5 10.5l2.1 2.1M3.4 12.6l2.1-2.1M10.5 5.5l2.1-2.1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <circle cx="8" cy="8" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3M11.3 11.3l1.3 1.3M3.4 12.6l1.3-1.3M11.3 4.7l1.3-1.3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path d="M13.5 9.8A6 6 0 0 1 6.2 2.5a6 6 0 1 0 7.3 7.3z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MonitorIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <rect x="2" y="2.5" width="12" height="8.5" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M5.5 14h5M8 11v3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function CpuIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <rect x="4" y="4" width="8" height="8" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M6 1.5v2.5M10 1.5v2.5M6 12v2.5M10 12v2.5M1.5 6h2.5M1.5 10h2.5M12 6h2.5M12 10h2.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M8 7v4.5M8 4.7v.2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
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

/* ---------------- Model Combobox with dropdown & manual input ---------------- */

function ModelCombobox({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string;
  onChange: (val: string) => void;
  options: string[];
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filtered = options.filter((opt) => opt.toLowerCase().includes(query.toLowerCase()));
  const MAX_SHOWN = 50;
  const displayed = filtered.slice(0, MAX_SHOWN);

  return (
    <div className="combobox-wrapper" ref={containerRef}>
      <div className="input-row">
        <input
          type="text"
          name="model_id_input"
          autoComplete="off"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          data-lpignore="true"
          data-1p-ignore="true"
          data-form-type="other"
          className="text-input"
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            setQuery(e.target.value);
          }}
          onFocus={() => {
            if (options.length > 0) setOpen(true);
          }}
          placeholder={placeholder || "e.g. deepseek-chat, gpt-4o, claude-3-5-sonnet"}
          required
        />
        {options.length > 0 && (
          <button
            type="button"
            className="btn-icon"
            onClick={() => setOpen(!open)}
            title="Toggle model list"
          >
            <ChevronDownIcon />
          </button>
        )}
      </div>

      {open && options.length > 0 && (
        <div className="combobox-dropdown">
          {displayed.length === 0 ? (
            <div style={{ padding: "8px 12px", fontSize: 12, color: "var(--muted)" }}>
              No matching model found in fetched list. You can type freely above.
            </div>
          ) : (
            displayed.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`combobox-option ${opt === value ? "active" : ""}`}
                onClick={() => {
                  onChange(opt);
                  setOpen(false);
                }}
              >
                {opt}
              </button>
            ))
          )}
          {filtered.length > MAX_SHOWN && (
            <div className="combobox-footer-info">
              Showing top {MAX_SHOWN} of {filtered.length} models. Type in the box to search.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ---------------- Provider Editor Form Component ---------------- */

function ProviderEditor({
  provider,
  isNew,
  onSave,
  onCancel,
  onDelete,
}: {
  provider: ProviderConfig;
  isNew?: boolean;
  onSave: (p: ProviderConfig) => void;
  onCancel: () => void;
  onDelete?: () => void;
}) {
  const [name, setName] = useState(provider.name);
  const [endpoint, setEndpoint] = useState(provider.endpoint);
  const [apiKey, setApiKey] = useState(provider.apiKey);
  const [selectedModel, setSelectedModel] = useState(provider.selectedModel);
  const [cachedModels, setCachedModels] = useState<string[]>(provider.cachedModels || []);
  const [isDefault, setIsDefault] = useState(!!provider.isDefault);
  const [showApiKey, setShowApiKey] = useState(false);

  // Advanced parameters
  const [showAdvanced, setShowAdvanced] = useState(
    provider.temperature !== undefined || provider.maxTokens !== undefined || provider.topP !== undefined
  );
  const [temperature, setTemperature] = useState<number | undefined>(provider.temperature);
  const [maxTokens, setMaxTokens] = useState<number | undefined>(provider.maxTokens);
  const [topP, setTopP] = useState<number | undefined>(provider.topP);

  const [loadingModels, setLoadingModels] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  const handleFetchModels = async () => {
    if (!endpoint.trim()) {
      setStatusMsg({ type: "err", text: "Please enter the Endpoint URL first." });
      return;
    }
    setLoadingModels(true);
    setStatusMsg(null);
    try {
      const res = await fetch("/api/models/fetch", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          endpoint: endpoint.trim(),
          apiKey: apiKey.trim() || undefined,
        }),
      });
      const data = (await res.json()) as { ok: boolean; models?: string[]; error?: string };
      if (!res.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      const list = data.models ?? [];
      setCachedModels(list);
      if (list.length > 0) {
        if (!selectedModel || !list.includes(selectedModel)) {
          setSelectedModel(list[0]);
        }
        setStatusMsg({ type: "ok", text: `Fetched ${list.length} models from endpoint.` });
      } else {
        setStatusMsg({ type: "ok", text: "Endpoint connected, but empty model list returned. You can type model ID manually." });
      }
    } catch (err: any) {
      setStatusMsg({ type: "err", text: `Failed to load models: ${err.message || String(err)}` });
    } finally {
      setLoadingModels(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (provider.id !== "cf-default" && !endpoint.trim()) {
      setStatusMsg({ type: "err", text: "Endpoint URL is required." });
      return;
    }
    if (!selectedModel.trim()) {
      setStatusMsg({ type: "err", text: "Model ID is required." });
      return;
    }

    const updated: ProviderConfig = {
      ...provider,
      name: name.trim() || selectedModel.trim(),
      endpoint: endpoint.trim(),
      apiKey: apiKey.trim(),
      selectedModel: selectedModel.trim(),
      cachedModels: cachedModels.length > 0 ? cachedModels : [selectedModel.trim()],
      isDefault,
      temperature,
      maxTokens,
      topP,
    };
    onSave(updated);
  };

  const isPreset = provider.id === "cf-default";

  return (
    <form className="form-grid" onSubmit={handleSubmit} autoComplete="off" data-lpignore="true" data-1p-ignore="true">
      <div className="form-group">
        <label className="form-label">
          Provider Name
          <span className="form-hint">e.g. OpenRouter, DeepSeek Official, OpenAI</span>
        </label>
        <input
          type="text"
          name="provider_name_field"
          autoComplete="off"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          data-lpignore="true"
          data-1p-ignore="true"
          data-form-type="other"
          className="text-input"
          placeholder="Provider Display Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>

      {isPreset ? (
        <div className="info-card" style={{ marginBottom: 6 }}>
          <div className="info-card-icon">
            <SparklesIcon />
          </div>
          <div className="info-card-content">
            <span className="info-card-title">Cloudflare Worker Native Preset</span>
            <span className="info-card-desc">
              Base URL: <code>{endpoint || "Loaded from wrangler.jsonc (AIG_BASE_URL)"}</code><br />
              API Key: <code>Stored in Cloudflare Secret (OPENCODE_GO_API_KEY)</code><br />
              Default Model: <code>{selectedModel}</code> (MODEL_ID in wrangler.jsonc)
            </span>
          </div>
        </div>
      ) : (
        <>
          <div className="form-group">
            <label className="form-label">
              AI Endpoint (Base URL)
              <span className="form-hint">OpenAI Compatible</span>
            </label>
            <input
              type="text"
              name="provider_endpoint_field"
              autoComplete="off"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
              data-lpignore="true"
              data-1p-ignore="true"
              data-form-type="other"
              className="text-input"
              placeholder="https://openrouter.ai/api/v1 or https://api.deepseek.com/v1"
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">API Key</label>
            <div className="input-row">
              <input
                type="text"
                name="provider_key_field"
                autoComplete="off"
                autoCapitalize="off"
                autoCorrect="off"
                spellCheck={false}
                data-lpignore="true"
                data-1p-ignore="true"
                data-form-type="other"
                className={showApiKey ? "text-input" : "text-input key-masked"}
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              <button
                type="button"
                className="btn-icon"
                onClick={() => setShowApiKey(!showApiKey)}
                title={showApiKey ? "Hide API key" : "Show API key"}
              >
                {showApiKey ? <EyeOffIcon /> : <EyeIcon />}
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={handleFetchModels}
                disabled={loadingModels || !endpoint.trim()}
              >
                {loadingModels ? <span className="spinner" /> : <SparklesIcon />}
                {loadingModels ? "Loading..." : "Load / Refresh Models"}
              </button>
            </div>
          </div>
        </>
      )}

      <div className="form-group">
        <label className="form-label">
          Model ID
          {cachedModels.length > 0 && <span className="form-hint">{cachedModels.length} models available</span>}
        </label>
        <ModelCombobox
          value={selectedModel}
          onChange={setSelectedModel}
          options={cachedModels}
          placeholder="Select from dropdown or type model ID"
        />
      </div>

      {/* Advanced Parameters Accordion */}
      <div className="accordion">
        <button
          type="button"
          className="accordion-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <span>Advanced Inference Parameters (Optional)</span>
          {showAdvanced ? <ChevronDownIcon /> : <ChevronRightIcon />}
        </button>
        {showAdvanced && (
          <div className="accordion-body">
            {/* Temperature */}
            <div className="param-row">
              <div className="param-row-head">
                <span className="param-label">Temperature (Creativity)</span>
                <span className="param-val">{temperature !== undefined ? temperature : "Default"}</span>
              </div>
              <div className="param-slider-row">
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.05"
                  className="param-slider"
                  value={temperature !== undefined ? temperature : 0.7}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                />
                <button
                  type="button"
                  className="btn-secondary btn-sm"
                  onClick={() => setTemperature(undefined)}
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Max Output Tokens */}
            <div className="param-row">
              <div className="param-row-head">
                <span className="param-label">Max Output Tokens</span>
                <span className="param-val">{maxTokens ? `${maxTokens}` : "Default"}</span>
              </div>
              <div className="input-row">
                <input
                  type="number"
                  min="256"
                  max="131072"
                  step="256"
                  className="text-input"
                  placeholder="e.g. 4096, 8192 (leave empty for model default)"
                  value={maxTokens ?? ""}
                  onChange={(e) => setMaxTokens(e.target.value ? parseInt(e.target.value, 10) : undefined)}
                />
              </div>
            </div>

            {/* Top P */}
            <div className="param-row">
              <div className="param-row-head">
                <span className="param-label">Top P (Nucleus Sampling)</span>
                <span className="param-val">{topP !== undefined ? topP : "Default"}</span>
              </div>
              <div className="param-slider-row">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  className="param-slider"
                  value={topP !== undefined ? topP : 1.0}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                />
                <button
                  type="button"
                  className="btn-secondary btn-sm"
                  onClick={() => setTopP(undefined)}
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="input-row" style={{ justifyContent: "space-between", marginTop: 4 }}>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={isDefault}
            onChange={(e) => setIsDefault(e.target.checked)}
          />
          Set as default provider
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          {!isNew && (
            <button type="button" className="btn-secondary" onClick={onCancel}>
              Close
            </button>
          )}
          {isNew && (
            <button type="button" className="btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          )}
          <button type="submit" className="btn-primary">
            {isNew ? "Add Provider" : "Save Changes"}
          </button>
        </div>
      </div>

      {statusMsg && (
        <div className={`banner-msg ${statusMsg.type}`}>
          {statusMsg.type === "ok" ? <CheckIcon /> : <XIcon />}
          <span>{statusMsg.text}</span>
        </div>
      )}
    </form>
  );
}

/* ---------------- Modular Tabbed Settings Modal ---------------- */

function SettingsModal({
  isOpen,
  onClose,
  theme,
  onSetTheme,
  providers,
  activeProviderId,
  onSaveProviders,
  onSelectActiveProvider,
  convosCount,
}: {
  isOpen: boolean;
  onClose: () => void;
  theme: ThemeMode;
  onSetTheme: (t: ThemeMode) => void;
  providers: ProviderConfig[];
  activeProviderId: string;
  onSaveProviders: (providers: ProviderConfig[], newActiveId?: string) => void;
  onSelectActiveProvider: (id: string) => void;
  convosCount: number;
}) {
  const [activeTab, setActiveTab] = useState<"general" | "models" | "tools" | "about">("models");
  const [expandedProviderId, setExpandedProviderId] = useState<string | null>(null);
  const [isAddingNew, setIsAddingNew] = useState(false);

  if (!isOpen) return null;

  const handleSaveProvider = (updatedProvider: ProviderConfig, isNew?: boolean) => {
    let updated = [...providers];
    if (updatedProvider.isDefault) {
      updated = updated.map((p) => ({ ...p, isDefault: false }));
    }

    if (isNew) {
      updated.push(updatedProvider);
      onSaveProviders(updated, updatedProvider.id);
      setIsAddingNew(false);
    } else {
      updated = updated.map((p) => (p.id === updatedProvider.id ? updatedProvider : p));
      onSaveProviders(updated);
      setExpandedProviderId(null);
    }
  };

  const handleDeleteProvider = (id: string) => {
    if (id === "cf-default") return;
    const updated = providers.filter((p) => p.id !== id);
    const nextActive = activeProviderId === id ? "cf-default" : undefined;
    onSaveProviders(updated, nextActive);
  };

  const handleSetDefaultProvider = (id: string) => {
    const updated = providers.map((p) => ({
      ...p,
      isDefault: p.id === id,
    }));
    onSaveProviders(updated);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2 className="modal-title">Settings</h2>
          <button type="button" className="ghost" onClick={onClose} aria-label="Close modal">
            <XIcon />
          </button>
        </div>

        {/* Tab Navigation - macOS/iOS Segmented Control */}
        <div className="modal-tabs">
          <div className="segmented-nav">
            <button
              type="button"
              className={`tab-btn ${activeTab === "general" ? "active" : ""}`}
              onClick={() => setActiveTab("general")}
            >
              <SettingsIcon />
              <span>General</span>
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "models" ? "active" : ""}`}
              onClick={() => setActiveTab("models")}
            >
              <SparklesIcon />
              <span>Providers</span>
              <span className="tab-count">{providers.length}</span>
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "tools" ? "active" : ""}`}
              onClick={() => setActiveTab("tools")}
            >
              <CpuIcon />
              <span>Tools & MCP</span>
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "about" ? "active" : ""}`}
              onClick={() => setActiveTab("about")}
            >
              <InfoIcon />
              <span>About</span>
            </button>
          </div>
        </div>

        <div className="modal-body">
          {/* TAB 1: GENERAL */}
          {activeTab === "general" && (
            <>
              <div className="modal-section">
                <h3 className="modal-section-title">Appearance & Theme</h3>
                <p className="modal-section-desc">Customize how the agent interface looks on your screen.</p>
                <div className="theme-grid">
                  <button
                    type="button"
                    className={`theme-card ${theme === "system" ? "active" : ""}`}
                    onClick={() => onSetTheme("system")}
                  >
                    <div className="theme-card-head">
                      <div className="theme-card-icon">
                        <MonitorIcon />
                      </div>
                      {theme === "system" && <span className="badge badge-active">Active</span>}
                    </div>
                    <span className="theme-card-title">System</span>
                    <span className="theme-card-desc">Syncs with your operating system preference</span>
                  </button>

                  <button
                    type="button"
                    className={`theme-card ${theme === "light" ? "active" : ""}`}
                    onClick={() => onSetTheme("light")}
                  >
                    <div className="theme-card-head">
                      <div className="theme-card-icon">
                        <SunIcon />
                      </div>
                      {theme === "light" && <span className="badge badge-active">Active</span>}
                    </div>
                    <span className="theme-card-title">Light</span>
                    <span className="theme-card-desc">Warm natural paper palette</span>
                  </button>

                  <button
                    type="button"
                    className={`theme-card ${theme === "dark" ? "active" : ""}`}
                    onClick={() => onSetTheme("dark")}
                  >
                    <div className="theme-card-head">
                      <div className="theme-card-icon">
                        <MoonIcon />
                      </div>
                      {theme === "dark" && <span className="badge badge-active">Active</span>}
                    </div>
                    <span className="theme-card-title">Dark</span>
                    <span className="theme-card-desc">Deep espresso & charcoal palette</span>
                  </button>
                </div>
              </div>

              <div className="modal-section">
                <h3 className="modal-section-title">Cloud Storage & Synchronization</h3>
                <div className="info-card">
                  <div className="info-card-icon">
                    <InfoIcon />
                  </div>
                  <div className="info-card-content">
                    <span className="info-card-title">Cloudflare Durable Objects SQLite</span>
                    <span className="info-card-desc">
                      Conversations ({convosCount}) and AI Providers ({providers.length}) are securely stored in Cloudflare edge SQLite. Synced across all your devices automatically.
                    </span>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* TAB 2: PROVIDERS & MODELS */}
          {activeTab === "models" && (
            <>
              {/* Header with Add Provider Action */}
              <div className="modal-section-head">
                <h3 className="modal-section-title">Saved Providers ({providers.length})</h3>
                {!isAddingNew && (
                  <button
                    type="button"
                    className="btn-secondary btn-sm"
                    onClick={() => setIsAddingNew(true)}
                  >
                    <PlusIcon />
                    <span>Add Provider</span>
                  </button>
                )}
              </div>

              {/* Add New Provider Form */}
              {isAddingNew && (
                <div className="provider-card active" style={{ padding: 16 }}>
                  <h4 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 600, color: "var(--ink)" }}>
                    Add New AI Provider
                  </h4>
                  <ProviderEditor
                    provider={{
                      id: "p-" + Math.random().toString(36).slice(2, 9),
                      name: "",
                      endpoint: "",
                      apiKey: "",
                      selectedModel: "",
                      cachedModels: [],
                    }}
                    isNew={true}
                    onSave={(p) => handleSaveProvider(p, true)}
                    onCancel={() => setIsAddingNew(false)}
                  />
                </div>
              )}

              {/* Providers List */}
              <div className="providers-list">
                {providers.map((p) => {
                  const isActive = p.id === activeProviderId;
                  const isPreset = p.id === "cf-default";
                  const isExpanded = expandedProviderId === p.id;

                  return (
                    <div key={p.id} className={`provider-card ${isActive ? "active" : ""}`}>
                      <div
                        className="provider-card-head"
                        onClick={() => setExpandedProviderId(isExpanded ? null : p.id)}
                      >
                        <div className="provider-meta">
                          <div className="provider-title-row">
                            <span className="provider-name">{p.name}</span>
                            {isActive && <span className="badge badge-active">Active</span>}
                            {p.isDefault && <span className="badge badge-default">Default</span>}
                            {isPreset && <span className="badge badge-preset">Cloudflare</span>}
                          </div>
                          <span className="provider-desc">
                            Model: <span className="provider-model-tag">{p.selectedModel}</span>
                            {p.endpoint && ` · ${p.endpoint}`}
                          </span>
                        </div>
                        <div className="provider-actions" onClick={(e) => e.stopPropagation()}>
                          {!isActive && (
                            <button
                              type="button"
                              className="btn-secondary btn-sm"
                              onClick={() => onSelectActiveProvider(p.id)}
                            >
                              Use
                            </button>
                          )}
                          {!p.isDefault && (
                            <button
                              type="button"
                              className="btn-secondary btn-sm"
                              onClick={() => handleSetDefaultProvider(p.id)}
                            >
                              Make Default
                            </button>
                          )}
                          <button
                            type="button"
                            className="btn-secondary btn-sm"
                            onClick={() => setExpandedProviderId(isExpanded ? null : p.id)}
                          >
                            {isExpanded ? "Collapse" : "Configure"}
                          </button>
                          {!isPreset && (
                            <button
                              type="button"
                              className="btn-text-del"
                              onClick={() => handleDeleteProvider(p.id)}
                              title="Delete provider"
                            >
                              <XIcon />
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Expandable Provider Configuration Panel */}
                      {isExpanded && (
                        <div className="provider-expand-body">
                          <ProviderEditor
                            provider={p}
                            onSave={(updated) => handleSaveProvider(updated, false)}
                            onCancel={() => setExpandedProviderId(null)}
                            onDelete={!isPreset ? () => handleDeleteProvider(p.id) : undefined}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {/* TAB 3: TOOLS & MCP */}
          {activeTab === "tools" && (
            <div className="modal-section">
              <h3 className="modal-section-title">Integrated MCP Servers & Tools</h3>
              <p className="modal-section-desc">Active tools and protocols available to your Cloudflare Think Agent.</p>

              <div className="providers-list">
                <div className="provider-card">
                  <div className="provider-card-head" style={{ cursor: "default" }}>
                    <div className="provider-meta">
                      <div className="provider-title-row">
                        <span className="provider-name">Parallel Web Search MCP</span>
                        <span className="badge badge-connected">Connected</span>
                      </div>
                      <span className="provider-desc">Real-time live web search and URL content extraction</span>
                    </div>
                  </div>
                </div>

                <div className="provider-card">
                  <div className="provider-card-head" style={{ cursor: "default" }}>
                    <div className="provider-meta">
                      <div className="provider-title-row">
                        <span className="provider-name">GBrain MCP Knowledge Base</span>
                        <span className="badge badge-connected">Connected</span>
                      </div>
                      <span className="provider-desc">Personal knowledge retrieval, search, memory recall & page management</span>
                    </div>
                  </div>
                </div>

                <div className="provider-card">
                  <div className="provider-card-head" style={{ cursor: "default" }}>
                    <div className="provider-meta">
                      <div className="provider-title-row">
                        <span className="provider-name">Cloudflare Computer Workspace</span>
                        <span className="badge badge-connected">Active</span>
                      </div>
                      <span className="provider-desc">Durable SQLite Virtual File System (read, write, edit, ls)</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="info-card" style={{ marginTop: 8 }}>
                <div className="info-card-icon">
                  <CpuIcon />
                </div>
                <div className="info-card-content">
                  <span className="info-card-title">Extensible MCP Infrastructure</span>
                  <span className="info-card-desc">
                    Support for custom user accounts, authentication tokens, and dynamic MCP server connections will be accessible in this section.
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: ABOUT */}
          {activeTab === "about" && (
            <div className="modal-section">
              <h3 className="modal-section-title">About cf-think-agent</h3>
              <div className="info-card">
                <div className="info-card-icon">
                  <SparklesIcon />
                </div>
                <div className="info-card-content">
                  <span className="info-card-title">Cloudflare Edge Autonomous Agent</span>
                  <span className="info-card-desc">
                    Built natively on Cloudflare Workers, Durable Objects SQLite, `@cloudflare/think`, and `@cloudflare/ai-chat`. Featuring full cloud session persistence, multi-provider inference routing, and real-time streaming tools.
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Composer ---------------- */

function Composer({
  draft,
  setDraft,
  onSubmit,
  busy,
  activeProvider,
  providers,
  onSelectProvider,
  onOpenSettings,
}: {
  draft: string;
  setDraft: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
  activeProvider: ProviderConfig;
  providers: ProviderConfig[];
  onSelectProvider: (id: string) => void;
  onOpenSettings: () => void;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const [menuOpen, setMenuOpen] = useState(false);

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
        <button
          type="button"
          className="pill pill-interactive"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Select provider and model"
        >
          <SparklesIcon />
          <span>{activeProvider.name}: {activeProvider.selectedModel}</span>
          <span className={`pill-caret ${menuOpen ? "open" : ""}`}>
            <ChevronDownIcon />
          </span>
        </button>

        {menuOpen && (
          <>
            <div className="model-popover-backdrop" onClick={() => setMenuOpen(false)} />
            <div className="model-popover">
              <div className="popover-head">Select Provider</div>
              {providers.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className={`popover-item ${p.id === activeProvider.id ? "active" : ""}`}
                  onClick={() => {
                    onSelectProvider(p.id);
                    setMenuOpen(false);
                  }}
                >
                  <div className="popover-item-title">
                    <span className="popover-item-name">{p.name}</span>
                    <span className="popover-item-sub">{p.selectedModel}</span>
                  </div>
                  {p.id === activeProvider.id && <CheckIcon />}
                </button>
              ))}
              <div className="popover-divider" />
              <button
                type="button"
                className="popover-btn"
                onClick={() => {
                  setMenuOpen(false);
                  onOpenSettings();
                }}
              >
                <SettingsIcon />
                Manage Providers & Models...
              </button>
            </div>
          </>
        )}

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
  activeProvider,
  providers,
  onSelectProvider,
  onOpenSettings,
}: {
  convoId: string;
  onFirstMessage: (text: string) => void;
  activeProvider: ProviderConfig;
  providers: ProviderConfig[];
  onSelectProvider: (id: string) => void;
  onOpenSettings: () => void;
}) {
  const agent = useAgent({ agent: "Assistant", name: convoId });
  const { messages, sendMessage, status } = useAgentChat({
    agent,
    body: () => ({
      providerId: activeProvider.id,
      customModel:
        activeProvider.id !== "cf-default" && activeProvider.endpoint && activeProvider.selectedModel
          ? {
              endpoint: activeProvider.endpoint,
              apiKey: activeProvider.apiKey,
              modelId: activeProvider.selectedModel,
              temperature: activeProvider.temperature,
              maxTokens: activeProvider.maxTokens,
              topP: activeProvider.topP,
            }
          : undefined,
    }),
  });
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
            <Composer
              draft={draft}
              setDraft={setDraft}
              onSubmit={submit}
              busy={busy}
              activeProvider={activeProvider}
              providers={providers}
              onSelectProvider={onSelectProvider}
              onOpenSettings={onOpenSettings}
            />
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
          <Composer
            draft={draft}
            setDraft={setDraft}
            onSubmit={submit}
            busy={busy}
            activeProvider={activeProvider}
            providers={providers}
            onSelectProvider={onSelectProvider}
            onOpenSettings={onOpenSettings}
          />
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
  const [sideOpen, setSideOpen] = useState<boolean>(
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 900px)").matches
  );
  const [cloudReady, setCloudReady] = useState(false);

  // Theme State
  const [theme, setTheme] = useState<ThemeMode>(loadLocalTheme);

  // Providers State (Local hot cache + Cloud DO source of truth)
  const [providers, setProviders] = useState<ProviderConfig[]>(loadLocalProviders);
  const [activeProviderId, setActiveProviderId] = useState<string>(() =>
    loadActiveProviderId(loadLocalProviders())
  );
  const [settingsOpen, setSettingsOpen] = useState(false);

  const activeProvider = providers.find((p) => p.id === activeProviderId) ?? providers[0] ?? DEFAULT_PRESET;

  // Sync Theme attribute
  useEffect(() => {
    saveLocalTheme(theme);
    if (theme === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
  }, [theme]);

  // Sidebar Quick Theme Toggle (cycles System -> Dark -> Light)
  const cycleTheme = () => {
    setTheme((prev) => {
      if (prev === "system") return "dark";
      if (prev === "dark") return "light";
      return "system";
    });
  };

  const handleSelectActiveProvider = (id: string) => {
    setActiveProviderId(id);
    saveActiveProviderId(id);
    cloudSetActiveProvider(id).catch(() => {});
  };

  const handleSaveProviders = (newProviders: ProviderConfig[], newActiveId?: string) => {
    setProviders(newProviders);
    saveLocalProviders(newProviders);
    if (newActiveId) {
      setActiveProviderId(newActiveId);
      saveActiveProviderId(newActiveId);
    }
    // Cloud sync
    cloudSaveAllProviders(newProviders).catch(() => {});
    if (newActiveId) {
      cloudSetActiveProvider(newActiveId).catch(() => {});
    }
  };

  const closeSideOnMobile = () => {
    if (typeof window !== "undefined" && !window.matchMedia("(min-width: 900px)").matches) {
      setSideOpen(false);
    }
  };

  // On mount: pull cloud list for both convos and providers
  useEffect(() => {
    let cancelled = false;
    (async () => {
      // 1. Convos sync
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
        if (merged.length > 0) {
          const localIds = new Set(local.map((c) => c.id));
          if (!localIds.has(active) && !merged.some((c) => c.id === active)) {
            setActive(merged[0].id);
          }
        }
        for (const o of orphans) {
          try {
            await cloudTouchConvs(o.id, o.title);
          } catch {
            /* best effort */
          }
        }
      } catch {
        if (!cancelled) setCloudReady(true);
      }

      // 2. Providers cloud sync
      try {
        const cloudProvidersRes = await cloudListProviders();
        if (cancelled) return;
        const localProviders = loadLocalProviders();
        const cloudList = cloudProvidersRes.providers || [];
        const cloudIds = new Set(cloudList.map((p) => p.id));
        const localUnsynced = localProviders.filter((p) => p.id !== "cf-default" && !cloudIds.has(p.id));

        if (localUnsynced.length > 0) {
          // Push local offline additions up to Cloudflare
          const merged = [...cloudList, ...localUnsynced];
          setProviders(merged);
          saveLocalProviders(merged);
          await cloudSaveAllProviders(merged);
        } else if (cloudList.length > 0) {
          setProviders(cloudList);
          saveLocalProviders(cloudList);
        }

        if (cloudProvidersRes.activeId && (cloudList.some((p) => p.id === cloudProvidersRes.activeId) || cloudProvidersRes.activeId === "cf-default")) {
          setActiveProviderId(cloudProvidersRes.activeId);
          saveActiveProviderId(cloudProvidersRes.activeId);
        }
      } catch {
        /* best effort */
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
          <div className="side-rail-bottom">
            <button
              type="button"
              className="ghost"
              onClick={cycleTheme}
              aria-label="Toggle Theme"
              title={`Current theme: ${theme} (Click to switch)`}
            >
              {theme === "dark" ? <MoonIcon /> : theme === "light" ? <SunIcon /> : <MonitorIcon />}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => setSettingsOpen(true)}
              aria-label="Settings"
              title="Settings"
            >
              <SettingsIcon />
            </button>
          </div>
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
          <div className="side-foot-actions">
            <button
              type="button"
              className="theme-quick-btn"
              onClick={cycleTheme}
              aria-label="Toggle Theme"
              title={`Theme: ${theme} (Click to toggle)`}
            >
              {theme === "dark" ? <MoonIcon /> : theme === "light" ? <SunIcon /> : <MonitorIcon />}
            </button>
            <button
              type="button"
              className="side-settings-btn"
              onClick={() => setSettingsOpen(true)}
              aria-label="Settings"
            >
              <SettingsIcon />
              <span>Settings</span>
            </button>
          </div>
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
          activeProvider={activeProvider}
          providers={providers}
          onSelectProvider={handleSelectActiveProvider}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </Suspense>

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        onSetTheme={setTheme}
        providers={providers}
        activeProviderId={activeProviderId}
        onSaveProviders={handleSaveProviders}
        onSelectActiveProvider={handleSelectActiveProvider}
        convosCount={convos.length}
      />
    </div>
  );
}
