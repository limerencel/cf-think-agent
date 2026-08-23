import { Component, Suspense, type ReactElement, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { getToolName, isToolUIPart } from "ai";
import type { UIMessage } from "ai";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { McpServerConfig, McpToolDef, McpAuthType, HindsightConfig } from "./mcp-types";
import type {
  MnemosyneConfig,
  EpisodicMemoryItem,
  TripleItem,
  WorkingMemoryItem,
  MnemosyneStats,
  MnemosyneRecallResult,
} from "./mnemosyne";

export type {
  McpServerConfig,
  McpToolDef,
  McpAuthType,
  HindsightConfig,
  MnemosyneConfig,
  EpisodicMemoryItem,
  TripleItem,
  WorkingMemoryItem,
  MnemosyneStats,
  MnemosyneRecallResult,
};

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
  useResponseApi?: boolean;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  reasoningEffort?: "low" | "medium" | "high" | "none";
}

const LS_KEY = "edgeagent.conversations";
const LS_THEME_KEY = "edgeagent.theme";
const LS_PROVIDERS_KEY = "edgeagent.providers";
const LS_ACTIVE_PROVIDER_KEY = "edgeagent.active_provider_id";
const LS_MCP_KEY = "edgeagent.mcp_servers";

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
      return list.filter((p) => p.id !== "cf-default");
    }
    return [];
  } catch {
    return [];
  }
}

function saveLocalProviders(providers: ProviderConfig[]): void {
  try {
    const clean = providers.filter((p) => p.id !== "cf-default");
    localStorage.setItem(LS_PROVIDERS_KEY, JSON.stringify(clean));
  } catch {
    /* ignore */
  }
}

function loadActiveProviderId(providers: ProviderConfig[]): string {
  try {
    const raw = localStorage.getItem(LS_ACTIVE_PROVIDER_KEY);
    if (raw && raw !== "cf-default" && providers.some((p) => p.id === raw)) return raw;
    const def = providers.find((p) => p.isDefault);
    return def?.id ?? providers[0]?.id ?? "";
  } catch {
    return "";
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

/* ---------------- MCP Servers Cloud API helpers ---------------- */

function loadLocalMcpServers(): McpServerConfig[] {
  try {
    const raw = localStorage.getItem(LS_MCP_KEY);
    if (raw) {
      const list = JSON.parse(raw) as McpServerConfig[];
      return list.filter((s) => s.id !== "gbrain-default");
    }
    return [];
  } catch {
    return [];
  }
}

function saveLocalMcpServers(servers: McpServerConfig[]): void {
  try {
    const clean = servers.filter((s) => s.id !== "gbrain-default");
    localStorage.setItem(LS_MCP_KEY, JSON.stringify(clean));
  } catch {
    /* ignore */
  }
}

const LOCAL_STORAGE_PROMPT_KEY = "think_custom_system_prompt";
const LOCAL_STORAGE_PROMPT_MODE_KEY = "think_system_prompt_mode";

function loadLocalSystemPrompt(): { prompt: string; mode: "append" | "override" } {
  if (typeof window === "undefined") return { prompt: "", mode: "append" };
  try {
    const prompt = localStorage.getItem(LOCAL_STORAGE_PROMPT_KEY) || "";
    const mode = (localStorage.getItem(LOCAL_STORAGE_PROMPT_MODE_KEY) as "append" | "override") || "append";
    return { prompt, mode };
  } catch {
    return { prompt: "", mode: "append" };
  }
}

function saveLocalSystemPrompt(prompt: string, mode: "append" | "override") {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(LOCAL_STORAGE_PROMPT_KEY, prompt);
    localStorage.setItem(LOCAL_STORAGE_PROMPT_MODE_KEY, mode);
  } catch {
    /* ignore */
  }
}

async function cloudGetSetting(key: string): Promise<string | null> {
  const res = await fetch(`/api/settings?key=${encodeURIComponent(key)}`);
  if (!res.ok) return null;
  const data = (await res.json()) as any;
  return data?.value ?? null;
}

async function cloudSetSetting(key: string, value: string): Promise<void> {
  await fetch("/api/settings", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ key, value }),
  });
}

async function cloudListMcpServers(): Promise<McpServerConfig[]> {
  const res = await fetch("/api/mcp/list", { headers: { accept: "application/json" } });
  if (!res.ok) throw new Error(`cloud list mcp ${res.status}`);
  const data = (await res.json()) as { ok: boolean; servers: McpServerConfig[] };
  if (!data.ok) throw new Error("cloud list mcp !ok");
  return data.servers || [];
}

async function cloudSaveMcpServer(server: McpServerConfig): Promise<McpServerConfig[]> {
  const res = await fetch("/api/mcp/save", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ server }),
  });
  if (!res.ok) throw new Error(`cloud save mcp ${res.status}`);
  const data = (await res.json()) as { ok: boolean; servers: McpServerConfig[] };
  if (!data.ok) throw new Error("cloud save mcp !ok");
  return data.servers;
}

async function cloudSaveAllMcpServers(servers: McpServerConfig[]): Promise<McpServerConfig[]> {
  const res = await fetch("/api/mcp/save", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ servers }),
  });
  if (!res.ok) throw new Error(`cloud save all mcp ${res.status}`);
  const data = (await res.json()) as { ok: boolean; servers: McpServerConfig[] };
  if (!data.ok) throw new Error("cloud save all mcp !ok");
  return data.servers;
}

async function cloudRemoveMcpServer(id: string): Promise<McpServerConfig[]> {
  const res = await fetch("/api/mcp/remove", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error(`cloud remove mcp ${res.status}`);
  const data = (await res.json()) as { ok: boolean; servers: McpServerConfig[] };
  if (!data.ok) throw new Error("cloud remove mcp !ok");
  return data.servers;
}

async function cloudFetchMcpTools(
  endpoint: string,
  authType: McpAuthType,
  bearerToken?: string,
  cfAccessClientId?: string,
  cfAccessClientSecret?: string,
  oauthTokens?: any,
  serverId?: string
): Promise<McpToolDef[]> {
  const res = await fetch("/api/mcp/fetch-tools", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      endpoint,
      authType,
      bearerToken,
      cfAccessClientId,
      cfAccessClientSecret,
      oauthTokens,
      serverId,
    }),
  });
  const data = (await res.json()) as { ok: boolean; tools?: McpToolDef[]; error?: string };
  if (!res.ok || !data.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data.tools || [];
}

async function cloudStartMcpOAuth(
  endpoint: string,
  serverId?: string,
  serverName?: string,
  clientId?: string,
  clientSecret?: string
): Promise<{ ok: boolean; authUrl: string; serverId: string; state: string; discovery: any }> {
  const res = await fetch("/api/mcp/oauth/start", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ endpoint, serverId, serverName, clientId, clientSecret }),
  });
  const data = (await res.json()) as any;
  if (!res.ok || !data.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

async function cloudGetHindsightConfig(): Promise<HindsightConfig> {
  const res = await fetch("/api/hindsight/config");
  if (!res.ok) throw new Error(`cloud get hindsight config ${res.status}`);
  const data = (await res.json()) as { ok: boolean; config: HindsightConfig };
  if (!data.ok) throw new Error("cloud get hindsight config !ok");
  return data.config;
}

async function cloudSaveHindsightConfig(config: HindsightConfig): Promise<HindsightConfig> {
  const res = await fetch("/api/hindsight/config", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`cloud save hindsight config ${res.status}`);
  const data = (await res.json()) as { ok: boolean; config: HindsightConfig };
  if (!data.ok) throw new Error("cloud save hindsight config !ok");
  return data.config;
}

/* ---------------- Mnemosyne Zero-Cloud Native Memory Client ---------------- */

async function cloudGetMnemosyneConfig(): Promise<MnemosyneConfig> {
  const res = await fetch("/api/mnemosyne/config");
  if (!res.ok) throw new Error(`cloud get mnemosyne config ${res.status}`);
  const data = (await res.json()) as { ok: boolean; config: MnemosyneConfig };
  if (!data.ok) throw new Error("cloud get mnemosyne config !ok");
  return data.config;
}

async function cloudSaveMnemosyneConfig(config: MnemosyneConfig): Promise<MnemosyneConfig> {
  const res = await fetch("/api/mnemosyne/config", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`cloud save mnemosyne config ${res.status}`);
  const data = (await res.json()) as { ok: boolean; config: MnemosyneConfig };
  if (!data.ok) throw new Error("cloud save mnemosyne config !ok");
  return data.config;
}

async function cloudGetMnemosyneStats(): Promise<MnemosyneStats> {
  const res = await fetch("/api/mnemosyne/stats");
  if (!res.ok) throw new Error(`cloud get mnemosyne stats ${res.status}`);
  const data = (await res.json()) as { ok: boolean; stats: MnemosyneStats };
  if (!data.ok) throw new Error("cloud get mnemosyne stats !ok");
  return data.stats;
}

async function cloudListMnemosyneMemories(): Promise<EpisodicMemoryItem[]> {
  const res = await fetch("/api/mnemosyne/memories");
  if (!res.ok) throw new Error(`cloud get mnemosyne memories ${res.status}`);
  const data = (await res.json()) as { ok: boolean; memories: EpisodicMemoryItem[] };
  if (!data.ok) throw new Error("cloud get mnemosyne memories !ok");
  return data.memories || [];
}

async function cloudListMnemosyneTriples(): Promise<TripleItem[]> {
  const res = await fetch("/api/mnemosyne/triples");
  if (!res.ok) throw new Error(`cloud get mnemosyne triples ${res.status}`);
  const data = (await res.json()) as { ok: boolean; triples: TripleItem[] };
  if (!data.ok) throw new Error("cloud get mnemosyne triples !ok");
  return data.triples || [];
}

async function cloudRememberMnemosyne(payload: {
  content: string;
  importance?: number;
  isWorkingMemory?: boolean;
  ttlSeconds?: number;
  triples?: Array<{ subject: string; predicate: string; object: string }>;
}): Promise<{ ok: boolean; id: string }> {
  const res = await fetch("/api/mnemosyne/remember", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = (await res.json()) as any;
  if (!res.ok || !data.ok) throw new Error(data.error || "Failed to save memory");
  return data;
}

async function cloudRecallMnemosyne(query: string, topK = 5): Promise<MnemosyneRecallResult> {
  const res = await fetch("/api/mnemosyne/recall", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, topK, includeTriples: true }),
  });
  const data = (await res.json()) as any;
  if (!res.ok || !data.ok) throw new Error(data.error || "Failed to recall memories");
  return data.result;
}

async function cloudDeleteMnemosyneMemory(id: string): Promise<void> {
  const res = await fetch("/api/mnemosyne/memories", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ action: "delete", id }),
  });
  const data = (await res.json()) as any;
  if (!res.ok || !data.ok) throw new Error(data.error || "Failed to delete memory");
}

async function cloudDeleteMnemosyneTriple(id: string): Promise<void> {
  const res = await fetch("/api/mnemosyne/triples", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ action: "delete", id }),
  });
  const data = (await res.json()) as any;
  if (!res.ok || !data.ok) throw new Error(data.error || "Failed to delete triple");
}

async function cloudClearMnemosyne(target?: "all" | "working" | "episodic" | "triples"): Promise<void> {
  const res = await fetch("/api/mnemosyne/clear", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ target: target || "all" }),
  });
  const data = (await res.json()) as any;
  if (!res.ok || !data.ok) throw new Error(data.error || "Failed to clear memory");
}

async function cloudConsolidateMnemosyne(): Promise<number> {
  const res = await fetch("/api/mnemosyne/consolidate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = (await res.json()) as any;
  if (!res.ok || !data.ok) throw new Error(data.error || "Failed to consolidate memory");
  return data.count || 0;
}


function newId(): string {
  return "c" + Math.random().toString(36).slice(2, 10);
}

function FolderIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function FileCodeIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="m10 13-2 2 2 2" />
      <path d="m14 17 2-2-2-2" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function ArchiveIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="21 8 21 21 3 21 3 8" />
      <rect x="1" y="3" width="22" height="5" />
      <line x1="10" y1="12" x2="14" y2="12" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z" />
    </svg>
  );
}

function ZapIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" stroke="none">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function TerminalIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  );
}

function KeyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 2l-2 2m-1.5 1.5L14 9l-1.5-1.5L11 9l-1.5-1.5L8 9a5 5 0 1 0 7 7l5.5-5.5a2.12 2.12 0 0 0 0-3L21 2z" />
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
      <path
        d="M3 4h10M6 4V2.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5V4M12.5 4v9a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 13V4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
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

function ImageIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <rect x="2" y="2.5" width="12" height="11" rx="2" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="5.5" cy="6" r="1.2" fill="currentColor" />
      <path d="M14 11l-3.5-3.5L4 14" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

function LayersIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

function NetworkIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="16" y="16" width="6" height="6" rx="1" />
      <rect x="2" y="16" width="6" height="6" rx="1" />
      <rect x="9" y="2" width="6" height="6" rx="1" />
      <path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3" />
      <path d="M12 12V8" />
    </svg>
  );
}

function CalendarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  );
}

function CompassIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}


function FileTextIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
      <path d="M3.5 2.5h6l3.5 3.5v7.5a1 1 0 0 1-1 1h-8.5a1 1 0 0 1-1-1v-10a1 1 0 0 1 1-1z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9.5 2.5v3.5h3.5M6 8h4M6 10.5h4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

function PaperclipIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path
        d="M13.5 7.5l-5.8 5.8a3.18 3.18 0 0 1-4.5-4.5l6.3-6.3a2.12 2.12 0 0 1 3 3l-6.3 6.3a1.06 1.06 0 0 1-1.5-1.5l5.3-5.3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
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

function AlertTriangleIcon() {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
      <path
        d="M10 2.5L1.8 16.5a1 1 0 0 0 .86 1.5h14.68a1 1 0 0 0 .86-1.5L10 2.5z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M10 8v4M10 14.5v.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
      <path
        d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9L13.5 5.5M13.5 2v3.5h-3.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
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

function ClockIcon() {
  return (
    <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
      <circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <path d="M8 4.5v3.8l2.5 1.5" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

function TokenIcon() {
  return (
    <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.3" strokeDasharray="3 1.5" />
      <path d="M6 7.2h4M6 9.2h4M7.2 5.5v5.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
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

function PanelRightIcon() {
  return (
    <svg viewBox="0 0 20 20" width="17" height="17" aria-hidden="true">
      <rect x="2.5" y="3.5" width="15" height="13" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M12 3.5v13" stroke="currentColor" strokeWidth="1.5" />
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
        const rawName = getToolName(part);
        const state = "state" in part ? (part.state as string) : "";
        const cls = state === "output-available" ? "ok" : state === "output-error" ? "err" : "run";
        let display = rawName;
        if (["read", "write", "edit", "ls", "rm", "mkdir", "stat"].includes(rawName)) {
          display = `workspace · ${rawName}`;
        }
        return (
          <span key={`${rawName}-${i}`} className={`chip ${cls}`}>
            {display}
          </span>
        );
      })}
    </div>
  );
}

/* ---------------- Reasoning & Chain of Thought ---------------- */

interface ExtractedMessageContent {
  thoughtText: string | null;
  mainText: string;
  isThinking: boolean;
}

function extractThoughtAndAnswer(message: UIMessage): ExtractedMessageContent {
  let thoughtPartsText = "";
  let textPartsText = "";

  for (const part of message.parts) {
    if (part.type === "reasoning" || (part as any).type === "thought") {
      thoughtPartsText += (part as any).text || (part as any).reasoning || (part as any).thought || "";
    } else if (part.type === "text") {
      textPartsText += (part as { type: "text"; text: string }).text || "";
    }
  }

  // Parse <think>...</think> blocks embedded in LLM text output (DeepSeek-R1,
  // QwQ, or our server-side reasoningAwareFetch which injects them). Handles:
  //  - multiple think blocks (one per agent step / tool round)
  //  - an unterminated trailing <think> (stream still in progress, or the
  //    model forgot to close): treated as in-progress thinking, everything
  //    before it stays as the visible answer.
  let combinedThought = thoughtPartsText;
  const cleanedParts: string[] = [];
  let rest = textPartsText;
  let hasOpenThink = false;

  while (rest.length > 0) {
    const start = rest.indexOf("<think>");
    if (start === -1) {
      cleanedParts.push(rest);
      break;
    }
    cleanedParts.push(rest.substring(0, start));
    const afterStart = rest.substring(start + 7);
    const end = afterStart.indexOf("</think>");
    if (end === -1) {
      // Unterminated: everything after is (in-progress) thinking.
      const tail = afterStart.trim();
      if (tail) {
        combinedThought = combinedThought ? combinedThought + "\n\n" + tail : tail;
        hasOpenThink = true;
      }
      break;
    }
    const block = afterStart.substring(0, end).trim();
    if (block) {
      combinedThought = combinedThought ? combinedThought + "\n\n" + block : block;
    }
    rest = afterStart.substring(end + 8);
  }

  const cleanedMain = cleanedParts.join("").trim();
  const thoughtText = combinedThought.trim() || null;

  if (thoughtText) {
    return {
      thoughtText,
      mainText: cleanedMain,
      // In-progress only when a think block is still open AND the model is
      // streaming (caller ORs in the busy state for the last message).
      isThinking: hasOpenThink,
    };
  }

  return {
    thoughtText: null,
    mainText: textPartsText,
    isThinking: false,
  };
}

function ThoughtBlock({ thought, isThinking }: { thought: string; isThinking?: boolean }) {
  const [open, setOpen] = useState(isThinking ?? false);

  useEffect(() => {
    if (isThinking) setOpen(true);
  }, [isThinking]);

  return (
    <div className={`thought-container ${open ? "open" : "collapsed"}`}>
      <button
        type="button"
        className="thought-toggle-btn"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <div className="thought-toggle-left">
          <span className="thought-sparkle-icon">
            <BrainIcon />
          </span>
          <span className="thought-title">
            {isThinking ? "Thinking…" : "Chain of Thought"}
          </span>
        </div>
        <div className="thought-toggle-right">
          <span className="thought-chevron">{open ? <ChevronDownIcon /> : <ChevronRightIcon />}</span>
        </div>
      </button>

      {open && (
        <div className="thought-content">
          <Markdown text={thought} />
        </div>
      )}
    </div>
  );
}

/**
 * Standard CJK + English Tokenizer Estimator
 */
function estimateTokenCount(text: string): number {
  if (!text) return 0;
  const cjkMatches = text.match(/[\u4e00-\u9fa5\u3040-\u30ff\uac00-\ud7af]/g);
  const cjkCount = cjkMatches ? cjkMatches.length : 0;
  const nonCjk = text.replace(/[\u4e00-\u9fa5\u3040-\u30ff\uac00-\ud7af]/g, " ");
  const wordMatches = nonCjk.match(/\S+/g);
  const wordCount = wordMatches ? wordMatches.length : 0;
  const punctMatches = text.match(/[^\w\s\u4e00-\u9fa5\u3040-\u30ff\uac00-\ud7af]/g);
  const punctCount = punctMatches ? punctMatches.length : 0;
  const est = Math.round(cjkCount * 1.2 + wordCount * 1.3 + punctCount * 0.5);
  return Math.max(1, est);
}

function MessageMetaFooter({
  rawText,
  durationSec,
}: {
  rawText: string;
  durationSec?: number;
}) {
  const [copied, setCopied] = useState(false);
  const tokenCount = useMemo(() => estimateTokenCount(rawText), [rawText]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(rawText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  };

  if (!rawText.trim()) return null;

  return (
    <div className="message-meta-footer">
      {durationSec !== undefined && durationSec > 0 && (
        <span className="message-meta-item" title={`Total turn latency: ${durationSec}s`}>
          <ClockIcon />
          <span>{durationSec}s</span>
        </span>
      )}
      {tokenCount > 0 && (
        <span className="message-meta-item" title={`Estimated token usage: ~${tokenCount.toLocaleString()} tokens`}>
          <TokenIcon />
          <span>{tokenCount.toLocaleString()} tok</span>
        </span>
      )}
      <button
        type="button"
        className="message-meta-copy-btn"
        onClick={handleCopy}
        title="Copy response markdown"
        aria-label="Copy response markdown"
      >
        {copied ? <CheckIcon /> : <CopyIcon />}
        <span>{copied ? "Copied" : "Copy"}</span>
      </button>
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

/* ---------------- Workspace File Explorer & Previewer ---------------- */

export interface WorkspaceFileItem {
  path: string;
  name: string;
  size: number;
  mtime?: number;
  isDirectory: boolean;
}

function createZipBlob(files: Array<{ path: string; content: string }>): Blob {
  const encoder = new TextEncoder();
  const fileRecords: Array<{
    nameBytes: Uint8Array;
    contentBytes: Uint8Array;
    crc32: number;
    offset: number;
  }> = [];

  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    crcTable[i] = c;
  }
  const calcCrc32 = (bytes: Uint8Array): number => {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
      crc = crcTable[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
  };

  const parts: Uint8Array[] = [];
  let currentOffset = 0;

  for (const f of files) {
    const cleanPath = f.path.replace(/^\/+/, "");
    const nameBytes = encoder.encode(cleanPath);
    const contentBytes = encoder.encode(f.content);
    const crc = calcCrc32(contentBytes);
    const size = contentBytes.length;

    const header = new Uint8Array(30 + nameBytes.length);
    const view = new DataView(header.buffer);
    view.setUint32(0, 0x04034b50, true);
    view.setUint16(4, 20, true);
    view.setUint16(6, 0, true);
    view.setUint16(8, 0, true);
    view.setUint16(10, 0, true);
    view.setUint16(12, 0, true);
    view.setUint32(14, crc, true);
    view.setUint32(18, size, true);
    view.setUint32(22, size, true);
    view.setUint16(26, nameBytes.length, true);
    view.setUint16(28, 0, true);
    header.set(nameBytes, 30);

    fileRecords.push({
      nameBytes,
      contentBytes,
      crc32: crc,
      offset: currentOffset,
    });

    parts.push(header);
    parts.push(contentBytes);
    currentOffset += header.length + contentBytes.length;
  }

  const cdOffset = currentOffset;
  let cdSize = 0;

  for (const rec of fileRecords) {
    const cdHeader = new Uint8Array(46 + rec.nameBytes.length);
    const view = new DataView(cdHeader.buffer);
    view.setUint32(0, 0x02014b50, true);
    view.setUint16(4, 20, true);
    view.setUint16(6, 20, true);
    view.setUint16(8, 0, true);
    view.setUint16(10, 0, true);
    view.setUint16(12, 0, true);
    view.setUint16(14, 0, true);
    view.setUint32(16, rec.crc32, true);
    view.setUint32(20, rec.contentBytes.length, true);
    view.setUint32(24, rec.contentBytes.length, true);
    view.setUint16(28, rec.nameBytes.length, true);
    view.setUint16(30, 0, true);
    view.setUint16(32, 0, true);
    view.setUint16(34, 0, true);
    view.setUint16(36, 0, true);
    view.setUint32(38, 0, true);
    view.setUint32(42, rec.offset, true);
    cdHeader.set(rec.nameBytes, 46);

    parts.push(cdHeader);
    cdSize += cdHeader.length;
  }

  const eocd = new Uint8Array(22);
  const eocdView = new DataView(eocd.buffer);
  eocdView.setUint32(0, 0x06054b50, true);
  eocdView.setUint16(4, 0, true);
  eocdView.setUint16(6, 0, true);
  eocdView.setUint16(8, fileRecords.length, true);
  eocdView.setUint16(10, fileRecords.length, true);
  eocdView.setUint32(12, cdSize, true);
  eocdView.setUint32(16, cdOffset, true);
  eocdView.setUint16(20, 0, true);

  parts.push(eocd);
  return new Blob(parts as BlobPart[], { type: "application/zip" });
}

function getFileLanguage(fileName: string): string {
  const ext = fileName.split(".").pop()?.toLowerCase() || "";
  const map: Record<string, string> = {
    ts: "typescript",
    tsx: "typescript",
    js: "javascript",
    jsx: "javascript",
    py: "python",
    json: "json",
    html: "html",
    css: "css",
    md: "markdown",
    sql: "sql",
    sh: "bash",
    yaml: "yaml",
    yml: "yaml",
    toml: "toml",
    rs: "rust",
    go: "go",
  };
  return map[ext] || "text";
}

export interface StagedAttachment {
  id: string;
  file: File;
  name: string;
  size: number;
  type: string;
  previewUrl?: string;
  isImage: boolean;
}

function getFileExtBadge(fileName: string): { label: string; color: string } {
  const ext = fileName.split(".").pop()?.toLowerCase() || "";
  switch (ext) {
    case "ts":
    case "tsx":
      return { label: "TS", color: "#3178C6" };
    case "js":
    case "jsx":
      return { label: "JS", color: "#E5A00D" };
    case "py":
      return { label: "PY", color: "#3776AB" };
    case "json":
      return { label: "JSON", color: "#E08A6C" };
    case "md":
      return { label: "MD", color: "#8E44AD" };
    case "html":
      return { label: "HTML", color: "#E34F26" };
    case "css":
      return { label: "CSS", color: "#1572B6" };
    case "sql":
      return { label: "SQL", color: "#336791" };
    case "sh":
    case "bash":
      return { label: "SH", color: "#4EAA25" };
    case "png":
    case "jpg":
    case "jpeg":
    case "gif":
    case "webp":
    case "svg":
    case "ico":
    case "bmp":
    case "avif":
      return { label: "IMG", color: "#D97757" };
    case "pdf":
      return { label: "PDF", color: "#E74C3C" };
    case "csv":
      return { label: "CSV", color: "#27AE60" };
    case "zip":
    case "tar":
    case "gz":
      return { label: "ZIP", color: "#F39C12" };
    case "rs":
      return { label: "RS", color: "#DEA584" };
    case "go":
      return { label: "GO", color: "#00ADD8" };
    default:
      return { label: ext.toUpperCase().slice(0, 4) || "FILE", color: "var(--muted)" };
  }
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes === 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function WorkspacePanel({
  convoId,
  isOpen,
  onClose,
  onFileCountUpdate,
  refreshToken,
}: {
  convoId: string;
  isOpen: boolean;
  onClose: () => void;
  onFileCountUpdate?: (count: number) => void;
  refreshToken?: number;
}) {
  const [files, setFiles] = useState<WorkspaceFileItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [downloadingZip, setDownloadingZip] = useState(false);
  const [copied, setCopied] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const refreshFiles = useCallback(async (autoSelect = false) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/workspace/files?convoId=${encodeURIComponent(convoId)}`);
      const data = (await res.json()) as { ok: boolean; files?: WorkspaceFileItem[]; error?: string };
      if (data.ok && data.files) {
        const fileOnlyList = data.files.filter((f) => !f.isDirectory);
        setFiles(fileOnlyList);
        onFileCountUpdate?.(fileOnlyList.length);
        if (autoSelect || !selectedPath || !fileOnlyList.some((f) => f.path === selectedPath)) {
          if (fileOnlyList.length > 0) {
            setSelectedPath(fileOnlyList[0].path);
          } else {
            setSelectedPath(null);
            setFileContent(null);
          }
        }
      }
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, [convoId, selectedPath, onFileCountUpdate]);

  useEffect(() => {
    if (convoId) {
      refreshFiles(true);
    }
  }, [convoId, refreshToken]);

  useEffect(() => {
    if (!selectedPath) {
      setFileContent(null);
      return;
    }
    let cancelled = false;
    setFileLoading(true);
    fetch(`/api/workspace/file?convoId=${encodeURIComponent(convoId)}&path=${encodeURIComponent(selectedPath)}`)
      .then((res) => res.json())
      .then((data: any) => {
        if (!cancelled && data.ok && data.content !== undefined) {
          setFileContent(data.content);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setFileLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [convoId, selectedPath]);

  const handleDownloadZip = async () => {
    setDownloadingZip(true);
    try {
      const res = await fetch(`/api/workspace/archive?convoId=${encodeURIComponent(convoId)}`);
      const data = (await res.json()) as { ok: boolean; files?: Array<{ path: string; content: string }> };
      if (data.ok && data.files && data.files.length > 0) {
        const blob = createZipBlob(data.files);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `workspace-${convoId}.zip`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      alert("Failed to export workspace zip: " + String(err));
    } finally {
      setDownloadingZip(false);
    }
  };

  const handleDownloadSingle = () => {
    if (!selectedPath || fileContent === null) return;
    const blob = new Blob([fileContent], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = selectedPath.split("/").pop() || "file.txt";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleCopyContent = async () => {
    if (fileContent === null) return;
    try {
      await navigator.clipboard.writeText(fileContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };

  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const handleDirectUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    setUploading(true);
    const formData = new FormData();
    for (let i = 0; i < e.target.files.length; i++) {
      formData.append("files", e.target.files[i], e.target.files[i].name);
    }
    try {
      const res = await fetch(`/api/workspace/upload?convoId=${encodeURIComponent(convoId)}`, {
        method: "POST",
        body: formData,
      });
      const data = (await res.json()) as any;
      if (data.ok) {
        refreshFiles(true);
      }
    } catch (err: any) {
      alert("Failed to upload file to workspace: " + (err.message || String(err)));
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const handleDeleteFile = async (path: string) => {
    if (!confirm(`Are you sure you want to delete ${path}?`)) return;
    try {
      const res = await fetch(`/api/workspace/delete?convoId=${encodeURIComponent(convoId)}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ path }),
      });
      const data = (await res.json()) as any;
      if (data.ok) {
        refreshFiles(true);
      }
    } catch (err: any) {
      alert("Failed to delete file: " + (err.message || String(err)));
    }
  };

  const filteredFiles = files.filter(
    (f) => !searchQuery || f.path.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const selectedFile = files.find((f) => f.path === selectedPath);
  const isSelectedImage =
    selectedFile &&
    ["png", "jpg", "jpeg", "gif", "webp", "svg", "ico", "bmp", "avif"].includes(
      selectedFile.name.split(".").pop()?.toLowerCase() || ""
    );
  const lang = selectedFile ? getFileLanguage(selectedFile.name) : "text";

  return (
    <aside className={"workspace-drawer" + (isOpen ? " open" : "")}>
      {/* Workspace Panel Header */}
      <div className="side-top workspace-side-top">
        <div className="workspace-top-title">
          <span className="wordmark">workspace</span>
          {files.length > 0 && <span className="workspace-count-badge">{files.length}</span>}
        </div>
        <div className="workspace-top-actions">
          <button
            type="button"
            className="ghost side-icon-btn"
            onClick={() => uploadInputRef.current?.click()}
            disabled={uploading}
            title="Upload files to workspace"
            aria-label="Upload files"
          >
            <PlusIcon />
          </button>
          <input
            ref={uploadInputRef}
            type="file"
            multiple
            style={{ display: "none" }}
            onChange={handleDirectUpload}
          />
          <button
            type="button"
            className="ghost side-icon-btn"
            onClick={() => refreshFiles(false)}
            disabled={loading || uploading}
            title="Refresh workspace files"
            aria-label="Refresh"
          >
            <RefreshIcon />
          </button>
          {files.length > 0 && (
            <button
              type="button"
              className="workspace-zip-pill"
              onClick={handleDownloadZip}
              disabled={downloadingZip}
              title="Download all files as .zip archive"
            >
              <DownloadIcon />
              <span>{downloadingZip ? "Archiving…" : "ZIP"}</span>
            </button>
          )}
          <button
            type="button"
            className="ghost side-collapse"
            onClick={onClose}
            title="Collapse workspace"
            aria-label="Collapse workspace"
          >
            <PanelRightIcon />
          </button>
        </div>
      </div>

      {/* Search bar if many files */}
      {files.length > 3 && (
        <div className="workspace-search-bar">
          <input
            type="text"
            placeholder="Filter files…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      )}

      {/* Main Workspace Body */}
      <div className="workspace-drawer-body">
        {files.length === 0 ? (
          <div className="workspace-empty">
            <div className="workspace-empty-icon">
              <FolderIcon />
            </div>
            <h3>Workspace is empty</h3>
            <p>
              Files uploaded or created by the agent in this session will appear here in real-time.
            </p>
          </div>
        ) : (
          <div className="workspace-content">
            {/* File List Strip */}
            <div className="workspace-file-list">
              {filteredFiles.map((f) => {
                const badge = getFileExtBadge(f.name);
                const isSelected = f.path === selectedPath;
                return (
                  <button
                    key={f.path}
                    type="button"
                    className={`workspace-file-tab ${isSelected ? "active" : ""}`}
                    onClick={() => setSelectedPath(f.path)}
                    title={f.path}
                  >
                    <span className="file-badge" style={{ color: badge.color }}>
                      {badge.label}
                    </span>
                    <span className="file-tab-name">{f.name}</span>
                    <span className="file-tab-size">{formatBytes(f.size)}</span>
                  </button>
                );
              })}
            </div>

            {/* File Preview Area */}
            <div className="workspace-preview">
              {selectedFile && (
                <div className="workspace-preview-header">
                  <div className="workspace-preview-path" title={selectedFile.path}>
                    <FileCodeIcon />
                    <span>{selectedFile.path}</span>
                    <span className="workspace-file-meta">{formatBytes(selectedFile.size)}</span>
                  </div>
                  <div className="workspace-preview-actions">
                    <button
                      type="button"
                      className="preview-action-btn"
                      onClick={handleCopyContent}
                      title="Copy file content"
                    >
                      {copied ? <CheckIcon /> : <CopyIcon />}
                      <span>{copied ? "Copied" : "Copy"}</span>
                    </button>
                    <button
                      type="button"
                      className="preview-action-btn"
                      onClick={handleDownloadSingle}
                      title="Download file"
                    >
                      <DownloadIcon />
                      <span>Download</span>
                    </button>
                    <button
                      type="button"
                      className="preview-action-btn delete-btn"
                      onClick={() => handleDeleteFile(selectedFile.path)}
                      title="Delete file"
                    >
                      <TrashIcon />
                    </button>
                  </div>
                </div>
              )}

              <div className="workspace-code-container">
                {fileLoading ? (
                  <div className="workspace-file-loading">Loading file…</div>
                ) : isSelectedImage && selectedFile ? (
                  <div className="workspace-image-container">
                    <img
                      src={`/api/workspace/raw?convoId=${encodeURIComponent(convoId)}&path=${encodeURIComponent(
                        selectedFile.path
                      )}`}
                      alt={selectedFile.name}
                      className="workspace-image-preview"
                    />
                    <div className="workspace-image-meta">
                      <span>{selectedFile.name}</span>
                      <span>{formatBytes(selectedFile.size)}</span>
                    </div>
                  </div>
                ) : fileContent === null ? (
                  <div className="workspace-file-loading">Select a file to preview</div>
                ) : (
                  <div className="workspace-code-view">
                    <Markdown text={`\`\`\`${lang}\n${fileContent}\n\`\`\``} />
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
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
  const [useResponseApi, setUseResponseApi] = useState(!!provider.useResponseApi);
  const [showApiKey, setShowApiKey] = useState(false);

  // Advanced parameters
  const [showAdvanced, setShowAdvanced] = useState(
    provider.temperature !== undefined ||
      provider.maxTokens !== undefined ||
      provider.topP !== undefined ||
      provider.reasoningEffort !== undefined
  );
  const [temperature, setTemperature] = useState<number | undefined>(provider.temperature);
  const [maxTokens, setMaxTokens] = useState<number | undefined>(provider.maxTokens);
  const [topP, setTopP] = useState<number | undefined>(provider.topP);
  const [reasoningEffort, setReasoningEffort] = useState<"low" | "medium" | "high" | undefined>(
    provider.reasoningEffort === "none" ? undefined : provider.reasoningEffort
  );

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
      const data = (await res.json()) as {
        ok: boolean;
        models?: string[];
        error?: string;
        statusCode?: number;
        statusText?: string;
        rawResponse?: any;
      };
      if (!res.ok || !data.ok) {
        const rawBody = data.rawResponse ? (typeof data.rawResponse === "object" ? JSON.stringify(data.rawResponse) : String(data.rawResponse)) : "";
        const codePrefix = data.statusCode || res.status;
        throw new Error(`[HTTP ${codePrefix}] ${data.error || "Failed to load models"}${rawBody && rawBody !== data.error ? ` — ${rawBody}` : ""}`);
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
    if (!endpoint.trim()) {
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
      useResponseApi,
      temperature,
      maxTokens,
      topP,
      reasoningEffort: reasoningEffort || undefined,
    };
    onSave(updated);
  };

  return (
    <form className="form-grid" onSubmit={handleSubmit} autoComplete="off" data-lpignore="true" data-1p-ignore="true">
      <div className="form-group">
        <label className="form-label">
          Provider Name
          <span className="form-hint">e.g. OpenRouter, DeepSeek Official, OpenAI, Anthropic, Ollama</span>
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
          placeholder="e.g. DeepSeek / OpenAI / OpenRouter"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>

      <div className="form-group">
        <label className="form-label">
          AI Endpoint (Base URL)
          <span className="form-hint">e.g. <code>https://api.deepseek.com/v1</code> or <code>https://openrouter.ai/api/v1</code></span>
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
          placeholder="https://api.deepseek.com/v1 or https://api.openai.com/v1"
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          required
        />
      </div>

      <div className="protocol-select-box">
        <label className="checkbox-label" style={{ alignItems: "flex-start", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={useResponseApi}
            onChange={(e) => setUseResponseApi(e.target.checked)}
            style={{ marginTop: 3 }}
          />
          <div className="checkbox-text-group">
            <span className="checkbox-title">Use OpenAI Response Protocol (Beta /responses)</span>
            <span className="checkbox-desc">
              Enable only if this provider/model explicitly requires OpenAI's /responses protocol. Unchecked by default (uses standard /v1/chat/completions, compatible with DeepSeek, OpenRouter, and standard endpoints).
            </span>
          </div>
        </label>
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
            placeholder="sk-... (optional for local Ollama)"
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

      <div className="form-group">
        <label className="form-label">
          Model ID
          {cachedModels.length > 0 && <span className="form-hint">{cachedModels.length} models available</span>}
        </label>
        <ModelCombobox
          value={selectedModel}
          onChange={setSelectedModel}
          options={cachedModels}
          placeholder="Select from dropdown or type model ID (e.g. deepseek-chat)"
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

            {/* Reasoning Effort (Chain of Thought) */}
            <div className="param-row">
              <div className="param-row-head">
                <span className="param-label">Reasoning Effort (Chain of Thought)</span>
                <span className="param-val">
                  {reasoningEffort ? reasoningEffort.toUpperCase() : "Default / Model Setting"}
                </span>
              </div>
              <div className="segmented-nav" style={{ width: "100%" }}>
                {(["none", "low", "medium", "high"] as const).map((lvl) => (
                  <button
                    key={lvl}
                    type="button"
                    className={`tab-btn ${(reasoningEffort || "none") === lvl ? "active" : ""}`}
                    style={{ flex: 1, justifyContent: "center", fontSize: 12 }}
                    onClick={() => setReasoningEffort(lvl === "none" ? undefined : lvl)}
                  >
                    {lvl === "none" ? "Default" : lvl.charAt(0).toUpperCase() + lvl.slice(1)}
                  </button>
                ))}
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
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {!isNew && onDelete && (
            <button
              type="button"
              className="btn-text-del"
              onClick={onDelete}
              style={{ marginRight: 8 }}
            >
              Delete Provider
            </button>
          )}
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

/* ---------------- MCP Server Editor ---------------- */
function McpServerEditor({
  server,
  isNew,
  onSave,
  onCancel,
  onDelete,
  onOAuthCompleted,
}: {
  server: McpServerConfig;
  isNew?: boolean;
  onSave: (updated: McpServerConfig) => void;
  onCancel: () => void;
  onDelete?: () => void;
  onOAuthCompleted?: () => void;
}) {
  const [name, setName] = useState(server.name);
  const [endpoint, setEndpoint] = useState(server.endpoint);
  const [authType, setAuthType] = useState<McpAuthType>(server.authType || "none");
  const [bearerToken, setBearerToken] = useState(server.bearerToken || "");
  const [cfAccessClientId, setCfAccessClientId] = useState(server.cfAccessClientId || "");
  const [cfAccessClientSecret, setCfAccessClientSecret] = useState(server.cfAccessClientSecret || "");
  const [showCfSecret, setShowCfSecret] = useState(false);
  const [oauthClientId, setOauthClientId] = useState(server.oauthClientId || "");
  const [oauthClientSecret, setOauthClientSecret] = useState(server.oauthClientSecret || "");
  const [oauthTokens, setOauthTokens] = useState(server.oauthTokens);
  const [showToken, setShowToken] = useState(false);
  const [showAdvancedOAuth, setShowAdvancedOAuth] = useState(false);
  const [enabled, setEnabled] = useState(server.enabled ?? true);
  const [cachedTools, setCachedTools] = useState<McpToolDef[]>(server.cachedTools || []);
  const [loadingTools, setLoadingTools] = useState(false);
  const [isAuthorizing, setIsAuthorizing] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  useEffect(() => {
    const handleMessage = async (e: MessageEvent) => {
      if (e.data?.type === "MCP_OAUTH_SUCCESS") {
        setIsAuthorizing(false);
        setStatusMsg({
          type: "ok",
          text: e.data?.message || "OAuth 2.0 connected and tools discovered successfully!",
        });
        if (onOAuthCompleted) onOAuthCompleted();
      } else if (e.data?.type === "MCP_OAUTH_ERROR") {
        setIsAuthorizing(false);
        setStatusMsg({
          type: "err",
          text: `OAuth authorization failed: ${e.data?.message || "Unknown error"}`,
        });
      }
    };
    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [onOAuthCompleted]);

  const handleStartOAuth = async () => {
    if (!endpoint.trim()) {
      setStatusMsg({ type: "err", text: "Endpoint URL is required before starting OAuth." });
      return;
    }
    setIsAuthorizing(true);
    setStatusMsg(null);
    try {
      const data = await cloudStartMcpOAuth(
        endpoint.trim(),
        server.id,
        name.trim() || "MCP Server",
        oauthClientId.trim() || undefined,
        oauthClientSecret.trim() || undefined
      );

      const w = 580;
      const h = 720;
      const left = Math.max(0, (window.screen.width - w) / 2);
      const top = Math.max(0, (window.screen.height - h) / 2);
      const popup = window.open(
        data.authUrl,
        "mcp_oauth_window",
        `width=${w},height=${h},top=${top},left=${left},scrollbars=yes,status=no`
      );
      if (!popup || popup.closed) {
        window.location.href = data.authUrl;
      }
    } catch (err: any) {
      setIsAuthorizing(false);
      setStatusMsg({ type: "err", text: `Failed to initiate OAuth: ${err.message || String(err)}` });
    }
  };

  const handleFetchTools = async () => {
    if (!endpoint.trim()) {
      setStatusMsg({ type: "err", text: "Endpoint URL is required." });
      return;
    }
    setLoadingTools(true);
    setStatusMsg(null);
    try {
      const tools = await cloudFetchMcpTools(
        endpoint.trim(),
        authType,
        bearerToken.trim() || undefined,
        cfAccessClientId.trim() || undefined,
        cfAccessClientSecret.trim() || undefined,
        oauthTokens,
        server.id
      );
      setCachedTools(tools);
      setStatusMsg({
        type: "ok",
        text: `Successfully discovered ${tools.length} tool${tools.length === 1 ? "" : "s"}.`,
      });
    } catch (err: any) {
      setStatusMsg({ type: "err", text: `Failed to connect: ${err.message || String(err)}` });
    } finally {
      setLoadingTools(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setStatusMsg({ type: "err", text: "Server Name is required." });
      return;
    }
    if (!endpoint.trim()) {
      setStatusMsg({ type: "err", text: "Endpoint URL is required." });
      return;
    }

    const updated: McpServerConfig = {
      ...server,
      name: name.trim(),
      endpoint: endpoint.trim(),
      authType,
      bearerToken: bearerToken.trim(),
      cfAccessClientId: cfAccessClientId.trim() || undefined,
      cfAccessClientSecret: cfAccessClientSecret.trim() || undefined,
      oauthClientId: oauthClientId.trim() || undefined,
      oauthClientSecret: oauthClientSecret.trim() || undefined,
      oauthTokens,
      enabled,
      cachedTools,
      isPreset: false,
      updatedAt: Date.now(),
    };
    onSave(updated);
  };

  return (
    <form className="form-grid" onSubmit={handleSubmit} autoComplete="off">
      <div className="form-group">
        <label className="form-label">
          MCP Server Name
          <span className="form-hint">e.g. Cloudflare MCP Portal, Weather Service, Inkstone</span>
        </label>
        <input
          type="text"
          className="text-input"
          placeholder="Server Display Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>

      <div className="form-group">
        <label className="form-label">
          MCP Endpoint URL
          <span className="form-hint">Streamable HTTP / SSE JSON-RPC endpoint</span>
        </label>
        <input
          type="text"
          className="text-input"
          placeholder="https://example.com/mcp"
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          required
        />
      </div>

      <div className="form-group">
        <label className="form-label">
          Authentication Method
          <span className="form-hint">Select how the Agent authenticates with this MCP server</span>
        </label>
        <div className="segmented-nav" style={{ width: "100%", justifyContent: "space-between" }}>
          <button
            type="button"
            className={`tab-btn ${authType === "none" ? "active" : ""}`}
            style={{ flex: 1, justifyContent: "center" }}
            onClick={() => setAuthType("none")}
          >
            No Auth
          </button>
          <button
            type="button"
            className={`tab-btn ${authType === "bearer" ? "active" : ""}`}
            style={{ flex: 1, justifyContent: "center" }}
            onClick={() => setAuthType("bearer")}
          >
            Bearer Token
          </button>
          <button
            type="button"
            className={`tab-btn ${authType === "oauth" ? "active" : ""}`}
            style={{ flex: 1, justifyContent: "center" }}
            onClick={() => setAuthType("oauth")}
          >
            OAuth 2.0
          </button>
        </div>
      </div>

      {authType === "bearer" && (
        <div className="form-group">
          <label className="form-label">
            Bearer Token / API Key
            <span className="form-hint">Included in Authorization: Bearer header</span>
          </label>
          <div className="input-row">
            <input
              type="text"
              className={showToken ? "text-input" : "text-input key-masked"}
              placeholder="sk-... or token"
              value={bearerToken}
              onChange={(e) => setBearerToken(e.target.value)}
            />
            <button
              type="button"
              className="btn-icon"
              onClick={() => setShowToken(!showToken)}
              title={showToken ? "Hide token" : "Show token"}
            >
              {showToken ? <EyeOffIcon /> : <EyeIcon />}
            </button>
          </div>
        </div>
      )}

      {authType === "oauth" && (
        <div className="protocol-select-box" style={{ marginTop: 2, marginBottom: 4 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <KeyIcon />
              <span style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>
                OAuth 2.0 PKCE Handshake
              </span>
            </div>
            {oauthTokens?.accessToken ? (
              <span className="badge badge-connected">Connected (Token Active)</span>
            ) : (
              <span className="badge badge-auth">Not Authorized</span>
            )}
          </div>
          <p style={{ fontSize: 12, color: "var(--muted)", margin: "0 0 10px", lineHeight: 1.45 }}>
            Automated discovery with PKCE and dynamic token refresh. Click below to authorize in a secure popup window.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <button
              type="button"
              className="btn-primary btn-sm"
              onClick={handleStartOAuth}
              disabled={isAuthorizing || !endpoint.trim()}
              style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
            >
              {isAuthorizing ? <span className="spinner" /> : <ExternalLinkIcon />}
              <span>{isAuthorizing ? "Connecting OAuth…" : oauthTokens?.accessToken ? "Reconnect OAuth" : "Connect with OAuth"}</span>
            </button>
            <button
              type="button"
              className="btn-secondary btn-sm"
              onClick={() => setShowAdvancedOAuth(!showAdvancedOAuth)}
            >
              {showAdvancedOAuth ? "Hide Advanced" : "Custom Client ID / Secret"}
            </button>
          </div>

          {showAdvancedOAuth && (
            <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 10, borderTop: "1px solid var(--line)", paddingTop: 10 }}>
              <div className="form-group">
                <label className="form-label" style={{ fontSize: 12 }}>
                  Custom Client ID (Optional)
                  <span className="form-hint">Leave blank for automatic registration</span>
                </label>
                <input
                  type="text"
                  className="text-input"
                  placeholder="e.g. 8RlTsw82PWv8K32I"
                  value={oauthClientId}
                  onChange={(e) => setOauthClientId(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label" style={{ fontSize: 12 }}>
                  Custom Client Secret (Optional)
                  <span className="form-hint">Optional for confidential clients</span>
                </label>
                <input
                  type="password"
                  className="text-input"
                  placeholder="Optional for confidential clients"
                  value={oauthClientSecret}
                  onChange={(e) => setOauthClientSecret(e.target.value)}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Discovered Tools Preview */}
      <div className="form-group">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
          <label className="form-label" style={{ margin: 0 }}>
            Discovered Tools ({cachedTools.length})
          </label>
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={handleFetchTools}
            disabled={loadingTools || !endpoint.trim()}
          >
            {loadingTools ? <span className="spinner" /> : <SparklesIcon />}
            {loadingTools ? "Connecting…" : "Probe / Refresh Tools"}
          </button>
        </div>

        {cachedTools.length === 0 ? (
          <p className="side-empty" style={{ margin: 0, padding: "8px 0" }}>
            No tools discovered yet. Click "Probe / Refresh Tools" or connect via OAuth to fetch exposed tools.
          </p>
        ) : (
          <div className="mcp-tools-list">
            {cachedTools.map((t) => (
              <div key={t.name} className="mcp-tool-badge-item">
                <span className="mcp-tool-tag">{t.name}</span>
                {t.description && <span className="mcp-tool-desc">{t.description}</span>}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="input-row" style={{ justifyContent: "space-between", marginTop: 4 }}>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
          />
          Enable this MCP server
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          {onDelete && (
            <button type="button" className="btn-text-del" onClick={onDelete}>
              <TrashIcon />
              <span>Delete</span>
            </button>
          )}
          <button type="button" className="btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="btn-primary">
            {isNew ? "Add MCP Server" : "Save Changes"}
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

/* ---------------- System Prompt Editor ---------------- */

const DEFAULT_SYSTEM_PROMPT = `You are Aki's Cloudflare edge agent.
Reply in the user's language (Chinese if they write Chinese).
You have access to these core systems and toolsets:
1) Hindsight (External Live Memory) — Aki's fresh, active, dynamic, frequently updated memory stream (current preferences, ongoing context, evolving thoughts and state).
2) GBrain (Library & Archive) — Aki's digital library and static archive (structured repository for permanent reference docs, server/infra facts, people, decisions, and historical records).
3) Cloudflare Computer workspace — durable session files in this Durable Object (read, write, edit, ls).
4) Parallel Web Search MCP — real-time live web search and URL content extraction.

System & Tool usage guidelines:
- For active context, current preferences, recent activities, or evolving state: check and update Hindsight (live dynamic memory).
- For structured archives, permanent knowledge, server infra, holdings, or long-term docs: query GBrain (library/archive).
- For notes, code artifacts, or working drafts created in this session: use Cloudflare workspace tools (prefer read/ls/write/edit over bash cat/ls/sed).
- If asked about unfamiliar topics, recent events, breaking news, new technologies/APIs, facts outside your training cutoff, or anything you are not 100% certain about, you MUST proactively use the Parallel MCP search tools to search the web before answering.
- Keep replies concise. Cite GBrain slugs or web source URLs when you use them.
- Do not hallucinate or invent holdings, keys, or infra facts — look them up.`;

const PROMPT_PRESETS = [
  {
    name: "Concise Expert",
    prompt: "Provide direct, high-signal responses. Minimize conversational filler. Always use markdown formatting and code blocks with syntax highlighting.",
  },
  {
    name: "Coding Architect",
    prompt: "You are a principal software engineer. Always analyze edge cases, performance implications, and verify code integrity before answering. Prefer modern TypeScript and clean architectural patterns.",
  },
  {
    name: "Bilingual Explainer",
    prompt: "Always respond in natural, professional Traditional/Simplified Chinese with clear technical terms explained where appropriate. Maintain clear structured headings.",
  },
  {
    name: "Research Analyst",
    prompt: "Structure answers with Executive Summary, Deep Dive, Key Findings, and Actionable Next Steps. Cite relevant sources and knowledge entries.",
  },
  {
    name: "Memory Proactive",
    prompt: "Actively maintain long-term memory. Whenever important decisions, user preferences, workflow habits, or system requirements are discussed, proactively retain them and weave relevant past memories into answers.",
  },
];

function SystemPromptEditor({
  initialPrompt,
  initialMode,
  onSave,
}: {
  initialPrompt: string;
  initialMode: "append" | "override";
  onSave: (prompt: string, mode: "append" | "override") => void;
}) {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [mode, setMode] = useState<"append" | "override">(initialMode);
  const [showDefault, setShowDefault] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  const handleApplyPreset = (text: string) => {
    setPrompt((prev) => (prev.trim() ? `${prev}\n\n${text}` : text));
  };

  const handleSave = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    onSave(prompt.trim(), mode);
    setStatusMsg({ type: "ok", text: "System prompt saved and synced successfully!" });
    setTimeout(() => setStatusMsg(null), 3000);
  };

  const handleReset = () => {
    setPrompt("");
    setMode("append");
    onSave("", "append");
    setStatusMsg({ type: "ok", text: "Reset to default system prompt." });
    setTimeout(() => setStatusMsg(null), 3000);
  };

  return (
    <form className="form-grid" onSubmit={handleSave}>
      <div className="modal-section-head" style={{ marginBottom: 4 }}>
        <h3 className="modal-section-title">Custom System Prompt & Persona</h3>
        <div style={{ display: "flex", gap: 8 }}>
          <button type="button" className="btn-secondary btn-sm" onClick={handleReset}>
            Reset to Default
          </button>
          <button type="submit" className="btn-primary btn-sm">
            Save Prompt
          </button>
        </div>
      </div>

      <div className="form-group">
        <label className="form-label">
          Prompt Execution Mode
          <span className="form-hint">Choose how custom instructions interact with built-in agent rules</span>
        </label>
        <div className="segmented-nav" style={{ width: "100%" }}>
          <button
            type="button"
            className={`tab-btn ${mode === "append" ? "active" : ""}`}
            style={{ flex: 1, justifyContent: "center" }}
            onClick={() => setMode("append")}
          >
            Append to Default (Recommended)
          </button>
          <button
            type="button"
            className={`tab-btn ${mode === "override" ? "active" : ""}`}
            style={{ flex: 1, justifyContent: "center" }}
            onClick={() => setMode("override")}
          >
            Override Default Completely
          </button>
        </div>
        <p style={{ fontSize: 12, color: "var(--muted)", margin: "4px 0 0" }}>
          {mode === "append"
            ? "Your instructions will be added to the base prompt, preserving VFS file workspace and GBrain MCP knowledge retrieval tools."
            : "Replaces the entire base prompt. Workspace tools remain available, but core system guidelines will be overwritten."}
        </p>
      </div>

      <div className="form-group">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
          <label className="form-label" style={{ margin: 0 }}>
            Custom Instructions / Persona
          </label>
          <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
            {prompt.length} chars · {prompt.trim() ? prompt.trim().split(/\s+/).length : 0} words
          </span>
        </div>
        <textarea
          className="prompt-textarea"
          placeholder="Enter custom instructions, tone of voice, formatting guidelines, or roleplay personas... (e.g. Always format responses in clean tables...)"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div className="prompt-templates-bar">
          <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 500 }}>Quick Templates:</span>
          {PROMPT_PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              className="prompt-template-chip"
              onClick={() => handleApplyPreset(p.prompt)}
              title={p.prompt}
            >
              + {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Built-in Default Prompt Inspector */}
      <div className="form-group" style={{ marginTop: 4 }}>
        <button
          type="button"
          className="btn-secondary btn-sm"
          style={{ width: "fit-content", display: "inline-flex", alignItems: "center", gap: 6 }}
          onClick={() => setShowDefault(!showDefault)}
        >
          {showDefault ? <ChevronDownIcon /> : <ChevronRightIcon />}
          <span>{showDefault ? "Hide Default System Prompt" : "Inspect Built-in Default System Prompt"}</span>
        </button>
        {showDefault && (
          <div className="prompt-default-viewer" style={{ marginTop: 8 }}>
            {DEFAULT_SYSTEM_PROMPT}
          </div>
        )}
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

/* ---------------- Mnemosyne Full Intelligence Dashboard ---------------- */

function MnemosyneFullDashboard({
  onClose,
  config,
  onSaveConfig,
}: {
  onClose: () => void;
  config: MnemosyneConfig;
  onSaveConfig: (updated: MnemosyneConfig) => void;
}) {
  const [activeTab, setActiveTab] = useState<
    "overview" | "today" | "context_bank" | "graph" | "debugger" | "activity" | "settings"
  >("overview");

  const [stats, setStats] = useState<MnemosyneStats | null>(null);
  const [memories, setMemories] = useState<EpisodicMemoryItem[]>([]);
  const [triples, setTriples] = useState<TripleItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchFilter, setSearchFilter] = useState("");
  const [statusMsg, setStatusMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  // Detail Modal
  const [inspectMemory, setInspectMemory] = useState<EpisodicMemoryItem | null>(null);

  // Forms
  const [showAddMemModal, setShowAddMemModal] = useState(false);
  const [showAddTripleModal, setShowAddTripleModal] = useState(false);

  // New Memory Form
  const [newContent, setNewContent] = useState("");
  const [newImportance, setNewImportance] = useState(0.8);
  const [isWorkingMem, setIsWorkingMem] = useState(false);
  const [ttlSec, setTtlSec] = useState("");

  // New Triple Form
  const [newSubj, setNewSubj] = useState("");
  const [newPred, setNewPred] = useState("");
  const [newObj, setNewObj] = useState("");

  // Debugger Form
  const [debugQuery, setDebugQuery] = useState("");
  const [debugResult, setDebugResult] = useState<MnemosyneRecallResult | null>(null);
  const [debugging, setDebugging] = useState(false);

  // Settings
  const [enabled, setEnabled] = useState(config.enabled ?? true);
  const [autoRecall, setAutoRecall] = useState(config.autoRecall ?? true);
  const [autoRetain, setAutoRetain] = useState(config.autoRetain ?? true);
  const [recallTopK, setRecallTopK] = useState(config.recallTopK || 5);
  const [scope, setScope] = useState(config.scope || "global");

  const refreshData = async () => {
    setLoading(true);
    try {
      const [s, m, t] = await Promise.all([
        cloudGetMnemosyneStats().catch(() => null),
        cloudListMnemosyneMemories().catch(() => []),
        cloudListMnemosyneTriples().catch(() => []),
      ]);
      if (s) setStats(s);
      setMemories(m);
      setTriples(t);
    } catch (err: any) {
      console.error("Failed to load Mnemosyne data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, []);

  // Compute Context Bank Inferred Sections
  const contextBank = useMemo(() => {
    const userPrefs: EpisodicMemoryItem[] = [];
    const techStack: EpisodicMemoryItem[] = [];
    const rulesDecisions: EpisodicMemoryItem[] = [];
    const generalFacts: EpisodicMemoryItem[] = [];

    for (const m of memories) {
      const text = m.content.toLowerCase();
      if (
        text.includes("prefer") ||
        text.includes("like") ||
        text.includes("user") ||
        text.includes("theme") ||
        text.includes("language") ||
        text.includes("ui") ||
        text.includes("style")
      ) {
        userPrefs.push(m);
      } else if (
        text.includes("cloudflare") ||
        text.includes("react") ||
        text.includes("python") ||
        text.includes("typescript") ||
        text.includes("api") ||
        text.includes("sqlite") ||
        text.includes("worker") ||
        text.includes("durable object")
      ) {
        techStack.push(m);
      } else if (
        text.includes("must") ||
        text.includes("always") ||
        text.includes("never") ||
        text.includes("rule") ||
        text.includes("decision") ||
        text.includes("constraint")
      ) {
        rulesDecisions.push(m);
      } else {
        generalFacts.push(m);
      }
    }

    return { userPrefs, techStack, rulesDecisions, generalFacts };
  }, [memories]);

  // Compute Today's Digest
  const todayDigest = useMemo(() => {
    const startOfToday = new Date();
    startOfToday.setHours(0, 0, 0, 0);
    const todayMems = memories.filter((m) => m.createdAt >= startOfToday.getTime());
    const highSalience = memories.filter((m) => (m.importance || 0.7) >= 0.8);
    return {
      addedToday: todayMems.length,
      todayMems,
      highSalienceCount: highSalience.length,
    };
  }, [memories]);

  const filteredMemories = useMemo(() => {
    if (!searchFilter.trim()) return memories;
    const q = searchFilter.toLowerCase();
    return memories.filter(
      (m) =>
        m.content.toLowerCase().includes(q) ||
        (m.scope && m.scope.toLowerCase().includes(q)) ||
        (m.source && m.source.toLowerCase().includes(q))
    );
  }, [memories, searchFilter]);

  const handleAddMemory = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newContent.trim()) return;
    try {
      await cloudRememberMnemosyne({
        content: newContent.trim(),
        importance: Number(newImportance),
        isWorkingMemory: isWorkingMem,
        ttlSeconds: ttlSec ? Number(ttlSec) : undefined,
      });
      setNewContent("");
      setShowAddMemModal(false);
      setStatusMsg({ type: "ok", text: "Memory successfully remembered!" });
      setTimeout(() => setStatusMsg(null), 3000);
      refreshData();
    } catch (err: any) {
      setStatusMsg({ type: "err", text: err.message || "Failed to add memory" });
    }
  };

  const handleAddTriple = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSubj.trim() || !newPred.trim() || !newObj.trim()) return;
    try {
      const res = await fetch("/api/mnemosyne/triples", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          subject: newSubj.trim(),
          predicate: newPred.trim(),
          object: newObj.trim(),
        }),
      });
      const data = (await res.json()) as any;
      if (!data.ok) throw new Error(data.error);
      setNewSubj("");
      setNewPred("");
      setNewObj("");
      setShowAddTripleModal(false);
      setStatusMsg({ type: "ok", text: "Knowledge Graph triple added!" });
      setTimeout(() => setStatusMsg(null), 3000);
      refreshData();
    } catch (err: any) {
      setStatusMsg({ type: "err", text: err.message || "Failed to add triple" });
    }
  };

  const handleDeleteMemory = async (id: string) => {
    if (!confirm("Are you sure you want to delete this memory?")) return;
    try {
      await cloudDeleteMnemosyneMemory(id);
      refreshData();
      if (inspectMemory?.id === id) setInspectMemory(null);
    } catch (err: any) {
      alert("Failed to delete memory: " + err.message);
    }
  };

  const handleDeleteTriple = async (id: string) => {
    if (!confirm("Are you sure you want to delete this triple?")) return;
    try {
      await cloudDeleteMnemosyneTriple(id);
      refreshData();
    } catch (err: any) {
      alert("Failed to delete triple: " + err.message);
    }
  };

  const handleRunRecallTest = async () => {
    if (!debugQuery.trim()) return;
    setDebugging(true);
    try {
      const res = await cloudRecallMnemosyne(debugQuery.trim(), Number(recallTopK) || 5);
      setDebugResult(res);
    } catch (err: any) {
      alert("Recall test failed: " + err.message);
    } finally {
      setDebugging(false);
    }
  };

  const handleConsolidate = async () => {
    try {
      const count = await cloudConsolidateMnemosyne();
      setStatusMsg({ type: "ok", text: `Consolidated ${count} working memory item(s) into episodic summaries.` });
      setTimeout(() => setStatusMsg(null), 4000);
      refreshData();
    } catch (err: any) {
      alert("Consolidation failed: " + err.message);
    }
  };

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    const updated: MnemosyneConfig = {
      enabled,
      autoRecall,
      autoRetain,
      recallTopK: Number(recallTopK) || 5,
      scope: scope.trim() || "global",
      updatedAt: Date.now(),
    };
    onSaveConfig(updated);
    try {
      await cloudSaveMnemosyneConfig(updated);
      setStatusMsg({ type: "ok", text: "Mnemosyne settings saved to Cloudflare DO SQLite!" });
      setTimeout(() => setStatusMsg(null), 3000);
    } catch (err: any) {
      setStatusMsg({ type: "err", text: err.message || "Failed to save configuration" });
    }
  };

  const handleExportJSON = () => {
    const data = {
      exportDate: new Date().toISOString(),
      stats,
      memories,
      triples,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mnemosyne-memories-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClearAll = async () => {
    if (!confirm("⚠️ Are you sure you want to clear ALL memories and Knowledge Graph triples? This action is irreversible.")) return;
    try {
      await cloudClearMnemosyne("all");
      setStatusMsg({ type: "ok", text: "All Mnemosyne memories cleared." });
      setTimeout(() => setStatusMsg(null), 3000);
      refreshData();
    } catch (err: any) {
      alert("Clear failed: " + err.message);
    }
  };

  return (
    <div className="mnemosyne-dashboard-page">
      {/* Top Navigation Bar */}
      <header className="mnemosyne-dashboard-header">
        <div className="dashboard-header-left">
          <div className="dashboard-logo">
            <BrainIcon />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <h2 className="dashboard-title">Mnemosyne Memory Intelligence</h2>
              <span className="badge badge-connected" style={{ fontSize: 11 }}>
                Cloudflare DO SQLite
              </span>
              <span className="badge badge-memory" style={{ fontSize: 11 }}>
                Workers AI BAAI-bge
              </span>
            </div>
            <p className="dashboard-subtitle">
              Universal BEAM memory layer (Episodic Associative Memory + Temporal Knowledge Graph). 100% serverless at the Cloudflare edge.
            </p>
          </div>
        </div>

        <div className="dashboard-header-actions">
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={refreshData}
            disabled={loading}
            title="Refresh memory state"
          >
            {loading ? <span className="spinner" /> : <RefreshIcon />}
            <span>Refresh</span>
          </button>
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={handleConsolidate}
            title="Synthesize and consolidate working memories into long-term summaries"
          >
            <SparklesIcon />
            <span>Consolidate (Sleep)</span>
          </button>
          <button
            type="button"
            className="btn-primary btn-sm"
            onClick={() => setShowAddMemModal(true)}
          >
            <PlusIcon />
            <span>Add Memory</span>
          </button>
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={onClose}
            style={{ fontWeight: 600 }}
          >
            <span>← Back to Chat</span>
          </button>
        </div>
      </header>

      {/* Global Status Banner */}
      {statusMsg && (
        <div className={`banner-msg ${statusMsg.type}`} style={{ margin: "0 24px 12px" }}>
          {statusMsg.type === "ok" ? <CheckIcon /> : <XIcon />}
          <span>{statusMsg.text}</span>
        </div>
      )}

      {/* Primary Tab Navigation */}
      <nav className="mnemosyne-dashboard-nav">
        <button
          type="button"
          className={`dash-nav-btn ${activeTab === "overview" ? "active" : ""}`}
          onClick={() => setActiveTab("overview")}
        >
          <MonitorIcon />
          <span>Overview</span>
        </button>
        <button
          type="button"
          className={`dash-nav-btn ${activeTab === "today" ? "active" : ""}`}
          onClick={() => setActiveTab("today")}
        >
          <CalendarIcon />
          <span>Today</span>
          {todayDigest.addedToday > 0 && (
            <span className="dash-nav-badge">{todayDigest.addedToday}</span>
          )}
        </button>
        <button
          type="button"
          className={`dash-nav-btn ${activeTab === "context_bank" ? "active" : ""}`}
          onClick={() => setActiveTab("context_bank")}
        >
          <LayersIcon />
          <span>Context Bank</span>
        </button>
        <button
          type="button"
          className={`dash-nav-btn ${activeTab === "graph" ? "active" : ""}`}
          onClick={() => setActiveTab("graph")}
        >
          <NetworkIcon />
          <span>Knowledge Graph</span>
          <span className="dash-nav-badge">{triples.length}</span>
        </button>
        <button
          type="button"
          className={`dash-nav-btn ${activeTab === "debugger" ? "active" : ""}`}
          onClick={() => setActiveTab("debugger")}
        >
          <CompassIcon />
          <span>BEAM Debugger</span>
        </button>
        <button
          type="button"
          className={`dash-nav-btn ${activeTab === "activity" ? "active" : ""}`}
          onClick={() => setActiveTab("activity")}
        >
          <ActivityIcon />
          <span>Activity</span>
        </button>
        <button
          type="button"
          className={`dash-nav-btn ${activeTab === "settings" ? "active" : ""}`}
          onClick={() => setActiveTab("settings")}
        >
          <SettingsIcon />
          <span>Settings & Safe Ops</span>
        </button>
      </nav>

      {/* Main Content Area */}
      <main className="mnemosyne-dashboard-body">
        {/* TAB 1: OVERVIEW */}
        {activeTab === "overview" && (
          <div className="dash-section-grid">
            {/* 4 Stat Cards */}
            <div className="dash-stats-row">
              <div className="dash-stat-box">
                <span className="dash-stat-label">Episodic Memories</span>
                <span className="dash-stat-value">{stats?.totalEpisodic ?? memories.length}</span>
                <span className="dash-stat-sub">Long-term associative facts</span>
              </div>
              <div className="dash-stat-box">
                <span className="dash-stat-label">Working Context</span>
                <span className="dash-stat-value">{stats?.totalWorking ?? 0}</span>
                <span className="dash-stat-sub">Hot temporary session facts</span>
              </div>
              <div className="dash-stat-box">
                <span className="dash-stat-label">KG Triples</span>
                <span className="dash-stat-value">{stats?.totalTriples ?? triples.length}</span>
                <span className="dash-stat-sub">Structured entity relationships</span>
              </div>
              <div className="dash-stat-box">
                <span className="dash-stat-label">BEAM Formula</span>
                <span className="dash-stat-value" style={{ fontSize: 17, color: "var(--ok)" }}>
                  50% Vec + 30% FTS
                </span>
                <span className="dash-stat-sub">15% Salience + 5% Recency</span>
              </div>
            </div>

            {/* Quick Actions & Search */}
            <div className="dash-filter-row">
              <div className="input-search-wrapper" style={{ flex: 1 }}>
                <input
                  type="text"
                  className="text-input"
                  placeholder="Search memories, entity concepts, scopes..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                />
              </div>
              <button
                type="button"
                className="btn-secondary btn-sm"
                onClick={() => setShowAddTripleModal(true)}
              >
                <PlusIcon />
                <span>Add KG Triple</span>
              </button>
              <button
                type="button"
                className="btn-secondary btn-sm"
                onClick={handleExportJSON}
                title="Export memories as JSON"
              >
                <DownloadIcon />
                <span>Export JSON</span>
              </button>
            </div>

            {/* Memories List */}
            <div className="dash-card">
              <div className="dash-card-head">
                <h3 className="dash-card-title">
                  All Active Memories ({filteredMemories.length})
                </h3>
              </div>

              {filteredMemories.length === 0 ? (
                <div className="empty-card" style={{ padding: "36px 16px", textAlign: "center" }}>
                  <p style={{ margin: "0 0 12px", color: "var(--muted)" }}>
                    No memories match your filter. The agent will proactively form memories during conversations.
                  </p>
                  <button
                    type="button"
                    className="btn-primary btn-sm"
                    onClick={() => setShowAddMemModal(true)}
                  >
                    + Add New Memory
                  </button>
                </div>
              ) : (
                <div className="dash-memories-feed">
                  {filteredMemories.map((m) => (
                    <div
                      key={m.id}
                      className="dash-memory-item"
                      onClick={() => setInspectMemory(m)}
                    >
                      <div className="dash-memory-content">
                        <span className="dash-memory-text">{m.content}</span>
                        <div className="dash-memory-meta">
                          <span className="badge badge-connected" style={{ fontSize: 10 }}>
                            {((m.importance || 0.7) * 100).toFixed(0)}% Salience
                          </span>
                          <span className="dash-tag">Scope: {m.scope}</span>
                          <span className="dash-tag">Accessed: {m.accessCount || 0}x</span>
                          <span className="dash-tag">
                            {new Date(m.createdAt).toLocaleDateString()} {new Date(m.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </span>
                        </div>
                      </div>
                      <div className="dash-memory-actions" onClick={(e) => e.stopPropagation()}>
                        <button
                          type="button"
                          className="btn-text-del"
                          onClick={() => handleDeleteMemory(m.id)}
                          title="Delete memory"
                        >
                          <TrashIcon />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: TODAY */}
        {activeTab === "today" && (
          <div className="dash-section-grid">
            <div className="dash-hero-banner">
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div className="dash-hero-icon">
                  <CalendarIcon />
                </div>
                <div>
                  <h3 style={{ margin: "0 0 4px", fontSize: 16, fontWeight: 600, color: "var(--ink)" }}>
                    Today's Memory Digest
                  </h3>
                  <p style={{ margin: 0, fontSize: 13, color: "var(--muted)" }}>
                    {todayDigest.addedToday} memories formed today · {todayDigest.highSalienceCount} high-salience permanent facts in active storage.
                  </p>
                </div>
              </div>
            </div>

            <div className="dash-card">
              <div className="dash-card-head">
                <h3 className="dash-card-title">Memories Formed Today ({todayDigest.todayMems.length})</h3>
              </div>

              {todayDigest.todayMems.length === 0 ? (
                <div className="empty-card" style={{ padding: "32px 16px", textAlign: "center" }}>
                  <p style={{ color: "var(--muted)", margin: 0 }}>
                    No new memories formed yet today. As you chat with the agent, key decisions and preferences will appear here.
                  </p>
                </div>
              ) : (
                <div className="dash-memories-feed">
                  {todayDigest.todayMems.map((m) => (
                    <div key={m.id} className="dash-memory-item">
                      <div className="dash-memory-content">
                        <span className="dash-memory-text">{m.content}</span>
                        <div className="dash-memory-meta">
                          <span className="badge badge-connected" style={{ fontSize: 10 }}>
                            {((m.importance || 0.7) * 100).toFixed(0)}% Salience
                          </span>
                          <span className="dash-tag">
                            {new Date(m.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: CONTEXT BANK */}
        {activeTab === "context_bank" && (
          <div className="dash-section-grid">
            <div className="dash-hero-banner">
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div className="dash-hero-icon">
                  <LayersIcon />
                </div>
                <div>
                  <h3 style={{ margin: "0 0 4px", fontSize: 16, fontWeight: 600, color: "var(--ink)" }}>
                    Context Bank & Inferred Profile
                  </h3>
                  <p style={{ margin: 0, fontSize: 13, color: "var(--muted)" }}>
                    Automatically synthesized user persona, tech stack preferences, and architectural rules derived from active memory items.
                  </p>
                </div>
              </div>
            </div>

            <div className="context-bank-grid">
              {/* Card 1: User Preferences */}
              <div className="context-bank-card">
                <div className="context-bank-card-head">
                  <h4 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: "var(--ink)" }}>
                    👤 User Preferences & Persona
                  </h4>
                  <span className="badge badge-connected" style={{ fontSize: 10 }}>
                    {contextBank.userPrefs.length} Items
                  </span>
                </div>
                <div className="context-bank-items">
                  {contextBank.userPrefs.length === 0 ? (
                    <p style={{ fontSize: 12.5, color: "var(--muted)", margin: 0 }}>No user preferences recorded yet.</p>
                  ) : (
                    contextBank.userPrefs.map((m) => (
                      <div key={m.id} className="context-bank-item">
                        {m.content}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Card 2: Tech Stack & Architecture */}
              <div className="context-bank-card">
                <div className="context-bank-card-head">
                  <h4 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: "var(--ink)" }}>
                    💻 Tech Stack & Systems
                  </h4>
                  <span className="badge badge-connected" style={{ fontSize: 10 }}>
                    {contextBank.techStack.length} Items
                  </span>
                </div>
                <div className="context-bank-items">
                  {contextBank.techStack.length === 0 ? (
                    <p style={{ fontSize: 12.5, color: "var(--muted)", margin: 0 }}>No tech stack preferences recorded yet.</p>
                  ) : (
                    contextBank.techStack.map((m) => (
                      <div key={m.id} className="context-bank-item">
                        {m.content}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Card 3: Key Decisions & Constraints */}
              <div className="context-bank-card">
                <div className="context-bank-card-head">
                  <h4 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: "var(--ink)" }}>
                    📌 Rules, Decisions & Constraints
                  </h4>
                  <span className="badge badge-connected" style={{ fontSize: 10 }}>
                    {contextBank.rulesDecisions.length} Items
                  </span>
                </div>
                <div className="context-bank-items">
                  {contextBank.rulesDecisions.length === 0 ? (
                    <p style={{ fontSize: 12.5, color: "var(--muted)", margin: 0 }}>No operational rules recorded yet.</p>
                  ) : (
                    contextBank.rulesDecisions.map((m) => (
                      <div key={m.id} className="context-bank-item">
                        {m.content}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Card 4: General Facts */}
              <div className="context-bank-card">
                <div className="context-bank-card-head">
                  <h4 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: "var(--ink)" }}>
                    🌐 General Long-Term Facts
                  </h4>
                  <span className="badge badge-connected" style={{ fontSize: 10 }}>
                    {contextBank.generalFacts.length} Items
                  </span>
                </div>
                <div className="context-bank-items">
                  {contextBank.generalFacts.length === 0 ? (
                    <p style={{ fontSize: 12.5, color: "var(--muted)", margin: 0 }}>No general facts recorded yet.</p>
                  ) : (
                    contextBank.generalFacts.map((m) => (
                      <div key={m.id} className="context-bank-item">
                        {m.content}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: KNOWLEDGE GRAPH */}
        {activeTab === "graph" && (
          <div className="dash-section-grid">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <p style={{ margin: 0, fontSize: 13, color: "var(--muted)" }}>
                Structured entity relationships represented as <code>(Subject) ──[predicate]──&gt; (Object)</code> triples.
              </p>
              <button
                type="button"
                className="btn-primary btn-sm"
                onClick={() => setShowAddTripleModal(true)}
              >
                <PlusIcon />
                <span>+ Add Triple</span>
              </button>
            </div>

            {triples.length === 0 ? (
              <div className="empty-card" style={{ padding: "36px 16px", textAlign: "center" }}>
                <p style={{ color: "var(--muted)", margin: "0 0 12px" }}>
                  No Knowledge Graph triples found.
                </p>
                <button
                  type="button"
                  className="btn-primary btn-sm"
                  onClick={() => setShowAddTripleModal(true)}
                >
                  + Add First Triple
                </button>
              </div>
            ) : (
              <div className="kg-triples-grid">
                {triples.map((t) => (
                  <div key={t.id} className="kg-triple-card">
                    <div className="kg-triple-body">
                      <div className="kg-node subject-node">{t.subject}</div>
                      <div className="kg-edge">
                        <span className="kg-edge-label">{t.predicate}</span>
                        <div className="kg-arrow">───►</div>
                      </div>
                      <div className="kg-node object-node">{t.object}</div>
                    </div>
                    <div className="kg-triple-foot">
                      <span className="dash-tag">
                        {t.validUntil ? `Valid until: ${t.validUntil}` : "Perpetual"}
                      </span>
                      <button
                        type="button"
                        className="btn-text-del"
                        onClick={() => handleDeleteTriple(t.id)}
                        title="Delete triple"
                      >
                        <TrashIcon />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 5: BEAM DEBUGGER */}
        {activeTab === "debugger" && (
          <div className="dash-section-grid">
            <div className="dash-card" style={{ padding: 18 }}>
              <h3 className="dash-card-title" style={{ marginBottom: 12 }}>
                BEAM Hybrid Recall Simulator
              </h3>
              <p style={{ margin: "0 0 14px", fontSize: 13, color: "var(--muted)" }}>
                Simulate how Mnemosyne ranks memories when the user sends a message. The BEAM algorithm combines Cloudflare Workers AI vector embeddings, FTS5 BM25, importance salience, and recency decay.
              </p>

              <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
                <input
                  type="text"
                  className="text-input"
                  style={{ flex: 1 }}
                  placeholder="Enter a test prompt (e.g. 'What tools and libraries do we use?')"
                  value={debugQuery}
                  onChange={(e) => setDebugQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleRunRecallTest()}
                />
                <button
                  type="button"
                  className="btn-primary"
                  onClick={handleRunRecallTest}
                  disabled={debugging || !debugQuery.trim()}
                >
                  {debugging ? <span className="spinner" /> : <SparklesIcon />}
                  <span>Test BEAM Recall</span>
                </button>
              </div>

              {debugResult && (
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  <div>
                    <h4 style={{ margin: "0 0 8px", fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>
                      Formatted Context Payload (Injected into System Prompt):
                    </h4>
                    <pre className="dash-code-block">
                      {debugResult.formattedContext || "No matching memories retrieved."}
                    </pre>
                  </div>

                  {debugResult.memories.length > 0 && (
                    <div>
                      <h4 style={{ margin: "0 0 8px", fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>
                        BEAM Score Breakdown ({debugResult.memories.length} matches):
                      </h4>
                      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        {debugResult.memories.map((m) => (
                          <div key={m.id} className="dash-score-card">
                            <div style={{ fontSize: 13.5, color: "var(--ink)", fontWeight: 500 }}>
                              {m.content}
                            </div>
                            <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6, flexWrap: "wrap" }}>
                              <span className="badge badge-connected" style={{ fontSize: 10.5 }}>
                                BEAM Total Match: {((m.score || 0) * 100).toFixed(1)}%
                              </span>
                              <span className="dash-tag">Scope: {m.scope}</span>
                              <span className="dash-tag">Salience: {((m.importance || 0.7) * 100).toFixed(0)}%</span>
                              <span className="dash-tag">Accesses: {m.accessCount || 0}x</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 6: ACTIVITY */}
        {activeTab === "activity" && (
          <div className="dash-section-grid">
            <div className="dash-card">
              <div className="dash-card-head">
                <h3 className="dash-card-title">Memory Activity Timeline</h3>
              </div>

              <div className="dash-timeline-feed">
                {memories.map((m) => (
                  <div key={m.id} className="dash-timeline-item">
                    <div className="dash-timeline-dot" />
                    <div className="dash-timeline-content">
                      <div style={{ fontSize: 13.5, color: "var(--ink)" }}>{m.content}</div>
                      <div className="dash-memory-meta" style={{ marginTop: 4 }}>
                        <span className="dash-tag">{new Date(m.createdAt).toLocaleString()}</span>
                        <span className="dash-tag">Source: {m.source || "agent_tool"}</span>
                        <span className="dash-tag">Salience: {((m.importance || 0.7) * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 7: SETTINGS & SAFE OPS */}
        {activeTab === "settings" && (
          <div className="dash-section-grid">
            <form onSubmit={handleSaveSettings} className="dash-card" style={{ padding: 18 }}>
              <h3 className="dash-card-title" style={{ marginBottom: 14 }}>
                Execution Controls & Dual-Loop Configuration
              </h3>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
                <label className="checkbox-label" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={autoRecall}
                    onChange={(e) => setAutoRecall(e.target.checked)}
                  />
                  <div>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>Pre-Inference Auto-Recall</span>
                    <p style={{ fontSize: 11.5, color: "var(--muted)", margin: "2px 0 0" }}>
                      Injects relevant memory context before model reasoning
                    </p>
                  </div>
                </label>

                <label className="checkbox-label" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={autoRetain}
                    onChange={(e) => setAutoRetain(e.target.checked)}
                  />
                  <div>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>Post-Turn Auto-Retain</span>
                    <p style={{ fontSize: 11.5, color: "var(--muted)", margin: "2px 0 0" }}>
                      Asynchronously commits turn highlights in background
                    </p>
                  </div>
                </label>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
                <div className="form-group">
                  <label className="form-label">
                    Recall Top-K Results: {recallTopK}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    value={recallTopK}
                    onChange={(e) => setRecallTopK(Number(e.target.value))}
                    style={{ width: "100%", accentColor: "var(--accent)" }}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Default Memory Scope</label>
                  <input
                    type="text"
                    className="text-input"
                    value={scope}
                    onChange={(e) => setScope(e.target.value)}
                    placeholder="global"
                  />
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button type="submit" className="btn-primary">
                  Save Mnemosyne Settings
                </button>
              </div>
            </form>

            <div className="dash-card" style={{ padding: 18 }}>
              <h3 className="dash-card-title" style={{ color: "var(--err)", marginBottom: 10 }}>
                Danger Zone & Database Maintenance
              </h3>
              <p style={{ fontSize: 12.5, color: "var(--muted)", margin: "0 0 14px" }}>
                Purging memories will erase all episodic facts and knowledge graph triples from Cloudflare DO SQLite.
              </p>
              <button type="button" className="btn-secondary btn-sm" onClick={handleClearAll} style={{ color: "var(--err)" }}>
                Clear All Memories
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Add Memory Modal */}
      {showAddMemModal && (
        <div className="modal-backdrop" onClick={() => setShowAddMemModal(false)}>
          <div className="dash-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="dash-dialog-head">
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Add New Memory</h3>
              <button type="button" className="ghost" onClick={() => setShowAddMemModal(false)}>
                <XIcon />
              </button>
            </div>
            <form onSubmit={handleAddMemory} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div className="form-group">
                <label className="form-label">Memory Content</label>
                <textarea
                  className="text-input"
                  rows={3}
                  placeholder="e.g. User prefers clean React code with Tailwind or custom CSS."
                  value={newContent}
                  onChange={(e) => setNewContent(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">
                  Importance / Salience: {((newImportance || 0.7) * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={newImportance}
                  onChange={(e) => setNewImportance(parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--accent)" }}
                />
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <input
                  type="checkbox"
                  checked={isWorkingMem}
                  onChange={(e) => setIsWorkingMem(e.target.checked)}
                />
                <span style={{ fontSize: 13 }}>Working Memory (Temporary hot context with TTL)</span>
              </label>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 8 }}>
                <button type="button" className="btn-secondary" onClick={() => setShowAddMemModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Commit Memory
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Add Triple Modal */}
      {showAddTripleModal && (
        <div className="modal-backdrop" onClick={() => setShowAddTripleModal(false)}>
          <div className="dash-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="dash-dialog-head">
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Add Knowledge Graph Triple</h3>
              <button type="button" className="ghost" onClick={() => setShowAddTripleModal(false)}>
                <XIcon />
              </button>
            </div>
            <form onSubmit={handleAddTriple} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div className="form-group">
                <label className="form-label">Subject</label>
                <input
                  type="text"
                  className="text-input"
                  placeholder="e.g. User"
                  value={newSubj}
                  onChange={(e) => setNewSubj(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">Predicate</label>
                <input
                  type="text"
                  className="text-input"
                  placeholder="e.g. prefers"
                  value={newPred}
                  onChange={(e) => setNewPred(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">Object</label>
                <input
                  type="text"
                  className="text-input"
                  placeholder="e.g. Dark Mode"
                  value={newObj}
                  onChange={(e) => setNewObj(e.target.value)}
                  required
                />
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 8 }}>
                <button type="button" className="btn-secondary" onClick={() => setShowAddTripleModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Add Triple
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Inspect Memory Detail Modal */}
      {inspectMemory && (
        <div className="modal-backdrop" onClick={() => setInspectMemory(null)}>
          <div className="dash-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="dash-dialog-head">
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Memory Raw Inspector</h3>
              <button type="button" className="ghost" onClick={() => setInspectMemory(null)}>
                <XIcon />
              </button>
            </div>
            <pre className="dash-code-block" style={{ maxHeight: 280, overflowY: "auto" }}>
              {JSON.stringify(inspectMemory, null, 2)}
            </pre>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
              <button type="button" className="btn-secondary" onClick={() => setInspectMemory(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------------- Mnemosyne Zero-Cloud Memory Panel (Modal tab) ---------------- */

function MnemosyneMemoryPanel({
  config,
  onSaveConfig,
  onOpenFullDashboard,
}: {
  config: MnemosyneConfig;
  onSaveConfig: (updated: MnemosyneConfig) => void;
  onOpenFullDashboard?: () => void;
}) {
  const [enabled, setEnabled] = useState(config.enabled ?? true);
  const [autoRecall, setAutoRecall] = useState(config.autoRecall ?? true);
  const [autoRetain, setAutoRetain] = useState(config.autoRetain ?? true);
  const [recallTopK, setRecallTopK] = useState(config.recallTopK || 5);
  const [scope, setScope] = useState(config.scope || "global");

  const [activeTab, setActiveTab] = useState<"memories" | "triples" | "test" | "add">("memories");
  const [stats, setStats] = useState<MnemosyneStats | null>(null);
  const [memories, setMemories] = useState<EpisodicMemoryItem[]>([]);
  const [triples, setTriples] = useState<TripleItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  // New Memory form
  const [newContent, setNewContent] = useState("");
  const [newImportance, setNewImportance] = useState(0.8);
  const [isWorkingMem, setIsWorkingMem] = useState(false);
  const [ttlSec, setTtlSec] = useState("");

  // New Triple form
  const [newSubj, setNewSubj] = useState("");
  const [newPred, setNewPred] = useState("");
  const [newObj, setNewObj] = useState("");

  // Test Recall state
  const [testQuery, setTestQuery] = useState("");
  const [testResults, setTestResults] = useState<MnemosyneRecallResult | null>(null);
  const [testing, setTesting] = useState(false);

  const refreshData = async () => {
    setLoading(true);
    try {
      const [s, m, t] = await Promise.all([
        cloudGetMnemosyneStats().catch(() => null),
        cloudListMnemosyneMemories().catch(() => []),
        cloudListMnemosyneTriples().catch(() => []),
      ]);
      if (s) setStats(s);
      setMemories(m);
      setTriples(t);
    } catch (err: any) {
      console.error("Failed to load Mnemosyne data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, []);

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    const updated: MnemosyneConfig = {
      enabled,
      autoRecall,
      autoRetain,
      recallTopK: Number(recallTopK) || 5,
      scope: scope.trim() || "global",
      updatedAt: Date.now(),
    };
    onSaveConfig(updated);
    try {
      await cloudSaveMnemosyneConfig(updated);
      setStatusMsg({ type: "ok", text: "Mnemosyne configuration saved to Cloudflare DO SQLite!" });
      setTimeout(() => setStatusMsg(null), 3000);
    } catch (err: any) {
      setStatusMsg({ type: "err", text: err.message || "Failed to save configuration" });
    }
  };

  const handleAddMemory = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newContent.trim()) return;
    try {
      await cloudRememberMnemosyne({
        content: newContent.trim(),
        importance: Number(newImportance),
        isWorkingMemory: isWorkingMem,
        ttlSeconds: ttlSec ? Number(ttlSec) : undefined,
      });
      setNewContent("");
      setStatusMsg({ type: "ok", text: "Memory successfully remembered!" });
      setTimeout(() => setStatusMsg(null), 3000);
      refreshData();
      setActiveTab("memories");
    } catch (err: any) {
      setStatusMsg({ type: "err", text: err.message || "Failed to add memory" });
    }
  };

  const handleAddTriple = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSubj.trim() || !newPred.trim() || !newObj.trim()) return;
    try {
      const res = await fetch("/api/mnemosyne/triples", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          subject: newSubj.trim(),
          predicate: newPred.trim(),
          object: newObj.trim(),
        }),
      });
      const data = (await res.json()) as any;
      if (!data.ok) throw new Error(data.error);
      setNewSubj("");
      setNewPred("");
      setNewObj("");
      setStatusMsg({ type: "ok", text: "Knowledge graph triple added!" });
      setTimeout(() => setStatusMsg(null), 3000);
      refreshData();
    } catch (err: any) {
      setStatusMsg({ type: "err", text: err.message || "Failed to add triple" });
    }
  };

  const handleDeleteMemory = async (id: string) => {
    if (!confirm("Are you sure you want to delete this memory?")) return;
    try {
      await cloudDeleteMnemosyneMemory(id);
      refreshData();
    } catch (err: any) {
      alert("Failed to delete memory: " + err.message);
    }
  };

  const handleDeleteTriple = async (id: string) => {
    if (!confirm("Are you sure you want to delete this triple?")) return;
    try {
      await cloudDeleteMnemosyneTriple(id);
      refreshData();
    } catch (err: any) {
      alert("Failed to delete triple: " + err.message);
    }
  };

  const handleRunRecallTest = async () => {
    if (!testQuery.trim()) return;
    setTesting(true);
    try {
      const res = await cloudRecallMnemosyne(testQuery.trim(), Number(recallTopK) || 5);
      setTestResults(res);
    } catch (err: any) {
      alert("Recall test failed: " + err.message);
    } finally {
      setTesting(false);
    }
  };

  const handleConsolidate = async () => {
    try {
      const count = await cloudConsolidateMnemosyne();
      setStatusMsg({ type: "ok", text: `Consolidated ${count} working memory item(s) into episodic summaries.` });
      setTimeout(() => setStatusMsg(null), 4000);
      refreshData();
    } catch (err: any) {
      alert("Consolidation failed: " + err.message);
    }
  };

  const handleClearAll = async () => {
    if (!confirm("⚠️ Are you sure you want to clear ALL memories & knowledge graph triples? This cannot be undone.")) return;
    try {
      await cloudClearMnemosyne("all");
      setStatusMsg({ type: "ok", text: "All Mnemosyne memories cleared." });
      setTimeout(() => setStatusMsg(null), 3000);
      refreshData();
    } catch (err: any) {
      alert("Clear failed: " + err.message);
    }
  };

  return (
    <div className="mnemosyne-panel">
      {/* Top Banner & Active Toggle */}
      <div className="modal-section" style={{ borderBottom: "none", paddingBottom: 0 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div>
            <h3 className="modal-section-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <BrainIcon />
              <span>Mnemosyne Zero-Cloud Memory</span>
              <span className="badge badge-connected" style={{ fontSize: 11 }}>Cloudflare DO SQLite</span>
            </h3>
            <p className="modal-section-desc">
              Universal BEAM memory layer (Working Memory + Episodic Memory with Hybrid Vector/FTS scoring + Temporal Knowledge Graph). Runs 100% serverless at the Cloudflare edge without external dependencies.
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {onOpenFullDashboard && (
              <button
                type="button"
                className="btn-primary btn-sm"
                onClick={onOpenFullDashboard}
                style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
              >
                <CompassIcon />
                <span>Open Full Intelligence Dashboard</span>
              </button>
            )}
            <label className="toggle-switch" title={enabled ? "Disable Mnemosyne Memory" : "Enable Mnemosyne Memory"}>
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
        </div>
      </div>

      {/* Live Stats Bar */}
      <div className="mnemosyne-stats-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 10, margin: "8px 0 16px" }}>
        <div className="mnemosyne-stat-card" style={{ padding: "10px 14px", background: "var(--panel-2)", border: "1px solid var(--line)", borderRadius: 10 }}>
          <span style={{ fontSize: 11, color: "var(--muted)", display: "block" }}>Episodic Memories</span>
          <span style={{ fontSize: 20, fontWeight: 700, color: "var(--ink)" }}>{stats?.totalEpisodic ?? memories.length}</span>
        </div>
        <div className="mnemosyne-stat-card" style={{ padding: "10px 14px", background: "var(--panel-2)", border: "1px solid var(--line)", borderRadius: 10 }}>
          <span style={{ fontSize: 11, color: "var(--muted)", display: "block" }}>Working Context</span>
          <span style={{ fontSize: 20, fontWeight: 700, color: "var(--ink)" }}>{stats?.totalWorking ?? 0}</span>
        </div>
        <div className="mnemosyne-stat-card" style={{ padding: "10px 14px", background: "var(--panel-2)", border: "1px solid var(--line)", borderRadius: 10 }}>
          <span style={{ fontSize: 11, color: "var(--muted)", display: "block" }}>KG Triples</span>
          <span style={{ fontSize: 20, fontWeight: 700, color: "var(--ink)" }}>{stats?.totalTriples ?? triples.length}</span>
        </div>
        <div className="mnemosyne-stat-card" style={{ padding: "10px 14px", background: "var(--panel-2)", border: "1px solid var(--line)", borderRadius: 10 }}>
          <span style={{ fontSize: 11, color: "var(--muted)", display: "block" }}>BEAM Scoring</span>
          <span style={{ fontSize: 12.5, fontWeight: 600, color: "var(--ok)" }}>50% Vec + 30% FTS</span>
        </div>
      </div>

      {/* Sub-Navigation */}
      <div className="segmented-nav" style={{ width: "100%", marginBottom: 16 }}>
        <button
          type="button"
          className={`tab-btn ${activeTab === "memories" ? "active" : ""}`}
          style={{ flex: 1, justifyContent: "center" }}
          onClick={() => setActiveTab("memories")}
        >
          Memories ({memories.length})
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "triples" ? "active" : ""}`}
          style={{ flex: 1, justifyContent: "center" }}
          onClick={() => setActiveTab("triples")}
        >
          Knowledge Graph ({triples.length})
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "test" ? "active" : ""}`}
          style={{ flex: 1, justifyContent: "center" }}
          onClick={() => setActiveTab("test")}
        >
          Hybrid Recall Test
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "add" ? "active" : ""}`}
          style={{ flex: 1, justifyContent: "center" }}
          onClick={() => setActiveTab("add")}
        >
          + Add Memory
        </button>
      </div>

      {/* SUB-TAB 1: MEMORIES LIST */}
      {activeTab === "memories" && (
        <div className="mnemosyne-memories-container">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <span style={{ fontSize: 12.5, color: "var(--muted)" }}>
              Enduring long-term facts retrieved during chat turns.
            </span>
            <div style={{ display: "flex", gap: 6 }}>
              <button
                type="button"
                className="btn-secondary btn-sm"
                onClick={handleConsolidate}
                title="Consolidate working memory into long-term insights"
              >
                <RefreshIcon />
                <span>Consolidate (Sleep)</span>
              </button>
              <button
                type="button"
                className="btn-secondary btn-sm"
                onClick={refreshData}
                disabled={loading}
              >
                {loading ? <span className="spinner" /> : <RefreshIcon />}
                <span>Refresh</span>
              </button>
            </div>
          </div>

          {memories.length === 0 ? (
            <div className="empty-card" style={{ padding: "28px 16px", textAlign: "center", border: "1px dashed var(--line)", borderRadius: 12 }}>
              <p style={{ margin: "0 0 10px", color: "var(--muted)", fontSize: 13.5 }}>
                No memories recorded yet. The agent will proactively remember salient user preferences and facts, or you can add one manually.
              </p>
              <button type="button" className="btn-secondary btn-sm" onClick={() => setActiveTab("add")}>
                + Add First Memory
              </button>
            </div>
          ) : (
            <div className="mnemosyne-cards-list" style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 320, overflowY: "auto" }}>
              {memories.map((m) => (
                <div
                  key={m.id}
                  className="provider-card"
                  style={{ padding: "10px 14px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}
                >
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontSize: 13.5, color: "var(--ink)", lineHeight: 1.4, wordBreak: "break-word" }}>
                      {m.content}
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6, flexWrap: "wrap" }}>
                      <span className="badge badge-connected" style={{ fontSize: 10 }}>
                        {((m.importance || 0.7) * 100).toFixed(0)}% Salience
                      </span>
                      <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                        Scope: {m.scope}
                      </span>
                      <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                        Accessed: {m.accessCount || 0}x
                      </span>
                      <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                        {new Date(m.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn-text-del"
                    onClick={() => handleDeleteMemory(m.id)}
                    title="Delete memory"
                  >
                    <TrashIcon />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* SUB-TAB 2: KNOWLEDGE GRAPH TRIPLES */}
      {activeTab === "triples" && (
        <div className="mnemosyne-triples-container">
          {/* Add Triple Inline */}
          <form onSubmit={handleAddTriple} style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
            <input
              type="text"
              className="text-input"
              style={{ flex: "1 1 120px" }}
              placeholder="Subject (e.g. User)"
              value={newSubj}
              onChange={(e) => setNewSubj(e.target.value)}
              required
            />
            <input
              type="text"
              className="text-input"
              style={{ flex: "1 1 120px" }}
              placeholder="Predicate (e.g. prefers)"
              value={newPred}
              onChange={(e) => setNewPred(e.target.value)}
              required
            />
            <input
              type="text"
              className="text-input"
              style={{ flex: "1 1 140px" }}
              placeholder="Object (e.g. Dark Mode)"
              value={newObj}
              onChange={(e) => setNewObj(e.target.value)}
              required
            />
            <button type="submit" className="btn-secondary btn-sm" style={{ flexShrink: 0 }}>
              + Add Triple
            </button>
          </form>

          {triples.length === 0 ? (
            <div className="empty-card" style={{ padding: "24px 16px", textAlign: "center", border: "1px dashed var(--line)", borderRadius: 12 }}>
              <p style={{ margin: "0", color: "var(--muted)", fontSize: 13 }}>
                No Knowledge Graph triples recorded yet. Add structured entity links above.
              </p>
            </div>
          ) : (
            <div style={{ maxHeight: 300, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--line)", textAlign: "left", color: "var(--muted)" }}>
                    <th style={{ padding: "6px 8px" }}>Subject</th>
                    <th style={{ padding: "6px 8px" }}>Predicate</th>
                    <th style={{ padding: "6px 8px" }}>Object</th>
                    <th style={{ padding: "6px 8px", width: 40 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {triples.map((t) => (
                    <tr key={t.id} style={{ borderBottom: "1px solid var(--line)" }}>
                      <td style={{ padding: "8px", fontWeight: 600, color: "var(--ink)" }}>{t.subject}</td>
                      <td style={{ padding: "8px", color: "var(--accent)" }}>{t.predicate}</td>
                      <td style={{ padding: "8px", color: "var(--ink)" }}>{t.object}</td>
                      <td style={{ padding: "8px" }}>
                        <button
                          type="button"
                          className="btn-text-del"
                          onClick={() => handleDeleteTriple(t.id)}
                          title="Delete triple"
                        >
                          <TrashIcon />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* SUB-TAB 3: HYBRID RECALL TEST */}
      {activeTab === "test" && (
        <div className="mnemosyne-test-container">
          <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
            <input
              type="text"
              className="text-input"
              style={{ flex: 1 }}
              placeholder="Enter search query (e.g. 'What are my UI preferences?')"
              value={testQuery}
              onChange={(e) => setTestQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleRunRecallTest()}
            />
            <button
              type="button"
              className="btn-primary"
              onClick={handleRunRecallTest}
              disabled={testing || !testQuery.trim()}
            >
              {testing ? <span className="spinner" /> : <SparklesIcon />}
              <span>Test BEAM Recall</span>
            </button>
          </div>

          {testResults && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink)" }}>
                Retrieved Context Payload ({testResults.memories.length} Episodic, {testResults.triples.length} Triples):
              </div>
              <pre
                style={{
                  padding: 12,
                  background: "var(--code-bg)",
                  color: "var(--code-text)",
                  borderRadius: 8,
                  fontSize: 12,
                  fontFamily: "var(--font-mono)",
                  whiteSpace: "pre-wrap",
                  maxHeight: 220,
                  overflowY: "auto",
                }}
              >
                {testResults.formattedContext || "No matching memories found for this query."}
              </pre>

              {testResults.memories.length > 0 && (
                <div>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--ink)" }}>Hybrid Score Breakdown:</span>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 6 }}>
                    {testResults.memories.map((m) => (
                      <div
                        key={m.id}
                        style={{
                          padding: "8px 12px",
                          background: "var(--panel-2)",
                          border: "1px solid var(--line)",
                          borderRadius: 8,
                          fontSize: 12,
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <span style={{ flex: 1 }}>{m.content}</span>
                        <span className="badge badge-connected" style={{ fontSize: 10, marginLeft: 8 }}>
                          BEAM Match: {((m.score || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* SUB-TAB 4: ADD MEMORY FORM */}
      {activeTab === "add" && (
        <form onSubmit={handleAddMemory} className="form-grid" style={{ gap: 12 }}>
          <div className="form-group">
            <label className="form-label">
              Memory Content / Fact
              <span className="form-hint">Concise statement, user preference, or architectural decision</span>
            </label>
            <textarea
              className="text-input"
              rows={3}
              style={{ resize: "vertical" }}
              placeholder="e.g. User prefers Python and React for backend and frontend workflows."
              value={newContent}
              onChange={(e) => setNewContent(e.target.value)}
              required
            />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div className="form-group">
              <label className="form-label">
                Salience / Importance: {((newImportance || 0.7) * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={newImportance}
                onChange={(e) => setNewImportance(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "var(--accent)" }}
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                Working Memory (Hot Context)
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                <input
                  type="checkbox"
                  checked={isWorkingMem}
                  onChange={(e) => setIsWorkingMem(e.target.checked)}
                />
                <span style={{ fontSize: 13 }}>Temporary short-term context</span>
              </label>
            </div>
          </div>

          <button type="submit" className="btn-primary" style={{ marginTop: 4 }}>
            Commit Memory to Mnemosyne
          </button>
        </form>
      )}

      {/* Global Configuration Controls Footer */}
      <form onSubmit={handleSaveSettings} style={{ marginTop: 18, borderTop: "1px solid var(--border)", paddingTop: 14 }}>
        <h4 style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)", margin: "0 0 10px" }}>
          Execution & Auto-Retention Settings
        </h4>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
          <label className="checkbox-label" style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={autoRecall}
              onChange={(e) => setAutoRecall(e.target.checked)}
            />
            <span style={{ fontSize: 12.5, fontWeight: 500 }}>Pre-Turn Auto-Recall</span>
          </label>
          <label className="checkbox-label" style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={autoRetain}
              onChange={(e) => setAutoRetain(e.target.checked)}
            />
            <span style={{ fontSize: 12.5, fontWeight: 500 }}>Post-Turn Auto-Retain</span>
          </label>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
          <button type="button" className="btn-secondary btn-sm" onClick={handleClearAll} style={{ color: "var(--err)" }}>
            Clear All Memories
          </button>
          <button type="submit" className="btn-primary">
            Save Mnemosyne Settings
          </button>
        </div>
      </form>

      {statusMsg && (
        <div className={`banner-msg ${statusMsg.type}`} style={{ marginTop: 12 }}>
          {statusMsg.type === "ok" ? <CheckIcon /> : <XIcon />}
          <span>{statusMsg.text}</span>
        </div>
      )}
    </div>
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
  mcpServers,
  onSaveMcpServers,
  mnemosyneConfig,
  onSaveMnemosyneConfig,
  onOpenFullDashboard,
  customSystemPrompt,
  systemPromptMode,
  onSaveSystemPrompt,
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
  mcpServers: McpServerConfig[];
  onSaveMcpServers: (servers: McpServerConfig[]) => void;
  mnemosyneConfig: MnemosyneConfig;
  onSaveMnemosyneConfig: (config: MnemosyneConfig) => void;
  onOpenFullDashboard?: () => void;
  customSystemPrompt: string;
  systemPromptMode: "append" | "override";
  onSaveSystemPrompt: (prompt: string, mode: "append" | "override") => void;
  convosCount: number;
}) {
  const [activeTab, setActiveTab] = useState<"general" | "models" | "memory" | "prompt" | "tools" | "about">("models");
  const [expandedProviderId, setExpandedProviderId] = useState<string | null>(null);
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [expandedMcpId, setExpandedMcpId] = useState<string | null>(null);
  const [isAddingMcp, setIsAddingMcp] = useState(false);
  const [refreshingMcpId, setRefreshingMcpId] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSaveProvider = async (updatedProvider: ProviderConfig, isNew?: boolean) => {
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
    try {
      const cloudList = await cloudSaveProvider(updatedProvider);
      if (Array.isArray(cloudList)) onSaveProviders(cloudList, isNew ? updatedProvider.id : undefined);
    } catch (err) {
      console.error("Failed to save provider to cloud:", err);
    }
  };

  const handleDeleteProvider = async (id: string) => {
    const updated = providers.filter((p) => p.id !== id);
    const nextActive = activeProviderId === id ? (updated[0]?.id || "") : undefined;
    onSaveProviders(updated, nextActive);
    try {
      const cloudList = await cloudRemoveProvider(id);
      if (Array.isArray(cloudList)) onSaveProviders(cloudList, nextActive);
    } catch (err) {
      console.error("Failed to remove provider from cloud:", err);
    }
  };

  const handleSetDefaultProvider = async (id: string) => {
    const updated = providers.map((p) => ({
      ...p,
      isDefault: p.id === id,
    }));
    onSaveProviders(updated, id);
    try {
      const target = updated.find((p) => p.id === id);
      if (target) {
        await cloudSaveProvider(target);
      }
      await cloudSetActiveProvider(id);
    } catch (err) {
      console.error("Failed to set default provider in cloud:", err);
    }
  };

  const handleSaveMcpServer = async (updatedServer: McpServerConfig, isNew?: boolean) => {
    let updated = [...mcpServers];
    if (isNew) {
      updated.push(updatedServer);
      onSaveMcpServers(updated);
      setIsAddingMcp(false);
    } else {
      updated = updated.map((s) => (s.id === updatedServer.id ? updatedServer : s));
      onSaveMcpServers(updated);
      setExpandedMcpId(null);
    }
    try {
      const cloudServers = await cloudSaveMcpServer(updatedServer);
      if (Array.isArray(cloudServers)) onSaveMcpServers(cloudServers);
    } catch (err) {
      console.error("Failed to save MCP server to cloud:", err);
    }
  };

  const handleDeleteMcpServer = async (id: string) => {
    const updated = mcpServers.filter((s) => s.id !== id);
    onSaveMcpServers(updated);
    try {
      const cloudServers = await cloudRemoveMcpServer(id);
      if (Array.isArray(cloudServers)) onSaveMcpServers(cloudServers);
    } catch (err) {
      console.error("Failed to remove MCP server from cloud:", err);
    }
  };

  const handleToggleMcpServer = async (id: string) => {
    const updated = mcpServers.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s));
    onSaveMcpServers(updated);
    const target = updated.find((s) => s.id === id);
    if (target) {
      cloudSaveMcpServer(target).catch(() => {});
    }
  };

  const handleOAuthSuccess = async () => {
    try {
      const refreshed = await cloudListMcpServers();
      onSaveMcpServers(refreshed);
      setIsAddingMcp(false);
      setExpandedMcpId(null);
    } catch {
      /* ignore */
    }
  };

  const handleRefreshMcpTools = async (server: McpServerConfig) => {
    setRefreshingMcpId(server.id);
    try {
      const tools = await cloudFetchMcpTools(
        server.endpoint,
        server.authType,
        server.bearerToken,
        server.cfAccessClientId,
        server.cfAccessClientSecret,
        server.oauthTokens,
        server.id
      );
      const updated = mcpServers.map((s) => (s.id === server.id ? { ...s, cachedTools: tools } : s));
      onSaveMcpServers(updated);
    } catch {
      /* ignore */
    } finally {
      setRefreshingMcpId(null);
    }
  };

  return (
    <div className="modal-backdrop">
      <div className="modal-card">
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
              className={`tab-btn ${activeTab === "memory" ? "active" : ""}`}
              onClick={() => setActiveTab("memory")}
            >
              <BrainIcon />
              <span>Memory</span>
              {mnemosyneConfig?.enabled && <span className="tab-count">Active</span>}
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "prompt" ? "active" : ""}`}
              onClick={() => setActiveTab("prompt")}
            >
              <TerminalIcon />
              <span>Prompt</span>
              {customSystemPrompt?.trim() && <span className="tab-count">Custom</span>}
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "tools" ? "active" : ""}`}
              onClick={() => setActiveTab("tools")}
            >
              <CpuIcon />
              <span>Tools & MCP</span>
              <span className="tab-count">{mcpServers.length}</span>
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
                      Conversations ({convosCount}), AI Providers ({providers.length}), and MCP Servers ({mcpServers.length}) are securely stored in Cloudflare edge SQLite. Synced across all your devices automatically.
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
              {providers.length === 0 && !isAddingNew && (
                <div className="empty-state-card" style={{ textAlign: "center", padding: "36px 16px", border: "1px dashed var(--border)", borderRadius: 12, margin: "12px 0" }}>
                  <div style={{ fontSize: 32, marginBottom: 8 }}>✨</div>
                  <h4 style={{ margin: "0 0 6px", fontSize: 16, fontWeight: 600, color: "var(--ink)" }}>No AI Providers Configured</h4>
                  <p style={{ margin: "0 0 16px", fontSize: 13, color: "var(--muted)", maxWidth: 440, marginLeft: "auto", marginRight: "auto", lineHeight: 1.5 }}>
                    Add your AI model provider (OpenAI, DeepSeek, OpenRouter, Anthropic proxy, Cloudflare AI Gateway, or local Ollama) to start generating responses.
                  </p>
                  <button type="button" className="btn-primary" onClick={() => setIsAddingNew(true)}>
                    <PlusIcon />
                    <span>Add AI Provider</span>
                  </button>
                </div>
              )}

              <div className="providers-list">
                {providers.map((p) => {
                  const isActive = p.id === activeProviderId;
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
                          <button
                            type="button"
                            className="btn-text-del"
                            onClick={() => handleDeleteProvider(p.id)}
                            title="Delete provider"
                          >
                            <XIcon />
                          </button>
                        </div>
                      </div>

                      {/* Expandable Provider Configuration Panel */}
                      {isExpanded && (
                        <div className="provider-expand-body">
                          <ProviderEditor
                            provider={p}
                            onSave={(updated) => handleSaveProvider(updated, false)}
                            onCancel={() => setExpandedProviderId(null)}
                            onDelete={() => handleDeleteProvider(p.id)}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {/* TAB: MNEMOSYNE ZERO-CLOUD MEMORY */}
          {activeTab === "memory" && (
            <MnemosyneMemoryPanel
              config={mnemosyneConfig}
              onSaveConfig={onSaveMnemosyneConfig}
              onOpenFullDashboard={onOpenFullDashboard}
            />
          )}

          {/* TAB 3: SYSTEM PROMPT */}
          {activeTab === "prompt" && (
            <SystemPromptEditor
              initialPrompt={customSystemPrompt}
              initialMode={systemPromptMode}
              onSave={onSaveSystemPrompt}
            />
          )}

          {/* TAB 4: TOOLS & MCP */}
          {activeTab === "tools" && (
            <>
              <div className="modal-section-head">
                <h3 className="modal-section-title">Connected MCP Servers ({mcpServers.length})</h3>
                {!isAddingMcp && (
                  <button
                    type="button"
                    className="btn-secondary btn-sm"
                    onClick={() => setIsAddingMcp(true)}
                  >
                    <PlusIcon />
                    <span>Add MCP Server</span>
                  </button>
                )}
              </div>

              {/* Add New MCP Server Form */}
              {isAddingMcp && (
                <div className="provider-card active" style={{ padding: 16 }}>
                  <h4 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 600, color: "var(--ink)" }}>
                    Add New MCP Server
                  </h4>
                  <McpServerEditor
                    server={{
                      id: "mcp-" + Math.random().toString(36).slice(2, 9),
                      name: "",
                      endpoint: "",
                      authType: "none",
                      bearerToken: "",
                      enabled: true,
                      cachedTools: [],
                    }}
                    isNew={true}
                    onSave={(s) => handleSaveMcpServer(s, true)}
                    onCancel={() => setIsAddingMcp(false)}
                    onOAuthCompleted={handleOAuthSuccess}
                  />
                </div>
              )}

              {/* MCP Servers List */}
              <div className="providers-list">
                {mcpServers.length === 0 && !isAddingMcp && (
                  <div className="empty-card" style={{ padding: "20px 16px", textAlign: "center", border: "1px dashed var(--line)", borderRadius: 12 }}>
                    <p style={{ margin: "0 0 10px", color: "var(--muted)", fontSize: 13.5 }}>
                      No external MCP servers connected. You can add any custom MCP server endpoint below.
                    </p>
                    <button
                      type="button"
                      className="btn-secondary btn-sm"
                      onClick={() => setIsAddingMcp(true)}
                    >
                      <PlusIcon />
                      <span>Add Your First MCP Server</span>
                    </button>
                  </div>
                )}
                {mcpServers.map((s) => {
                  const isExpanded = expandedMcpId === s.id;
                  const isRefreshing = refreshingMcpId === s.id;

                  return (
                    <div key={s.id} className={`provider-card ${s.enabled ? "" : "disabled-card"}`}>
                      <div
                        className="provider-card-head"
                        onClick={() => setExpandedMcpId(isExpanded ? null : s.id)}
                      >
                        <div className="provider-meta">
                          <div className="provider-title-row">
                            <span className="provider-name">{s.name}</span>
                            {s.enabled ? (
                              <span className="badge badge-connected">Active</span>
                            ) : (
                              <span className="badge badge-disabled">Disabled</span>
                            )}
                            <span className="badge badge-auth">
                              {s.authType === "oauth" ? "OAuth 2.0" : s.authType === "bearer" ? "Bearer" : "No Auth"}
                            </span>
                            <span className="badge badge-tools-count">
                              {s.cachedTools.length} Tool{s.cachedTools.length === 1 ? "" : "s"}
                            </span>
                          </div>
                          <span className="provider-desc">
                            Endpoint: <code>{s.endpoint}</code>
                          </span>
                        </div>
                        <div className="provider-actions" onClick={(e) => e.stopPropagation()}>
                          <button
                            type="button"
                            className="btn-secondary btn-sm"
                            onClick={() => handleRefreshMcpTools(s)}
                            title="Probe & Refresh Tools"
                            disabled={isRefreshing}
                          >
                            {isRefreshing ? <span className="spinner" /> : <RefreshIcon />}
                            <span>{isRefreshing ? "Probing…" : "Refresh"}</span>
                          </button>
                          <button
                            type="button"
                            className="btn-secondary btn-sm"
                            onClick={() => handleToggleMcpServer(s.id)}
                          >
                            {s.enabled ? "Disable" : "Enable"}
                          </button>
                          <button
                            type="button"
                            className="btn-secondary btn-sm"
                            onClick={() => setExpandedMcpId(isExpanded ? null : s.id)}
                          >
                            {isExpanded ? "Collapse" : "Configure"}
                          </button>
                          <button
                            type="button"
                            className="btn-text-del"
                            onClick={() => handleDeleteMcpServer(s.id)}
                            title="Delete MCP server"
                          >
                            <TrashIcon />
                          </button>
                        </div>
                      </div>

                      {/* Expandable MCP Server Editor Panel */}
                      {isExpanded && (
                        <div className="provider-expand-body">
                          <McpServerEditor
                            server={s}
                            onSave={(updated) => handleSaveMcpServer(updated, false)}
                            onCancel={() => setExpandedMcpId(null)}
                            onDelete={() => handleDeleteMcpServer(s.id)}
                            onOAuthCompleted={handleOAuthSuccess}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Built-in Edge Agent Capabilities */}
              <div className="modal-section" style={{ marginTop: 16 }}>
                <h4 className="modal-section-title">Built-in Edge Agent Capabilities</h4>
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
                          <span className="provider-name">Cloudflare Computer Workspace</span>
                          <span className="badge badge-connected">Active</span>
                        </div>
                        <span className="provider-desc">Durable SQLite Virtual File System (read, write, edit, ls)</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
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

/* ---------------- User Message with Attachments ---------------- */

function UserMessageBody({ text, convoId }: { text: string; convoId: string }) {
  // Check if text has [Attached Workspace File(s): ...] or [Uploaded Workspace File: ...]
  const match = text.match(/^\[Attached Workspace File(?:\(s\))?:\s*([^\]]+)\]\s*\n*/i);

  if (!match) {
    return <div className="body">{text}</div>;
  }

  const rawAttachList = match[1];
  const remainingText = text.slice(match[0].length).trim();

  // Parse items separated by comma
  const items = rawAttachList.split(/,\s*(?=\/[^\s])/).map((s) => s.trim()).filter(Boolean);

  const attachments = items.map((itemStr) => {
    const itemMatch = itemStr.match(/^(\/[^\s(]+)(?:\s*\(([^,]+),\s*([^)]+)\))?/);
    const path = itemMatch ? itemMatch[1] : itemStr.split(" ")[0];
    const mime = itemMatch ? itemMatch[2] : "";
    const sizeStr = itemMatch ? itemMatch[3] : "";
    const name = path.split("/").pop() || path;
    const isImage =
      (mime && mime.startsWith("image/")) ||
      /\.(png|jpe?g|gif|webp|svg|ico|bmp|avif)$/i.test(name);
    return { path, name, mime, sizeStr, isImage };
  });

  return (
    <div className="body user-message-container">
      {attachments.length > 0 && (
        <div className="user-msg-attachments">
          {attachments.map((att, i) => {
            const rawUrl = `/api/workspace/raw?convoId=${encodeURIComponent(convoId)}&path=${encodeURIComponent(att.path)}`;
            if (att.isImage) {
              return (
                <div
                  key={i}
                  className="user-msg-img-wrap"
                  onClick={() => window.open(rawUrl, "_blank")}
                  title={`Click to view ${att.name}`}
                >
                  <img src={rawUrl} alt={att.name} />
                  <span className="user-msg-img-name">{att.name}</span>
                </div>
              );
            }
            const badge = getFileExtBadge(att.name);
            return (
              <a
                key={i}
                href={rawUrl}
                target="_blank"
                rel="noreferrer"
                download={att.name}
                className="user-msg-file-pill"
                title={`Download ${att.name}`}
              >
                <span className="file-badge" style={{ color: badge.color }}>
                  {badge.label}
                </span>
                <span className="user-msg-file-name">{att.name}</span>
                {att.sizeStr && <span className="user-msg-file-size">{att.sizeStr}</span>}
              </a>
            );
          })}
        </div>
      )}
      {remainingText && <div className="user-msg-text">{remainingText}</div>}
    </div>
  );
}

/* ---------------- Composer ---------------- */

function Composer({
  draft,
  setDraft,
  stagedFiles,
  onAddFiles,
  onRemoveFile,
  onSubmit,
  busy,
  uploading,
  activeProvider,
  providers,
  onSelectProvider,
  onOpenSettings,
  onUpdateReasoningEffort,
}: {
  draft: string;
  setDraft: (v: string) => void;
  stagedFiles: StagedAttachment[];
  onAddFiles: (files: File[]) => void;
  onRemoveFile: (id: string) => void;
  onSubmit: () => void;
  busy: boolean;
  uploading?: boolean;
  activeProvider?: ProviderConfig;
  providers: ProviderConfig[];
  onSelectProvider: (id: string) => void;
  onOpenSettings: () => void;
  onUpdateReasoningEffort?: (effort: "none" | "low" | "medium" | "high") => void;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 176) + "px";
  }, [draft, stagedFiles]);

  const handlePaste = (e: React.ClipboardEvent) => {
    const clipboardData = e.clipboardData;
    if (!clipboardData) return;
    const files: File[] = [];

    if (clipboardData.files && clipboardData.files.length > 0) {
      for (let i = 0; i < clipboardData.files.length; i++) {
        files.push(clipboardData.files[i]);
      }
    } else if (clipboardData.items && clipboardData.items.length > 0) {
      for (let i = 0; i < clipboardData.items.length; i++) {
        const item = clipboardData.items[i];
        if (item.kind === "file") {
          const file = item.getAsFile();
          if (file) {
            if (file.name === "image.png" || !file.name) {
              const now = new Date();
              const timeStr = now.toISOString().replace(/[-:T]/g, "").slice(0, 14);
              const named = new File([file], `pasted_image_${timeStr}.png`, { type: file.type || "image/png" });
              files.push(named);
            } else {
              files.push(file);
            }
          }
        }
      }
    }

    if (files.length > 0) {
      if (!clipboardData.getData("text/plain")) {
        e.preventDefault();
      }
      onAddFiles(files);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onAddFiles(Array.from(e.dataTransfer.files));
    }
  };

  return (
    <div
      className={`composer ${isDragging ? "drag-over" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onPaste={handlePaste}
    >
      {/* Gemini-like small attachment preview cards */}
      {stagedFiles.length > 0 && (
        <div className="composer-attachments-preview">
          {stagedFiles.map((att) => (
            <div
              key={att.id}
              className={`composer-attach-card ${att.isImage ? "is-image" : "is-file"}`}
              title={`${att.name} (${formatBytes(att.size)})`}
            >
              {att.isImage && att.previewUrl ? (
                <div className="composer-attach-thumb">
                  <img src={att.previewUrl} alt={att.name} />
                </div>
              ) : (
                <div className="composer-attach-file-info">
                  <span className="file-badge" style={{ color: getFileExtBadge(att.name).color }}>
                    {getFileExtBadge(att.name).label}
                  </span>
                  <div className="composer-attach-meta">
                    <span className="composer-attach-name">{att.name}</span>
                    <span className="composer-attach-size">{formatBytes(att.size)}</span>
                  </div>
                </div>
              )}
              <button
                type="button"
                className="composer-attach-del"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemoveFile(att.id);
                }}
                title="Remove attachment"
                aria-label="Remove attachment"
              >
                <XIcon />
              </button>
            </div>
          ))}
        </div>
      )}

      <textarea
        ref={ref}
        rows={1}
        value={draft}
        placeholder={
          stagedFiles.length > 0
            ? "Add instructions for attached file(s)…"
            : "How can I help you today? (Paste or upload files)"
        }
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
      />

      <div className="composer-row">
        <div className="composer-controls-left">
          <button
            type="button"
            className="composer-attach-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Upload image or files (+)"
            aria-label="Upload files"
            disabled={busy || uploading}
          >
            <PlusIcon />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                onAddFiles(Array.from(e.target.files));
                e.target.value = "";
              }
            }}
          />

          <button
            type="button"
            className="pill pill-interactive"
            onClick={() => (providers.length > 0 ? setMenuOpen(!menuOpen) : onOpenSettings())}
            aria-label="Select provider and model"
          >
            <SparklesIcon />
            <span>{activeProvider ? `${activeProvider.name}: ${activeProvider.selectedModel}` : "Configure AI Provider"}</span>
            {activeProvider?.reasoningEffort && activeProvider.reasoningEffort !== "none" && (
              <span className="reasoning-effort-badge" title={`Reasoning Effort: ${activeProvider.reasoningEffort}`}>
                <ZapIcon />
                <span>{activeProvider.reasoningEffort.toUpperCase()}</span>
              </span>
            )}
            {providers.length > 0 && (
              <span className={`pill-caret ${menuOpen ? "open" : ""}`}>
                <ChevronDownIcon />
              </span>
            )}
          </button>
        </div>

        {menuOpen && (
          <>
            <div className="model-popover-backdrop" onClick={() => setMenuOpen(false)} />
            <div className="model-popover">
              <div className="popover-head">Select Provider</div>
              {providers.length === 0 ? (
                <div style={{ padding: "12px 14px", fontSize: 13, color: "var(--muted)" }}>
                  No providers configured yet.
                </div>
              ) : (
                providers.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className={`popover-item ${p.id === activeProvider?.id ? "active" : ""}`}
                    onClick={() => {
                      onSelectProvider(p.id);
                      setMenuOpen(false);
                    }}
                  >
                    <div className="popover-item-title">
                      <span className="popover-item-name">{p.name}</span>
                      <span className="popover-item-sub">{p.selectedModel}</span>
                    </div>
                    {p.id === activeProvider?.id && <CheckIcon />}
                  </button>
                ))
              )}

              {activeProvider && (
                <>
                  <div className="popover-divider" />
                  <div className="popover-section-label">Reasoning Effort (CoT)</div>
                  <div className="segmented-nav" style={{ margin: "4px 8px 8px", padding: 2 }}>
                    {(["none", "low", "medium", "high"] as const).map((lvl) => (
                      <button
                        key={lvl}
                        type="button"
                        className={`tab-btn ${(activeProvider.reasoningEffort || "none") === lvl ? "active" : ""}`}
                        style={{ flex: 1, justifyContent: "center", fontSize: 11, padding: "4px 2px" }}
                        onClick={() => {
                          if (onUpdateReasoningEffort) onUpdateReasoningEffort(lvl);
                        }}
                      >
                        {lvl === "none" ? "Off" : lvl.charAt(0).toUpperCase() + lvl.slice(1)}
                      </button>
                    ))}
                  </div>
                </>
              )}

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

        <button
          type="button"
          className="send"
          onClick={onSubmit}
          disabled={busy || uploading || (!draft.trim() && stagedFiles.length === 0)}
          aria-label="Send"
        >
          {uploading ? <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> : <SendIcon />}
        </button>
      </div>
    </div>
  );
}

const HINTS = [
  "Search the web for the latest Cloudflare Agents SDK features",
  "Write today's meeting notes and save them in the workspace",
  "Summarize the latest AI architecture breakthroughs",
];

/* ---------------- Diagnostic Error Card ---------------- */

interface ParsedBackendError {
  statusCode: number | string;
  statusCategory: string;
  primaryReason: string;
  errorType?: string;
  errorCode?: string | number;
  param?: string;
  endpointUrl: string;
  modelId: string;
  providerName: string;
  timestamp: string;
  rawJson: Record<string, unknown> | null;
  rawOutput: string;
}

function parseBackendError(
  rawError: string,
  activeProvider?: ProviderConfig
): ParsedBackendError {
  let explicitStatus: number | string | null = null;
  let rawJson: Record<string, unknown> | null = null;

  // 1. Try to extract JSON from rawError
  const jsonMatch = rawError.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]);
      if (typeof parsed === "object" && parsed !== null) {
        rawJson = parsed;
      }
    } catch {
      /* ignore */
    }
  }

  // 2. Extract HTTP status code from text or JSON
  if (rawJson) {
    if ((rawJson as any).status) explicitStatus = (rawJson as any).status;
    else if ((rawJson as any).statusCode) explicitStatus = (rawJson as any).statusCode;
    else if ((rawJson as any).error && typeof (rawJson as any).error === "object" && (rawJson as any).error.code && typeof (rawJson as any).error.code === "number") {
      explicitStatus = (rawJson as any).error.code;
    }
  }

  if (!explicitStatus) {
    const statusMatch = rawError.match(/\b([45]\d{2})\b/);
    if (statusMatch) {
      explicitStatus = parseInt(statusMatch[1], 10);
    }
  }

  const statusCode = explicitStatus || (rawError.toLowerCase().includes("websocket") ? "WS_DISCONNECT" : "ERROR");

  let statusCategory = "Request Failed";
  if (statusCode === 400) statusCategory = "Bad Request (Invalid parameters / model name / context limit)";
  else if (statusCode === 401) statusCategory = "Unauthorized (Invalid or missing API key)";
  else if (statusCode === 403) statusCategory = "Forbidden (Permission denied / Country blocked)";
  else if (statusCode === 404) statusCategory = "Not Found (Model ID not found on endpoint / invalid path)";
  else if (statusCode === 422) statusCategory = "Unprocessable Entity (Schema / body validation error)";
  else if (statusCode === 429) statusCategory = "Too Many Requests (Rate limited or quota exhausted)";
  else if (statusCode === 500) statusCategory = "Internal Server Error (Upstream provider crashed)";
  else if (statusCode === 502) statusCategory = "Bad Gateway (Upstream endpoint unreachable / proxy error)";
  else if (statusCode === 503) statusCategory = "Service Unavailable (Model overloaded / under maintenance)";
  else if (statusCode === 504) statusCategory = "Gateway Timeout (Upstream inference timed out)";
  else if (statusCode === 524) statusCategory = "Cloudflare Timeout (Inference took too long to first token)";
  else if (statusCode === "WS_DISCONNECT") statusCategory = "WebSocket Disconnected";

  let primaryReason = "";
  let errorType = "";
  let errorCode: string | number | undefined = undefined;
  let param: string | undefined = undefined;

  if (rawJson) {
    const errSub = (rawJson.error || rawJson) as any;
    if (typeof errSub === "object" && errSub !== null) {
      primaryReason = errSub.message || errSub.msg || errSub.detail || errSub.error || "";
      errorType = errSub.type || "";
      errorCode = errSub.code || "";
      param = errSub.param || "";
    } else if (typeof errSub === "string") {
      primaryReason = errSub;
    }
  }

  if (!primaryReason) {
    primaryReason = rawError
      .replace(/^Failed to process request:\s*/i, "")
      .replace(/^Upstream error\s*(\(\d+\))?:\s*/i, "")
      .trim() || "An unexpected error occurred during model inference.";
  }

  const endpointUrl = activeProvider?.endpoint || (activeProvider?.id === "cf-default" ? "Cloudflare AI Gateway (AIG_BASE_URL)" : "Default Endpoint");
  const modelId = activeProvider?.selectedModel || "deepseek-v4-flash";
  const providerName = activeProvider?.name || "Configured Provider";
  const protocol = activeProvider?.useResponseApi ? "openai.responses (/responses)" : "openai.chat (/chat/completions)";

  const completeJson = rawJson || {
    statusCode,
    statusCategory,
    error: {
      message: primaryReason,
      type: errorType || undefined,
      code: errorCode || undefined,
      param: param || undefined,
    },
    target: {
      provider: providerName,
      model: modelId,
      endpoint: endpointUrl,
      protocol,
    },
    raw: rawError,
  };

  const rawOutput = JSON.stringify(completeJson, null, 2);

  return {
    statusCode,
    statusCategory,
    primaryReason,
    errorType: errorType || undefined,
    errorCode: errorCode || undefined,
    param: param || undefined,
    endpointUrl,
    modelId,
    providerName,
    timestamp: new Date().toISOString(),
    rawJson: completeJson,
    rawOutput,
  };
}

function ChatErrorCard({
  rawError,
  activeProvider,
  onRetry,
  onDismiss,
  onOpenSettings,
}: {
  rawError: string;
  activeProvider?: ProviderConfig;
  onRetry?: () => void;
  onDismiss: () => void;
  onOpenSettings: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const parsed = parseBackendError(rawError, activeProvider);

  const handleCopy = () => {
    try {
      navigator.clipboard.writeText(parsed.rawOutput);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="chat-error-card" role="alert">
      {/* Header with HTTP Status & Target Metadata */}
      <div className="chat-error-head">
        <div className="chat-error-title-row">
          <div className="chat-error-icon">
            <AlertTriangleIcon />
          </div>
          <span className="chat-error-http-badge">
            {typeof parsed.statusCode === "number" ? `HTTP ${parsed.statusCode}` : parsed.statusCode}
          </span>
          <span className="chat-error-category">{parsed.statusCategory}</span>
        </div>
        <button type="button" className="chat-error-close" onClick={onDismiss} title="Dismiss error">
          <XIcon />
        </button>
      </div>

      {/* Target Metadata Pills */}
      <div className="chat-error-meta-tags">
        <span className="chat-error-meta-tag">
          Provider: <strong>{parsed.providerName}</strong>
        </span>
        <span className="chat-error-meta-tag">
          Model: <code>{parsed.modelId}</code>
        </span>
        {parsed.endpointUrl && (
          <span className="chat-error-meta-tag endpoint-tag" title={parsed.endpointUrl}>
            Endpoint: <code>{parsed.endpointUrl}</code>
          </span>
        )}
      </div>

      {/* Exact Error Reason Box */}
      <div className="chat-error-reason-box">
        <div className="chat-error-reason-label">Exact Error Reason:</div>
        <div className="chat-error-reason-text">{parsed.primaryReason}</div>
        {(parsed.errorType || parsed.errorCode || parsed.param) && (
          <div className="chat-error-reason-extra">
            {parsed.errorType && <span>Type: <code>{parsed.errorType}</code></span>}
            {parsed.errorCode && <span>Code: <code>{String(parsed.errorCode)}</code></span>}
            {parsed.param && <span>Param: <code>{parsed.param}</code></span>}
          </div>
        )}
      </div>

      {/* Direct Backend Response / Raw JSON output */}
      <div className="chat-error-json-card">
        <div className="chat-error-json-head">
          <span className="chat-error-json-title">Backend Response (Raw JSON Payload)</span>
          <button type="button" className="btn-copy-raw-log" onClick={handleCopy}>
            {copied ? <CheckIcon /> : <CopyIcon />}
            <span>{copied ? "Copied" : "Copy JSON"}</span>
          </button>
        </div>
        <pre className="chat-error-raw-json">{parsed.rawOutput}</pre>
      </div>

      {/* Actions */}
      <div className="chat-error-actions">
        {onRetry && (
          <button type="button" className="btn-error-action primary" onClick={onRetry}>
            <RefreshIcon />
            Retry Request
          </button>
        )}
        <button type="button" className="btn-error-action" onClick={onOpenSettings}>
          <SettingsIcon />
          Fix in Provider Settings
        </button>
      </div>
    </div>
  );
}

/* ---------------- one conversation ---------------- */

function Chat({
  convoId,
  onFirstMessage,
  activeProvider,
  providers,
  onSelectProvider,
  onOpenSettings,
  onUpdateProviderReasoningEffort,
  mcpServers,
  mnemosyneConfig,
  customSystemPrompt,
  systemPromptMode,
  onTriggerWorkspaceRefresh,
}: {
  convoId: string;
  onFirstMessage: (text: string) => void;
  activeProvider?: ProviderConfig;
  providers: ProviderConfig[];
  onSelectProvider: (id: string) => void;
  onOpenSettings: () => void;
  onUpdateProviderReasoningEffort?: (effort: "none" | "low" | "medium" | "high") => void;
  mcpServers: McpServerConfig[];
  mnemosyneConfig: MnemosyneConfig;
  customSystemPrompt: string;
  systemPromptMode: "append" | "override";
  onTriggerWorkspaceRefresh?: () => void;
}) {
  const agent = useAgent({ agent: "Assistant", name: convoId });
  const [dismissedError, setDismissedError] = useState<string | null>(null);
  const lastSentUserTextRef = useRef<string>("");

  const { messages, sendMessage, status, error, regenerate, clearError, connectionError } = useAgentChat({
    agent,
    body: () => ({
      providerId: activeProvider?.id,
      mcpServers,
      mnemosyneConfig,
      userMessage: lastSentUserTextRef.current,
      customSystemPrompt,
      promptMode: systemPromptMode,
      customModel:
        activeProvider?.endpoint && activeProvider?.selectedModel
          ? {
              endpoint: activeProvider.endpoint,
              apiKey: activeProvider.apiKey,
              modelId: activeProvider.selectedModel,
              useResponseApi: activeProvider.useResponseApi,
              temperature: activeProvider.temperature,
              maxTokens: activeProvider.maxTokens,
              topP: activeProvider.topP,
              reasoningEffort: activeProvider.reasoningEffort,
            }
          : undefined,
    }),
  });

  // Automatically refresh workspace files & record turn durations when agent finishes a turn
  const msgDurationsRef = useRef<Record<string, number>>({});
  const turnStartTimeRef = useRef<number | null>(null);

  const prevStatusRef = useRef(status);
  useEffect(() => {
    if (status === "submitted" || status === "streaming") {
      if (!turnStartTimeRef.current) {
        turnStartTimeRef.current = Date.now();
      }
    } else if (prevStatusRef.current === "streaming" && status === "ready") {
      if (turnStartTimeRef.current) {
        const lastMsg = messages[messages.length - 1];
        if (lastMsg && lastMsg.role !== "user" && !msgDurationsRef.current[lastMsg.id]) {
          const elapsed = Number(((Date.now() - turnStartTimeRef.current) / 1000).toFixed(1));
          msgDurationsRef.current[lastMsg.id] = Math.max(0.2, elapsed);
        }
        turnStartTimeRef.current = null;
      }
      onTriggerWorkspaceRefresh?.();

      // Mnemosyne Post-Turn Auto-Retain: Asynchronously retain turn in background
      if (mnemosyneConfig?.enabled && mnemosyneConfig.autoRetain) {
        const userMsgs = messages.filter((m) => m.role === "user");
        const asstMsgs = messages.filter((m) => m.role === "assistant");
        const lastUser = userMsgs[userMsgs.length - 1];
        const lastAsst = asstMsgs[asstMsgs.length - 1];
        if (lastUser && lastAsst) {
          const userText = textOf(lastUser);
          const asstText = textOf(lastAsst);
          if (userText && asstText) {
            cloudRememberMnemosyne({
              content: `User: ${userText.slice(0, 300)}\nAssistant: ${asstText.slice(0, 300)}`,
              importance: 0.6,
            }).catch((err: any) => console.warn("Mnemosyne auto-retain failed:", err));
          }
        }
      }
    }
    prevStatusRef.current = status;
  }, [status, messages, onTriggerWorkspaceRefresh, mnemosyneConfig]);

  const [draft, setDraft] = useState("");
  const [stagedFiles, setStagedFiles] = useState<StagedAttachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const busy = status === "submitted" || status === "streaming";
  const titled = useRef(false);
  const empty = messages.length === 0;

  const handleAddFiles = (files: File[]) => {
    const next: StagedAttachment[] = [];
    for (const f of files) {
      const isImg = f.type.startsWith("image/") || /\.(png|jpe?g|gif|webp|svg|ico|bmp|avif)$/i.test(f.name);
      const id = "att-" + Math.random().toString(36).slice(2, 9);
      const previewUrl = isImg ? URL.createObjectURL(f) : undefined;
      next.push({
        id,
        file: f,
        name: f.name,
        size: f.size,
        type: f.type || (isImg ? "image/png" : "application/octet-stream"),
        isImage: isImg,
        previewUrl,
      });
    }
    setStagedFiles((prev) => [...prev, ...next]);
  };

  const handleRemoveFile = (id: string) => {
    setStagedFiles((prev) => {
      const item = prev.find((x) => x.id === id);
      if (item?.previewUrl) {
        URL.revokeObjectURL(item.previewUrl);
      }
      return prev.filter((x) => x.id !== id);
    });
  };

  const activeErrorStr = connectionError
    ? connectionError.message || `WebSocket Connection Error (${connectionError.code || "closed"})`
    : error
    ? error.message || String(error)
    : null;

  const hasError = !!activeErrorStr && dismissedError !== activeErrorStr;

  const submit = async () => {
    const text = draft.trim();
    if ((!text && stagedFiles.length === 0) || busy || uploading) return;
    if (!activeProvider || !activeProvider.endpoint || !activeProvider.selectedModel) {
      onOpenSettings();
      return;
    }

    let finalPromptText = text;
    const filesToUpload = [...stagedFiles];

    if (filesToUpload.length > 0) {
      setUploading(true);
      try {
        const formData = new FormData();
        for (const item of filesToUpload) {
          formData.append("files", item.file, item.name);
        }
        const res = await fetch(`/api/workspace/upload?convoId=${encodeURIComponent(convoId)}`, {
          method: "POST",
          body: formData,
        });
        const data = (await res.json()) as any;
        if (!data.ok) {
          throw new Error(data.error || "Failed to upload files");
        }
        onTriggerWorkspaceRefresh?.();

        const fileSummary = filesToUpload
          .map((f) => `/${f.name.replace(/^\/+/, "")} (${f.type || "file"}, ${formatBytes(f.size)})`)
          .join(", ");

        const attachHeader = `[Attached Workspace File(s): ${fileSummary}]`;
        if (text) {
          finalPromptText = `${attachHeader}\n\n${text}`;
        } else {
          finalPromptText = `${attachHeader}\nPlease inspect and analyze the attached workspace file(s).`;
        }
      } catch (err: any) {
        alert(`File upload failed: ${err.message || String(err)}`);
        setUploading(false);
        return;
      } finally {
        for (const item of filesToUpload) {
          if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
        }
        setStagedFiles([]);
        setUploading(false);
      }
    }

    turnStartTimeRef.current = Date.now();
    lastSentUserTextRef.current = finalPromptText;
    setDismissedError(null);
    clearError?.();
    if (!titled.current) {
      titled.current = true;
      onFirstMessage(text || (filesToUpload[0] ? `Upload ${filesToUpload[0].name}` : "Upload files"));
    }
    sendMessage({ text: finalPromptText });
    setDraft("");
  };

  const handleRetry = () => {
    turnStartTimeRef.current = Date.now();
    setDismissedError(null);
    clearError?.();
    if (regenerate) {
      regenerate();
    }
  };

  return (
    <div className={empty ? "page home" : "page chat"}>
      {empty ? (
        <div className="home-inner">
          <h1>{greeting()}, Aki</h1>
          <div className="home-composer">
            {hasError && activeErrorStr && (
              <div style={{ marginBottom: 16 }}>
                <ChatErrorCard
                  rawError={activeErrorStr}
                  activeProvider={activeProvider}
                  onRetry={handleRetry}
                  onDismiss={() => {
                    setDismissedError(activeErrorStr);
                    clearError?.();
                  }}
                  onOpenSettings={onOpenSettings}
                />
              </div>
            )}
            <Composer
              draft={draft}
              setDraft={setDraft}
              stagedFiles={stagedFiles}
              onAddFiles={handleAddFiles}
              onRemoveFile={handleRemoveFile}
              onSubmit={submit}
              busy={busy}
              uploading={uploading}
              activeProvider={activeProvider}
              providers={providers}
              onSelectProvider={onSelectProvider}
              onOpenSettings={onOpenSettings}
              onUpdateReasoningEffort={onUpdateProviderReasoningEffort}
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
                    turnStartTimeRef.current = Date.now();
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
                {m.role === "user" ? (
                  <UserMessageBody text={textOf(m)} convoId={convoId} />
                ) : (
                  (() => {
                    const { thoughtText, mainText, isThinking } = extractThoughtAndAnswer(m);
                    const isLast = m.id === messages[messages.length - 1]?.id;
                    // An unterminated <think> only means "still thinking" while
                    // the stream is live; after the turn ends it renders as a
                    // completed CoT block.
                    const activelyThinking = busy && isLast;
                    const responseText = mainText || textOf(m);
                    const fullText = (thoughtText ? `<think>\n${thoughtText}\n</think>\n\n` : "") + responseText;
                    const durationSec = msgDurationsRef.current[m.id];

                    return (
                      <>
                        <ToolBits message={m} />
                        {thoughtText && (
                          <ThoughtBlock
                            thought={thoughtText}
                            isThinking={isThinking && activelyThinking}
                          />
                        )}
                        {mainText ? (
                          <Markdown text={mainText} />
                        ) : isThinking || (busy && isLast) ? null : (
                          <Markdown text={textOf(m)} />
                        )}
                        {!activelyThinking && responseText && (
                          <MessageMetaFooter rawText={fullText} durationSec={durationSec} />
                        )}
                      </>
                    );
                  })()
                )}
              </article>
            ))}
            {busy && (
              <div className="thinking" aria-live="polite">
                <span className="thinking-dot" />
                <span className="thinking-dot" style={{ animationDelay: "0.15s" }} />
                <span className="thinking-dot" style={{ animationDelay: "0.3s" }} />
              </div>
            )}
            {hasError && activeErrorStr && (
              <ChatErrorCard
                rawError={activeErrorStr}
                activeProvider={activeProvider}
                onRetry={handleRetry}
                onDismiss={() => {
                  setDismissedError(activeErrorStr);
                  clearError?.();
                }}
                onOpenSettings={onOpenSettings}
              />
            )}
          </main>
          <Composer
            draft={draft}
            setDraft={setDraft}
            stagedFiles={stagedFiles}
            onAddFiles={handleAddFiles}
            onRemoveFile={handleRemoveFile}
            onSubmit={submit}
            busy={busy}
            uploading={uploading}
            activeProvider={activeProvider}
            providers={providers}
            onSelectProvider={onSelectProvider}
            onOpenSettings={onOpenSettings}
            onUpdateReasoningEffort={onUpdateProviderReasoningEffort}
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

  // MCP Servers State (Local hot cache + Cloud DO source of truth)
  const [mcpServers, setMcpServers] = useState<McpServerConfig[]>(loadLocalMcpServers);

  // System Prompt State (Local hot cache + Cloud DO app_settings)
  const [customSystemPrompt, setCustomSystemPrompt] = useState<string>(
    () => loadLocalSystemPrompt().prompt
  );
  const [systemPromptMode, setSystemPromptMode] = useState<"append" | "override">(
    () => loadLocalSystemPrompt().mode
  );

  // Settings Modal State
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [currentView, setCurrentView] = useState<"chat" | "memory-dashboard">("chat");

  // Mnemosyne Zero-Cloud Memory State (Cloud DO SQLite source of truth)
  const [mnemosyneConfig, setMnemosyneConfig] = useState<MnemosyneConfig>({
    enabled: true,
    autoRecall: true,
    autoRetain: true,
    recallTopK: 5,
    scope: "global",
  });

  // Right Workspace Drawer State
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [workspaceFileCount, setWorkspaceFileCount] = useState(0);
  const [workspaceRefreshToken, setWorkspaceRefreshToken] = useState(0);

  const cycleTheme = () => {
    const next: ThemeMode = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
    setTheme(next);
  };

  useEffect(() => {
    saveLocalTheme(theme);
    if (theme === "system") {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.setAttribute("data-theme", prefersDark ? "dark" : "light");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
  }, [theme]);

  // Sync settings and cloud data
  useEffect(() => {
    let cancelled = false;

    cloudListConvs()
      .then((cloudList) => {
        if (!cancelled && cloudList.length) {
          applyCloud(cloudList);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setCloudReady(true);
      });

    cloudListProviders()
      .then((cloudProvidersRes) => {
        const cloudProviders = (cloudProvidersRes.providers || []).filter((p) => p.id !== "cf-default");
        if (!cancelled) {
          setProviders(cloudProviders);
          saveLocalProviders(cloudProviders);
          const activeItem =
            (cloudProvidersRes.activeId && cloudProviders.find((p) => p.id === cloudProvidersRes.activeId)) ||
            cloudProviders.find((p) => p.isDefault) ||
            cloudProviders[0];
          if (activeItem) {
            setActiveProviderId(activeItem.id);
            saveActiveProviderId(activeItem.id);
          }
        }
      })
      .catch(() => {});

    cloudListMcpServers()
      .then((servers) => {
        if (!cancelled && servers.length) {
          setMcpServers(servers);
          saveLocalMcpServers(servers);
        }
      })
      .catch(() => {});

    cloudGetMnemosyneConfig()
      .then((cfg) => {
        if (!cancelled && cfg) {
          setMnemosyneConfig(cfg);
        }
      })
      .catch(() => {});

    cloudGetSetting("custom_system_prompt")
      .then((promptVal) => {
        if (!cancelled && promptVal) {
          setCustomSystemPrompt(promptVal);
          saveLocalSystemPrompt(promptVal, systemPromptMode);
        }
      })
      .catch(() => {});

    cloudGetSetting("system_prompt_mode")
      .then((modeVal) => {
        if (!cancelled && (modeVal === "append" || modeVal === "override")) {
          setSystemPromptMode(modeVal);
          saveLocalSystemPrompt(customSystemPrompt, modeVal);
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, []);

  const activeProvider = providers.find((p) => p.id === activeProviderId) || providers[0];

  const handleSelectActiveProvider = (id: string) => {
    setActiveProviderId(id);
    saveActiveProviderId(id);
    cloudSetActiveProvider(id).catch((err) => {
      console.error("Failed to sync active provider with cloud:", err);
    });
  };

  const handleUpdateProviderReasoningEffort = (effort: "none" | "low" | "medium" | "high") => {
    if (!activeProvider) return;
    const updated = providers.map((p) => (p.id === activeProvider.id ? { ...p, reasoningEffort: effort } : p));
    setProviders(updated);
    saveLocalProviders(updated);
    cloudSaveProvider({ ...activeProvider, reasoningEffort: effort }).catch((err) => {
      console.error("Failed to sync reasoning effort with cloud:", err);
    });
  };

  const handleSaveProviders = (nextProviders: ProviderConfig[], newActiveId?: string) => {
    const clean = nextProviders.filter((p) => p.id !== "cf-default");
    setProviders(clean);
    if (newActiveId) {
      setActiveProviderId(newActiveId);
      saveActiveProviderId(newActiveId);
      cloudSetActiveProvider(newActiveId).catch(() => {});
    } else if (!clean.some((p) => p.id === activeProviderId)) {
      const fallbackId = clean[0]?.id;
      if (fallbackId) {
        setActiveProviderId(fallbackId);
        saveActiveProviderId(fallbackId);
        cloudSetActiveProvider(fallbackId).catch(() => {});
      }
    }
    saveLocalProviders(clean);
    cloudSaveAllProviders(clean).catch((err) => {
      console.error("Failed to sync providers with cloud:", err);
    });
  };

  const handleSaveMcpServers = async (nextServers: McpServerConfig[]) => {
    setMcpServers(nextServers);
    saveLocalMcpServers(nextServers);
    cloudSaveAllMcpServers(nextServers).catch((err) => {
      console.error("Failed to sync MCP servers with cloud:", err);
    });
  };

  const handleSaveMnemosyneConfig = async (nextConfig: MnemosyneConfig) => {
    setMnemosyneConfig(nextConfig);
    try {
      const saved = await cloudSaveMnemosyneConfig(nextConfig);
      setMnemosyneConfig(saved);
    } catch (err) {
      console.error("Failed to sync Mnemosyne config with cloud:", err);
    }
  };

  const handleSaveSystemPrompt = async (prompt: string, mode: "append" | "override") => {
    setCustomSystemPrompt(prompt);
    setSystemPromptMode(mode);
    saveLocalSystemPrompt(prompt, mode);
    await cloudSetSetting("custom_system_prompt", prompt);
    await cloudSetSetting("system_prompt_mode", mode);
  };

  const closeSideOnMobile = () => {
    if (window.innerWidth < 900) setSideOpen(false);
  };

  const applyCloud = (next: Convo[]) => {
    setConvos(next);
    saveLocalConvs(next);
  };

  useEffect(() => {
    saveLocalConvs(convos);
  }, [convos]);

  const touch = (id: string, title?: string) => {
    const now = Date.now();
    setConvos((prev) => {
      const existing = prev.find((c) => c.id === id);
      const nextTitle =
        title && (!existing || existing.title === "New chat" || existing.title.startsWith("c-"))
          ? title.slice(0, 48)
          : existing?.title ?? title ?? "New chat";
      const next = [{ id, title: nextTitle, ts: now }, ...prev.filter((c) => c.id !== id)];
      return next.slice(0, 100);
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
    const id = newId();
    touch(id, "New chat");
    setActive(id);
    setCurrentView("chat");
    closeSideOnMobile();
  };

  return (
    <div className={"shell" + (sideOpen ? " side-open" : "") + (workspaceOpen ? " workspace-open" : "")}>
      {/* Mobile Top Header */}
      <header className="app-header">
        <div className="header-left">
          <button
            type="button"
            className="ghost"
            onClick={() => setSideOpen(true)}
            aria-label="Open sidebar"
            title="Open sidebar"
          >
            <SideCollapseIcon />
          </button>
          <span className="header-wordmark">edge agent</span>
        </div>
        <div className="header-right">
          <button
            type="button"
            className={`ghost header-workspace-btn ${currentView === "memory-dashboard" ? "active" : ""}`}
            onClick={() => setCurrentView((v) => (v === "memory-dashboard" ? "chat" : "memory-dashboard"))}
            aria-label="Memory Dashboard"
            title="Mnemosyne Memory Dashboard"
          >
            <BrainIcon />
          </button>
          <button
            type="button"
            className="btn-new-chat-header"
            onClick={newChat}
            aria-label="New chat"
            title="Start new chat"
          >
            <PlusIcon />
            <span>New chat</span>
          </button>
          <button
            type="button"
            className={`ghost header-workspace-btn ${workspaceOpen ? "active" : ""}`}
            onClick={() => setWorkspaceOpen((prev) => !prev)}
            aria-label="Toggle Workspace"
            title="Workspace Files"
          >
            <PanelRightIcon />
            {workspaceFileCount > 0 && <span className="workspace-indicator-dot" />}
          </button>
        </div>
      </header>

      {/* Desktop Floating Top-Right Workspace Toggle Button */}
      {!workspaceOpen && (
        <div className="top-right-bar">
          <button
            type="button"
            className="ghost top-right-workspace-btn"
            onClick={() => setWorkspaceOpen(true)}
            aria-label="Open Workspace Panel"
            title="Workspace Files"
          >
            <PanelRightIcon />
            {workspaceFileCount > 0 && <span className="workspace-indicator-dot" />}
          </button>
        </div>
      )}

      {/* Desktop Collapsed Floating Rail (Left) */}
      {!sideOpen && (
        <div className="side-rail">
          <div className="side-rail-top">
            <button
              type="button"
              className="ghost"
              onClick={() => setSideOpen(true)}
              aria-label="Expand sidebar"
              title="Expand sidebar"
            >
              <SideCollapseIcon />
            </button>
            <button
              type="button"
              className="ghost side-rail-new"
              onClick={newChat}
              aria-label="New chat"
              title="New conversation"
            >
              <PlusIcon />
            </button>
            <button
              type="button"
              className={`ghost side-rail-new ${currentView === "memory-dashboard" ? "active" : ""}`}
              onClick={() => setCurrentView((v) => (v === "memory-dashboard" ? "chat" : "memory-dashboard"))}
              aria-label="Memory Dashboard"
              title="Mnemosyne Memory Dashboard"
              style={{ color: "var(--accent)" }}
            >
              <BrainIcon />
            </button>
          </div>
          <div className="side-rail-bottom">
            <button
              type="button"
              className="theme-quick-btn"
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

      {/* Left Sidebar (Conversations) */}
      <aside className={"left-sidebar" + (sideOpen ? " open" : "")}>
        <div className="side-top">
          <span className="wordmark">edge agent</span>
          <button
            type="button"
            className="ghost side-collapse"
            onClick={() => setSideOpen(false)}
            aria-label="Collapse sidebar"
            title="Collapse sidebar"
          >
            <SideCollapseIcon />
          </button>
        </div>
        <nav className="side-list">
          <button type="button" className="side-item new-chat" onClick={newChat}>
            <PlusIcon />
            New chat
          </button>
          <button
            type="button"
            className={`side-item memory-nav-item ${currentView === "memory-dashboard" ? "active" : ""}`}
            onClick={() => {
              setCurrentView("memory-dashboard");
              closeSideOnMobile();
            }}
            style={{ color: "var(--accent)", fontWeight: 600, display: "flex", alignItems: "center", gap: 8 }}
          >
            <BrainIcon />
            <span>🧠 Memory Dashboard</span>
          </button>
          {convos.length === 0 && <p className="side-empty">No conversations yet</p>}
          {convos.map((c) => (
            <div
              key={c.id}
              className={"side-item" + (c.id === active && currentView === "chat" ? " active" : "")}
              onClick={() => {
                setActive(c.id);
                setCurrentView("chat");
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

      {/* Mobile Backdrops */}
      {sideOpen && <div className="tap-away" onClick={() => setSideOpen(false)} />}
      {workspaceOpen && <div className="tap-away" onClick={() => setWorkspaceOpen(false)} />}

      {/* Center Main Viewport: Chat or Mnemosyne Full Dashboard */}
      {currentView === "memory-dashboard" ? (
        <MnemosyneFullDashboard
          onClose={() => setCurrentView("chat")}
          config={mnemosyneConfig}
          onSaveConfig={handleSaveMnemosyneConfig}
        />
      ) : (
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
            onUpdateProviderReasoningEffort={handleUpdateProviderReasoningEffort}
            mcpServers={mcpServers}
            mnemosyneConfig={mnemosyneConfig}
            customSystemPrompt={customSystemPrompt}
            systemPromptMode={systemPromptMode}
            onTriggerWorkspaceRefresh={() => setWorkspaceRefreshToken((n) => n + 1)}
          />
        </Suspense>
      )}

      {/* Right Workspace Drawer */}
      <WorkspacePanel
        convoId={active}
        isOpen={workspaceOpen}
        onClose={() => setWorkspaceOpen(false)}
        onFileCountUpdate={(count) => setWorkspaceFileCount(count)}
        refreshToken={workspaceRefreshToken}
      />

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        onSetTheme={setTheme}
        providers={providers}
        activeProviderId={activeProviderId}
        onSaveProviders={handleSaveProviders}
        onSelectActiveProvider={handleSelectActiveProvider}
        mcpServers={mcpServers}
        onSaveMcpServers={handleSaveMcpServers}
        mnemosyneConfig={mnemosyneConfig}
        onSaveMnemosyneConfig={handleSaveMnemosyneConfig}
        onOpenFullDashboard={() => {
          setSettingsOpen(false);
          setCurrentView("memory-dashboard");
        }}
        customSystemPrompt={customSystemPrompt}
        systemPromptMode={systemPromptMode}
        onSaveSystemPrompt={handleSaveSystemPrompt}
        convosCount={convos.length}
      />
    </div>
  );
}
