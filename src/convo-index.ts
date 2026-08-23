/**
 * Cloud conversation-list registry & settings store.
 *
 * Persists conversations and AI Service Providers (endpoints, API keys, models,
 * parameters) in Cloudflare Durable Object SQLite, making Cloudflare the true
 * cloud source of truth across all devices.
 */
import { Agent, callable } from "agents";
import type { McpServerConfig, McpToolDef } from "./mcp-types";

export type ConvoMeta = { id: string; title: string; ts: number };

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

export type { McpServerConfig, McpToolDef, HindsightConfig } from "./mcp-types";
import type { HindsightConfig } from "./mcp-types";
export type {
  MnemosyneConfig,
  WorkingMemoryItem,
  EpisodicMemoryItem,
  TripleItem,
  MnemosyneStats,
  MnemosyneRecallResult,
} from "./mnemosyne";
import {
  cosineSimilarity,
  tokenizeText,
  tokenOverlapScore,
  generateEmbedding,
  type MnemosyneConfig,
  type WorkingMemoryItem,
  type EpisodicMemoryItem,
  type TripleItem,
  type MnemosyneStats,
  type MnemosyneRecallResult,
} from "./mnemosyne";

const MAX_CONVOS = 30;

export class ConvoIndex extends Agent<Env> {
  private async ensureTables() {
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS convos (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        ts INTEGER NOT NULL
      )`
    );
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS providers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        api_key TEXT NOT NULL,
        selected_model TEXT NOT NULL,
        cached_models TEXT NOT NULL,
        is_default INTEGER NOT NULL DEFAULT 0,
        use_response_api INTEGER NOT NULL DEFAULT 0,
        temperature REAL,
        max_tokens INTEGER,
        top_p REAL,
        reasoning_effort TEXT,
        updated_at INTEGER NOT NULL
      )`
    );
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS mcp_servers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        auth_type TEXT NOT NULL,
        bearer_token TEXT,
        oauth_client_id TEXT,
        oauth_client_secret TEXT,
        oauth_auth_url TEXT,
        oauth_token_url TEXT,
        oauth_scopes TEXT,
        oauth_tokens TEXT,
        enabled INTEGER NOT NULL DEFAULT 1,
        cached_tools TEXT,
        is_preset INTEGER NOT NULL DEFAULT 0,
        updated_at INTEGER NOT NULL
      )`
    );
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )`
    );
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS mcp_oauth_sessions (
        state TEXT PRIMARY KEY,
        server_id TEXT NOT NULL,
        server_name TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        token_endpoint TEXT NOT NULL,
        client_id TEXT NOT NULL,
        client_secret TEXT,
        redirect_uri TEXT NOT NULL,
        code_verifier TEXT NOT NULL,
        created_at INTEGER NOT NULL
      )`
    );
    // Mnemosyne Zero-Cloud AI Memory Tables (BEAM Architecture)
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS mnemosyne_working_memory (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        content TEXT NOT NULL,
        importance REAL NOT NULL DEFAULT 0.5,
        created_at INTEGER NOT NULL,
        expires_at INTEGER
      )`
    );
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS mnemosyne_episodic_memory (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        importance REAL NOT NULL DEFAULT 0.5,
        scope TEXT NOT NULL DEFAULT 'global',
        source TEXT NOT NULL DEFAULT 'direct',
        embedding TEXT,
        access_count INTEGER NOT NULL DEFAULT 0,
        last_accessed INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        valid_until INTEGER
      )`
    );
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS mnemosyne_triples (
        id TEXT PRIMARY KEY,
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        object TEXT NOT NULL,
        valid_from TEXT,
        valid_until TEXT,
        confidence REAL NOT NULL DEFAULT 1.0,
        source TEXT,
        created_at INTEGER NOT NULL
      )`
    );
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS mnemosyne_banks (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at INTEGER NOT NULL
      )`
    );
    // Migration: ensure use_response_api and reasoning_effort columns exist in SQLite
    try {
      await this.ctx.storage.sql.exec(
        "ALTER TABLE providers ADD COLUMN use_response_api INTEGER NOT NULL DEFAULT 0"
      );
    } catch {
      /* ignore if column already exists */
    }
    try {
      await this.ctx.storage.sql.exec(
        "ALTER TABLE providers ADD COLUMN reasoning_effort TEXT"
      );
    } catch {
      /* ignore if column already exists */
    }
    try {
      await this.ctx.storage.sql.exec(
        "ALTER TABLE mcp_servers ADD COLUMN cf_access_client_id TEXT"
      );
    } catch {
      /* ignore if column already exists */
    }
    try {
      await this.ctx.storage.sql.exec(
        "ALTER TABLE mcp_servers ADD COLUMN cf_access_client_secret TEXT"
      );
    } catch {
      /* ignore if column already exists */
    }
    // Clean up deprecated hardcoded default provider if present
    try {
      await this.ctx.storage.sql.exec("DELETE FROM providers WHERE id = 'cf-default'");
      await this.ctx.storage.sql.exec("DELETE FROM mcp_servers WHERE id = 'gbrain-default'");
    } catch {
      /* ignore */
    }
  }

  /* ---------------- conversation list ---------------- */

  @callable()
  async list(): Promise<ConvoMeta[]> {
    await this.ensureTables();
    const rows = await this.ctx.storage.sql
      .exec("SELECT id, title, ts FROM convos ORDER BY ts DESC LIMIT ?", MAX_CONVOS)
      .toArray();
    return rows.map((r) => ({ id: r.id as string, title: r.title as string, ts: r.ts as number }));
  }

  @callable()
  async touch(id: string, title?: string): Promise<ConvoMeta[]> {
    await this.ensureTables();
    const existing = await this.ctx.storage.sql
      .exec("SELECT title FROM convos WHERE id = ?", id)
      .toArray();
    const finalTitle = title ?? (existing[0]?.title as string | undefined) ?? "New chat";
    await this.ctx.storage.sql.exec(
      `INSERT INTO convos (id, title, ts) VALUES (?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET title = excluded.title, ts = excluded.ts`,
      id,
      finalTitle,
      Date.now()
    );
    // Enforce cap
    await this.ctx.storage.sql.exec(
      `DELETE FROM convos WHERE id NOT IN (
         SELECT id FROM convos ORDER BY ts DESC LIMIT ?
       )`,
      MAX_CONVOS
    );
    return this.list();
  }

  @callable()
  async remove(id: string): Promise<ConvoMeta[]> {
    await this.ensureTables();
    await this.ctx.storage.sql.exec("DELETE FROM convos WHERE id = ?", id);
    return this.list();
  }

  /* ---------------- providers & cloud config ---------------- */

  @callable()
  async listProviders(): Promise<ProviderConfig[]> {
    await this.ensureTables();
    const rows = await this.ctx.storage.sql
      .exec("SELECT * FROM providers ORDER BY updated_at ASC")
      .toArray();

    return rows.map((r) => {
      let cached: string[] = [];
      try {
        cached = JSON.parse(r.cached_models as string) as string[];
      } catch {
        cached = [r.selected_model as string];
      }
      return {
        id: r.id as string,
        name: r.name as string,
        endpoint: r.endpoint as string,
        apiKey: r.api_key as string,
        selectedModel: r.selected_model as string,
        cachedModels: cached,
        isDefault: Number(r.is_default) === 1,
        useResponseApi: Number(r.use_response_api) === 1,
        temperature: r.temperature !== null ? Number(r.temperature) : undefined,
        maxTokens: r.max_tokens !== null ? Number(r.max_tokens) : undefined,
        topP: r.top_p !== null ? Number(r.top_p) : undefined,
        reasoningEffort: (r.reasoning_effort as "low" | "medium" | "high" | "none") || undefined,
      };
    });
  }

  @callable()
  async getProvider(id: string): Promise<ProviderConfig | null> {
    await this.ensureTables();
    const rows = await this.ctx.storage.sql
      .exec("SELECT * FROM providers WHERE id = ?", id)
      .toArray();
    if (!rows.length) return null;
    const r = rows[0];
    let cached: string[] = [];
    try {
      cached = JSON.parse(r.cached_models as string) as string[];
    } catch {
      cached = [r.selected_model as string];
    }
    return {
      id: r.id as string,
      name: r.name as string,
      endpoint: r.endpoint as string,
      apiKey: r.api_key as string,
      selectedModel: r.selected_model as string,
      cachedModels: cached,
      isDefault: Number(r.is_default) === 1,
      useResponseApi: Number(r.use_response_api) === 1,
      temperature: r.temperature !== null ? Number(r.temperature) : undefined,
      maxTokens: r.max_tokens !== null ? Number(r.max_tokens) : undefined,
      topP: r.top_p !== null ? Number(r.top_p) : undefined,
      reasoningEffort: (r.reasoning_effort as "low" | "medium" | "high" | "none") || undefined,
    };
  }

  @callable()
  async saveProvider(provider: ProviderConfig): Promise<ProviderConfig[]> {
    await this.ensureTables();
    if (provider.isDefault) {
      await this.ctx.storage.sql.exec("UPDATE providers SET is_default = 0");
    }
    const cachedJson = JSON.stringify(provider.cachedModels || [provider.selectedModel]);
    await this.ctx.storage.sql.exec(
      `INSERT INTO providers (id, name, endpoint, api_key, selected_model, cached_models, is_default, use_response_api, temperature, max_tokens, top_p, reasoning_effort, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         name = excluded.name,
         endpoint = excluded.endpoint,
         api_key = excluded.api_key,
         selected_model = excluded.selected_model,
         cached_models = excluded.cached_models,
         is_default = excluded.is_default,
         use_response_api = excluded.use_response_api,
         temperature = excluded.temperature,
         max_tokens = excluded.max_tokens,
         top_p = excluded.top_p,
         reasoning_effort = excluded.reasoning_effort,
         updated_at = excluded.updated_at`,
      provider.id,
      provider.name,
      provider.endpoint,
      provider.apiKey,
      provider.selectedModel,
      cachedJson,
      provider.isDefault ? 1 : 0,
      provider.useResponseApi ? 1 : 0,
      provider.temperature ?? null,
      provider.maxTokens ?? null,
      provider.topP ?? null,
      provider.reasoningEffort ?? null,
      Date.now()
    );
    return this.listProviders();
  }

  @callable()
  async saveAllProviders(providers: ProviderConfig[]): Promise<ProviderConfig[]> {
    await this.ensureTables();
    for (const p of providers) {
      await this.saveProvider(p);
    }
    return this.listProviders();
  }

  @callable()
  async removeProvider(id: string): Promise<ProviderConfig[]> {
    await this.ensureTables();
    await this.ctx.storage.sql.exec("DELETE FROM providers WHERE id = ?", id);
    return this.listProviders();
  }

  @callable()
  async getSetting(key: string): Promise<string | null> {
    await this.ensureTables();
    const rows = await this.ctx.storage.sql
      .exec("SELECT value FROM app_settings WHERE key = ?", key)
      .toArray();
    return rows.length ? (rows[0].value as string) : null;
  }

  @callable()
  async setSetting(key: string, value: string): Promise<void> {
    await this.ensureTables();
    await this.ctx.storage.sql.exec(
      `INSERT INTO app_settings (key, value) VALUES (?, ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
      key,
      value
    );
  }

  /* ---------------- MCP Servers Cloud Storage ---------------- */

  @callable()
  async listMcpServers(): Promise<McpServerConfig[]> {
    await this.ensureTables();
    const rows = await this.ctx.storage.sql
      .exec("SELECT * FROM mcp_servers ORDER BY updated_at ASC")
      .toArray();

    const list: McpServerConfig[] = rows.map((r) => {
      let cachedTools: McpToolDef[] = [];
      try {
        cachedTools = JSON.parse(r.cached_tools as string) as McpToolDef[];
      } catch {
        cachedTools = [];
      }
      let oauthTokens: any = undefined;
      try {
        if (r.oauth_tokens) oauthTokens = JSON.parse(r.oauth_tokens as string);
      } catch {
        /* ignore */
      }
      let oauthScopes: string[] | undefined = undefined;
      try {
        if (r.oauth_scopes) oauthScopes = JSON.parse(r.oauth_scopes as string);
      } catch {
        /* ignore */
      }

      return {
        id: r.id as string,
        name: r.name as string,
        endpoint: r.endpoint as string,
        authType: (r.auth_type as any) || "none",
        bearerToken: (r.bearer_token as string) || "",
        cfAccessClientId: (r.cf_access_client_id as string) || undefined,
        cfAccessClientSecret: (r.cf_access_client_secret as string) || undefined,
        oauthClientId: (r.oauth_client_id as string) || undefined,
        oauthClientSecret: (r.oauth_client_secret as string) || undefined,
        oauthAuthUrl: (r.oauth_auth_url as string) || undefined,
        oauthTokenUrl: (r.oauth_token_url as string) || undefined,
        oauthScopes,
        oauthTokens,
        enabled: Number(r.enabled) === 1,
        cachedTools,
        isPreset: false,
        updatedAt: Number(r.updated_at) || Date.now(),
      };
    });

    return list.filter((s) => s.id !== "gbrain-default");
  }

  @callable()
  async getMcpServer(id: string): Promise<McpServerConfig | null> {
    if (id === "gbrain-default") return null;
    await this.ensureTables();
    const rows = await this.ctx.storage.sql
      .exec("SELECT * FROM mcp_servers WHERE id = ?", id)
      .toArray();
    if (!rows.length) return null;
    const r = rows[0];
    let cachedTools: McpToolDef[] = [];
    try {
      cachedTools = JSON.parse(r.cached_tools as string) as McpToolDef[];
    } catch {
      cachedTools = [];
    }
    let oauthTokens: any = undefined;
    try {
      if (r.oauth_tokens) oauthTokens = JSON.parse(r.oauth_tokens as string);
    } catch {
      /* ignore */
    }
    let oauthScopes: string[] | undefined = undefined;
    try {
      if (r.oauth_scopes) oauthScopes = JSON.parse(r.oauth_scopes as string);
    } catch {
      /* ignore */
    }

    return {
      id: r.id as string,
      name: r.name as string,
      endpoint: r.endpoint as string,
      authType: (r.auth_type as any) || "none",
      bearerToken: (r.bearer_token as string) || "",
      cfAccessClientId: (r.cf_access_client_id as string) || undefined,
      cfAccessClientSecret: (r.cf_access_client_secret as string) || undefined,
      oauthClientId: (r.oauth_client_id as string) || undefined,
      oauthClientSecret: (r.oauth_client_secret as string) || undefined,
      oauthAuthUrl: (r.oauth_auth_url as string) || undefined,
      oauthTokenUrl: (r.oauth_token_url as string) || undefined,
      oauthScopes,
      oauthTokens,
      enabled: Number(r.enabled) === 1,
      cachedTools,
      isPreset: Number(r.is_preset) === 1,
      updatedAt: Number(r.updated_at) || Date.now(),
    };
  }

  @callable()
  async saveMcpServer(server: McpServerConfig): Promise<McpServerConfig[]> {
    await this.ensureTables();
    const cachedToolsJson = JSON.stringify(server.cachedTools || []);
    const oauthTokensJson = server.oauthTokens ? JSON.stringify(server.oauthTokens) : null;
    const oauthScopesJson = server.oauthScopes ? JSON.stringify(server.oauthScopes) : null;

    await this.ctx.storage.sql.exec(
      `INSERT INTO mcp_servers (id, name, endpoint, auth_type, bearer_token, cf_access_client_id, cf_access_client_secret, oauth_client_id, oauth_client_secret, oauth_auth_url, oauth_token_url, oauth_scopes, oauth_tokens, enabled, cached_tools, is_preset, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         name = excluded.name,
         endpoint = excluded.endpoint,
         auth_type = excluded.auth_type,
         bearer_token = excluded.bearer_token,
         cf_access_client_id = excluded.cf_access_client_id,
         cf_access_client_secret = excluded.cf_access_client_secret,
         oauth_client_id = excluded.oauth_client_id,
         oauth_client_secret = excluded.oauth_client_secret,
         oauth_auth_url = excluded.oauth_auth_url,
         oauth_token_url = excluded.oauth_token_url,
         oauth_scopes = excluded.oauth_scopes,
         oauth_tokens = excluded.oauth_tokens,
         enabled = excluded.enabled,
         cached_tools = excluded.cached_tools,
         is_preset = excluded.is_preset,
         updated_at = excluded.updated_at`,
      server.id,
      server.name,
      server.endpoint,
      server.authType || "none",
      server.bearerToken ?? null,
      server.cfAccessClientId ?? null,
      server.cfAccessClientSecret ?? null,
      server.oauthClientId ?? null,
      server.oauthClientSecret ?? null,
      server.oauthAuthUrl ?? null,
      server.oauthTokenUrl ?? null,
      oauthScopesJson,
      oauthTokensJson,
      server.enabled ? 1 : 0,
      cachedToolsJson,
      server.isPreset ? 1 : 0,
      Date.now()
    );
    return this.listMcpServers();
  }

  @callable()
  async saveAllMcpServers(servers: McpServerConfig[]): Promise<McpServerConfig[]> {
    await this.ensureTables();
    for (const s of servers) {
      await this.saveMcpServer(s);
    }
    return this.listMcpServers();
  }

  @callable()
  async deleteMcpServer(id: string): Promise<McpServerConfig[]> {
    await this.ensureTables();
    await this.ctx.storage.sql.exec("DELETE FROM mcp_servers WHERE id = ?", id);
    return this.listMcpServers();
  }

  @callable()
  async saveOAuthSession(session: {
    state: string;
    serverId: string;
    serverName: string;
    endpoint: string;
    tokenEndpoint: string;
    clientId: string;
    clientSecret?: string;
    redirectUri: string;
    codeVerifier: string;
  }): Promise<void> {
    await this.ensureTables();
    await this.ctx.storage.sql.exec(
      `INSERT OR REPLACE INTO mcp_oauth_sessions
      (state, server_id, server_name, endpoint, token_endpoint, client_id, client_secret, redirect_uri, code_verifier, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      session.state,
      session.serverId,
      session.serverName,
      session.endpoint,
      session.tokenEndpoint,
      session.clientId,
      session.clientSecret ?? null,
      session.redirectUri,
      session.codeVerifier,
      Date.now()
    );
  }

  @callable()
  async consumeOAuthSession(state: string): Promise<{
    state: string;
    serverId: string;
    serverName: string;
    endpoint: string;
    tokenEndpoint: string;
    clientId: string;
    clientSecret?: string;
    redirectUri: string;
    codeVerifier: string;
    createdAt: number;
  } | null> {
    await this.ensureTables();
    const rows = await this.ctx.storage.sql
      .exec("SELECT * FROM mcp_oauth_sessions WHERE state = ?", state)
      .toArray();
    if (rows.length === 0) return null;
    const r = rows[0];
    await this.ctx.storage.sql.exec("DELETE FROM mcp_oauth_sessions WHERE state = ?", state);
    return {
      state: r.state as string,
      serverId: r.server_id as string,
      serverName: r.server_name as string,
      endpoint: r.endpoint as string,
      tokenEndpoint: r.token_endpoint as string,
      clientId: r.client_id as string,
      clientSecret: (r.client_secret as string) || undefined,
      redirectUri: r.redirect_uri as string,
      codeVerifier: r.code_verifier as string,
      createdAt: Number(r.created_at),
    };
  }

  /* ---------------- Hindsight Memory configuration ---------------- */

  @callable()
  async getHindsightConfig(): Promise<HindsightConfig> {
    await this.ensureTables();
    const raw = await this.getSetting("hindsight_config");
    if (!raw) {
      return {
        enabled: false,
        endpoint: "",
        authType: "none",
        autoRecall: true,
        autoRetain: true,
        recallTopK: 5,
      };
    }
    try {
      const parsed = JSON.parse(raw);
      return {
        enabled: !!parsed.enabled,
        endpoint: parsed.endpoint || "",
        authType: parsed.authType || "none",
        bearerToken: parsed.bearerToken,
        cfAccessClientId: parsed.cfAccessClientId,
        cfAccessClientSecret: parsed.cfAccessClientSecret,
        oauthClientId: parsed.oauthClientId,
        oauthClientSecret: parsed.oauthClientSecret,
        oauthTokens: parsed.oauthTokens,
        bankId: parsed.bankId || undefined,
        autoRecall: parsed.autoRecall ?? true,
        autoRetain: parsed.autoRetain ?? true,
        recallTopK: parsed.recallTopK || 5,
        cachedTools: parsed.cachedTools,
        updatedAt: parsed.updatedAt,
      };
    } catch {
      return {
        enabled: false,
        endpoint: "",
        authType: "none",
        autoRecall: true,
        autoRetain: true,
        recallTopK: 5,
      };
    }
  }

  @callable()
  async saveHindsightConfig(config: HindsightConfig): Promise<HindsightConfig> {
    await this.ensureTables();
    const updated: HindsightConfig = {
      ...config,
      updatedAt: Date.now(),
    };
    await this.setSetting("hindsight_config", JSON.stringify(updated));
    return updated;
  }

  /* ---------------- Mnemosyne Zero-Cloud Memory Engine (BEAM) ---------------- */

  @callable()
  async mnemosyneGetConfig(): Promise<MnemosyneConfig> {
    await this.ensureTables();
    const raw = await this.getSetting("mnemosyne_config");
    if (!raw) {
      return {
        enabled: true, // Enabled by default for native Cloudflare memory
        autoRecall: true,
        autoRetain: true,
        recallTopK: 5,
        scope: "global",
      };
    }
    try {
      const parsed = JSON.parse(raw);
      return {
        enabled: parsed.enabled ?? true,
        autoRecall: parsed.autoRecall ?? true,
        autoRetain: parsed.autoRetain ?? true,
        recallTopK: parsed.recallTopK || 5,
        scope: parsed.scope || "global",
        updatedAt: parsed.updatedAt,
      };
    } catch {
      return {
        enabled: true,
        autoRecall: true,
        autoRetain: true,
        recallTopK: 5,
        scope: "global",
      };
    }
  }

  @callable()
  async mnemosyneSaveConfig(config: MnemosyneConfig): Promise<MnemosyneConfig> {
    await this.ensureTables();
    const updated: MnemosyneConfig = {
      ...config,
      updatedAt: Date.now(),
    };
    await this.setSetting("mnemosyne_config", JSON.stringify(updated));
    return updated;
  }

  @callable()
  async mnemosyneRemember(payload: {
    content: string;
    importance?: number;
    scope?: string;
    source?: string;
    sessionId?: string;
    isWorkingMemory?: boolean;
    ttlSeconds?: number;
    triples?: Array<{
      subject: string;
      predicate: string;
      object: string;
      validFrom?: string;
      validUntil?: string;
    }>;
  }): Promise<{ ok: boolean; id: string; isWorkingMemory: boolean }> {
    await this.ensureTables();
    const content = (payload.content || "").trim();
    if (!content) throw new Error("Memory content cannot be empty");

    const importance = Math.max(0.1, Math.min(1.0, payload.importance ?? 0.7));
    const scope = payload.scope || "global";
    const source = payload.source || "direct";
    const now = Date.now();
    const id = "mem_" + Math.random().toString(36).slice(2, 11) + "_" + now.toString(36);

    if (payload.isWorkingMemory) {
      const expiresAt = payload.ttlSeconds ? now + payload.ttlSeconds * 1000 : undefined;
      await this.ctx.storage.sql.exec(
        `INSERT OR REPLACE INTO mnemosyne_working_memory (id, session_id, content, importance, created_at, expires_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
        id,
        payload.sessionId || "global",
        content,
        importance,
        now,
        expiresAt ?? null
      );
    } else {
      const validUntil = payload.ttlSeconds ? now + payload.ttlSeconds * 1000 : undefined;
      const embedding = await generateEmbedding((this.env as any).AI, content);
      const embeddingJson = embedding ? JSON.stringify(embedding) : null;

      await this.ctx.storage.sql.exec(
        `INSERT OR REPLACE INTO mnemosyne_episodic_memory (
          id, content, importance, scope, source, embedding, access_count, last_accessed, created_at, valid_until
        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)`,
        id,
        content,
        importance,
        scope,
        source,
        embeddingJson,
        now,
        now,
        validUntil ?? null
      );
    }

    // Insert associated triples if provided
    if (payload.triples && Array.isArray(payload.triples)) {
      for (const t of payload.triples) {
        if (t.subject && t.predicate && t.object) {
          const tripleId = "trp_" + Math.random().toString(36).slice(2, 11);
          await this.ctx.storage.sql.exec(
            `INSERT INTO mnemosyne_triples (id, subject, predicate, object, valid_from, valid_until, confidence, source, created_at)
             VALUES (?, ?, ?, ?, ?, ?, 1.0, ?, ?)`,
            tripleId,
            t.subject.trim(),
            t.predicate.trim(),
            t.object.trim(),
            t.validFrom ?? null,
            t.validUntil ?? null,
            source,
            now
          );
        }
      }
    }

    return { ok: true, id, isWorkingMemory: !!payload.isWorkingMemory };
  }

  @callable()
  async mnemosyneRecall(payload: {
    query: string;
    topK?: number;
    scope?: string;
    sessionId?: string;
    includeTriples?: boolean;
    temporalWeight?: number;
  }): Promise<MnemosyneRecallResult> {
    await this.ensureTables();
    const query = (payload.query || "").trim();
    const topK = payload.topK || 5;
    const scope = payload.scope || "global";
    const sessionId = payload.sessionId;
    const includeTriples = payload.includeTriples !== false;
    const now = Date.now();

    if (!query) {
      return {
        memories: [],
        workingMemories: [],
        triples: [],
        formattedContext: "",
      };
    }

    // 1. Working Memory Recall (hot context, purge expired)
    try {
      await this.ctx.storage.sql.exec(
        "DELETE FROM mnemosyne_working_memory WHERE expires_at IS NOT NULL AND expires_at < ?",
        now
      );
    } catch {}

    let wmRows: any[] = [];
    if (sessionId) {
      wmRows = await this.ctx.storage.sql
        .exec(
          "SELECT * FROM mnemosyne_working_memory WHERE session_id = ? OR session_id = 'global' ORDER BY created_at DESC LIMIT 5",
          sessionId
        )
        .toArray();
    } else {
      wmRows = await this.ctx.storage.sql
        .exec("SELECT * FROM mnemosyne_working_memory ORDER BY created_at DESC LIMIT 5")
        .toArray();
    }

    const workingMemories: WorkingMemoryItem[] = wmRows.map((r: any) => ({
      id: r.id as string,
      sessionId: r.session_id as string,
      content: r.content as string,
      importance: Number(r.importance),
      createdAt: Number(r.created_at),
      expiresAt: r.expires_at ? Number(r.expires_at) : undefined,
    }));

    // 2. Episodic Memory Hybrid Recall
    const queryVec = await generateEmbedding((this.env as any).AI, query);

    const episodicRows = await this.ctx.storage.sql
      .exec(
        `SELECT * FROM mnemosyne_episodic_memory 
         WHERE (scope = ? OR scope = 'global') 
           AND (valid_until IS NULL OR valid_until > ?) 
         ORDER BY created_at DESC LIMIT 100`,
        scope,
        now
      )
      .toArray();

    const scoredEpisodic: EpisodicMemoryItem[] = episodicRows.map((r: any) => {
      const content = r.content as string;
      const importance = Number(r.importance) || 0.5;
      const lastAccessed = Number(r.last_accessed) || Number(r.created_at);
      let docVec: number[] | undefined = undefined;
      if (r.embedding) {
        try {
          docVec = JSON.parse(r.embedding);
        } catch {}
      }

      // Hybrid components
      const vecScore = queryVec && docVec ? cosineSimilarity(queryVec, docVec) : 0;
      const ftsScore = tokenOverlapScore(query, content);
      const recencyBoost = Math.exp(-(now - lastAccessed) / (1000 * 60 * 60 * 24 * 7)); // 7-day half-life

      let totalScore = 0;
      if (queryVec && docVec) {
        totalScore = 0.50 * vecScore + 0.30 * ftsScore + 0.15 * importance + 0.05 * recencyBoost;
      } else {
        totalScore = 0.65 * ftsScore + 0.25 * importance + 0.10 * recencyBoost;
      }

      return {
        id: r.id as string,
        content,
        importance,
        scope: r.scope as string,
        source: r.source as string,
        embedding: docVec,
        accessCount: Number(r.access_count) || 0,
        lastAccessed,
        createdAt: Number(r.created_at),
        validUntil: r.valid_until ? Number(r.valid_until) : undefined,
        score: totalScore,
      };
    });

    // Filter relevant memories (must match tokens or vector) and sort descending
    scoredEpisodic.sort((a, b) => (b.score || 0) - (a.score || 0));
    const topMemories = scoredEpisodic.filter((m) => (m.score || 0) > 0.08).slice(0, topK);

    // Update access metadata
    if (topMemories.length > 0) {
      for (const m of topMemories) {
        await this.ctx.storage.sql.exec(
          "UPDATE mnemosyne_episodic_memory SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
          now,
          m.id
        );
      }
    }

    // 3. Temporal TripleStore Recall
    let triples: TripleItem[] = [];
    if (includeTriples) {
      const qTokens = tokenizeText(query);
      if (qTokens.length > 0) {
        const allTriples = await this.ctx.storage.sql
          .exec("SELECT * FROM mnemosyne_triples ORDER BY created_at DESC LIMIT 50")
          .toArray();

        triples = allTriples
          .filter((t: any) => {
            const subj = (t.subject as string).toLowerCase();
            const pred = (t.predicate as string).toLowerCase();
            const obj = (t.object as string).toLowerCase();
            return qTokens.some(
              (token) => subj.includes(token) || pred.includes(token) || obj.includes(token)
            );
          })
          .slice(0, 6)
          .map((t: any) => ({
            id: t.id as string,
            subject: t.subject as string,
            predicate: t.predicate as string,
            object: t.object as string,
            validFrom: (t.valid_from as string) || undefined,
            validUntil: (t.valid_until as string) || undefined,
            confidence: Number(t.confidence) || 1.0,
            source: (t.source as string) || undefined,
            createdAt: Number(t.created_at),
          }));
      }
    }

    // 4. Format structured context for prompt injection
    const contextSections: string[] = [];

    if (workingMemories.length > 0) {
      contextSections.push(
        `[Working Memory (Recent Hot Context)]:\n` +
          workingMemories.map((wm) => `- ${wm.content}`).join("\n")
      );
    }

    if (topMemories.length > 0) {
      contextSections.push(
        `[Episodic Memory (Enduring Facts & User Preferences)]:\n` +
          topMemories
            .map((m) => `- ${m.content} (importance: ${(m.importance * 100).toFixed(0)}%)`)
            .join("\n")
      );
    }

    if (triples.length > 0) {
      contextSections.push(
        `[Knowledge Graph Triples]:\n` +
          triples
            .map(
              (t) =>
                `- (${t.subject}, ${t.predicate}, ${t.object})${
                  t.validUntil ? ` [Valid until: ${t.validUntil}]` : ""
                }`
            )
            .join("\n")
      );
    }

    const formattedContext = contextSections.join("\n\n");

    return {
      memories: topMemories,
      workingMemories,
      triples,
      formattedContext,
    };
  }

  @callable()
  async mnemosyneAddTriple(payload: {
    subject: string;
    predicate: string;
    object: string;
    validFrom?: string;
    validUntil?: string;
    confidence?: number;
    source?: string;
  }): Promise<{ ok: boolean; id: string }> {
    await this.ensureTables();
    if (!payload.subject || !payload.predicate || !payload.object) {
      throw new Error("Subject, predicate, and object are required");
    }
    const id = "trp_" + Math.random().toString(36).slice(2, 11);
    const now = Date.now();
    await this.ctx.storage.sql.exec(
      `INSERT INTO mnemosyne_triples (id, subject, predicate, object, valid_from, valid_until, confidence, source, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      id,
      payload.subject.trim(),
      payload.predicate.trim(),
      payload.object.trim(),
      payload.validFrom ?? null,
      payload.validUntil ?? null,
      payload.confidence ?? 1.0,
      payload.source ?? "direct",
      now
    );
    return { ok: true, id };
  }

  @callable()
  async mnemosyneQueryTriples(payload: {
    subject?: string;
    predicate?: string;
    object?: string;
    asOf?: string;
    limit?: number;
  }): Promise<TripleItem[]> {
    await this.ensureTables();
    const rows = await this.ctx.storage.sql
      .exec("SELECT * FROM mnemosyne_triples ORDER BY created_at DESC LIMIT ?", payload.limit || 50)
      .toArray();

    return rows
      .filter((r: any) => {
        if (payload.subject && !(r.subject as string).toLowerCase().includes(payload.subject.toLowerCase())) {
          return false;
        }
        if (payload.predicate && !(r.predicate as string).toLowerCase().includes(payload.predicate.toLowerCase())) {
          return false;
        }
        if (payload.object && !(r.object as string).toLowerCase().includes(payload.object.toLowerCase())) {
          return false;
        }
        return true;
      })
      .map((r: any) => ({
        id: r.id as string,
        subject: r.subject as string,
        predicate: r.predicate as string,
        object: r.object as string,
        validFrom: (r.valid_from as string) || undefined,
        validUntil: (r.valid_until as string) || undefined,
        confidence: Number(r.confidence) || 1.0,
        source: (r.source as string) || undefined,
        createdAt: Number(r.created_at),
      }));
  }

  @callable()
  async mnemosyneDeleteMemory(id: string): Promise<{ ok: boolean }> {
    await this.ensureTables();
    await this.ctx.storage.sql.exec("DELETE FROM mnemosyne_episodic_memory WHERE id = ?", id);
    await this.ctx.storage.sql.exec("DELETE FROM mnemosyne_working_memory WHERE id = ?", id);
    return { ok: true };
  }

  @callable()
  async mnemosyneDeleteTriple(id: string): Promise<{ ok: boolean }> {
    await this.ensureTables();
    await this.ctx.storage.sql.exec("DELETE FROM mnemosyne_triples WHERE id = ?", id);
    return { ok: true };
  }

  @callable()
  async mnemosyneClearAll(target?: "all" | "working" | "episodic" | "triples"): Promise<{ ok: boolean }> {
    await this.ensureTables();
    const mode = target || "all";
    if (mode === "all" || mode === "working") {
      await this.ctx.storage.sql.exec("DELETE FROM mnemosyne_working_memory");
    }
    if (mode === "all" || mode === "episodic") {
      await this.ctx.storage.sql.exec("DELETE FROM mnemosyne_episodic_memory");
    }
    if (mode === "all" || mode === "triples") {
      await this.ctx.storage.sql.exec("DELETE FROM mnemosyne_triples");
    }
    return { ok: true };
  }

  @callable()
  async mnemosyneGetStats(): Promise<MnemosyneStats> {
    await this.ensureTables();
    const epCountRows = await this.ctx.storage.sql
      .exec("SELECT COUNT(*) as count FROM mnemosyne_episodic_memory")
      .toArray();
    const wmCountRows = await this.ctx.storage.sql
      .exec("SELECT COUNT(*) as count FROM mnemosyne_working_memory")
      .toArray();
    const trCountRows = await this.ctx.storage.sql
      .exec("SELECT COUNT(*) as count FROM mnemosyne_triples")
      .toArray();
    const cfg = await this.mnemosyneGetConfig();

    return {
      totalEpisodic: Number(epCountRows[0]?.count || 0),
      totalWorking: Number(wmCountRows[0]?.count || 0),
      totalTriples: Number(trCountRows[0]?.count || 0),
      totalBanks: 1,
      enabled: cfg.enabled,
      updatedAt: cfg.updatedAt,
    };
  }

  @callable()
  async mnemosyneListMemories(params?: { scope?: string; limit?: number; offset?: number }): Promise<EpisodicMemoryItem[]> {
    await this.ensureTables();
    const limit = params?.limit || 50;
    const offset = params?.offset || 0;
    const rows = await this.ctx.storage.sql
      .exec(
        "SELECT * FROM mnemosyne_episodic_memory ORDER BY created_at DESC LIMIT ? OFFSET ?",
        limit,
        offset
      )
      .toArray();

    return rows.map((r: any) => ({
      id: r.id as string,
      content: r.content as string,
      importance: Number(r.importance),
      scope: r.scope as string,
      source: r.source as string,
      accessCount: Number(r.access_count) || 0,
      lastAccessed: Number(r.last_accessed) || Number(r.created_at),
      createdAt: Number(r.created_at),
      validUntil: r.valid_until ? Number(r.valid_until) : undefined,
    }));
  }

  @callable()
  async mnemosyneListTriples(params?: { limit?: number; offset?: number }): Promise<TripleItem[]> {
    await this.ensureTables();
    const limit = params?.limit || 50;
    const offset = params?.offset || 0;
    const rows = await this.ctx.storage.sql
      .exec("SELECT * FROM mnemosyne_triples ORDER BY created_at DESC LIMIT ? OFFSET ?", limit, offset)
      .toArray();

    return rows.map((r: any) => ({
      id: r.id as string,
      subject: r.subject as string,
      predicate: r.predicate as string,
      object: r.object as string,
      validFrom: (r.valid_from as string) || undefined,
      validUntil: (r.valid_until as string) || undefined,
      confidence: Number(r.confidence) || 1.0,
      source: (r.source as string) || undefined,
      createdAt: Number(r.created_at),
    }));
  }

  @callable()
  async mnemosyneConsolidate(payload?: { sessionId?: string; summary?: string }): Promise<{ ok: boolean; consolidatedCount: number }> {
    await this.ensureTables();
    const sessionId = payload?.sessionId;
    let wmRows: any[] = [];
    if (sessionId) {
      wmRows = await this.ctx.storage.sql
        .exec("SELECT * FROM mnemosyne_working_memory WHERE session_id = ?", sessionId)
        .toArray();
    } else {
      wmRows = await this.ctx.storage.sql
        .exec("SELECT * FROM mnemosyne_working_memory")
        .toArray();
    }

    if (wmRows.length === 0) {
      return { ok: true, consolidatedCount: 0 };
    }

    // Merge contents into an episodic memory item
    const mergedContent =
      payload?.summary ||
      wmRows
        .map((r: any) => r.content as string)
        .filter(Boolean)
        .join("; ");

    if (mergedContent) {
      await this.mnemosyneRemember({
        content: `[Consolidated]: ${mergedContent}`,
        importance: 0.8,
        source: "consolidation",
        isWorkingMemory: false,
      });
    }

    // Clear consolidated working memory
    if (sessionId) {
      await this.ctx.storage.sql.exec("DELETE FROM mnemosyne_working_memory WHERE session_id = ?", sessionId);
    } else {
      await this.ctx.storage.sql.exec("DELETE FROM mnemosyne_working_memory");
    }

    return { ok: true, consolidatedCount: wmRows.length };
  }
}
