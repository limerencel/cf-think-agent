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

export type { McpServerConfig, McpToolDef };

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
        cf_access_client_id TEXT,
        cf_access_client_secret TEXT,
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
}
