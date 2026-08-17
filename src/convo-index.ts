/**
 * Cloud conversation-list registry & settings store.
 *
 * Persists conversations and AI Service Providers (endpoints, API keys, models,
 * parameters) in Cloudflare Durable Object SQLite, making Cloudflare the true
 * cloud source of truth across all devices.
 */
import { Agent, callable } from "agents";

export type ConvoMeta = { id: string; title: string; ts: number };

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

const MAX_CONVOS = 30;

export class ConvoIndex extends Agent<Env> {
  private async ensureTables() {
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS convos (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        ts INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS providers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        api_key TEXT NOT NULL,
        selected_model TEXT NOT NULL,
        cached_models TEXT NOT NULL,
        is_default INTEGER NOT NULL DEFAULT 0,
        temperature REAL,
        max_tokens INTEGER,
        top_p REAL,
        updated_at INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );`
    );
  }

  private getDefaultPreset(): ProviderConfig {
    const endpoint = this.env.AIG_BASE_URL || "";
    const modelId = this.env.MODEL_ID || "deepseek-v4-flash";
    return {
      id: "cf-default",
      name: "Cloudflare AI Gateway (Env Default)",
      endpoint,
      apiKey: "•••••••• (Cloudflare Secret: OPENCODE_GO_API_KEY)",
      selectedModel: modelId,
      cachedModels: [modelId],
      isDefault: true,
    };
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

    const defaultPreset = this.getDefaultPreset();

    const list: ProviderConfig[] = rows.map((r) => {
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
        temperature: r.temperature !== null ? Number(r.temperature) : undefined,
        maxTokens: r.max_tokens !== null ? Number(r.max_tokens) : undefined,
        topP: r.top_p !== null ? Number(r.top_p) : undefined,
      };
    });

    if (!list.some((p) => p.id === "cf-default")) {
      return [defaultPreset, ...list];
    }

    return list.map((p) =>
      p.id === "cf-default"
        ? {
            ...defaultPreset,
            ...p,
            endpoint: defaultPreset.endpoint || p.endpoint,
            apiKey: defaultPreset.apiKey,
            selectedModel: p.selectedModel || defaultPreset.selectedModel,
          }
        : p
    );
  }

  @callable()
  async getProvider(id: string): Promise<ProviderConfig | null> {
    if (id === "cf-default") return this.getDefaultPreset();
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
      temperature: r.temperature !== null ? Number(r.temperature) : undefined,
      maxTokens: r.max_tokens !== null ? Number(r.max_tokens) : undefined,
      topP: r.top_p !== null ? Number(r.top_p) : undefined,
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
      `INSERT INTO providers (id, name, endpoint, api_key, selected_model, cached_models, is_default, temperature, max_tokens, top_p, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         name = excluded.name,
         endpoint = excluded.endpoint,
         api_key = excluded.api_key,
         selected_model = excluded.selected_model,
         cached_models = excluded.cached_models,
         is_default = excluded.is_default,
         temperature = excluded.temperature,
         max_tokens = excluded.max_tokens,
         top_p = excluded.top_p,
         updated_at = excluded.updated_at`,
      provider.id,
      provider.name,
      provider.endpoint,
      provider.apiKey,
      provider.selectedModel,
      cachedJson,
      provider.isDefault ? 1 : 0,
      provider.temperature ?? null,
      provider.maxTokens ?? null,
      provider.topP ?? null,
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
    if (id === "cf-default") return this.listProviders();
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
}
