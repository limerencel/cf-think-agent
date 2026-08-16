/**
 * Cloud conversation-list registry.
 *
 * Message bodies already persist per-conversation in each Assistant DO's
 * SQLite. What was missing: the LIST of conversations (id + title) lived only
 * in browser localStorage, so embedded browsers that disable storage looked
 * like they lost all history. This DO is the cloud source of truth for that
 * list. localStorage (client.tsx) remains only as an offline cache.
 */
import { Agent, callable } from "agents";

export type ConvoMeta = { id: string; title: string; ts: number };

const MAX_CONVOS = 30;

export class ConvoIndex extends Agent<Env> {
  private async ensureTable() {
    await this.ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS convos (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        ts INTEGER NOT NULL
      )`
    );
  }

  @callable()
  async list(): Promise<ConvoMeta[]> {
    await this.ensureTable();
    const rows = await this.ctx.storage.sql
      .exec("SELECT id, title, ts FROM convos ORDER BY ts DESC LIMIT ?", MAX_CONVOS)
      .toArray();
    return rows.map((r) => ({ id: r.id as string, title: r.title as string, ts: r.ts as number }));
  }

  @callable()
  async touch(id: string, title?: string): Promise<ConvoMeta[]> {
    await this.ensureTable();
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
    await this.ensureTable();
    await this.ctx.storage.sql.exec("DELETE FROM convos WHERE id = ?", id);
    return this.list();
  }
}
