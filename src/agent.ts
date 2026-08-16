/**
 * Think agent + Cloudflare Computer VFS + GBrain MCP + AI Gateway DeepSeek.
 *
 * Worker-shell / Containers need the `experimental` flag, which Cloudflare
 * rejects on production deploys (error 10021). Workspace still uses
 * @cloudflare/computer SQLite VFS; shell uses Think's built-in just-bash.
 */
import {
  type DurableObjectStorageLike,
  type ThinkWorkspaceCompatibility,
  Workspace,
} from "@cloudflare/computer";
import { createAITools } from "@cloudflare/computer/tools";
import { Think } from "@cloudflare/think";
import { createOpenAI } from "@ai-sdk/openai";
import { tool, type ToolSet } from "ai";
import { z } from "zod";
import { gbrainCall } from "./gbrain";

export class Assistant extends Think<Env> {
  override maxSteps = 16;

  override workspace = new Workspace({
    storage: this.ctx.storage as unknown as DurableObjectStorageLike,
    useThink: true,
  }) as Workspace & ThinkWorkspaceCompatibility;

  override getModel() {
    const openai = createOpenAI({
      apiKey: this.env.OPENCODE_GO_API_KEY,
      baseURL: this.env.AIG_BASE_URL,
    });
    return openai(this.env.MODEL_ID);
  }

  override getSystemPrompt(): string {
    return [
      "You are Aki's Cloudflare edge agent.",
      "Reply in the user's language (Chinese if they write Chinese).",
      "You have two systems:",
      "1) Cloudflare Computer workspace — durable files in this Durable Object.",
      "2) GBrain — Aki's personal knowledge base (query / search / get_page / put_page / recall / get_health).",
      "Prefer GBrain for anything Aki already stored (infra, people, decisions, workflows).",
      "Prefer workspace tools for notes you create in this session.",
      "Prefer read/ls/write/edit over bash cat/ls/sed.",
      "Keep replies concise. Cite GBrain slugs when you use them.",
      "Do not invent holdings, keys, or infra facts — look them up.",
    ].join("\n");
  }

  override getTools(): ToolSet {
    const url = this.env.GBRAIN_MCP_URL;
    const token = this.env.GBRAIN_MCP_TOKEN;

    return {
      ...createAITools({ workspace: this.workspace }),
      gbrain_health: tool({
        description: "GBrain health: page counts, embed coverage, brain score.",
        inputSchema: z.object({}),
        execute: async () => gbrainCall(url, token, "get_health", {}),
      }),
      gbrain_query: tool({
        description: "Ask GBrain a natural-language question over stored knowledge. Use for 'what do we know about X'.",
        inputSchema: z.object({
          query: z.string().describe("Natural language question"),
        }),
        execute: async ({ query }) => gbrainCall(url, token, "query", { query }),
      }),
      gbrain_search: tool({
        description: "Keyword / semantic search over GBrain pages. Returns slugs and snippets.",
        inputSchema: z.object({
          query: z.string().describe("Search query"),
        }),
        execute: async ({ query }) => gbrainCall(url, token, "search", { query }),
      }),
      gbrain_get_page: tool({
        description: "Read one GBrain page by slug.",
        inputSchema: z.object({
          slug: z.string().describe("Page slug, e.g. infra/ntfy"),
        }),
        execute: async ({ slug }) => gbrainCall(url, token, "get_page", { slug }),
      }),
      gbrain_put_page: tool({
        description: "Write or update a GBrain page. Use only when the user asks to remember / save knowledge.",
        inputSchema: z.object({
          slug: z.string().describe("Page slug"),
          content: z.string().describe("Full markdown page including optional YAML frontmatter"),
        }),
        execute: async ({ slug, content }) =>
          gbrainCall(url, token, "put_page", { slug, content, ingested_via: "cf-think-agent" }),
      }),
      gbrain_recall: tool({
        description: "Recall extracted facts from GBrain (not full pages).",
        inputSchema: z.object({
          query: z.string().describe("What to recall"),
        }),
        execute: async ({ query }) => gbrainCall(url, token, "recall", { query }),
      }),
    };
  }
}
