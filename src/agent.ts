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
import { Think, type TurnConfig, type TurnContext } from "@cloudflare/think";
import { createOpenAI } from "@ai-sdk/openai";
import { tool, type ToolSet } from "ai";
import { z } from "zod";
import { gbrainCall } from "./gbrain";

import { mcpCallTool } from "./mcp-client";
import type { McpServerConfig } from "./mcp-types";

function cleanBaseUrl(raw: string): string {
  let url = raw.trim();
  // Strip trailing slashes
  url = url.replace(/\/+$/, "");
  // Strip trailing /chat/completions, /responses, or /models if pasted by user
  url = url.replace(/\/(chat\/completions|responses|models)$/, "");
  url = url.replace(/\/+$/, "");
  return url;
}

export class Assistant extends Think<Env> {
  override maxSteps = 16;

  // Wait for MCP servers (Parallel search) to be connected before inference.
  override waitForMcpConnections = { timeout: 10_000 };

  private currentCustomPrompt?: string;
  private currentPromptMode: "append" | "override" = "append";

  override async onStart() {
    await super.onStart();
    // Free web search / URL extraction via Parallel MCP (no API key needed).
    // addMcpServer persists the connection across hibernation; only add if
    // it is not already present so repeated wakes don't stack duplicates.
    if (!this.getMcpServers().servers["parallel"]) {
      await this.addMcpServer("parallel", "https://search.parallel.ai/mcp");
    }
  }

  override workspace = new Workspace({
    storage: this.ctx.storage as unknown as DurableObjectStorageLike,
    useThink: true,
  }) as Workspace & ThinkWorkspaceCompatibility;

  override getModel() {
    const openai = createOpenAI({
      apiKey: this.env.OPENCODE_GO_API_KEY,
      baseURL: cleanBaseUrl(this.env.AIG_BASE_URL),
      headers: {
        // Tag every AI Gateway request as coming from the Think edge agent,
        // with the conversation id, so hermes-aig logs are filterable.
        "cf-aig-metadata": JSON.stringify({
          source: "think-edge-agent",
          convo: this.name,
          protocol: "chat_completions",
        }),
      },
    });
    return openai.chat(this.env.MODEL_ID);
  }

  override beforeTurn(ctx: TurnContext): TurnConfig | void {
    const customPrompt = ctx.body?.customSystemPrompt as string | undefined;
    const promptMode = ctx.body?.promptMode as "append" | "override" | undefined;
    this.currentCustomPrompt = customPrompt?.trim() || undefined;
    this.currentPromptMode = promptMode || "append";

    const custom = ctx.body?.customModel as
      | {
          endpoint?: string;
          apiKey?: string;
          modelId?: string;
          useResponseApi?: boolean;
          temperature?: number;
          maxTokens?: number;
          topP?: number;
          reasoningEffort?: "low" | "medium" | "high" | "none";
        }
      | undefined;

    let model: any = undefined;

    if (custom?.endpoint && custom?.modelId) {
      const cleanEndpoint = cleanBaseUrl(custom.endpoint);
      const isResponse = !!custom.useResponseApi;
      const openai = createOpenAI({
        apiKey: custom.apiKey?.trim() || "dummy-key",
        baseURL: cleanEndpoint,
        headers: {
          "cf-aig-metadata": JSON.stringify({
            source: "think-edge-agent",
            convo: this.name,
            custom_model: custom.modelId,
            reasoning_effort: custom.reasoningEffort || undefined,
            protocol: isResponse ? "responses" : "chat_completions",
          }),
        },
      });

      model = isResponse
        ? openai.responses(custom.modelId.trim())
        : openai.chat(custom.modelId.trim());
    }

    const mcpServers = (ctx.body?.mcpServers as McpServerConfig[] | undefined) || [];
    const dynamicTools: ToolSet = {};

    for (const s of mcpServers) {
      if (!s.enabled || s.id === "gbrain-default") continue;
      const prefix = (s.name || "mcp").toLowerCase().replace(/[^a-z0-9]/g, "_").slice(0, 20);
      for (const t of s.cachedTools || []) {
        const toolName = `${prefix}_${t.name}`.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
        dynamicTools[toolName] = tool({
          description: t.description || `Tool ${t.name} from MCP server ${s.name}`,
          inputSchema: z.record(z.string(), z.unknown()),
          execute: async (args) => {
            const res = await mcpCallTool(s, t.name, args as Record<string, unknown>);
            if (!res.ok) {
              return { error: res.error || "MCP tool execution failed" };
            }
            return res.text || "Success";
          },
        });
      }
    }

    if (model || Object.keys(dynamicTools).length > 0) {
      return {
        ...(model ? { model } : {}),
        ...(Object.keys(dynamicTools).length > 0
          ? { tools: { ...this.getTools(), ...dynamicTools } }
          : {}),
      };
    }
  }

  override getSystemPrompt(): string {
    const defaultPrompt = [
      "You are Aki's Cloudflare edge agent.",
      "Reply in the user's language (Chinese if they write Chinese).",
      "You have access to these core systems and toolsets:",
      "1) Hindsight (External Live Memory) — Aki's fresh, active, dynamic, frequently updated memory stream (current preferences, ongoing context, evolving thoughts and state).",
      "2) GBrain (Library & Archive) — Aki's digital library and static archive (structured repository for permanent reference docs, server/infra facts, people, decisions, and historical records).",
      "3) Cloudflare Computer workspace — durable session files in this Durable Object (read, write, edit, ls).",
      "4) Parallel Web Search MCP — real-time live web search and URL content extraction.",
      "",
      "System & Tool usage guidelines:",
      "- For active context, current preferences, recent activities, or evolving state: check and update Hindsight (live dynamic memory).",
      "- For structured archives, permanent knowledge, server infra, holdings, or long-term docs: query GBrain (library/archive).",
      "- For notes, code artifacts, or working drafts created in this session: use Cloudflare workspace tools (prefer read/ls/write/edit over bash cat/ls/sed).",
      "- If asked about unfamiliar topics, recent events, breaking news, new technologies/APIs, facts outside your training cutoff, or anything you are not 100% certain about, you MUST proactively use the Parallel MCP search tools to search the web before answering.",
      "- Keep replies concise. Cite GBrain slugs or web source URLs when you use them.",
      "- Do not hallucinate or invent holdings, keys, or infra facts — look them up.",
    ].join("\n");

    if (this.currentCustomPrompt) {
      if (this.currentPromptMode === "override") {
        return this.currentCustomPrompt;
      }
      return `${defaultPrompt}\n\n[Custom User Instructions / Persona]\n${this.currentCustomPrompt}`;
    }

    return defaultPrompt;
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
