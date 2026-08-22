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
import { callable } from "agents";
import { createOpenAI } from "@ai-sdk/openai";
import { tool, type ToolSet } from "ai";
import { z } from "zod";

import { mcpCallTool } from "./mcp-client";
import type { McpServerConfig, HindsightConfig } from "./mcp-types";

function cleanBaseUrl(raw: string): string {
  let url = raw.trim();
  // Strip trailing slashes
  url = url.replace(/\/+$/, "");
  // Strip trailing /chat/completions, /responses, or /models if pasted by user
  url = url.replace(/\/(chat\/completions|responses|models)$/, "");
  url = url.replace(/\/+$/, "");
  return url;
}

/**
 * Universal Reasoning / CoT Stream Adapter.
 *
 * Intercepts OpenAI-compatible SSE streams (DeepSeek-R1, SiliconFlow, OpenRouter, QwQ, etc.)
 * that return reasoning tokens in `delta.reasoning_content` (or `reasoning` / `thought`),
 * which @ai-sdk/openai drops by default. Seamlessly wraps them into `<think>...</think>` inside
 * `delta.content` so they are delivered to the frontend thinking UI without breaking standard parsers.
 */
function createReasoningAwareFetch(customFetch: typeof fetch = fetch): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    // Inject agent harness headers (required by OpenRouter for agentic models like thinkingmachines/inkling:free)
    const headers = new Headers(init?.headers);
    if (!headers.has("User-Agent")) {
      headers.set("User-Agent", "pi/1.0");
    }
    if (!headers.has("HTTP-Referer")) {
      headers.set("HTTP-Referer", "https://pi.dev");
    }
    if (!headers.has("X-Title") && !headers.has("X-OpenRouter-Title")) {
      headers.set("X-Title", "pi");
    }

    const modifiedInit = {
      ...init,
      headers,
    };

    const res = await customFetch(input, modifiedInit);
    const contentType = res.headers.get("content-type") || "";

    if (contentType.includes("text/event-stream") && res.body) {
      let isInsideReasoning = false;
      let buffer = "";

      const transformStream = new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, controller) {
          buffer += new TextDecoder().decode(chunk, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          const outputLines: string[] = [];

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:") || trimmed === "data: [DONE]") {
              outputLines.push(line);
              continue;
            }

            const jsonStr = trimmed.slice(5).trim();
            try {
              const data = JSON.parse(jsonStr);
              const choice = data.choices?.[0];
              if (choice?.delta) {
                const reasoning =
                  (typeof choice.delta.reasoning_content === "string" && choice.delta.reasoning_content) ||
                  (typeof choice.delta.reasoning === "string" && choice.delta.reasoning) ||
                  (typeof choice.delta.thought === "string" && choice.delta.thought) ||
                  (choice.delta.reasoning_details?.[0]?.text);

                const content = typeof choice.delta.content === "string" ? choice.delta.content : undefined;

                if (reasoning) {
                  let injected = "";
                  if (!isInsideReasoning) {
                    isInsideReasoning = true;
                    injected += "<think>";
                  }
                  injected += reasoning;
                  choice.delta.content = (content || "") + injected;
                  delete choice.delta.reasoning_content;
                  delete choice.delta.reasoning;
                  delete choice.delta.thought;
                  delete choice.delta.reasoning_details;
                } else if (isInsideReasoning) {
                  if (content !== undefined && content.length > 0) {
                    isInsideReasoning = false;
                    choice.delta.content = "</think>" + content;
                  } else if (choice.finish_reason) {
                    isInsideReasoning = false;
                    choice.delta.content = "</think>";
                  }
                }
              }
              outputLines.push("data: " + JSON.stringify(data));
            } catch {
              outputLines.push(line);
            }
          }

          if (outputLines.length > 0) {
            controller.enqueue(new TextEncoder().encode(outputLines.join("\n") + "\n"));
          }
        },
        flush(controller) {
          if (buffer) {
            controller.enqueue(new TextEncoder().encode(buffer));
          }
          if (isInsideReasoning) {
            isInsideReasoning = false;
            const closeChunk = {
              id: "think-close",
              choices: [{ delta: { content: "</think>" }, index: 0 }],
            };
            controller.enqueue(
              new TextEncoder().encode("data: " + JSON.stringify(closeChunk) + "\n\n")
            );
          }
        },
      });

      return new Response(res.body.pipeThrough(transformStream), {
        status: res.status,
        statusText: res.statusText,
        headers: res.headers,
      });
    }

    if (contentType.includes("application/json")) {
      try {
        const text = await res.text();
        const data = JSON.parse(text);
        const choice = data.choices?.[0];
        if (choice?.message) {
          const reasoning =
            (typeof choice.message.reasoning_content === "string" && choice.message.reasoning_content) ||
            (typeof choice.message.reasoning === "string" && choice.message.reasoning) ||
            (typeof choice.message.thought === "string" && choice.message.thought) ||
            (choice.message.reasoning_details?.[0]?.text);
          if (reasoning) {
            choice.message.content = `<think>${reasoning}</think>\n\n${choice.message.content || ""}`;
            delete choice.message.reasoning_content;
            delete choice.message.reasoning;
            delete choice.message.thought;
            delete choice.message.reasoning_details;
            return new Response(JSON.stringify(data), {
              status: res.status,
              statusText: res.statusText,
              headers: res.headers,
            });
          }
        }
        return new Response(text, {
          status: res.status,
          statusText: res.statusText,
          headers: res.headers,
        });
      } catch {
        return res;
      }
    }

    return res;
  };
}

export class Assistant extends Think<Env> {
  override maxSteps = 16;

  // Record message + tool payloads in traces (Agents dashboard session replay)
  override storeMessages = true;
  override storeTools = true;

  // Wait for MCP servers (Parallel search) to be connected before inference.
  override waitForMcpConnections = { timeout: 10_000 };

  private currentCustomPrompt?: string;
  private currentPromptMode: "append" | "override" = "append";
  private currentMemoryContext?: string;
  private currentHindsightConfig?: HindsightConfig;

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
      fetch: createReasoningAwareFetch(),
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

  override async beforeTurn(ctx: TurnContext): Promise<TurnConfig | void> {
    const customPrompt = ctx.body?.customSystemPrompt as string | undefined;
    const promptMode = ctx.body?.promptMode as "append" | "override" | undefined;
    this.currentCustomPrompt = customPrompt?.trim() || undefined;
    this.currentPromptMode = promptMode || "append";

    const hindsightConfig = ctx.body?.hindsightConfig as HindsightConfig | undefined;
    this.currentHindsightConfig = hindsightConfig?.enabled && hindsightConfig.endpoint ? hindsightConfig : undefined;
    this.currentMemoryContext = undefined;

    // 1. Auto-Recall: Hermes-style pre-inference memory injection
    if (this.currentHindsightConfig?.autoRecall) {
      const userMessage = ctx.body?.userMessage as string | undefined;
      if (userMessage?.trim()) {
        try {
          const bankId = this.currentHindsightConfig.bankId || this.name;
          const recallRes = await mcpCallTool(
            this.currentHindsightConfig,
            "hindsight_recall",
            {
              query: userMessage.trim(),
              bank_id: bankId,
              limit: this.currentHindsightConfig.recallTopK || 5,
            }
          );

          if (recallRes.ok && recallRes.text && recallRes.text.trim()) {
            this.currentMemoryContext = recallRes.text.trim();
          } else {
            const fallbackRes = await mcpCallTool(
              this.currentHindsightConfig,
              "recall",
              {
                query: userMessage.trim(),
                bank_id: bankId,
              }
            );
            if (fallbackRes.ok && fallbackRes.text && fallbackRes.text.trim()) {
              this.currentMemoryContext = fallbackRes.text.trim();
            }
          }
        } catch (err) {
          console.warn("Hindsight auto-recall failed:", err);
        }
      }
    }

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
        fetch: createReasoningAwareFetch(),
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
      if (!s.enabled) continue;
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

    // Register Hermes-style explicit Hindsight Memory tools
    if (this.currentHindsightConfig) {
      const hConfig = this.currentHindsightConfig;
      const bankId = hConfig.bankId || this.name;

      dynamicTools["hindsight_recall"] = tool({
        description: "Search and recall long-term memories, user preferences, and facts from the Hindsight memory bank.",
        inputSchema: z.object({
          query: z.string().describe("The memory search query or concept to recall"),
          bank_id: z.string().optional().describe("Optional memory bank ID (defaults to current session)"),
        }),
        execute: async (args) => {
          const res = await mcpCallTool(hConfig, "hindsight_recall", {
            bank_id: args.bank_id || bankId,
            query: args.query,
          });
          if (!res.ok) {
            const fallback = await mcpCallTool(hConfig, "recall", {
              bank_id: args.bank_id || bankId,
              query: args.query,
            });
            return fallback.ok ? fallback.text || "No memories found" : { error: fallback.error || res.error };
          }
          return res.text || "No memories found";
        },
      });

      dynamicTools["hindsight_retain"] = tool({
        description: "Retain, save, or commit a salient fact, preference, or decision into the persistent Hindsight memory bank.",
        inputSchema: z.object({
          content: z.string().describe("The concise fact, decision, or user preference to retain"),
          bank_id: z.string().optional().describe("Optional memory bank ID (defaults to current session)"),
        }),
        execute: async (args) => {
          const res = await mcpCallTool(hConfig, "hindsight_retain", {
            bank_id: args.bank_id || bankId,
            content: args.content,
            timestamp: Date.now(),
          });
          if (!res.ok) {
            const fallback = await mcpCallTool(hConfig, "retain", {
              bank_id: args.bank_id || bankId,
              content: args.content,
              timestamp: Date.now(),
            });
            return fallback.ok ? fallback.text || "Memory retained" : { error: fallback.error || res.error };
          }
          return res.text || "Memory retained successfully";
        },
      });

      dynamicTools["hindsight_reflect"] = tool({
        description: "Reflect and consolidate memories on a specific topic or synthesize insights across memory graphs.",
        inputSchema: z.object({
          topic: z.string().optional().describe("Topic or query to reflect upon"),
          bank_id: z.string().optional().describe("Optional memory bank ID"),
        }),
        execute: async (args) => {
          const res = await mcpCallTool(hConfig, "hindsight_reflect", {
            bank_id: args.bank_id || bankId,
            topic: args.topic,
          });
          if (!res.ok) {
            const fallback = await mcpCallTool(hConfig, "reflect", {
              bank_id: args.bank_id || bankId,
              topic: args.topic,
            });
            return fallback.ok ? fallback.text || "Reflection complete" : { error: fallback.error || res.error };
          }
          return res.text || "Reflection complete";
        },
      });
    }

    if (model || Object.keys(dynamicTools).length > 0) {
      // Pass sampling + reasoning parameters through to streamText.
      // providerOptions.openai.reasoningEffort maps to the request body's
      // reasoning_effort field (verified in @ai-sdk/openai internal source).
      const turnParams = custom
        ? {
            ...(custom.temperature !== undefined ? { temperature: custom.temperature } : {}),
            ...(custom.maxTokens !== undefined ? { maxOutputTokens: custom.maxTokens } : {}),
            ...(custom.topP !== undefined ? { topP: custom.topP } : {}),
            ...(custom.reasoningEffort && custom.reasoningEffort !== "none"
              ? { providerOptions: { openai: { reasoningEffort: custom.reasoningEffort } } }
              : {}),
          }
        : {};

      return {
        ...(model ? { model } : {}),
        ...(Object.keys(dynamicTools).length > 0
          ? { tools: { ...this.getTools(), ...dynamicTools } }
          : {}),
        ...turnParams,
      };
    }
  }

  override getSystemPrompt(): string {
    const defaultPrompt = [
      "You are Aki's Cloudflare edge agent.",
      "Reply in the user's language (Chinese if they write Chinese).",
      "You have access to these core systems and built-in capabilities:",
      "1) Cloudflare Computer workspace — durable session files in this Durable Object (read, write, edit, ls).",
      "2) Parallel Web Search MCP — real-time live web search and URL content extraction.",
      "",
      "System & Tool usage guidelines:",
      "- For notes, code artifacts, or working drafts created in this session: use Cloudflare workspace tools (prefer read/ls/write/edit over bash cat/ls/sed).",
      "- If asked about unfamiliar topics, recent events, breaking news, new technologies/APIs, facts outside your training cutoff, or anything you are not 100% certain about, you MUST proactively use the Parallel MCP search tools to search the web before answering.",
      "- Keep replies concise and helpful.",
    ].join("\n");

    let prompt = defaultPrompt;

    if (this.currentCustomPrompt) {
      prompt =
        this.currentPromptMode === "override"
          ? this.currentCustomPrompt
          : `${defaultPrompt}\n\n[Custom User Instructions / Persona]\n${this.currentCustomPrompt}`;
    }

    if (this.currentMemoryContext) {
      prompt += `\n\n<hindsight_memory_context>\n[Active Hindsight Long-term Memories]\n${this.currentMemoryContext}\n</hindsight_memory_context>`;
    }

    return prompt;
  }

  @callable()
  async autoRetainTurn(payload: {
    userMessage: string;
    assistantResponse: string;
    hindsightConfig?: HindsightConfig;
  }): Promise<{ ok: boolean; message?: string }> {
    const config = payload.hindsightConfig || this.currentHindsightConfig;
    if (!config?.enabled || !config?.endpoint || !config?.autoRetain) {
      return { ok: true, message: "Auto-retain not enabled" };
    }
    const bankId = config.bankId || this.name;
    const content = `User: ${payload.userMessage}\nAssistant: ${payload.assistantResponse}`;

    let res = await mcpCallTool(config, "hindsight_retain", {
      bank_id: bankId,
      content,
      user: payload.userMessage,
      assistant: payload.assistantResponse,
      timestamp: Date.now(),
    });
    if (!res.ok) {
      res = await mcpCallTool(config, "retain", {
        bank_id: bankId,
        content,
        user: payload.userMessage,
        assistant: payload.assistantResponse,
        timestamp: Date.now(),
      });
    }
    return { ok: res.ok, message: res.text || res.error };
  }

  override getTools(): ToolSet {
    return {
      ...createAITools({ workspace: this.workspace }),
    };
  }

  @callable()
  async getWorkspaceFiles(): Promise<{
    files: Array<{
      path: string;
      name: string;
      size: number;
      mtime?: number;
      isDirectory: boolean;
    }>;
  }> {
    const list: Array<{
      path: string;
      name: string;
      size: number;
      mtime?: number;
      isDirectory: boolean;
    }> = [];

    const visited = new Set<string>();

    const scanDirectory = async (dirPath: string) => {
      if (visited.has(dirPath)) return;
      visited.add(dirPath);
      try {
        const entries = (await (this.workspace as any).readdir?.(dirPath)) ||
                        (await (this.workspace as any).fs?.readdir?.(dirPath)) ||
                        [];
        for (const entry of entries) {
          const name = entry.name || "";
          if (!name || name === "." || name === "..") continue;
          const fullPath = dirPath === "/" ? `/${name}` : `${dirPath}/${name}`;
          const isDir = !!entry.isDirectory;
          list.push({
            path: fullPath,
            name,
            size: entry.size || 0,
            mtime: entry.mtime,
            isDirectory: isDir,
          });
          if (isDir) {
            await scanDirectory(fullPath);
          }
        }
      } catch {
        /* ignore error on single dir scan */
      }
    };

    await scanDirectory("/");

    // Fallback: If readdir didn't return items, try ls("/")
    if (list.length === 0) {
      try {
        const lsItems = (await (this.workspace as any).ls?.("/")) || [];
        for (const item of lsItems) {
          const itemPath = typeof item === "string" ? item : item.path || item.name;
          if (itemPath) {
            const clean = itemPath.startsWith("/") ? itemPath : `/${itemPath}`;
            let size = 0;
            try {
              const st = await (this.workspace as any).stat?.(clean);
              size = st?.size || 0;
            } catch {
              /* ignore stat error */
            }
            list.push({
              path: clean,
              name: clean.split("/").pop() || clean,
              size,
              isDirectory: false,
            });
          }
        }
      } catch {
        /* best effort */
      }
    }

    return { files: list };
  }

  @callable()
  async getWorkspaceFile(path: string): Promise<{ ok: boolean; content?: string; error?: string }> {
    try {
      const cleanPath = path.startsWith("/") ? path : `/${path}`;
      const raw = await (this.workspace as any).readFile?.(cleanPath, { encoding: "utf8" });
      const content = typeof raw === "string" ? raw : raw instanceof Uint8Array ? new TextDecoder().decode(raw) : String(raw || "");
      return { ok: true, content };
    } catch (err: any) {
      return { ok: false, error: err.message || String(err) };
    }
  }

  @callable()
  async getWorkspaceZipArchive(): Promise<{ ok: boolean; files: Array<{ path: string; content: string }> }> {
    const { files: fileList } = await this.getWorkspaceFiles();
    const result: Array<{ path: string; content: string }> = [];
    for (const file of fileList) {
      if (!file.isDirectory) {
        const res = await this.getWorkspaceFile(file.path);
        if (res.ok && res.content !== undefined) {
          result.push({ path: file.path.replace(/^\/+/, ""), content: res.content });
        }
      }
    }
    return { ok: true, files: result };
  }
}
