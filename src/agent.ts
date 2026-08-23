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

import { mcpCallTool, resolveHindsightEndpoint } from "./mcp-client";
import type { McpServerConfig, HindsightConfig } from "./mcp-types";
import type { MnemosyneConfig } from "./mnemosyne";

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

    let mnemosyneConfig: MnemosyneConfig | undefined;
    if ((this.env as any).ConvoIndex) {
      try {
        const stub = (this.env as any).ConvoIndex.get((this.env as any).ConvoIndex.idFromName("convo-index"));
        mnemosyneConfig = await stub.mnemosyneGetConfig();
      } catch (err) {
        console.warn("Failed to load MnemosyneConfig from ConvoIndex DO:", err);
      }
    }

    this.currentMemoryContext = undefined;

    // Extract last user message from body or conversation history
    let userMessage = ctx.body?.userMessage as string | undefined;
    if (!userMessage?.trim() && ctx.messages && ctx.messages.length > 0) {
      const lastUser = [...ctx.messages].reverse().find((m: any) => m.role === "user");
      if (lastUser) {
        userMessage =
          typeof lastUser.content === "string"
            ? lastUser.content
            : Array.isArray(lastUser.content)
            ? lastUser.content.map((c: any) => c.text || "").join(" ")
            : "";
      }
    }

    // 1. Mnemosyne Auto-Recall: Pre-inference memory injection (Working Memory + Episodic + Knowledge Graph)
    if (mnemosyneConfig?.enabled && mnemosyneConfig?.autoRecall && userMessage?.trim() && (this.env as any).ConvoIndex) {
      try {
        const stub = (this.env as any).ConvoIndex.get((this.env as any).ConvoIndex.idFromName("convo-index"));
        const recallRes = await stub.mnemosyneRecall({
          query: userMessage.trim(),
          topK: mnemosyneConfig.recallTopK || 5,
          sessionId: this.name,
          scope: mnemosyneConfig.scope || "global",
          includeTriples: true,
        });

        if (recallRes && recallRes.formattedContext) {
          this.currentMemoryContext = recallRes.formattedContext;
        }
      } catch (err) {
        console.warn("Mnemosyne auto-recall failed:", err);
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

    let mcpServers = (ctx.body?.mcpServers as McpServerConfig[] | undefined) || [];
    if (mcpServers.length === 0 && (this.env as any).ConvoIndex) {
      try {
        const stub = (this.env as any).ConvoIndex.get((this.env as any).ConvoIndex.idFromName("convo-index"));
        mcpServers = await stub.listMcpServers();
      } catch (err) {
        console.warn("Failed to load McpServers from ConvoIndex DO:", err);
      }
    }

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

    // Register Mnemosyne Zero-Cloud Native Memory Tools
    if ((this.env as any).ConvoIndex && (mnemosyneConfig?.enabled ?? true)) {
      const convoStub = (this.env as any).ConvoIndex.get((this.env as any).ConvoIndex.idFromName("convo-index"));

      const rememberTool = tool({
        description: "Store a permanent memory, user preference, architectural decision, constraint, or fact into Mnemosyne Zero-Cloud memory. Call this proactively whenever the user shares enduring information.",
        inputSchema: z.object({
          content: z.string().describe("The concise fact, decision, or preference to remember"),
          importance: z.number().min(0.1).max(1.0).optional().describe("Salience weight from 0.1 to 1.0 (default 0.7)"),
          isWorkingMemory: z.boolean().optional().describe("Set true for hot temporary context with TTL"),
          ttlSeconds: z.number().optional().describe("Optional time-to-live in seconds"),
        }),
        execute: async (args) => {
          const res = await convoStub.mnemosyneRemember({
            content: args.content,
            importance: args.importance ?? 0.7,
            isWorkingMemory: args.isWorkingMemory,
            ttlSeconds: args.ttlSeconds,
            sessionId: this.name,
            source: "agent_tool",
          });
          return res.ok ? `Memory retained successfully in Mnemosyne (ID: ${res.id})` : { error: "Failed to remember" };
        },
      });

      const recallTool = tool({
        description: "Search long-term episodic memory, working memory, and knowledge graph triples using Mnemosyne's BEAM hybrid scoring. Call this whenever you need past knowledge or user context.",
        inputSchema: z.object({
          query: z.string().describe("Search query or concept to recall"),
          top_k: z.number().optional().describe("Maximum number of memories to return (default 5)"),
        }),
        execute: async (args) => {
          const res = await convoStub.mnemosyneRecall({
            query: args.query,
            topK: args.top_k || 5,
            sessionId: this.name,
            includeTriples: true,
          });
          return res.formattedContext || "No relevant memories found in Mnemosyne.";
        },
      });

      const queryTriplesTool = tool({
        description: "Query structured entity relationships from the temporal knowledge graph (subject, predicate, object).",
        inputSchema: z.object({
          subject: z.string().optional().describe("Filter by subject entity name"),
          predicate: z.string().optional().describe("Filter by relationship predicate"),
          object: z.string().optional().describe("Filter by object value or entity"),
        }),
        execute: async (args) => {
          const triples = await convoStub.mnemosyneQueryTriples({
            subject: args.subject,
            predicate: args.predicate,
            object: args.object,
          });
          if (!triples || triples.length === 0) return "No matching knowledge graph triples found.";
          return triples
            .map((t: any) => `(${t.subject}, ${t.predicate}, ${t.object})${t.validUntil ? ` [Valid until: ${t.validUntil}]` : ""}`)
            .join("\n");
        },
      });

      const addTripleTool = tool({
        description: "Add a structured entity relationship (Subject, Predicate, Object) into the temporal knowledge graph.",
        inputSchema: z.object({
          subject: z.string().describe("Subject entity (e.g. 'User', 'Project', 'API')"),
          predicate: z.string().describe("Predicate relationship (e.g. 'prefers', 'uses', 'assigned_to')"),
          object: z.string().describe("Target object or attribute (e.g. 'Dark Mode', 'Cloudflare Workers')"),
          valid_from: z.string().optional().describe("Optional valid from date"),
          valid_until: z.string().optional().describe("Optional valid until date"),
        }),
        execute: async (args) => {
          const res = await convoStub.mnemosyneAddTriple({
            subject: args.subject,
            predicate: args.predicate,
            object: args.object,
            validFrom: args.valid_from,
            validUntil: args.valid_until,
            source: "agent_tool",
          });
          return res.ok ? "Knowledge graph triple added successfully" : { error: "Failed to add triple" };
        },
      });

      dynamicTools["mnemosyne_remember"] = rememberTool;
      dynamicTools["mnemosyne_recall"] = recallTool;
      dynamicTools["mnemosyne_triples_query"] = queryTriplesTool;
      dynamicTools["mnemosyne_triples_add"] = addTripleTool;

      // Provide short aliases
      if (!dynamicTools["remember"]) dynamicTools["remember"] = rememberTool;
      if (!dynamicTools["recall"]) dynamicTools["recall"] = recallTool;
    }

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

    const allTools: ToolSet = {
      ...this.getTools(),
      ...dynamicTools,
    };

    return {
      ...(model ? { model } : {}),
      tools: allTools,
      ...turnParams,
      system: this.getSystemPrompt(),
    };
  }

  /* ---------------- System Prompt ---------------- */

  getSystemPrompt(): string {
    const promptParts = [
      "You are cf-think-agent, a fast, proactive, intelligent autonomous agent running on Cloudflare Workers and Durable Objects.",
      "You have access to persistent workspace file tools, real-time web search, and Mnemosyne Zero-Cloud AI memory.",
      "",
      "System & Tool usage guidelines:",
      "- For notes, code artifacts, or working drafts created in this session: use Cloudflare workspace tools (prefer read/ls/write/edit over bash cat/ls/sed).",
      "- Uploaded user files & images: Files or images uploaded/pasted by the user in chat are stored directly in your Cloudflare Computer workspace root (e.g. `/image.png`, `/data.csv`, `/app.py`). You have full access to inspect, read, analyze, edit, and modify them using your workspace tools (`read`, `write`, `edit`, `ls`). When the user asks you to update or modify an uploaded file, apply the changes directly to the workspace file using `edit` or `write` so the updated file is saved.",
      "- If asked about unfamiliar topics, recent events, breaking news, new technologies/APIs, facts outside your training cutoff, or anything you are not 100% certain about, you MUST proactively use the Parallel MCP search tools to search the web before answering.",
      "- Mnemosyne Zero-Cloud Memory: Long-term memories, user preferences, and knowledge graph triples are automatically retrieved into <mnemosyne_memory_context>. Use them seamlessly to maintain continuity across all conversations without asking the user for already-known preferences.",
      "- Proactive Retention: When the user shares enduring personal preferences, project architecture decisions, constraints, or requests you to remember something, proactively invoke `mnemosyne_remember` or `mnemosyne_triples_add`.",
      "- Memory Search: Use `mnemosyne_recall` or `mnemosyne_triples_query` whenever you need historical context or user knowledge.",
      "- Keep replies concise, structured, and helpful."
    ];

    const defaultPrompt = promptParts.join("\n");
    let prompt = defaultPrompt;

    if (this.currentCustomPrompt) {
      prompt =
        this.currentPromptMode === "override"
          ? this.currentCustomPrompt
          : `${defaultPrompt}\n\n[Custom User Instructions / Persona]\n${this.currentCustomPrompt}`;
    }

    if (this.currentMemoryContext) {
      prompt += `\n\n<mnemosyne_memory_context>\n[Mnemosyne Persistent Memories & Knowledge Graph]\n${this.currentMemoryContext}\n</mnemosyne_memory_context>`;
    }

    return prompt;
  }

  @callable()
  async autoRetainTurn(payload: {
    userMessage: string;
    assistantResponse: string;
    mnemosyneConfig?: MnemosyneConfig;
  }): Promise<{ ok: boolean; message?: string }> {
    if ((this.env as any).ConvoIndex) {
      try {
        const stub = (this.env as any).ConvoIndex.get((this.env as any).ConvoIndex.idFromName("convo-index"));
        const cfg = payload.mnemosyneConfig || (await stub.mnemosyneGetConfig());
        if (cfg?.enabled && cfg?.autoRetain) {
          const userSummary = payload.userMessage.slice(0, 300);
          const asstSummary = payload.assistantResponse.slice(0, 300);
          await stub.mnemosyneRemember({
            content: `User: ${userSummary}\nAssistant: ${asstSummary}`,
            importance: 0.6,
            source: "auto_retain",
            sessionId: this.name,
          });
          return { ok: true, message: "Mnemosyne turn retained" };
        }
      } catch (err: any) {
        console.warn("Mnemosyne autoRetainTurn failed:", err);
      }
    }
    return { ok: true };
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
  async getWorkspaceFile(path: string): Promise<{
    ok: boolean;
    content?: string;
    isBinary?: boolean;
    base64?: string;
    mimeType?: string;
    error?: string;
  }> {
    try {
      const cleanPath = path.startsWith("/") ? path : `/${path}`;
      const ext = cleanPath.split(".").pop()?.toLowerCase() || "";
      const isImg = ["png", "jpg", "jpeg", "gif", "webp", "svg", "ico", "bmp", "avif"].includes(ext);
      const isPdf = ext === "pdf";
      const isBin = isImg || isPdf || ["zip", "tar", "gz", "wasm", "bin", "mp3", "mp4", "wav", "webm"].includes(ext);

      let rawBytes: Uint8Array | null = null;
      if (typeof (this.workspace as any).readFileBytes === "function") {
        try {
          rawBytes = await (this.workspace as any).readFileBytes(cleanPath);
        } catch {
          rawBytes = null;
        }
      }

      if (!rawBytes && (this.workspace as any).fs?.readFile) {
        try {
          const stream = await (this.workspace as any).fs.readFile(cleanPath);
          if (stream && typeof stream.getReader === "function") {
            const reader = stream.getReader();
            const chunks: Uint8Array[] = [];
            let totalLen = 0;
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              if (value) {
                chunks.push(value);
                totalLen += value.length;
              }
            }
            rawBytes = new Uint8Array(totalLen);
            let offset = 0;
            for (const chunk of chunks) {
              rawBytes.set(chunk, offset);
              offset += chunk.length;
            }
          }
        } catch {
          rawBytes = null;
        }
      }

      let mime = "application/octet-stream";
      if (ext === "png") mime = "image/png";
      else if (ext === "jpg" || ext === "jpeg") mime = "image/jpeg";
      else if (ext === "gif") mime = "image/gif";
      else if (ext === "webp") mime = "image/webp";
      else if (ext === "svg") mime = "image/svg+xml";
      else if (ext === "ico") mime = "image/x-icon";
      else if (ext === "bmp") mime = "image/bmp";
      else if (ext === "avif") mime = "image/avif";
      else if (ext === "pdf") mime = "application/pdf";
      else if (ext === "json") mime = "application/json";
      else if (ext === "html" || ext === "htm") mime = "text/html";
      else if (ext === "css") mime = "text/css";
      else if (ext === "js" || ext === "mjs") mime = "application/javascript";
      else if (ext === "ts" || ext === "tsx") mime = "text/plain";
      else if (ext === "txt" || ext === "md" || ext === "py" || ext === "sh") mime = "text/plain";

      if (rawBytes) {
        if (isBin) {
          let binary = "";
          const len = rawBytes.length;
          for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(rawBytes[i]);
          }
          const base64 = btoa(binary);
          return { ok: true, isBinary: true, base64, mimeType: mime };
        } else {
          const content = new TextDecoder().decode(rawBytes);
          return { ok: true, content, isBinary: false, mimeType: mime };
        }
      }

      const raw = await (this.workspace as any).readFile?.(cleanPath, { encoding: "utf8" });
      const content = typeof raw === "string" ? raw : raw instanceof Uint8Array ? new TextDecoder().decode(raw) : String(raw || "");
      return { ok: true, content, isBinary: false, mimeType: mime };
    } catch (err: any) {
      return { ok: false, error: err.message || String(err) };
    }
  }

  @callable()
  async writeWorkspaceFile(payload: {
    path: string;
    content: string; // utf-8 string or base64
    encoding?: "utf8" | "base64";
  }): Promise<{ ok: boolean; path: string; size?: number; error?: string }> {
    try {
      const cleanPath = payload.path.startsWith("/") ? payload.path : `/${payload.path}`;
      const lastSlash = cleanPath.lastIndexOf("/");
      if (lastSlash > 0) {
        const dir = cleanPath.slice(0, lastSlash);
        try {
          await (this.workspace as any).fs?.mkdir?.(dir, { recursive: true });
        } catch {
          /* ignore mkdir error */
        }
      }

      let bytes: Uint8Array;
      if (payload.encoding === "base64") {
        const binaryStr = atob(payload.content);
        bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
          bytes[i] = binaryStr.charCodeAt(i);
        }
      } else {
        bytes = new TextEncoder().encode(payload.content);
      }

      if ((this.workspace as any).fs?.writeFile) {
        await (this.workspace as any).fs.writeFile(cleanPath, bytes);
      } else if ((this.workspace as any).writeFile) {
        await (this.workspace as any).writeFile(
          cleanPath,
          payload.encoding === "base64" ? new TextDecoder().decode(bytes) : payload.content
        );
      }

      return { ok: true, path: cleanPath, size: bytes.length };
    } catch (err: any) {
      return { ok: false, path: payload.path, error: err.message || String(err) };
    }
  }

  @callable()
  async deleteWorkspaceFile(payload: { path: string }): Promise<{ ok: boolean; error?: string }> {
    try {
      const cleanPath = payload.path.startsWith("/") ? payload.path : `/${payload.path}`;
      if ((this.workspace as any).fs?.rm) {
        await (this.workspace as any).fs.rm(cleanPath, { force: true });
      } else if ((this.workspace as any).rm) {
        await (this.workspace as any).rm(cleanPath, { force: true });
      }
      return { ok: true };
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
