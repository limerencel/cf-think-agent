# cf-think-agent

A state-of-the-art, cloud-native Edge Autonomous AI Agent built directly on Cloudflare Workers, Durable Objects, SQLite Virtual File System (VFS), Universal Model Context Protocol (MCP), and multi-provider inference routing.

Production Domain: [https://think.itsuhiro.com](https://think.itsuhiro.com) (Protected by Cloudflare Zero Trust Access)

---

## 🏛️ System Architecture

`cf-think-agent` utilizes a **Dual Durable Objects Architecture** designed to separate session-isolated execution from globally synchronized state.

```
                                  [ User WebUI (React 19 + Vite) ]
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  │                                                               │
     [ Global API / Sync / OAuth ]                                    [ Real-time WebSocket ]
                  │                                                               │
                  ▼                                                               ▼
  ┌──────────────────────────────┐                              ┌───────────────────────────────────┐
  │  ConvoIndex (Global Singleton)│                              │   Assistant (Per-Session DO)      │
  │  Durable Object: convo-index │                              │   Durable Object: c<convo_id>     │
  ├──────────────────────────────┤                              ├───────────────────────────────────┤
  │ • Conversation Index (SQLite)│                              │ • Realtime Think Agent Loop       │
  │ • AI Providers & Credentials │                              │ • LLM Multi-turn Reasoning        │
  │ • MCP Servers Registry       │                              │ • Cloudflare Computer VFS (SQLite)│
  │ • OAuth Tokens & PKCE State  │                              │ • Dynamic MCP Tool Execution      │
  │ • Custom System Prompts      │                              │ • Conversation Message History    │
  └──────────────────────────────┘                              └───────────────────────────────────┘
```

---

## 🧩 Core Technical Components

### 1. Dual Durable Objects Subsystem

- **`Assistant` (Session DO Instance)**:
  - **Scale**: One independent Durable Object instance per conversation (`name = c<convo_id>`).
  - **Realtime Agent Loop**: Backed by `@cloudflare/think` and `@cloudflare/ai-chat`. Manages the step-by-step reasoning cycle, tool calling decisions, and streaming response token generation.
  - **Cloudflare Computer VFS**: Houses a private SQLite-backed Virtual File System (`@cloudflare/computer`) per session, ensuring generated files, temporary code, and drafts are fully isolated.
  
- **`ConvoIndex` (Global Singleton DO)**:
  - **Scale**: Global singleton (`idFromName("convo-index")`).
  - **Persistent SQLite Tables**:
    - `convos`: Cross-device synced conversation index with timestamps and titles.
    - `providers`: Custom AI provider configurations, endpoints, API keys, selected models, and inference hyperparameters (`temperature`, `max_tokens`, `top_p`, `use_response_api`).
    - `mcp_servers`: Universal MCP server registry with auth types, dynamic tokens, and cached tool signatures.
    - `mcp_oauth_sessions`: Transient OAuth 2.0 PKCE verification state storage.
    - `app_settings`: Global application settings, custom system prompts, and prompt execution modes.

---

### 2. Cloudflare Virtual File System (VFS)

- **Engine**: `@cloudflare/computer` SQLite VFS on Durable Object storage.
- **Capabilities**:
  - Exposes standard file operations to the agent: `read`, `write`, `edit`, `ls`, `rm`, `mkdir`, `stat`.
  - **Sub-millisecond Edge Persistence**: Files created by the agent persist across page reloads and browser sessions within that specific conversation.
  - **Clean Separation from Long-term Memory**:
    - **Cloudflare VFS**: Session-scoped workspace for code artifacts, data tables, and working drafts.
    - **GBrain MCP**: Global personal knowledge base for infrastructure facts, credentials, and persistent documents.

---

### 3. Multi-Provider Inference & Routing Engine

- **Protocols Supported**:
  1. **Cloudflare AI Gateway / Chat Completions** (`openai.chat` protocol via `/chat/completions`).
  2. **OpenAI Responses API** (`openai.responses` protocol via `/responses`).
- **Dynamic Endpoint Probing**:
  - Automatically queries upstream `/models` or `/v1/models` endpoints to discover and auto-populate available models.
- **Tunable Hyperparameters**:
  - Per-provider customization of `Temperature`, `Max Tokens`, and `Top-P`.
- **Zero-downtime Fallback**:
  - Default preset routes to Cloudflare AI Gateway + DeepSeek (`deepseek-v4-flash`).

---

### 4. Universal MCP (Model Context Protocol) Client

The engine includes a full-featured MCP client (`src/mcp-client.ts`) compliant with the latest Model Context Protocol specifications:

- **Transport**: Supports Streamable HTTP, Server-Sent Events (SSE), and JSON-RPC 2.0 over standard HTTP.
- **Authentication Modes**:
  1. **No Auth**: Open endpoints.
  2. **Static Bearer Token / API Key**: For server tokens (e.g., GBrain MCP).
  3. **Automated OAuth 2.0 + PKCE (RFC 9728 + RFC 8414 + RFC 7591)**:
     - **Discovery**: Automatically probes `/.well-known/oauth-protected-resource/mcp` and `/.well-known/oauth-authorization-server`.
     - **Dynamic Registration**: Automatically registers dynamic clients via RFC 7591 when supported (e.g., Inkstone MCP).
     - **PKCE Security**: Cryptographically generates SHA-256 code verifiers and challenges (`S256`).
     - **Popup Handshake**: One-click authorization popup that exchanges authorization codes for access and refresh tokens, synchronizes discovered tools, and updates Durable Object SQLite.

---

### 5. Custom System Prompt & Persona System

- **Dedicated Prompt Tab in Settings**:
  - Customize agent instructions, personas, formatting rules, and behavior.
- **Two Execution Modes**:
  1. **Append to Default (Recommended)**: Appends custom instructions to core agent guidelines, preserving VFS file operations and GBrain MCP knowledge retrieval tools.
  2. **Override Default Completely**: Replaces the entire base prompt with user-supplied text.
- **Quick Preset Templates**:
  - *Concise Expert*, *Coding Architect*, *Bilingual Explainer*, *Research Analyst*.
- **Built-in Inspector**:
  - View and copy the built-in system prompt anytime.

---

## 🛠️ Tech Stack

- **Runtime**: Cloudflare Workers & Durable Objects (`nodejs_compat`).
- **AI Agent Framework**: `@cloudflare/think`, `@cloudflare/ai-chat`, `@cloudflare/computer`, `ai` (Vercel AI SDK).
- **Persistence**: Durable Objects SQLite (`ctx.storage.sql`).
- **Frontend**: React 19 + Vite 8 + Rolldown (`@cloudflare/vite-plugin`) + Custom Zero-dependency CSS.
- **Security & Access**: Cloudflare Zero Trust Access.

---

## 🚀 Development & Deployment

### Local Development

```bash
# 1. Install dependencies
npm install

# 2. Configure local secrets in .dev.vars
cp .dev.vars.example .dev.vars
# Fill OPENCODE_GO_API_KEY and GBRAIN_MCP_TOKEN

# 3. Start local development server
npm run dev
```

### Type Checking & Building

```bash
npx tsc --noEmit
npm run build
```

### Production Deployment

```bash
# Upload Cloudflare secrets
npx wrangler secret put OPENCODE_GO_API_KEY
npx wrangler secret put GBRAIN_MCP_TOKEN

# Deploy Worker & Assets
CLOUDFLARE_API_TOKEN="..." npm run deploy
```

> **⚠️ Domain Notice**: Always access and test against the production domain `https://think.itsuhiro.com`.
