# CF Think Agent - Coding & Agent Guidelines

## ⚠️ Critical Testing & Domain Rule (STRICT)

- **NEVER use the default Cloudflare `*.workers.dev` domain for testing or verification.**
  - ❌ `https://cf-think-agent.islala.workers.dev`
- **ALWAYS use the custom production domain for all tests, probes, and verification:**
  - ✅ `https://think.itsuhiro.com`
- **Zero Trust Authentication**: Note that `think.itsuhiro.com` is protected behind Cloudflare Access / Zero Trust (returns 302/login when accessed via non-authenticated external curl). Automated browser tests and local mock requests should account for Access auth tokens or test against local unit tests / Vite dev environment.

---

## 🛠️ Architecture & Tech Stack

- **Runtime**: Cloudflare Workers & Durable Objects (`agents` SDK, `@cloudflare/think`, `@cloudflare/ai-chat`).
- **Persistence**: Durable Objects SQLite (`ConvoIndex` DO for conversations, AI providers, and MCP servers).
- **Frontend**: React 19 + Vite + Rolldown (`@cloudflare/vite-plugin`) + Custom zero-dependency CSS.
- **Inference & Routing**: Multi-provider support via Cloudflare AI Gateway, OpenAI Chat Completions, and OpenAI Responses API protocols.
- **MCP Client**: Universal Streamable-HTTP / SSE / JSON-RPC MCP Client (`src/mcp-client.ts`).

---

## 📐 UI & Code Quality Guidelines

1. **Language Consistency**:
   - **ALL UI text, error messages, placeholders, badges, and diagnostic logs MUST be in English.**
2. **Responsive Design**:
   - Modals and settings must remain fully scrollable and legible on compact screens (`min-height: 0`, `overflow-y: auto`, `@media (max-height: 760px)`).
   - Long model names must truncate properly without overflowing pills or displacing action buttons.
   - Mobile devices utilize the sticky `.app-header` with quick new chat, while desktop views maintain the collapsible `.side-rail`.
3. **TypeScript & Build Integrity**:
   - Always verify `npx tsc --noEmit` and `npm run build` with zero errors before deploying.
4. **Deployment Command**:
   - Deploy using `CLOUDFLARE_API_TOKEN="..." npm run deploy`.

---

## ⚡ Tooling & Optimization (RTK)

- Use `rtk` (Rust Token Killer) for dev operations and token optimization.
- Commands: `rtk gain`, `rtk discover`, `rtk proxy <cmd>`.
