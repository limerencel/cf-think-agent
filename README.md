# cf-think-agent

Cloudflare edge agent for Aki — Think + Computer VFS + GBrain MCP + AI Gateway.

```
Browser WebUI (Think WebSocket)
  → Worker cf-think-agent
    → Durable Object "Assistant" (Think agent loop)
         ├─ Model: deepseek-v4-flash
         │    via Cloudflare AI Gateway hermes-aig / custom-opencode
         ├─ Files: @cloudflare/computer SQLite VFS (durable, per conversation)
         └─ GBrain MCP tools: query / search / get_page / put_page / recall / get_health
```

- URL: https://think.itsuhiro.com/ (Cloudflare Access; allowed emails itsushimamura@gmail.com, hjy2441217743@gmail.com)
- Each conversation = one Durable Object instance (`name=c<id>`); chat history persists in the DO.
- Conversation list stored in localStorage (max 30). Delete only clears the local entry.

## Frontend

React + Vite. Claude-style layout: centered greeting empty state, one 24px-rounded
composer card (model pill bottom-left, circular send button bottom-right), mobile
drawer sidebar (hamburger top-left) with history + New chat; sidebar is permanent
on ≥900px.

## Dev

```sh
npm install
cp .dev.vars.example .dev.vars   # fill the two secrets
npm run dev                      # vite dev
```

Typecheck: `npx tsc --noEmit`

## Deploy

```sh
npx wrangler secret put OPENCODE_GO_API_KEY
npx wrangler secret put GBRAIN_MCP_TOKEN
npm run build && npx wrangler deploy
```

Custom domain: `think.itsuhiro.com` attached via Workers custom domain + Access app.
(The workers.dev URL 1101s on this account — use the custom domain.)

## Known limits

- `@cloudflare/computer` worker-shell / container backends need the `experimental`
  compatibility flag, which Cloudflare rejects on production deploys (error 10021).
  So the Workspace runs as SQLite VFS only; shell is Think's built-in bash.
- GBrain is wired as 6 explicit tools, not the full 102-tool MCP surface
  (keeps deepseek-v4-flash context small).
- The delete button in the sidebar clears the local list only; messages remain in
  the DO until a purge callable is added.
