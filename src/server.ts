import { routeAgentRequest } from "agents";
import { Assistant } from "./agent";
import { ConvoIndex, type ProviderConfig } from "./convo-index";
import {
  mcpListTools,
  discoverMcpOAuth,
  generatePkcePair,
  generateRandomString,
  exchangeOAuthCode,
} from "./mcp-client";

export { Assistant, ConvoIndex };

const CONVO_INDEX = "convo-index";

function renderOAuthResultHtml(params: {
  ok: boolean;
  title: string;
  message: string;
  serverId?: string;
}): string {
  const iconSvg = params.ok
    ? `<svg style="width:48px;height:48px;color:#6B8F5E;margin:0 auto 16px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`
    : `<svg style="width:48px;height:48px;color:#B0544A;margin:0 auto 16px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${params.title} - Think Agent</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      background: #14120F;
      color: #EDECE9;
      padding: 20px;
    }
    .card {
      background: #1C1A17;
      border: 1px solid #2E2B26;
      border-radius: 16px;
      padding: 36px 32px;
      max-width: 420px;
      width: 100%;
      text-align: center;
      box-shadow: 0 20px 48px rgba(0, 0, 0, 0.4);
    }
    h2 { margin: 0 0 8px; font-size: 19px; font-weight: 600; color: #EDECE9; }
    p { margin: 0 0 24px; color: #9E9A90; font-size: 13.5px; line-height: 1.5; }
    .btn {
      display: inline-block;
      background: #E58235;
      color: #FFFFFF;
      border: none;
      padding: 9px 22px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      text-decoration: none;
      transition: background 0.15s ease;
    }
    .btn:hover { background: #F0934A; }
  </style>
</head>
<body>
  <div class="card">
    ${iconSvg}
    <h2>${params.title}</h2>
    <p>${params.message}</p>
    <button class="btn" onclick="window.close()">Close Window</button>
  </div>
  <script>
    try {
      if (window.opener) {
        window.opener.postMessage({
          type: "${params.ok ? "MCP_OAUTH_SUCCESS" : "MCP_OAUTH_ERROR"}",
          serverId: "${params.serverId || ""}",
          message: "${params.message.replace(/"/g, '\\"')}"
        }, "*");
      }
    } catch (e) {}
    ${params.ok ? "setTimeout(function() { window.close(); }, 1400);" : ""}
  </script>
</body>
</html>`;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Cloud conversation-list REST API (backed by ConvoIndex DO SQLite).
    // GET  /api/convos           -> list
    // POST /api/convos           -> { id, title? } touch
    // POST /api/convos/remove    -> { id } remove
    if (url.pathname.startsWith("/api/convos")) {
      const stub = env.ConvoIndex.get(env.ConvoIndex.idFromName(CONVO_INDEX));
      if (request.method === "GET") {
        const list = await stub.list();
        return Response.json({ ok: true, convos: list });
      }
      if (request.method === "POST") {
        let body: { id?: string; title?: string } = {};
        try {
          body = (await request.json()) as typeof body;
        } catch {
          /* ignore */
        }
        if (!body.id) return Response.json({ ok: false, error: "missing id" }, { status: 400 });
        if (url.pathname.endsWith("/remove")) {
          const list = await stub.remove(body.id);
          return Response.json({ ok: true, convos: list });
        }
        const list = await stub.touch(body.id, body.title);
        return Response.json({ ok: true, convos: list });
      }
      return Response.json({ ok: false, error: "method not allowed" }, { status: 405 });
    }

    // Cloud AI Providers & Settings REST API (backed by ConvoIndex DO SQLite).
    // GET  /api/providers        -> { ok: true, providers: ProviderConfig[], activeId: string | null }
    // POST /api/providers        -> { provider?: ProviderConfig, providers?: ProviderConfig[], activeId?: string }
    // POST /api/providers/remove -> { id: string }
    // POST /api/providers/active -> { id: string }
    if (url.pathname.startsWith("/api/providers")) {
      const stub = env.ConvoIndex.get(env.ConvoIndex.idFromName(CONVO_INDEX));

      if (request.method === "GET") {
        const [list, activeId] = await Promise.all([
          stub.listProviders(),
          stub.getSetting("active_provider_id"),
        ]);
        return Response.json({ ok: true, providers: list, activeId });
      }

      if (request.method === "POST") {
        let body: any = {};
        try {
          body = await request.json();
        } catch {
          /* ignore */
        }

        if (url.pathname.endsWith("/remove")) {
          if (!body?.id) return Response.json({ ok: false, error: "missing id" }, { status: 400 });
          const list = await stub.removeProvider(body.id);
          return Response.json({ ok: true, providers: list });
        }

        if (url.pathname.endsWith("/active")) {
          if (!body?.id) return Response.json({ ok: false, error: "missing id" }, { status: 400 });
          await stub.setSetting("active_provider_id", body.id);
          return Response.json({ ok: true, activeId: body.id });
        }

        if (body?.activeId) {
          await stub.setSetting("active_provider_id", body.activeId);
        }

        if (Array.isArray(body?.providers)) {
          const list = await stub.saveAllProviders(body.providers);
          return Response.json({ ok: true, providers: list });
        }

        if (body?.provider) {
          const list = await stub.saveProvider(body.provider);
          return Response.json({ ok: true, providers: list });
        }

        // Single provider payload fallback
        if (body?.id && body?.name && body?.selectedModel) {
          const list = await stub.saveProvider(body as ProviderConfig);
          return Response.json({ ok: true, providers: list });
        }

        const currentList = await stub.listProviders();
        return Response.json({ ok: true, providers: currentList });
      }

      return Response.json({ ok: false, error: "method not allowed" }, { status: 405 });
    }

    // Model list fetching proxy to bypass browser CORS restrictions.
    // POST /api/models/fetch     -> { endpoint, apiKey? } -> { ok: true, models: string[] }
    if (url.pathname === "/api/models/fetch") {
      if (request.method !== "POST") {
        return Response.json({ ok: false, error: "method not allowed" }, { status: 405 });
      }
      try {
        let body: { endpoint?: string; apiKey?: string } = {};
        try {
          body = (await request.json()) as typeof body;
        } catch {
          return Response.json({ ok: false, error: "invalid json body" }, { status: 400 });
        }

        const endpoint = body.endpoint?.trim();
        if (!endpoint) {
          return Response.json({ ok: false, error: "endpoint is required" }, { status: 400 });
        }

        let cleanEndpoint = endpoint
          .replace(/\/+$/, "")
          .replace(/\/(chat\/completions|responses|models)$/, "")
          .replace(/\/+$/, "");
        let targetUrl = `${cleanEndpoint}/models`;

        const headers: Record<string, string> = {
          accept: "application/json",
          "user-agent": "cf-think-agent/1.0",
        };
        if (body.apiKey?.trim()) {
          headers["authorization"] = `Bearer ${body.apiKey.trim()}`;
        }

        const upstreamRes = await fetch(targetUrl, {
          method: "GET",
          headers,
        });

        if (!upstreamRes.ok) {
          const errText = await upstreamRes.text().catch(() => "");
          let parsedJson: any = null;
          try {
            parsedJson = JSON.parse(errText);
          } catch {
            /* ignore */
          }
          return Response.json(
            {
              ok: false,
              statusCode: upstreamRes.status,
              statusText: upstreamRes.statusText,
              error: parsedJson?.error?.message || parsedJson?.message || errText || `Upstream error (${upstreamRes.status}): ${upstreamRes.statusText}`,
              rawResponse: parsedJson || errText || upstreamRes.statusText,
              url: targetUrl,
            },
            { status: upstreamRes.status }
          );
        }

        const rawData = (await upstreamRes.json()) as any;
        let modelList: string[] = [];

        if (Array.isArray(rawData?.data)) {
          modelList = rawData.data
            .map((item: any) => (typeof item === "string" ? item : item?.id || item?.name))
            .filter(Boolean);
        } else if (Array.isArray(rawData?.models)) {
          modelList = rawData.models
            .map((item: any) => (typeof item === "string" ? item : item?.name || item?.id || item?.model))
            .filter(Boolean);
        } else if (Array.isArray(rawData)) {
          modelList = rawData
            .map((item: any) => (typeof item === "string" ? item : item?.id || item?.name))
            .filter(Boolean);
        }

        // Deduplicate and sort
        const uniqueModels = Array.from(new Set(modelList)).sort((a, b) => a.localeCompare(b));

        return Response.json({ ok: true, models: uniqueModels });
      } catch (err: any) {
        return Response.json(
          {
            ok: false,
            statusCode: 500,
            error: err?.message || "Failed to fetch models",
            rawResponse: {
              name: err?.name,
              message: err?.message,
              stack: err?.stack,
            },
          },
          { status: 500 }
        );
      }
    }

    // Workspace File Explorer & Code Preview REST API (backed by Assistant DO SQLite VFS)
    // GET /api/workspace/files?convoId=xyz          -> { ok: true, files: [...] }
    // GET /api/workspace/file?convoId=xyz&path=abc  -> { ok: true, content: "..." }
    // GET /api/workspace/archive?convoId=xyz       -> { ok: true, files: [...] }
    if (url.pathname.startsWith("/api/workspace")) {
      const convoId = url.searchParams.get("convoId");
      if (!convoId) {
        return Response.json({ ok: false, error: "Missing convoId parameter" }, { status: 400 });
      }
      const stub = env.Assistant.get(env.Assistant.idFromName(convoId));

      if (url.pathname === "/api/workspace/files" && request.method === "GET") {
        try {
          const res = await stub.getWorkspaceFiles();
          return Response.json({ ok: true, files: res.files });
        } catch (err: any) {
          return Response.json({ ok: false, error: err.message || String(err) }, { status: 500 });
        }
      }

      if (url.pathname === "/api/workspace/file" && request.method === "GET") {
        const filePath = url.searchParams.get("path") || "";
        try {
          const res = await stub.getWorkspaceFile(filePath);
          return Response.json(res);
        } catch (err: any) {
          return Response.json({ ok: false, error: err.message || String(err) }, { status: 500 });
        }
      }

      if (url.pathname === "/api/workspace/archive" && request.method === "GET") {
        try {
          const res = await stub.getWorkspaceZipArchive();
          return Response.json(res);
        } catch (err: any) {
          return Response.json({ ok: false, error: err.message || String(err) }, { status: 500 });
        }
      }
    }

    // App Settings REST API (backed by ConvoIndex DO SQLite)
    // GET  /api/settings?key=xyz -> { ok: true, key, value }
    // POST /api/settings        -> { key, value } -> { ok: true }
    if (url.pathname.startsWith("/api/settings")) {
      const stub = env.ConvoIndex.get(env.ConvoIndex.idFromName(CONVO_INDEX));
      if (request.method === "GET") {
        const key = url.searchParams.get("key") || "custom_system_prompt";
        const value = await stub.getSetting(key);
        return Response.json({ ok: true, key, value });
      }
      if (request.method === "POST") {
        let body: any = {};
        try {
          body = await request.json();
        } catch {
          /* ignore */
        }
        if (body?.key && typeof body.value === "string") {
          await stub.setSetting(body.key, body.value);
          return Response.json({ ok: true });
        }
        return Response.json({ ok: false, error: "missing key or value" }, { status: 400 });
      }
    }

    // Cloud MCP Servers & Tools REST API (backed by ConvoIndex DO SQLite).
    // GET  /api/mcp/list        -> { ok: true, servers: McpServerConfig[] }
    // POST /api/mcp/save        -> { server: McpServerConfig } -> { ok: true, servers: McpServerConfig[] }
    // POST /api/mcp/remove      -> { id: string } -> { ok: true, servers: McpServerConfig[] }
    // POST /api/mcp/fetch-tools -> { endpoint, authType, bearerToken?, oauthTokens? } -> { ok: true, tools: McpToolDef[] }
    if (url.pathname.startsWith("/api/mcp")) {
      const stub = env.ConvoIndex.get(env.ConvoIndex.idFromName(CONVO_INDEX));

      if (url.pathname === "/api/mcp/list" && request.method === "GET") {
        const servers = await stub.listMcpServers();
        return Response.json({ ok: true, servers });
      }

      if (url.pathname === "/api/mcp/save" && request.method === "POST") {
        let body: any = {};
        try {
          body = await request.json();
        } catch {
          /* ignore */
        }
        if (body?.server) {
          const servers = await stub.saveMcpServer(body.server);
          return Response.json({ ok: true, servers });
        }
        if (Array.isArray(body?.servers)) {
          const servers = await stub.saveAllMcpServers(body.servers);
          return Response.json({ ok: true, servers });
        }
        return Response.json({ ok: false, error: "missing server config" }, { status: 400 });
      }

      if (url.pathname === "/api/mcp/remove" && request.method === "POST") {
        let body: any = {};
        try {
          body = await request.json();
        } catch {
          /* ignore */
        }
        if (!body?.id) return Response.json({ ok: false, error: "missing id" }, { status: 400 });
        const servers = await stub.deleteMcpServer(body.id);
        return Response.json({ ok: true, servers });
      }

      if (url.pathname === "/api/mcp/fetch-tools" && request.method === "POST") {
        try {
          let body: any = {};
          try {
            body = await request.json();
          } catch {
            return Response.json({ ok: false, error: "invalid json body" }, { status: 400 });
          }
          const endpoint = body?.endpoint?.trim();
          if (!endpoint) {
            return Response.json({ ok: false, error: "endpoint is required" }, { status: 400 });
          }

          const bearerToken = body.bearerToken?.trim();

          const tools = await mcpListTools({
            endpoint,
            authType: body.authType || "bearer",
            bearerToken,
            cfAccessClientId: body.cfAccessClientId?.trim(),
            cfAccessClientSecret: body.cfAccessClientSecret?.trim(),
            oauthTokens: body.oauthTokens,
          });

          return Response.json({ ok: true, tools });
        } catch (err: any) {
          return Response.json(
            { ok: false, error: err?.message || "Failed to fetch tools from MCP endpoint" },
            { status: 500 }
          );
        }
      }

      // ---------------- OAuth 2.0 PKCE Handshake Endpoints ----------------
      if (url.pathname === "/api/mcp/oauth/start" && request.method === "POST") {
        try {
          let body: any = {};
          try {
            body = await request.json();
          } catch {
            return Response.json({ ok: false, error: "invalid json body" }, { status: 400 });
          }
          const endpoint = body?.endpoint?.trim();
          if (!endpoint) {
            return Response.json({ ok: false, error: "endpoint is required" }, { status: 400 });
          }

          const serverId = body?.serverId || "mcp-" + generateRandomString(8);
          const serverName = body?.serverName?.trim() || "MCP Server";
          const origin = url.origin;
          const redirectUri = `${origin}/api/mcp/oauth/callback`;

          const discovery = await discoverMcpOAuth(endpoint, redirectUri);
          const { verifier, challenge } = await generatePkcePair();
          const state = generateRandomString(32);
          const clientId = body?.clientId?.trim() || discovery.clientId;

          if (!clientId) {
            const extraHint = discovery.registrationError
              ? `${discovery.registrationError}. Note: Cloudflare Access requires '${redirectUri}' to be added in Zero Trust Dashboard -> Settings -> Authentication -> Allowed OAuth redirect URIs.`
              : `Unable to obtain client_id from ${discovery.authorizationServer}. Please specify a Custom Client ID under Advanced settings or configure DCR Allowed Redirect URIs.`;
            return Response.json(
              { ok: false, error: extraHint, discovery },
              { status: 400 }
            );
          }

          await stub.saveOAuthSession({
            state,
            serverId,
            serverName,
            endpoint,
            tokenEndpoint: discovery.tokenEndpoint,
            clientId,
            clientSecret: body?.clientSecret?.trim(),
            redirectUri,
            codeVerifier: verifier,
          });

          const authUrl = new URL(discovery.authorizationEndpoint);
          authUrl.searchParams.set("response_type", "code");
          authUrl.searchParams.set("client_id", clientId);
          authUrl.searchParams.set("redirect_uri", redirectUri);
          authUrl.searchParams.set("state", state);
          authUrl.searchParams.set("code_challenge", challenge);
          authUrl.searchParams.set("code_challenge_method", "S256");
          const resourceUri = discovery.resource || endpoint;
          authUrl.searchParams.set("resource", resourceUri);
          if (discovery.scopesSupported && discovery.scopesSupported.length > 0) {
            authUrl.searchParams.set("scope", discovery.scopesSupported.join(" "));
          }

          return Response.json({
            ok: true,
            authUrl: authUrl.toString(),
            state,
            serverId,
            discovery,
          });
        } catch (err: any) {
          return Response.json(
            { ok: false, error: err?.message || "Failed to initiate OAuth authorization" },
            { status: 500 }
          );
        }
      }

      if (url.pathname === "/api/mcp/oauth/callback" && request.method === "GET") {
        const error = url.searchParams.get("error");
        const errorDesc = url.searchParams.get("error_description");
        const state = url.searchParams.get("state");
        const code = url.searchParams.get("code");

        if (error) {
          return new Response(
            renderOAuthResultHtml({
              ok: false,
              title: "Authorization Failed",
              message: errorDesc || error,
            }),
            { headers: { "content-type": "text/html; charset=utf-8" } }
          );
        }

        if (!state || !code) {
          return new Response(
            renderOAuthResultHtml({
              ok: false,
              title: "Missing Parameters",
              message: "State or authorization code was missing in callback.",
            }),
            { headers: { "content-type": "text/html; charset=utf-8" } }
          );
        }

        const session = await stub.consumeOAuthSession(state);
        if (!session) {
          return new Response(
            renderOAuthResultHtml({
              ok: false,
              title: "Invalid Session",
              message: "OAuth session expired or is invalid. Please try connecting again.",
            }),
            { headers: { "content-type": "text/html; charset=utf-8" } }
          );
        }

        try {
          // 1. Exchange code for tokens
          const tokens = await exchangeOAuthCode({
            tokenEndpoint: session.tokenEndpoint,
            clientId: session.clientId,
            clientSecret: session.clientSecret,
            code,
            redirectUri: session.redirectUri,
            codeVerifier: session.codeVerifier,
            resource: session.endpoint,
          });

          // 2. Fetch exposed tools
          let tools: any[] = [];
          try {
            tools = await mcpListTools({
              endpoint: session.endpoint,
              authType: "oauth",
              oauthTokens: {
                accessToken: tokens.accessToken,
                refreshToken: tokens.refreshToken,
              },
            });
          } catch {
            /* tools can be refreshed later */
          }

          // 3. Save MCP server into DO SQLite
          const expiresAt = tokens.expiresIn ? Date.now() + tokens.expiresIn * 1000 : undefined;
          await stub.saveMcpServer({
            id: session.serverId,
            name: session.serverName,
            endpoint: session.endpoint,
            authType: "oauth",
            oauthClientId: session.clientId,
            oauthClientSecret: session.clientSecret,
            oauthTokens: {
              accessToken: tokens.accessToken,
              refreshToken: tokens.refreshToken,
              expiresAt,
            },
            enabled: true,
            cachedTools: tools,
            updatedAt: Date.now(),
          });

          return new Response(
            renderOAuthResultHtml({
              ok: true,
              title: "Connected Successfully",
              message: `Connected to ${session.serverName}. Discovered ${tools.length} tool${tools.length === 1 ? "" : "s"}.`,
              serverId: session.serverId,
            }),
            { headers: { "content-type": "text/html; charset=utf-8" } }
          );
        } catch (err: any) {
          return new Response(
            renderOAuthResultHtml({
              ok: false,
              title: "Token Exchange Failed",
              message: err?.message || String(err),
            }),
            { headers: { "content-type": "text/html; charset=utf-8" } }
          );
        }
      }
    }

    return (
      (await routeAgentRequest(request, env)) ??
      new Response("cf-think-agent", {
        headers: { "content-type": "text/plain" },
      })
    );
  },
} satisfies ExportedHandler<Env>;
