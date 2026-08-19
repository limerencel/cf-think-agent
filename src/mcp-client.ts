/**
 * Universal Streamable-HTTP & JSON-RPC MCP Client for Cloudflare Workers.
 * Supports None, Bearer Token, and OAuth 2.0 authentications.
 */
import type { McpServerConfig, McpToolDef } from "./mcp-types";

export type McpRpcResult = {
  ok: boolean;
  tool: string;
  text?: string;
  error?: string;
};

function parseSseOrJson(raw: string): any {
  const trimmed = raw.trim();
  if (trimmed.startsWith("{")) {
    try {
      return JSON.parse(trimmed);
    } catch {
      /* fallback to line parsing */
    }
  }
  let last: any = null;
  for (const line of trimmed.split("\n")) {
    const clean = line.trim();
    if (clean.startsWith("data: ")) {
      try {
        last = JSON.parse(clean.slice(6));
      } catch {
        /* ignore invalid line */
      }
    }
  }
  return last || { raw: trimmed };
}

export function buildMcpHeaders(
  config: Pick<McpServerConfig, "authType" | "bearerToken" | "oauthTokens">
): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json, text/event-stream",
    "User-Agent": "cf-think-agent/0.1.0",
  };

  const token =
    config.authType === "oauth"
      ? config.oauthTokens?.accessToken || config.bearerToken?.trim()
      : config.authType === "bearer"
      ? config.bearerToken?.trim()
      : undefined;

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  return headers;
}

/* ---------------- OAuth 2.0 PKCE & Discovery Helpers ---------------- */

export function base64UrlEncode(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let str = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    str += String.fromCharCode(bytes[i]);
  }
  return btoa(str).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function generateRandomString(length = 43): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => chars[b % chars.length]).join("");
}

export async function generatePkcePair(): Promise<{ verifier: string; challenge: string }> {
  const verifier = generateRandomString(64);
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const hash = await crypto.subtle.digest("SHA-256", data);
  const challenge = base64UrlEncode(hash);
  return { verifier, challenge };
}

export interface McpOAuthDiscoveryResult {
  authorizationServer: string;
  authorizationEndpoint: string;
  tokenEndpoint: string;
  registrationEndpoint?: string;
  scopesSupported: string[];
  clientId?: string;
}

export async function discoverMcpOAuth(
  endpoint: string,
  redirectUri: string
): Promise<McpOAuthDiscoveryResult> {
  const urlObj = new URL(endpoint);
  const origin = urlObj.origin;

  let authServer = origin;
  let scopes: string[] = [];

  // 1. Try protected resource metadata endpoint (RFC 9728)
  try {
    const resourceRes = await fetch(`${origin}/.well-known/oauth-protected-resource/mcp`);
    if (resourceRes.ok) {
      const data: any = await resourceRes.json();
      if (Array.isArray(data.authorization_servers) && data.authorization_servers[0]) {
        authServer = data.authorization_servers[0];
      }
      if (Array.isArray(data.scopes_supported)) {
        scopes = data.scopes_supported;
      }
    }
  } catch {
    /* fallback to authServer = origin */
  }

  // 2. Fetch authorization server metadata (RFC 8414)
  let authEndpoint = `${authServer}/authorize`;
  let tokenEndpoint = `${authServer}/oauth/token`;
  let registrationEndpoint: string | undefined = undefined;

  try {
    const serverRes = await fetch(`${authServer}/.well-known/oauth-authorization-server`);
    if (serverRes.ok) {
      const serverData: any = await serverRes.json();
      if (serverData.authorization_endpoint) authEndpoint = serverData.authorization_endpoint;
      if (serverData.token_endpoint) tokenEndpoint = serverData.token_endpoint;
      if (serverData.registration_endpoint) registrationEndpoint = serverData.registration_endpoint;
      if (Array.isArray(serverData.scopes_supported) && scopes.length === 0) {
        scopes = serverData.scopes_supported;
      }
    } else {
      const oidcRes = await fetch(`${authServer}/.well-known/openid-configuration`);
      if (oidcRes.ok) {
        const oidcData: any = await oidcRes.json();
        if (oidcData.authorization_endpoint) authEndpoint = oidcData.authorization_endpoint;
        if (oidcData.token_endpoint) tokenEndpoint = oidcData.token_endpoint;
        if (oidcData.registration_endpoint) registrationEndpoint = oidcData.registration_endpoint;
        if (Array.isArray(oidcData.scopes_supported) && scopes.length === 0) {
          scopes = oidcData.scopes_supported;
        }
      }
    }
  } catch {
    /* use standard defaults */
  }

  // 3. Dynamic client registration (RFC 7591) if supported
  let clientId: string | undefined = undefined;
  if (registrationEndpoint) {
    try {
      const regRes = await fetch(registrationEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_name: "Think Agent",
          redirect_uris: [redirectUri],
          grant_types: ["authorization_code", "refresh_token"],
          response_types: ["code"],
          token_endpoint_auth_method: "none",
        }),
      });
      if (regRes.ok) {
        const regData: any = await regRes.json();
        if (regData.client_id) {
          clientId = regData.client_id;
        }
      }
    } catch {
      /* ignore registration failure */
    }
  }

  return {
    authorizationServer: authServer,
    authorizationEndpoint: authEndpoint,
    tokenEndpoint,
    registrationEndpoint,
    scopesSupported: scopes,
    clientId,
  };
}

export async function exchangeOAuthCode(params: {
  tokenEndpoint: string;
  clientId: string;
  clientSecret?: string;
  code: string;
  redirectUri: string;
  codeVerifier: string;
}): Promise<{
  accessToken: string;
  refreshToken?: string;
  expiresIn?: number;
  scope?: string;
}> {
  const body = new URLSearchParams();
  body.set("grant_type", "authorization_code");
  body.set("client_id", params.clientId);
  if (params.clientSecret) body.set("client_secret", params.clientSecret);
  body.set("code", params.code);
  body.set("redirect_uri", params.redirectUri);
  body.set("code_verifier", params.codeVerifier);

  const res = await fetch(params.tokenEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Accept: "application/json",
    },
    body: body.toString(),
  });

  const raw = await res.text();
  let data: any = {};
  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(`Token exchange failed (HTTP ${res.status}): ${raw.slice(0, 200)}`);
  }

  if (!res.ok || data.error) {
    throw new Error(data.error_description || data.error || `Token exchange error (HTTP ${res.status})`);
  }

  const accessToken = data.access_token || data.accessToken;
  if (!accessToken) {
    throw new Error("Token endpoint did not return access_token");
  }

  return {
    accessToken,
    refreshToken: data.refresh_token || data.refreshToken,
    expiresIn: data.expires_in || data.expiresIn,
    scope: data.scope,
  };
}

export async function refreshOAuthToken(params: {
  tokenEndpoint: string;
  clientId: string;
  clientSecret?: string;
  refreshToken: string;
}): Promise<{
  accessToken: string;
  refreshToken?: string;
  expiresIn?: number;
}> {
  const body = new URLSearchParams();
  body.set("grant_type", "refresh_token");
  body.set("client_id", params.clientId);
  if (params.clientSecret) body.set("client_secret", params.clientSecret);
  body.set("refresh_token", params.refreshToken);

  const res = await fetch(params.tokenEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Accept: "application/json",
    },
    body: body.toString(),
  });

  const data: any = await res.json().catch(() => ({}));
  if (!res.ok || !data.access_token) {
    throw new Error(data.error_description || data.error || "Failed to refresh OAuth token");
  }

  return {
    accessToken: data.access_token,
    refreshToken: data.refresh_token || params.refreshToken,
    expiresIn: data.expires_in,
  };
}

/**
 * Probes an MCP endpoint and retrieves the list of exposed tools.
 */
export async function mcpListTools(
  config: Pick<McpServerConfig, "endpoint" | "authType" | "bearerToken" | "oauthTokens">
): Promise<McpToolDef[]> {
  const url = config.endpoint.trim();
  if (!url) throw new Error("Endpoint URL is required");

  const headers = buildMcpHeaders(config);

  // 1. Initialize Handshake
  const initRes = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-03-26",
        capabilities: {},
        clientInfo: { name: "cf-think-agent", version: "0.1.0" },
      },
    }),
  });

  if (!initRes.ok) {
    const errText = await initRes.text().catch(() => "");
    throw new Error(
      `MCP handshake failed (HTTP ${initRes.status}): ${errText.slice(0, 250) || initRes.statusText}`
    );
  }

  const session =
    initRes.headers.get("mcp-session-id") || initRes.headers.get("Mcp-Session-Id");
  if (session) headers["mcp-session-id"] = session;

  // 2. Query tools/list
  const listRes = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    }),
  });

  const raw = await listRes.text();
  if (!listRes.ok) {
    throw new Error(
      `tools/list failed (HTTP ${listRes.status}): ${raw.slice(0, 250) || listRes.statusText}`
    );
  }

  const parsed = parseSseOrJson(raw);
  if (parsed?.error?.message) {
    throw new Error(`MCP Error: ${parsed.error.message}`);
  }

  const toolsArray = parsed?.result?.tools || parsed?.tools || [];
  if (!Array.isArray(toolsArray)) {
    return [];
  }

  return toolsArray.map((t: any) => ({
    name: String(t?.name || "unnamed_tool"),
    description: t?.description ? String(t.description) : undefined,
    inputSchema: t?.inputSchema && typeof t.inputSchema === "object" ? t.inputSchema : undefined,
  }));
}

/**
 * Executes a tool call against an MCP endpoint.
 */
export async function mcpCallTool(
  config: Pick<McpServerConfig, "endpoint" | "authType" | "bearerToken" | "oauthTokens">,
  toolName: string,
  args: Record<string, unknown> = {}
): Promise<McpRpcResult> {
  const url = config.endpoint.trim();
  const headers = buildMcpHeaders(config);

  try {
    // 1. Initialize
    const initRes = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2025-03-26",
          capabilities: {},
          clientInfo: { name: "cf-think-agent", version: "0.1.0" },
        },
      }),
    });

    if (!initRes.ok) {
      return { ok: false, tool: toolName, error: `initialize HTTP ${initRes.status}` };
    }

    const session =
      initRes.headers.get("mcp-session-id") || initRes.headers.get("Mcp-Session-Id");
    if (session) headers["mcp-session-id"] = session;

    // 2. Call tool
    const callRes = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: { name: toolName, arguments: args },
      }),
    });

    const raw = await callRes.text();
    if (!callRes.ok) {
      return {
        ok: false,
        tool: toolName,
        error: `tools/call HTTP ${callRes.status}: ${raw.slice(0, 300)}`,
      };
    }

    const parsed = parseSseOrJson(raw);
    if (parsed?.error?.message) {
      return { ok: false, tool: toolName, error: parsed.error.message };
    }

    const texts = (parsed?.result?.content || [])
      .filter((c: any) => c?.type === "text" && c?.text)
      .map((c: any) => c.text as string);

    const text = texts.join("\n") || raw.slice(0, 4000);
    return { ok: !parsed?.result?.isError, tool: toolName, text };
  } catch (err: any) {
    return { ok: false, tool: toolName, error: err?.message || String(err) };
  }
}
