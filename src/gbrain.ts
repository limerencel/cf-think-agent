/** Minimal Streamable-HTTP MCP client for GBrain. */

export type GbrainRpcResult = {
  ok: boolean;
  tool: string;
  text?: string;
  error?: string;
};

function parseSseOrJson(raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed.startsWith("{")) return JSON.parse(trimmed);
  let last: unknown = null;
  for (const line of trimmed.split("\n")) {
    if (line.startsWith("data: ")) {
      last = JSON.parse(line.slice(6));
    }
  }
  return last;
}

export async function gbrainCall(
  url: string,
  token: string,
  tool: string,
  args: Record<string, unknown> = {},
): Promise<GbrainRpcResult> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    Accept: "application/json, text/event-stream",
    "User-Agent": "cf-think-agent/0.1",
  };

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
    return { ok: false, tool, error: `initialize HTTP ${initRes.status}` };
  }
  const session =
    initRes.headers.get("mcp-session-id") || initRes.headers.get("Mcp-Session-Id");
  if (session) headers["mcp-session-id"] = session;

  const callRes = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { name: tool, arguments: args },
    }),
  });
  const raw = await callRes.text();
  if (!callRes.ok) {
    return { ok: false, tool, error: `tools/call HTTP ${callRes.status}: ${raw.slice(0, 300)}` };
  }

  try {
    const parsed = parseSseOrJson(raw) as {
      error?: { message?: string };
      result?: { content?: Array<{ type?: string; text?: string }>; isError?: boolean };
    };
    if (parsed?.error?.message) {
      return { ok: false, tool, error: parsed.error.message };
    }
    const texts = (parsed?.result?.content || [])
      .filter((c) => c.type === "text" && c.text)
      .map((c) => c.text as string);
    const text = texts.join("\n") || raw.slice(0, 4000);
    return { ok: !parsed?.result?.isError, tool, text };
  } catch (err) {
    return { ok: false, tool, error: String(err), text: raw.slice(0, 500) };
  }
}
