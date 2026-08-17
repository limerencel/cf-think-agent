import { routeAgentRequest } from "agents";
import { Assistant } from "./agent";
import { ConvoIndex, type ProviderConfig } from "./convo-index";

export { Assistant, ConvoIndex };

const CONVO_INDEX = "convo-index";

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

        let targetUrl = endpoint.replace(/\/+$/, "");
        if (!targetUrl.endsWith("/models")) {
          targetUrl = `${targetUrl}/models`;
        }

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
          return Response.json(
            {
              ok: false,
              error: `Upstream error (${upstreamRes.status}): ${errText.slice(0, 200) || upstreamRes.statusText}`,
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
        return Response.json({ ok: false, error: err?.message || "Failed to fetch models" }, { status: 500 });
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
