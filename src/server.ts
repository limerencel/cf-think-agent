import { routeAgentRequest } from "agents";
import { Assistant } from "./agent";
import { ConvoIndex } from "./convo-index";

export { Assistant, ConvoIndex };

const CONVO_INDEX = "convo-index";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Cloud conversation-list REST API (backed by ConvoIndex DO).
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

    return (
      (await routeAgentRequest(request, env)) ??
      new Response("cf-think-agent", {
        headers: { "content-type": "text/plain" },
      })
    );
  },
} satisfies ExportedHandler<Env>;
