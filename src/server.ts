import { routeAgentRequest } from "agents";
import { Assistant } from "./agent";

export { Assistant };

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return (
      (await routeAgentRequest(request, env)) ??
      new Response("cf-think-agent", {
        headers: { "content-type": "text/plain" },
      })
    );
  },
} satisfies ExportedHandler<Env>;
