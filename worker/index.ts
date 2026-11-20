interface Env {
  ASSETS: {
    fetch: (request: Request) => Promise<Response>;
  };
  TWIN_API_URL?: string;
}

/**
 * Cloudflare Worker entrypoint for CardioTwin:
 * - Proxies `/twin-api/*` to `TWIN_API_URL` (FastAPI backend)
 * - Delegates all other requests to Cloudflare Static Assets (`./dist`)
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname.startsWith('/twin-api/')) {
      const backendBase = (env.TWIN_API_URL || '').replace(/\/+$/, '');
      if (!backendBase) {
        return new Response(
          JSON.stringify({
            error: 'TWIN_API_URL is not configured in Cloudflare Worker environment variables.',
          }),
          { status: 500, headers: { 'Content-Type': 'application/json' } },
        );
      }

      const upstreamPath = url.pathname.replace(/^\/twin-api/, '');
      const targetUrl = `${backendBase}${upstreamPath}${url.search}`;

      const headers = new Headers(request.headers);
      headers.delete('host');

      const init: RequestInit = {
        method: request.method,
        headers,
        redirect: 'follow',
      };

      if (request.method !== 'GET' && request.method !== 'HEAD') {
        init.body = await request.arrayBuffer();
      }

      const upstream = await fetch(targetUrl, init);
      const respHeaders = new Headers(upstream.headers);
      respHeaders.set('Access-Control-Allow-Origin', '*');

      return new Response(upstream.body, {
        status: upstream.status,
        statusText: upstream.statusText,
        headers: respHeaders,
      });
    }

    return env.ASSETS.fetch(request);
  },
};
