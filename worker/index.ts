interface Env {
  ASSETS: {
    fetch: (request: Request) => Promise<Response>;
  };
  TWIN_API_URL?: string;
}

/**
 * Cloudflare Worker entrypoint for CardioTwin:
 * - Proxies `/twin-api/*` to `TWIN_API_URL` (FastAPI backend)
 * - Normalizes upstream HTML 5xx pages into clean JSON errors
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
          { status: 503, headers: { 'Content-Type': 'application/json' } },
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

      try {
        const upstream = await fetch(targetUrl, init);
        const contentType = upstream.headers.get('content-type') || '';

        if (upstream.status >= 500 && !contentType.includes('application/json')) {
          return new Response(
            JSON.stringify({
              error: `Backend returned HTTP ${upstream.status}`,
              upstream_url: backendBase,
            }),
            {
              status: upstream.status,
              headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
              },
            },
          );
        }

        const respHeaders = new Headers(upstream.headers);
        respHeaders.set('Access-Control-Allow-Origin', '*');

        return new Response(upstream.body, {
          status: upstream.status,
          statusText: upstream.statusText,
          headers: respHeaders,
        });
      } catch (err: any) {
        return new Response(
          JSON.stringify({
            error: 'Failed to reach backend service',
            detail: err?.message || 'Network error',
          }),
          {
            status: 502,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*',
            },
          },
        );
      }
    }

    return env.ASSETS.fetch(request);
  },
};
