interface Env {
  TWIN_API_URL?: string;
}

/**
 * Cloudflare Pages Function proxying `/twin-api/*` to the FastAPI backend
 * configured via `TWIN_API_URL` in Cloudflare Pages environment variables.
 */
export const onRequest = async (context: {
  request: Request;
  env: Env;
  params: { path?: string[] | string };
}): Promise<Response> => {
  const backendBase = (context.env.TWIN_API_URL || '').replace(/\/+$/, '');
  if (!backendBase) {
    return new Response(
      JSON.stringify({
        error: 'TWIN_API_URL is not configured in Cloudflare Pages environment variables.',
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    );
  }

  const incomingUrl = new URL(context.request.url);
  const pathSegments = Array.isArray(context.params.path)
    ? context.params.path.join('/')
    : context.params.path || '';
  const targetUrl = `${backendBase}/${pathSegments}${incomingUrl.search}`;

  const headers = new Headers(context.request.headers);
  headers.delete('host');

  const init: RequestInit = {
    method: context.request.method,
    headers,
    redirect: 'follow',
  };

  if (context.request.method !== 'GET' && context.request.method !== 'HEAD') {
    init.body = await context.request.arrayBuffer();
  }

  const upstream = await fetch(targetUrl, init);
  const respHeaders = new Headers(upstream.headers);
  respHeaders.set('Access-Control-Allow-Origin', '*');

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: respHeaders,
  });
};
