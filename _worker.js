export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const backend = "https://disks-rugby-tips-friend.trycloudflare.com";
    const targetUrl = new URL(url.pathname + url.search, backend);

    const newHeaders = new Headers(request.headers);
    newHeaders.set("Host", new URL(backend).host);
    newHeaders.set("X-Forwarded-Host", url.host);

    const reqInit = {
      method: request.method,
      headers: newHeaders,
      redirect: "follow"
    };

    if (request.method !== "GET" && request.method !== "HEAD" && request.body) {
      reqInit.body = request.body;
      reqInit.duplex = "half";
    }

    return fetch(targetUrl.toString(), reqInit);
  }
};
