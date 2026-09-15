export default {
  async fetch(request, env, ctx) {
    let backend = "https://footage-inputs-decrease-smart.trycloudflare.com";
    try {
      const res = await fetch("https://raw.githubusercontent.com/ShivaneshV/ClearDrive-AI/main/tunnel_url.txt?t=" + Date.now(), {
        headers: { "User-Agent": "ClearDriveWorker/1.0" },
        cf: { cacheTtl: 0 }
      });
      if (res.ok) {
        const text = (await res.text()).trim();
        if (text.startsWith("https://")) backend = text;
      }
    } catch (e) {}

    const url = new URL(request.url);
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
