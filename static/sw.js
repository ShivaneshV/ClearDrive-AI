// ClearDrive AI Service Worker - Auto-Updating PWA Engine (v4)
const CACHE_NAME = 'cleardrive-pwa-v4';
const STATIC_ASSETS = [
  '/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

// Instantly install and skip waiting so new updates take effect immediately
self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS).catch(() => {});
    })
  );
});

// Purge all stale caches on activation and claim all open clients
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[SW] Purged old cache:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim()).then(() => {
      // Force all open client tabs and installed PWAs to immediately navigate/reload to fresh code
      return self.clients.matchAll({ type: 'window' }).then((clients) => {
        clients.forEach((client) => {
          if ('navigate' in client) {
            client.navigate(client.url);
          }
        });
      });
    })
  );
});


self.addEventListener('fetch', (event) => {
  const url = event.request.url;

  // 1. Live video feed and telemetry APIs always bypass cache
  if (url.includes('/video_feed') || url.includes('/api/')) {
    return;
  }

  // 2. NETWORK-FIRST for HTML navigation & page reloads
  // Guarantees mobile users ALWAYS see the latest UI/UX automatically without manual cache clearing
  if (event.request.mode === 'navigate' || event.request.headers.get('accept')?.includes('text/html') || url.endsWith('/')) {
    event.respondWith(
      fetch(event.request)
        .then((networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            const responseClone = networkResponse.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, responseClone);
            });
          }
          return networkResponse;
        })
        .catch(() => {
          // Offline fallback only if completely disconnected
          return caches.match(event.request);
        })
    );
    return;
  }

  // 3. Stale-While-Revalidate for static assets & icons
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      const fetchPromise = fetch(event.request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200) {
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
        }
        return networkResponse;
      }).catch(() => {});
      return cachedResponse || fetchPromise;
    })
  );
});
