#!/usr/bin/env node
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer, request as httpRequest } from 'node:http';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../dist');
const host = process.env.PREVIEW_HOST || '127.0.0.1';
const port = Number(process.env.PREVIEW_PORT || 4173);
const backend = new URL(process.env.ASK_AI_BACKEND_URL || 'http://127.0.0.1:8787');
if (backend.protocol !== 'http:') throw new Error('ASK_AI_BACKEND_URL must use HTTP');
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PREVIEW_PORT is invalid');

const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
  '.woff2': 'font/woff2',
};

function proxyApi(req, res, url) {
  const path = url.pathname.slice('/api'.length) || '/';
  const target = new URL(backend);
  target.pathname = path;
  target.search = url.search;
  const headers = { ...req.headers, host: backend.host };
  delete headers.origin;
  delete headers.referer;
  delete headers.connection;
  const upstream = httpRequest(target, { method: req.method, headers }, response => {
    const responseHeaders = { ...response.headers };
    delete responseHeaders['access-control-allow-origin'];
    res.writeHead(response.statusCode || 502, responseHeaders);
    response.pipe(res);
  });
  upstream.on('error', () => {
    if (res.headersSent) return res.destroy();
    res.writeHead(502, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
    res.end(JSON.stringify({ error: { code: 'backend_unavailable', message: 'Ask AI backend is unavailable' } }));
  });
  res.on('close', () => upstream.destroy());
  req.pipe(upstream);
}

async function serveFile(req, res, url) {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, { allow: 'GET, HEAD' });
    return res.end();
  }
  let path;
  try { path = decodeURIComponent(url.pathname); }
  catch { res.writeHead(400); return res.end(); }
  const filename = resolve(root, '.' + (path === '/' ? '/index.html' : path));
  if (filename !== root && !filename.startsWith(root + sep)) {
    res.writeHead(403);
    return res.end();
  }
  let info;
  try { info = await stat(filename); }
  catch { res.writeHead(404); return res.end(); }
  if (!info.isFile()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, {
    'content-type': contentTypes[extname(filename)] || 'application/octet-stream',
    'content-length': info.size,
    'cache-control': 'no-store',
    'x-content-type-options': 'nosniff',
  });
  if (req.method === 'HEAD') return res.end();
  createReadStream(filename).pipe(res);
}

const server = createServer((req, res) => {
  let url;
  try { url = new URL(req.url, `http://${req.headers.host || 'localhost'}`); }
  catch { res.writeHead(400); return res.end(); }
  if (url.pathname === '/api' || url.pathname.startsWith('/api/')) return proxyApi(req, res, url);
  void serveFile(req, res, url).catch(error => {
    console.error('[preview] File error:', error?.message || String(error));
    if (!res.headersSent) res.writeHead(500);
    res.end();
  });
});

server.listen(port, host, () => {
  console.log(`Model Zoo preview: http://${host}:${port}/ (Ask AI via /api)`);
});
