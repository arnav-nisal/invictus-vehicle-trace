import { defineConfig, Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';

function getFinalClipsDirectory(): { dirPath: string; dirName: string } {
  const candidate1 = path.resolve(__dirname, '../data/Final');
  if (fs.existsSync(candidate1)) {
    return { dirPath: candidate1, dirName: 'data/Final' };
  }
  const candidate2 = path.resolve(__dirname, '../data/final');
  if (fs.existsSync(candidate2)) {
    return { dirPath: candidate2, dirName: 'data/final' };
  }
  return { dirPath: candidate1, dirName: 'data/Final' };
}

const MIME_MAP: Record<string, string> = {
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.avi': 'video/x-msvideo',
  '.mkv': 'video/x-matroska',
  '.json': 'application/json',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
};

function localClipsPlugin(): Plugin {
  return {
    name: 'invictus-local-clips-plugin',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next();

        const urlPath = req.url.split('?')[0];

        // 1. GET /local-clips
        if (req.method === 'GET' && urlPath === '/local-clips') {
          try {
            const { dirPath, dirName } = getFinalClipsDirectory();
            if (!fs.existsSync(dirPath)) {
              res.writeHead(200, { 'Content-Type': 'application/json' });
              return res.end(JSON.stringify({ directory: dirName, clips: [] }));
            }

            const videoExts = new Set(['.mp4', '.mov', '.avi', '.mkv']);
            const entries = fs.readdirSync(dirPath, { withFileTypes: true });

            const validFiles = entries
              .filter((e) => e.isFile() && videoExts.has(path.extname(e.name).toLowerCase()))
              .map((e) => e.name);

            // Natural filename sort (file_01, file_02, etc.)
            validFiles.sort((a, b) =>
              a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
            );

            const clips = validFiles.map((filename) => {
              const fullPath = path.join(dirPath, filename);
              const stat = fs.statSync(fullPath);
              return {
                id: filename,
                filename,
                size_bytes: stat.size,
                video_url: `/local-clips/stream/${encodeURIComponent(filename)}`,
              };
            });

            res.writeHead(200, {
              'Content-Type': 'application/json',
              'Cache-Control': 'no-cache',
            });
            return res.end(JSON.stringify({ directory: dirName, clips }));
          } catch (err) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: (err as Error).message }));
          }
        }

        // 2. GET /local-clips/stream/:filename
        if (req.method === 'GET' && urlPath.startsWith('/local-clips/stream/')) {
          const rawParam = urlPath.slice('/local-clips/stream/'.length);
          let filename = '';
          try {
            filename = decodeURIComponent(rawParam);
          } catch {
            filename = rawParam;
          }

          // Path traversal protection
          const safeName = path.basename(filename);
          if (
            !filename ||
            safeName !== filename ||
            filename.includes('..') ||
            filename.includes('/') ||
            filename.includes('\\')
          ) {
            res.writeHead(400, { 'Content-Type': 'text/plain' });
            return res.end('Invalid filename parameter');
          }

          const { dirPath } = getFinalClipsDirectory();
          const filePath = path.join(dirPath, safeName);

          if (!fs.existsSync(filePath)) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            return res.end(`Clip "${safeName}" not found`);
          }

          const ext = path.extname(safeName).toLowerCase();
          const mimeType = MIME_MAP[ext] || 'application/octet-stream';
          const stat = fs.statSync(filePath);
          const fileSize = stat.size;
          const range = req.headers.range;

          if (range) {
            const parts = range.replace(/bytes=/, '').split('-');
            const start = parseInt(parts[0], 10);
            const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;

            if (isNaN(start) || start >= fileSize || (parts[1] && isNaN(end)) || start > end) {
              res.writeHead(416, {
                'Content-Range': `bytes */${fileSize}`,
              });
              return res.end();
            }

            const chunkSize = end - start + 1;
            res.writeHead(206, {
              'Content-Range': `bytes ${start}-${end}/${fileSize}`,
              'Accept-Ranges': 'bytes',
              'Content-Length': chunkSize,
              'Content-Type': mimeType,
            });

            const stream = fs.createReadStream(filePath, { start, end });
            stream.on('error', () => {
              if (!res.headersSent) {
                res.writeHead(500);
              }
              res.end();
            });
            return stream.pipe(res);
          } else {
            res.writeHead(200, {
              'Content-Length': fileSize,
              'Content-Type': mimeType,
              'Accept-Ranges': 'bytes',
            });

            const stream = fs.createReadStream(filePath);
            stream.on('error', () => {
              if (!res.headersSent) {
                res.writeHead(500);
              }
              res.end();
            });
            return stream.pipe(res);
          }
        }

        // 3. Dev fallback for /outputs/ if backend is offline
        if (req.method === 'GET' && urlPath.startsWith('/outputs/')) {
          const relPath = urlPath.slice('/outputs/'.length);
          const candidatePath = path.resolve(__dirname, '../outputs', relPath);
          if (fs.existsSync(candidatePath) && fs.statSync(candidatePath).isFile()) {
            const ext = path.extname(candidatePath).toLowerCase();
            const mimeType = MIME_MAP[ext] || 'application/octet-stream';
            res.writeHead(200, { 'Content-Type': mimeType });
            return fs.createReadStream(candidatePath).pipe(res);
          }
        }

        return next();
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), localClipsPlugin()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/outputs': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/media': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
