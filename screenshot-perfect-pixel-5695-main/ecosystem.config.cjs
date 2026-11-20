// PM2 config for Telugu Shallow Parser
// Usage: pm2 start ecosystem.config.cjs
//
// Two processes:
//   shallow-parser-api  — FastAPI backend  (port 8000)
//   shallow-parser-web  — Node.js frontend (port 3000)
//
// If port 3000 or 8000 is already in use on the server,
// change the port numbers below (and rebuild the frontend
// with the new VITE_API_BASE if you change the API port).

"use strict";

module.exports = {
  apps: [
    // ── 1. Python / FastAPI backend ─────────────────────────────────────
    {
      name: "shallow-parser-api",
      cwd: "./backend",
      interpreter: "./venv/bin/python3",   // uses virtualenv python
      script: "main.py",

      // Environment variables the backend reads
      env: {
        PORT: "8000",
        // Allow the browser (served from port 3000) to call the API.
        // Replace SERVER_IP with the actual server IP or domain when deploying.
        // e.g. "http://192.168.1.100:3000,http://localhost:3000"
        CORS_ORIGINS: "http://localhost:3000,http://127.0.0.1:3000",
      },

      // Keep the process alive; restart if it crashes
      autorestart: true,
      watch: false,
      max_memory_restart: "4G",

      // Log files
      out_file: "../logs/api-out.log",
      error_file: "../logs/api-err.log",
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },

    // ── 2. Node.js / Nitro frontend ─────────────────────────────────────
    {
      name: "shallow-parser-web",
      cwd: "./.output/server",             // built by NITRO_PRESET=node-server
      interpreter: "node",
      script: "index.mjs",

      env: {
        PORT: "3000",
        HOST: "0.0.0.0",
        NODE_ENV: "production",
      },

      autorestart: true,
      watch: false,
      max_memory_restart: "1G",

      out_file: "../logs/web-out.log",
      error_file: "../logs/web-err.log",
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};
