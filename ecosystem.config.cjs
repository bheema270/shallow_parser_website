// PM2 config for Telugu Shallow Parser (Root directory runner)
// Usage: pm2 start ecosystem.config.cjs
"use strict";

module.exports = {
  apps: [
    // ── 1. Python / FastAPI backend ─────────────────────────────────────
    {
      name: "shallow-parser-api",
      cwd: "./screenshot-perfect-pixel-5695-main/backend",
      interpreter: "./venv/bin/python3",
      script: "main.py",
      env: {
        PORT: "8000",
        CORS_ORIGINS: "http://localhost:3000,http://127.0.0.1:3000",
      },
      autorestart: true,
      watch: false,
      max_memory_restart: "4G",
      out_file: "../logs/api-out.log",
      error_file: "../logs/api-err.log",
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },

    // ── 2. Node.js / Nitro frontend ─────────────────────────────────────
    {
      name: "shallow-parser-web",
      cwd: "./screenshot-perfect-pixel-5695-main/.output/server",
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
