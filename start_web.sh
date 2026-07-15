#!/bin/bash
# PIPELINE_BACKEND defaults to "direct" (auto-imports pipeline); falls back to "proxy"
# if the pipeline module is not importable from PIPELINE_SERVER_URL.
export PIPELINE_SERVER_URL=http://localhost:8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/../.venv_qwen/bin/uvicorn" server_qwen:app --host 0.0.0.0 --port 8002 --log-level info
