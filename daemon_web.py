#!/usr/bin/env python3
"""Daemon script to start the web UI server and keep it running."""
import os, time, sys

os.environ["PIPELINE_BACKEND"] = "proxy"
os.environ["PIPELINE_SERVER_URL"] = "http://localhost:8000"

import uvicorn

if __name__ == "__main__":
    uvicorn.run("server_qwen:app", host="0.0.0.0", port=8002, log_level="info")
