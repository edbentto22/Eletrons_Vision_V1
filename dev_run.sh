#!/usr/bin/env bash
set -euo pipefail

# Simple dev runner
export APP_NAME="Eletrons Vision Service"
export MODEL_VARIANT="yolov8n.pt"
export SAVE_ANNOTATIONS="true"

# Optional auth and webhook
# export AUTH_TOKEN="changeme"
# export N8N_WEBHOOK_URL="https://n8n.example/webhook/xxxx"

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000