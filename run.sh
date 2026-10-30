#!/usr/bin/env bash
# Start the API and the dashboard together for the live demo.
# Usage: ./run.sh          (Ctrl-C stops both)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-./venv/bin/python}

# /assess and /audit need an API key since Day 16. No RISK_API_KEYS (env or .env)? Make a one-off key for this demo
# run, shared by the API and the dashboard.
if [ -z "${RISK_API_KEYS:-}" ] && ! grep -qs '^RISK_API_KEYS=.' .env; then
  RISK_API_KEY="demo-$("$PY" -c 'import secrets; print(secrets.token_urlsafe(18))')"
  export RISK_API_KEY RISK_API_KEYS="dashboard:$RISK_API_KEY"
fi
"$PY" -m uvicorn api.main:app --port 8000 --log-level warning &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

# wait for the API to answer
for _ in $(seq 1 20); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
  sleep 0.5
done

exec "$PY" -m streamlit run dashboard/app.py --server.port 8501
