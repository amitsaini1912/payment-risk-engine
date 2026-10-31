# Risk engine image (Portfolio Day 16). Build from the repo root:
#   docker build -t risk-engine .
#   docker run --rm -p 8000:8000 -e RISK_API_KEYS=billing:<16+ chars> -e REDIS_URL=redis://<host>:6379/0 risk-engine
# The model is TRAINED during the build (model.pkl is not in git): same generator + seed as local, so the same
# model and threshold; the runner-up becomes the shadow model (shadow.pkl, Day 17). One uvicorn worker per container (Prometheus metrics are per process): scale with replicas.

FROM python:3.12.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

# ---- train: data + model, thrown away except the model files ----
FROM base AS train
RUN pip install matplotlib==3.9.4
COPY data/generate_data.py data/
COPY model/ model/
RUN python data/generate_data.py --rows 15000 --seed 42 --out data/transactions.csv \
 && python model/train.py

# ---- runtime ----
FROM base AS runtime
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin risk \
 && mkdir -p /app/logs /app/data && chown risk /app/logs /app/data
COPY agent/ agent/
COPY api/ api/
COPY risk_v2/ risk_v2/
COPY model/features.py model/scorer.py model/drift.py model/
COPY --from=train /app/model/model.pkl /app/model/shadow.pkl /app/model/metrics.json model/
COPY rules.yaml .
USER risk
# /app/data holds the feedback store (decisions + chargebacks, Day 18): mount a volume there to keep the labels
ENV MODEL_THREADS=1 AUDIT_LOG_PATH=/app/logs/audit_trail.jsonl RULES_PATH=/app/rules.yaml \
    FEEDBACK_DB_PATH=/app/data/feedback.db
EXPOSE 8000
# liveness only; orchestrators should use GET /ready for traffic (rules, model, Redis, API keys)
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
