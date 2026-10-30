# ADR 0003 — `/v2/assess` in production: stored decisions, 300 ms budget, API keys, metrics, container

- Status: Accepted
- Date: 2026-10-31
- Builds on: ADR 0001 (contract + Redis signals), ADR 0002 (rules + model + reason codes)

## Context
Billing (Day 19) will call `/v2/assess` before every capture with a ~300 ms timeout and retry once. So a retry must
get the same answer, the engine must answer inside the budget, only billing may call it, ops must see latency and
fallbacks, and it must run as a container in compose. Weaknesses: #18 (one orphan thread per timed-out LLM call),
#20 (`GET /audit` count = the limit).

## Decision
1. **Stored decisions** (`risk_v2/decisions.py`): the full response is stored in Redis under
   `risk:decision:<transaction_id>` with the SHA-256 of the canonical request, 25 h TTL (= feature snapshot).
   - Same id + same facts → stored decision, `replayed: true` — even if rules.yaml or the model changed since.
   - Same id + different facts → **409** (client bug; never a silent reuse).
   - Racing copies: `SET NX`, the first stored decision wins and every copy returns it; one audit entry, one
     decision counted.
   - Decided but the store write fails → **503** (a retry could otherwise get a different answer).
2. **Budget** (`risk_v2/budget.py`, `RISK_BUDGET_MS=300`): Redis (200 ms socket timeouts), rules and the model
   always run — a decision is always returned, even late (`risk_budget_exceeded_total`). Only the LLM explanation
   gets what's left minus 30 ms; < 50 ms left → not tried (`llm_no_budget`). Model + rules are loaded at startup
   (the first request doesn't pay ~1 s of model loading).
3. **Bounded LLM calls** (`agent/llm_pool.py`, #18): at most `LLM_POOL_SIZE` (4) calls at once, each on a daemon
   thread; a full pool refuses at once (`llm_busy`, template/fallback) instead of queueing. A timed-out call keeps
   its slot until it really ends; the Gemini client has an HTTP timeout (`LLM_HTTP_TIMEOUT_MS`, 10 s) so it ends.
   Used by v1 `/assess` too. Not a `ThreadPoolExecutor`: its workers are joined at exit, so a hung call would block
   shutdown.
4. **API keys** (`risk_v2/auth.py`): `X-API-Key` on `/assess`, `/v2/assess`, `/audit*`. `RISK_API_KEYS=name:key,…`
   (16+ chars), one per caller, constant-time compare over all keys; caller name in the audit entry. None configured
   → 503 (never open by accident). Wrong/missing → 401 before Redis is touched. `/health`, `/ready`,
   `/metrics/prometheus` are open (network-restricted in a real deployment).
5. **Metrics** (`risk_v2/metrics.py`, `GET /metrics/prometheus`): request duration histogram (endpoint, status;
   buckets around 300 ms), decisions (api_version, action, path_used), replays, 503s by reason (= billing fallback
   rate), reason codes on hold/block, explanation source, budget exceeded, auth failures, LLM calls in flight.
   p50/p95/p99 are computed by Prometheus (`histogram_quantile`), not in the app.
6. **`/ready`**: 200 only if API keys, rules, model and Redis are all usable; `/health` stays plain liveness.
7. **Container** (`Dockerfile`): `python:3.12.12-slim-bookworm`, multi-stage — the build generates the data and
   trains the model (seed 42; `model.pkl` is not in git), runtime has only `requirements-runtime.txt`, non-root
   user, 1 uvicorn worker (metrics are per process; scale with replicas), HEALTHCHECK on `/health`.
8. **#20**: `GET /audit` returns the real number of entries.

## Consequences
- New dependency: `prometheus-client==0.21.1`. `requirements.txt` now = `requirements-runtime.txt` + matplotlib,
  dashboard and tests.
- The dashboard needs `RISK_API_KEY` (one of `RISK_API_KEYS`) to call the API.
- The stored decision is the source of truth for 25 h; after that the same id is decided again (features from a
  new snapshot). Billing never retries that late.
- With a 300 ms budget the Gemini explanation (~1 s) almost never fits: the template is the normal `/v2`
  explanation; the LLM explanation is effectively off on the hot path.
- Transitive dependencies are not locked (the image resolved e.g. `requests 2.34.2`); a lock file is Day 24a.
- Single API keys, no rotation overlap beyond "add the new key, remove the old one"; no mTLS (internal network).
- Capacity (measured by hand, 2026-10-31): one container ~50 decisions/s; 10 parallel clients → p50 189 ms,
  p95 224 ms. Scoring is CPU-bound Python (GIL), so concurrency becomes queueing inside the process, and
  `decided_in_ms` / the budget start when the handler runs — queueing before that isn't counted (the HTTP
  histogram includes it). Day 22 load test.
- The rules loader had a first-load race (concurrent requests got 503 "no rules") — fixed: "seen" is recorded
  only after the parse.

