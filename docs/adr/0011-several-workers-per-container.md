# ADR 0011 — Several worker processes per container

- Status: Accepted
- Date: 2026-11-01
- Amends: ADR 0003 ("1 uvicorn worker, scale with replicas"), ADR 0010 consequences ("still one worker").

## Context
One worker is one CPU (pandas/scikit-learn per decision, the GIL). Measured with k6 against `/v2/assess` alone
(`load-test/assess.js`, 30 s per rate, k6 on the same laptop, i5-13500H):

| workers | 40/s | 60/s | 80/s | 100/s |
|---|---|---|---|---|
| 1 (before) | p99 196 ms | **collapsed**: p99 9.1 s, 496 dropped | - | - |
| 2 | p99 64 ms | collapsed: p99 1.2 s, 17 dropped | p99 4.8 s | p99 5.9 s |
| 4 | p99 52 ms | p99 70 ms | **p99 140 ms** | collapsed: p99 1.1 s, 46 dropped |

(0 failed requests in every run. Billing's budget for the whole check is 300 ms.) The built image with its default
CMD (4 workers): 60/s at p99 43 ms. The 2-worker row is worse than "2 × 1 worker" would suggest; one run each, so
read it as "2 is not enough", not as a precise number.

What stood in the way (ADR 0010): metrics are per process, and the feedback store is a SQLite file.

## Decision
- `WEB_CONCURRENCY=4` workers in the image (uvicorn reads it as `--workers`); one container still = one SQLite file.
- **Prometheus multiprocess mode**: `PROMETHEUS_MULTIPROC_DIR` set in the image; each worker writes its values there,
  and `/metrics/prometheus` (answered by any worker) adds them up. Gauges: LLM calls in flight = sum over live
  workers; drift / shadow-agreement = most recent write (every worker computes the same value from Redis). The CMD
  empties the directory at start, so a restarted container doesn't count the old workers. Without the variable
  (tests, `run.sh`) nothing changes.
- **SQLite stays**: WAL mode already lets the workers write one file; 0 write failures in all the runs above
  (`risk_feedback_write_failures_total`), and the store had exactly one row per decision (8,357 = 8,357).
- Kafka: each worker runs the feedback consumer in the same group, so partitions are split, not duplicated.
- Audit JSONL: several processes append to one file. Nothing had to change - Python hands a line larger than its
  buffer to one `write()`, and Linux appends one O_APPEND write whole; a test guards it (4 processes, 20 KB lines).
- **No engine-side warm-up.** Scoring a synthetic payment at startup was tried and measured: the first request after a
  fresh start took 24-29 ms with it and 24-26 ms without (then 11-20 ms either way). Loading model + rules at startup
  (already done) is enough; the Day 23 cold start was billing's side (billing ADR 0026).

## Consequences
- ~2× the capacity per container on this laptop (40/s -> 80/s within budget). Memory: one model copy per worker.
- Per-process state is now per worker: rules hot reload (each worker stats the file), the LLM pool (4 slots per
  worker), in-process caches. Behaviour is the same, the limits multiply.
- Several CONTAINERS still need a shared feedback database (Postgres): each container has its own SQLite file.
- The billing checkout k6 test was not repeated with 4 workers (only the engine alone).
- Tests: `tests/test_workers.py` (4, real processes): a scrape adds up two workers' counters and histograms and takes
  the latest gauge; without the directory a scrape is this process only; 4 processes append whole audit lines; 4
  processes write 400 decisions to one SQLite file, all kept.
