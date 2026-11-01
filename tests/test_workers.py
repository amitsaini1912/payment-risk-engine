"""
Several worker processes per container (ADR 0011): what only goes wrong across PROCESSES, so every test here starts
real ones (prometheus_client picks single- or multi-process mode at import time - it can't be switched in-process).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = os.pathsep.join([ROOT] + [os.path.join(ROOT, sub) for sub in ("agent", "model")])


def run(code: str, **env) -> str:
    """Run `code` in a fresh interpreter with the project on sys.path; returns stdout. env value None = unset."""
    full = {**os.environ, "PYTHONPATH": PATHS, **env}
    # set-but-empty still means multiprocess mode to prometheus_client (files in the current directory)
    full = {k: v for k, v in full.items() if v is not None and not k.startswith("_")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=full,
                            cwd=str(env.get("_cwd") or ROOT))
    assert result.returncode == 0, result.stderr
    return result.stdout


def worker(n_decisions: int, llm_in_flight: int) -> str:
    return f"""
from risk_v2 import metrics as m
for _ in range({n_decisions}):
    m.DECISIONS.labels("v2", "allow", "rules+model").inc()
m.REQUEST_SECONDS.labels("/v2/assess", "200").observe(0.02)
m.LLM_IN_FLIGHT.set({llm_in_flight})
m.DRIFT_PSI.labels("rf-1", "amount").set(0.{n_decisions})
"""


def sample(text: str, name: str, **labels) -> float:
    """The one sample of `name` carrying these labels (label order differs between the two modes)."""
    lines = [line for line in text.splitlines() if line.startswith(name + "{")
             and all(f'{k}="{v}"' in line for k, v in labels.items())]
    assert len(lines) == 1, (name, labels, lines)
    return float(lines[0].rsplit(" ", 1)[1])


def test_a_scrape_adds_up_every_workers_metrics(tmp_path):
    env = {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    run(worker(3, 1), **env)    # two "workers", each with its own numbers
    run(worker(5, 2), **env)

    text = run("from risk_v2 import metrics as m; print(m.exposition().decode())", **env)   # any worker answers

    assert sample(text, "risk_decisions_total", action="allow") == 8                   # 3 + 5, not one worker's
    assert sample(text, "risk_request_duration_seconds_count", endpoint="/v2/assess") == 2
    assert sample(text, "risk_drift_psi", feature="amount") == 0.5                     # most recent write, not 0.3+0.5


def test_without_the_directory_a_scrape_is_this_process_only(tmp_path):
    text = run(worker(2, 0) + "print(m.exposition().decode())", PROMETHEUS_MULTIPROC_DIR=None, _cwd=str(tmp_path))
    assert list(tmp_path.iterdir()) == []                      # really single-process: no value files written
    assert sample(text, "risk_decisions_total", action="allow") == 2


def test_audit_lines_from_several_processes_never_interleave(tmp_path):
    """Entries bigger than one 8 KB buffer, written at once by 4 processes: every line must still be one JSON object.
    (Holds with the plain append: Python hands a line bigger than its buffer to ONE write() call, and Linux appends
    one O_APPEND write whole. Checked on 2026-11-01 - this test guards that, nothing in audit.py had to change.)"""
    path = tmp_path / "audit.jsonl"
    code = f"""
import audit
for i in range(50):
    audit.write_entry({{"proc": __import__("os").getpid(), "i": i, "explanation": "x" * 20000}}, path={str(path)!r})
"""
    procs = [subprocess.Popen([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": PATHS}) for _ in range(4)]
    assert all(p.wait(timeout=60) == 0 for p in procs)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    entries = [json.loads(line) for line in lines]       # a split/interleaved line would not parse
    assert len({(e["proc"], e["i"]) for e in entries}) == 200


def test_several_processes_share_one_feedback_db(tmp_path):
    """SQLite WAL across processes: 4 writers at once, every decision kept exactly once (no 'database is locked')."""
    db = tmp_path / "feedback.db"
    code = f"""
import os, sys
from risk_v2.feedback import OutcomeStore
store = OutcomeStore({str(db)!r})
for i in range(100):
    tid = f"p{{os.getpid()}}-{{i}}"
    ok = store.record_decision({{"transaction_id": tid, "signals": {{"observed_at_ms": 1}}, "action": "allow",
        "model_action": "allow", "model_score": 0.1, "threshold": 0.5, "model_version": "m", "rules_version": "r",
        "model_input": {{"customer_id": "c", "amount": 10.0}}}})
    assert ok, tid
"""
    procs = [subprocess.Popen([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": PATHS}) for _ in range(4)]
    assert all(p.wait(timeout=60) == 0 for p in procs)

    count = run(f"import sqlite3; print(sqlite3.connect({str(db)!r}).execute('select count(*) from decisions').fetchone()[0])")
    assert int(count) == 400
