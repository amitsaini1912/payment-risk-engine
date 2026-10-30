"""
The /v2/assess time budget (Day 16): billing waits at most ~300 ms for us (Day 19), so we plan for that.

Redis (200 ms socket timeouts), rules and the model always run - a decision is always made, even late. Only the
optional LLM explanation is cut to what is left, minus a reserve for storing and sending the answer.
"""

from __future__ import annotations

import os
import time
from typing import Optional

BUDGET_S = float(os.environ.get("RISK_BUDGET_MS", "300")) / 1000
RESERVE_S = 0.03   # storing the decision + writing the response


class Deadline:
    def __init__(self, budget_s: Optional[float] = None, clock=time.monotonic):
        self._clock = clock
        self.started = clock()
        self.budget_s = BUDGET_S if budget_s is None else budget_s

    def elapsed_s(self) -> float:
        return self._clock() - self.started

    def left_for_optional_work(self) -> float:
        return self.budget_s - self.elapsed_s() - RESERVE_S
