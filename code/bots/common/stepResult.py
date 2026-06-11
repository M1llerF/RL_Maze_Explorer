from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StepResult:
    """Unified result returned by every action execution."""
    reward: float
    done: bool
    info: dict[str, Any]
    duration: int = 1  # primitive steps consumed (>1 for macro-actions/options)
    trainingReward: float | None = None
