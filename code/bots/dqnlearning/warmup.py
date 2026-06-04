from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerObservationContext:
    plannerActions: tuple[int, ...]
    validActionMask: tuple[bool, ...]


class PlannerWarmupPolicy:
    def selectAction(self, context: PlannerObservationContext) -> int | None:
        for action in context.plannerActions:
            if 0 <= int(action) <= 3 and context.validActionMask[int(action)]:
                return int(action)
        return None
