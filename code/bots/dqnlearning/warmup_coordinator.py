from __future__ import annotations

from typing import Any, cast

from bots.common.decision import ActionChoice, DecisionInput

from .agent import DqnAgent
from .planner import WarmupPlanner


class WarmupCoordinator:
    """Owns planner-guided warmup action selection and planner lifecycle."""

    def __init__(self, planner: WarmupPlanner) -> None:
        self._planner = planner

    def reset(self) -> None:
        self._planner.reset()

    def onWallDiscovered(self) -> None:
        self._planner.onWallDiscovered()

    def selectAction(
        self,
        bot: Any,
        decision: DecisionInput,
        *,
        training: bool,
        allowWarmupPlanner: bool = True,
    ) -> ActionChoice | None:
        if not isinstance(bot.agent, DqnAgent):
            return None
        if (
            not training
            or not allowWarmupPlanner
            or not bool(getattr(bot, "_collectingWarmup", False))
            or bool(getattr(bot, "usesMacroOnlyPolicy", False))
        ):
            return None
        plannerAction = self._planner.nextAction(cast(tuple[int, int], bot.position))
        plannerLocal = decision.actionSpace.localId(plannerAction) if plannerAction is not None else None
        return bot.agent.selectWarmupAction(decision, plannerLocal)
