from __future__ import annotations

from typing import Any, cast

from bots.common.decision import ActionChoice, DecisionInput

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
        if (
            not training
            or not allowWarmupPlanner
            or not getattr(bot, "isWarmingUp", False)
        ):
            return None
        plannerAction = self._planner.nextAction(cast(tuple[int, int], bot.position))
        if getattr(bot, "usesMacroOnlyPolicy", False):
            # Policy action space contains only options. Translate the planner's primitive
            # direction to the option that executes it as its first step.
            plannerLocal = (
                bot.plannerDirectionToOptionLocalId(decision, plannerAction)
                if plannerAction is not None else None
            )
        else:
            plannerLocal = decision.actionSpace.localId(plannerAction) if plannerAction is not None else None
        selector = getattr(bot, "selectWarmupAction", None)
        if callable(selector):
            return selector(decision, plannerLocal)
        agentSelector = getattr(getattr(bot, "agent", None), "selectWarmupAction", None)
        if callable(agentSelector):
            return agentSelector(decision, plannerLocal)
        return None
