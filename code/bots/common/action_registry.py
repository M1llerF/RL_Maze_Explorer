from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .actions import ActionSpec, PrimitiveActionSpec
from .decision import LocalActionSpace
from .options import OptionSpec
from .step_result import StepResult

if TYPE_CHECKING:
    from environment.context import EnvironmentContext


class ActionRegistry:
    """
    Central registry for all actions available to a bot.

    Primitive actions are mapped to executor callables.
    Options (macro-actions) are stored as OptionSpec instances.
    The learning agent only sees action IDs, valid masks, and rewards —
    it does not know what any action means.
    """

    def __init__(self) -> None:
        self._primitives: dict[int, PrimitiveActionSpec] = {}
        self._executors: dict[int, Any] = {}   # int -> callable(context) -> StepResult
        self._options: dict[int, OptionSpec] = {}
        self._nameToId: dict[str, int] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def registerPrimitive(
        self,
        action: PrimitiveActionSpec,
        executor: Any,  # callable(context: EnvironmentContext) -> StepResult
    ) -> None:
        if action.id in self._primitives or action.id in self._options:
            existing_name = next((n for n, i in self._nameToId.items() if i == action.id), "?")
            raise ValueError(f"Action id {action.id} is already registered as '{existing_name}'")
        self._primitives[action.id] = action
        self._executors[action.id] = executor
        self._nameToId[action.name] = action.id

    def registerOption(self, option: OptionSpec) -> None:
        if option.id in self._primitives or option.id in self._options:
            raise ValueError(f"Action id {option.id} is already registered")
        self._options[option.id] = option
        self._nameToId[option.name] = option.id

    # ── Queries ───────────────────────────────────────────────────────────────

    def getActionSpace(self) -> list[ActionSpec]:
        actions: list[ActionSpec] = list(self._primitives.values())
        for opt in self._options.values():
            actions.append(ActionSpec(id=opt.id, name=opt.name, kind=opt.kind, description=opt.description))
        return sorted(actions, key=lambda a: a.id)

    def buildLocalActionSpace(
        self,
        context: EnvironmentContext,
        *,
        includePrimitives: bool = True,
        includeOptions: bool = True,
        optionIds: set[int] | None = None,
    ) -> LocalActionSpace:
        semanticSpecs: list[ActionSpec] = []
        if includePrimitives:
            semanticSpecs.extend(sorted(self._primitives.values(), key=lambda spec: spec.id))
        if includeOptions:
            for option in sorted(self._options.values(), key=lambda spec: spec.id):
                if optionIds is not None and int(option.id) not in optionIds:
                    continue
                semanticSpecs.append(
                    ActionSpec(
                        id=option.id,
                        name=option.name,
                        kind=option.kind,
                        description=option.description,
                    )
                )

        localizedSpecs: list[ActionSpec] = []
        semanticIds: list[int] = []
        validMask: list[bool] = []
        for localId, spec in enumerate(semanticSpecs):
            localizedSpecs.append(
                ActionSpec(
                    id=localId,
                    name=spec.name,
                    kind=spec.kind,
                    description=spec.description,
                )
            )
            semanticIds.append(int(spec.id))
            validMask.append(bool(self._isValidAction(int(spec.id), context)))
        return LocalActionSpace(
            actionSpecs=tuple(localizedSpecs),
            semanticActionIds=tuple(semanticIds),
            validActionMask=tuple(validMask),
        )

    def getValidActionMask(self, context: EnvironmentContext) -> list[bool]:
        """
        Returns a bool list indexed by action ID (dense, 0 .. max_id).
        Primitive actions are valid when the movement model allows the move.
        Options are valid when their canStart() returns True.
        """
        all_ids = sorted(list(self._primitives) + list(self._options))
        if not all_ids:
            return []
        max_id = all_ids[-1]
        mask = [False] * (max_id + 1)
        for actionId in all_ids:
            mask[actionId] = bool(self._isValidAction(int(actionId), context))
        return mask

    def isOption(self, action_id: int) -> bool:
        return action_id in self._options

    def getOption(self, action_id: int) -> OptionSpec | None:
        return self._options.get(action_id)

    def actionId(self, name: str) -> int:
        try:
            return self._nameToId[name]
        except KeyError:
            raise KeyError(f"No action named '{name}' is registered") from None

    def numActions(self) -> int:
        return len(self._primitives) + len(self._options)

    def actionVectorSize(self) -> int:
        """
        Return the dense vector length required to index values by action_id.
        """
        all_ids = sorted(list(self._primitives) + list(self._options))
        if not all_ids:
            return 0
        return int(all_ids[-1]) + 1

    def semanticIdForLocal(self, actionSpace: LocalActionSpace, localId: int) -> int:
        local = int(localId)
        if local < 0 or local >= actionSpace.numActions:
            raise IndexError(f"local action id out of range: {local}")
        return actionSpace.semanticId(local)

    def executeLocal(
        self,
        actionSpace: LocalActionSpace,
        localId: int,
        context: EnvironmentContext,
        gamma: float = 0.99,
    ) -> StepResult:
        return self.execute(self.semanticIdForLocal(actionSpace, localId), context, gamma=gamma)

    # ── Execution ─────────────────────────────────────────────────────────────

    def executePrimitive(self, action_id: int, context: EnvironmentContext) -> StepResult:
        """Execute a single primitive action and return the StepResult."""
        if action_id not in self._executors:
            raise KeyError(f"No primitive executor for action_id={action_id}")
        context.tickPushCooldown()
        return self._executors[action_id].execute(context)

    def execute(
        self,
        action_id: int,
        context: EnvironmentContext,
        gamma: float = 0.99,
    ) -> StepResult:
        """
        Execute any registered action through a single shared entrypoint.
        """
        if action_id in self._executors:
            return self.executePrimitive(action_id, context)
        option = self.getOption(action_id)
        if option is None:
            raise KeyError(f"No action registered for action_id={action_id}")
        reward, duration, done = self.executeOption(option, context, gamma=gamma)
        return StepResult(
            reward=reward,
            done=done,
            info={"option": option.name},
            duration=duration,
        )

    def executeOption(
        self,
        option: OptionSpec,
        context: EnvironmentContext,
        gamma: float = 0.99,
    ) -> tuple[float, int, bool]:
        """
        Run an option to termination.  Returns (cumulative_discounted_reward, duration, done).
        The context's bot position is updated in-place via executePrimitive().
        Callers use the return values for SMDP Q-updates.
        """
        cum_reward = 0.0
        duration = 0
        done = False
        while not option.shouldTerminate(context, duration) and not done:
            prim_id = option.choosePrimitiveAction(context)
            result = self.executePrimitive(prim_id, context)
            # Intrinsic reward added on top of environmental reward
            env_reward = result.reward + option.intrinsicReward(context, result)
            cum_reward += (gamma ** duration) * env_reward
            duration += 1
            done = result.done
            if done:
                break
        return cum_reward, duration, done

    def _isValidAction(self, action_id: int, context: EnvironmentContext) -> bool:
        if action_id in self._executors:
            executor = self._executors[action_id]
            try:
                return bool(executor.isValid(context))
            except AttributeError:
                return True
        option = self._options.get(action_id)
        if option is None:
            return False
        try:
            return bool(option.canStart(context))
        except Exception:
            return True
