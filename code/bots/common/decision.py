from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import ActionSpec


@dataclass(frozen=True)
class ActionChoice:
    local_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_id", int(self.local_id))


@dataclass(frozen=True)
class LocalActionSpace:
    actionSpecs: tuple[ActionSpec, ...]
    semanticActionIds: tuple[int, ...]
    validActionMask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.actionSpecs) != len(self.semanticActionIds):
            raise ValueError("actionSpecs and semanticActionIds must have the same length")
        if len(self.actionSpecs) != len(self.validActionMask):
            raise ValueError("actionSpecs and validActionMask must have the same length")
        for index, spec in enumerate(self.actionSpecs):
            if int(spec.id) != index:
                raise ValueError("Local action specs must be reindexed densely from 0..N-1")

    @property
    def numActions(self) -> int:
        return len(self.actionSpecs)

    def semanticId(self, local_id: int) -> int:
        return int(self.semanticActionIds[int(local_id)])

    def localId(self, semantic_id: int) -> int | None:
        semantic = int(semantic_id)
        for index, candidate in enumerate(self.semanticActionIds):
            if int(candidate) == semantic:
                return index
        return None

    def containsSemantic(self, semantic_id: int) -> bool:
        return self.localId(semantic_id) is not None

    def actionList(self) -> list[ActionSpec]:
        return list(self.actionSpecs)

    def maskList(self) -> list[bool]:
        return [bool(v) for v in self.validActionMask]


@dataclass(frozen=True)
class DecisionInput:
    state: Any
    actionSpace: LocalActionSpace
    level: str = "flat"
    activeOptionLocalId: int | None = None

    @property
    def action_space(self) -> list[ActionSpec]:
        return self.actionSpace.actionList()

    @property
    def action_mask(self) -> list[bool]:
        return self.actionSpace.maskList()
