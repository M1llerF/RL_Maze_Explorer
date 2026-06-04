from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CurriculumConfig:
    windowSize: int = 20
    minSuccessRateToGraduate: float = 0.7
    maxAStarRatioToGraduate: float = 2.0
    maxMazeSize: int = 31


@dataclass
class CurriculumWindowEntry:
    success: bool
    astarRatio: float


@dataclass
class CurriculumProfileState:
    size: int
    episodesAtSize: int = 0
    history: deque[CurriculumWindowEntry] = field(default_factory=lambda: deque[CurriculumWindowEntry]())


@dataclass(frozen=True)
class CurriculumDecision:
    graduated: bool
    previousSize: int
    nextSize: int
    successRate: float
    avgAStarRatio: float


class CurriculumManager:
    """
    Simple curriculum controller:
    - Tracks rolling success and A*-relative efficiency per profile.
    - Graduates maze size by +1 when both metrics clear thresholds.
    """

    def __init__(self, config: CurriculumConfig | None = None) -> None:
        self.config = config or CurriculumConfig()
        self._states: dict[str, CurriculumProfileState] = {}

    def resetProfile(self, profileName: str, initialSize: int) -> None:
        state = CurriculumProfileState(size=max(5, int(initialSize)))
        state.history = deque(maxlen=max(1, int(self.config.windowSize)))
        self._states[profileName] = state

    def currentSize(self, profileName: str, fallback: int) -> int:
        state = self._states.get(profileName)
        return int(state.size if state is not None else fallback)

    def update(
        self,
        profileName: str,
        *,
        success: bool,
        episodeSteps: int,
        astarSteps: int,
    ) -> CurriculumDecision:
        if profileName not in self._states:
            self.resetProfile(profileName, initialSize=10)
        state = self._states[profileName]
        state.episodesAtSize += 1

        safeEpisodeSteps = max(1, int(episodeSteps))
        safeAStarSteps = max(1, int(astarSteps))
        ratio = float(safeEpisodeSteps) / float(safeAStarSteps)
        state.history.append(CurriculumWindowEntry(success=bool(success), astarRatio=ratio))

        entries = list(state.history)
        successRate = float(sum(1 for e in entries if e.success)) / float(len(entries))
        avgRatio = float(sum(e.astarRatio for e in entries)) / float(len(entries))

        canEvaluate = len(entries) >= max(1, int(self.config.windowSize))
        shouldGraduate = (
            canEvaluate
            and successRate >= float(self.config.minSuccessRateToGraduate)
            and avgRatio <= float(self.config.maxAStarRatioToGraduate)
            and state.size < int(self.config.maxMazeSize)
        )

        prev = int(state.size)
        if shouldGraduate:
            state.size += 1
            state.episodesAtSize = 0
            state.history.clear()
        nxt = int(state.size)
        return CurriculumDecision(
            graduated=bool(shouldGraduate),
            previousSize=prev,
            nextSize=nxt,
            successRate=successRate,
            avgAStarRatio=avgRatio,
        )
