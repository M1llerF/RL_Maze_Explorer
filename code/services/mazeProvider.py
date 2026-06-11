from __future__ import annotations

from typing import Any, Optional, cast


class MazeProvider:
    """Owns maze lifecycle state and drives which maze is used on each reset.

    Three modes:
    - Random: generates a new maze each reset (default)
    - Pool: cycles through a pre-built fixed pool (stable MDP per phase)
    - Fixed: always restores the same snapshot (builder maze or visualization)

    configureMode() is the main entry point for TrainingController — it sets
    up the correct mode in one call, replacing repeated inline field mutation.

    resetMaze(isPaused) is called by GameEnvironment.resetEnvironment to
    apply the correct maze state before a bot resets its position.
    """

    def __init__(self, maze: Any) -> None:
        self.maze = maze

        # Pool mode
        self.trainingPoolActive: bool = False
        self.trainingPool: list[dict[str, Any]] = []
        self.trainingPoolIndex: int = 0

        # Fixed maze mode
        self.fixedMazeActive: bool = False
        self.fixedMazeState: Optional[dict[str, Any]] = None
        self.fixedMazeLabel: str = ""

        # Random / curriculum mode
        self.curriculumActive: bool = False
        self.randomGenerationMinLength: Optional[int] = None
        self.randomGenerationMaxLength: Optional[int] = None
        self.randomIncludeEnemies: bool = False

    # ── Mode configuration ────────────────────────────────────────────────────

    def configureMode(
        self,
        mode: str,
        *,
        poolSize: int = 20,
        minLength: Optional[int] = None,
        maxLength: Optional[int] = None,
        includeEnemies: bool = False,
    ) -> None:
        """Configure maze source for a training run.

        Caller must validate preconditions (e.g. fixed maze is loaded) before
        calling this. The mode string matches the UI combobox values:
        "Pool", "Fixed (Builder)", or anything else for Random.
        """
        self.randomIncludeEnemies = bool(includeEnemies)
        if mode == "Pool":
            self.setRandomGenerationLengthRange(None, None)
            self.configureTrainingPool(size=poolSize)
            self.curriculumActive = False
        elif mode == "Fixed (Builder)":
            self.setRandomGenerationLengthRange(None, None)
            self.trainingPoolActive = False
            self.curriculumActive = False
        else:  # Random (curriculum)
            self.setRandomGenerationLengthRange(minLength, maxLength)
            self.trainingPoolActive = False
            self.fixedMazeActive = False
            self.fixedMazeState = None
            self.fixedMazeLabel = ""
            self.curriculumActive = True
            if minLength is not None:
                self.setMazeSize(minLength, minLength)
            else:
                self.setupRandomMaze()

    # ── Fixed maze ────────────────────────────────────────────────────────────

    def setFixedMaze(self, state: dict[str, Any], *, label: str | None = None) -> None:
        self.fixedMazeState = state
        self.fixedMazeActive = True
        self.trainingPoolActive = False
        self.fixedMazeLabel = str(label or "Builder Maze").strip() or "Builder Maze"

    def clearFixedMaze(self) -> None:
        self.fixedMazeState = None
        self.fixedMazeActive = False
        self.fixedMazeLabel = ""

    # ── Random generation ─────────────────────────────────────────────────────

    def setRandomGenerationLengthRange(
        self,
        minLength: Optional[int],
        maxLength: Optional[int],
    ) -> None:
        self.randomGenerationMinLength = None if minLength is None else max(0, int(minLength))
        self.randomGenerationMaxLength = None if maxLength is None else max(0, int(maxLength))
        if (
            self.randomGenerationMinLength is not None
            and self.randomGenerationMaxLength is not None
            and self.randomGenerationMinLength > self.randomGenerationMaxLength
        ):
            raise ValueError("Minimum generation length must be <= maximum generation length.")

    def setRandomIncludeEnemies(self, includeEnemies: bool) -> None:
        self.randomIncludeEnemies = bool(includeEnemies)

    def setupRandomMaze(self) -> None:
        cast(Any, self.maze).setupSimpleMaze(
            minGenerationLength=self.randomGenerationMinLength,
            maxGenerationLength=self.randomGenerationMaxLength,
            includeEnemies=self.randomIncludeEnemies,
        )

    # ── Pool ──────────────────────────────────────────────────────────────────

    def configureTrainingPool(self, size: int = 20) -> None:
        """Build a fixed pool of random mazes to cycle through."""
        self.trainingPool = []
        for _ in range(max(1, size)):
            self.setupRandomMaze()
            self.trainingPool.append(cast(Any, self.maze).getState())
        self.trainingPoolIndex = 0
        self.trainingPoolActive = True
        self.fixedMazeActive = False
        self.fixedMazeLabel = ""
        self.curriculumActive = False

    # ── Size / curriculum ─────────────────────────────────────────────────────

    def setMazeSize(self, width: int, height: int) -> None:
        """Resize and regenerate; intended for curriculum progression."""
        cast(Any, self.maze).resize(int(width), int(height), regenerate=False)
        self.setupRandomMaze()
        self.trainingPoolActive = False
        self.fixedMazeActive = False
        self.fixedMazeState = None
        self.fixedMazeLabel = ""

    def getMazeSize(self) -> tuple[int, int]:
        return int(self.maze.width), int(self.maze.height)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def resetMaze(self, isPaused: bool) -> None:
        """Apply the correct maze state for the next episode.

        isPaused — True when visualization is open for this bot's profile.
        Visualization always uses fixed/random (never pool) to keep the UI stable.
        """
        if isPaused:
            if self.fixedMazeActive and self.fixedMazeState is not None:
                cast(Any, self.maze).setState(self.fixedMazeState)
            else:
                self.setupRandomMaze()
        elif self.trainingPoolActive and self.trainingPool:
            state = self.trainingPool[self.trainingPoolIndex]
            cast(Any, self.maze).setState(state)
            self.trainingPoolIndex = (self.trainingPoolIndex + 1) % len(self.trainingPool)
        else:
            if self.fixedMazeActive and self.fixedMazeState is not None:
                cast(Any, self.maze).setState(self.fixedMazeState)
            else:
                self.setupRandomMaze()
