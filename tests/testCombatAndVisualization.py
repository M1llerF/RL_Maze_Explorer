from __future__ import annotations

from typing import Any, cast

from botFactory import BotCreateContext
from botStatistics import BotStatistics
from bots.common.actionRegistry import ActionRegistry
from bots.common.actions import ATTACK_RIGHT, DEFAULT_PRIMITIVE_ACTIONS, RIGHT
from bots.common.optionsLibrary import MoveRightUntilJunction
from bots.dqnlearning import DQNBot, DQNConfig
from bots.qlearning import QLearningBot, QLearningConfig
from environment.context import ENEMY_KILLED, ENEMY_MOVED, EnvironmentContext
from environment.entityFactory import build_entity_registry
from environment.movement import GridMovement4Way, buildAttackExecutors, buildDefaultMovementExecutors
from environment.traversal import build_traversal_policy
from maze import Maze
from rewardSystem import RewardConfig, RewardSystem
from services.repository import ArtifactsRepository
from services.visualizationService import EnemyRenderState, VisualizationService


class _ContextDummyStats:
    def __init__(self) -> None:
        self.timesHitEnemy = 0
        self.timesRevisitedSquares = 0
        self.nonRepeatingStepsTaken = 0
        self._visited: dict[tuple[int, int], int] = {}

    def getVisitedPositions(self) -> dict[tuple[int, int], int]:
        return self._visited

    def updateLastVisited(self, _position: tuple[int, int]) -> None:
        return

    def updateVisitedPositions(self, position: tuple[int, int]) -> None:
        self._visited[position] = self._visited.get(position, 0) + 1


class _ContextDummyBot:
    def __init__(self) -> None:
        self.position = (0, 0)
        self.previousPosition: tuple[int, int] | None = None
        self.profileName = "combat"
        self.statistics = _ContextDummyStats()
        self._pushCooldownRemaining = 0

    @property
    def pushCooldownSteps(self) -> int:
        return 3

    def isPushReady(self) -> bool:
        return self._pushCooldownRemaining <= 0

    def consumePushCooldown(self) -> None:
        self._pushCooldownRemaining = self.pushCooldownSteps

    def tickPushCooldown(self) -> None:
        if self._pushCooldownRemaining > 0:
            self._pushCooldownRemaining -= 1


class _RepositoryDouble:
    def __init__(self) -> None:
        self.persistCalls = 0

    def loadQTable(self, _profile: str) -> dict[Any, Any]:
        return {}

    def ensureMazeFile(self, _profile: str) -> None:
        return

    def loadMazeData(self, _profile: str) -> dict[str, Any]:
        return {"highest": {"reward": float("-inf")}, "lowest": {"reward": float("inf")}}

    def saveQTable(self, _profile: str, _q_table: dict[Any, Any]) -> None:
        self.persistCalls += 1

    def saveMazeEpisode(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1

    def updateStepsFromHeatmap(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1

    def incrementTimesHitWall(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1

    def incrementTimesHitEnemy(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1

    def appendReward(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1


def _build_combat_maze() -> Maze:
    maze = Maze(5, 5)
    maze.grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    maze.start = (1, 1)
    maze.end = (3, 3)
    return maze


def test_attack_action_kills_adjacent_enemy() -> None:
    maze = _build_combat_maze()
    maze.setEntities(
        [{"type": "enemy", "position": [1, 2], "damage": 5.0, "behavior": {"kind": "stationary"}}]
    )
    registry = build_entity_registry(maze)
    bot = _ContextDummyBot()
    bot.position = (1, 1)
    context = EnvironmentContext(maze=maze, bot=bot, entity_registry=registry)
    killed: list[dict[str, Any]] = []
    context.on(ENEMY_KILLED, lambda **payload: killed.append(payload))

    result = buildAttackExecutors()[ATTACK_RIGHT].execute(context)

    assert result.info["enemy_kills"] == 1
    assert len(killed) == 1
    assert killed[0]["cause"] == "sword_push_wall"
    assert bot.position == (1, 1)


def test_dqn_runner_attack_kills_chase_enemy_before_enemy_tick() -> None:
    maze = _build_combat_maze()
    maze.setEntities([{"type": "enemy", "position": [1, 3], "damage": 5.0, "behavior": {"kind": "chase"}}])
    bot = DQNBot(
        BotCreateContext(
            maze=maze,
            config=DQNConfig(useAttackActions=True, hiddenSize=32),
            rewardSystem=RewardSystem(maze, RewardConfig()),
            statistics=BotStatistics(),
            profileName="attack_chase_kill_dqn",
            repository=cast(ArtifactsRepository, _RepositoryDouble()),
            loadCheckpoint=False,
        )
    )
    bot.position = (1, 2)
    bot.state = bot.calculateState()
    runtime = bot.runner._startEpisode()
    bot.position = (1, 2)
    bot.state = bot.calculateState()

    action_space = bot.buildDecisionInput(training=True).actionSpace
    local_attack_right = action_space.localId(ATTACK_RIGHT)
    assert local_attack_right is not None

    original_choose_action = bot.chooseAction
    bot.chooseAction = lambda **_kwargs: type("_Choice", (), {"local_id": int(local_attack_right)})()
    try:
        continued = bot.runner._runStep(runtime, mode="training")
    finally:
        bot.chooseAction = original_choose_action

    alive_enemies = [
        entity
        for entity in bot._context.entityRegistry.allEntities()
        if bool(getattr(entity, "_alive", False))
    ]
    assert continued is True
    assert alive_enemies == []
    assert bot.statistics.timesHitEnemy == 1
    assert runtime.outcome != "death_by_enemy"


def test_qlearning_auto_attack_chooses_adjacent_attack_action() -> None:
    maze = _build_combat_maze()
    maze.setEntities(
        [{"type": "enemy", "position": [1, 2], "damage": 5.0, "behavior": {"kind": "stationary"}}]
    )
    bot = QLearningBot(
        BotCreateContext(
            maze=maze,
            config=QLearningConfig(useAttackActions=True, autoAttackAdjacentEnemy=True),
            rewardSystem=RewardSystem(maze, RewardConfig()),
            statistics=BotStatistics(),
            profileName="auto_attack_q",
            repository=cast(ArtifactsRepository, _RepositoryDouble()),
        )
    )
    bot.position = (1, 1)
    bot.state = bot.calculateState()
    bot.qLearning.chooseAction = lambda _decision: (_ for _ in ()).throw(
        AssertionError("policy should be bypassed")
    )

    decision = bot.buildDecisionInput()
    choice = bot.selectAction(decision)

    assert decision.actionSpace.semanticId(choice.local_id) == ATTACK_RIGHT


def test_push_displaces_enemy_and_starts_cooldown() -> None:
    maze = _build_combat_maze()
    maze.setEntities(
        [{"type": "enemy", "position": [1, 2], "damage": 5.0, "behavior": {"kind": "stationary"}}]
    )
    registry = build_entity_registry(maze)
    bot = _ContextDummyBot()
    bot.position = (1, 1)
    context = EnvironmentContext(maze=maze, bot=bot, entity_registry=registry)
    moved: list[dict[str, Any]] = []
    context.on(ENEMY_MOVED, lambda **payload: moved.append(payload))

    result = GridMovement4Way().applyMove(context, RIGHT)

    assert result.info.get("hit_wall") is False
    assert result.info.get("enemy_kills", 0) == 0
    assert bot.position == (1, 2)
    assert moved[0]["new_position"] == (1, 3)
    assert bot._pushCooldownRemaining > 0


def test_macro_stops_when_enemy_is_ahead() -> None:
    maze = _build_combat_maze()
    maze.setEntities([{"type": "enemy", "position": [2, 3], "damage": 5.0, "behavior": {"kind": "stationary"}}])
    registry = build_entity_registry(maze)
    bot = _ContextDummyBot()
    bot.position = (2, 1)
    action_registry = ActionRegistry()
    executors = buildDefaultMovementExecutors(GridMovement4Way())
    for spec in DEFAULT_PRIMITIVE_ACTIONS:
        action_registry.registerPrimitive(spec, executors[spec.id])
    context = EnvironmentContext(
        maze=maze,
        bot=bot,
        entity_registry=registry,
        action_registry=action_registry,
        traversal_policy=build_traversal_policy(maze),
    )

    _, _, duration, _ = action_registry.executeOption(MoveRightUntilJunction(), context)

    assert bot.position == (2, 2)
    assert duration == 1


def test_visualization_snapshot_includes_enemy_positions(tmp_path) -> None:
    maze = _build_combat_maze()
    maze.setEntities(
        [
            {"type": "enemy", "position": [2, 2], "damage": 5.0, "behavior": {"kind": "chase"}, "alive": True},
            {
                "type": "enemy",
                "position": [1, 3],
                "damage": 5.0,
                "behavior": {"kind": "stationary"},
                "alive": False,
            },
        ]
    )
    repo = ArtifactsRepository(baseDir=str(tmp_path))
    repo.ensureMazeFile("viz_enemy_test")
    bot = QLearningBot(
        BotCreateContext(
            maze=maze,
            config=QLearningConfig(),
            rewardSystem=RewardSystem(maze, RewardConfig()),
            statistics=BotStatistics(),
            profileName="viz_enemy_test",
            repository=repo,
        )
    )

    snapshot = VisualizationService(repo).build_snapshot("viz_enemy_test", bot)

    alive_enemies = [enemy for enemy in snapshot.enemies if enemy.alive]
    dead_enemies = [enemy for enemy in snapshot.enemies if not enemy.alive]
    assert len(snapshot.enemies) == 2
    assert len(alive_enemies) == 1
    assert len(dead_enemies) == 1
    assert alive_enemies[0].position == (2, 2)
    assert alive_enemies[0].behavior_kind == "chase"
    assert dead_enemies[0].position == (1, 3)
    assert dead_enemies[0].behavior_kind == "stationary"
    assert all(isinstance(enemy, EnemyRenderState) for enemy in snapshot.enemies)
