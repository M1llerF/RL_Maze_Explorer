import os
import sys
import tempfile
import unittest
from typing import Any, cast


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODEDIR = os.path.join(ROOT, "code")
if CODEDIR not in sys.path:
    sys.path.insert(0, CODEDIR)

from botConfigs import QLearningConfig, botConfigs
from botFactory import BotFactory
from botProfile import BotProfile, ProfileManager
from botStatistics import BotStatistics
from qLearningBot import QLearning, QLearningBot
from rewardSystem import RewardConfig
from rewardSystem import RewardSystem
from services.runners import QLearningEpisodeRunner
from services.repository import ArtifactsRepository


class _DummyMaze:
    def __init__(self) -> None:
        self.width = 2
        self.height = 2
        self.grid = [[0, 0], [0, 0]]
        self.end = (1, 1)

    def isValidPosition(self, _profileName: str, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width


class _CtorWithRepository:
    def __init__(
        self,
        maze: Any,
        config: Any,
        rewardSystem: Any,
        statistics: Any,
        profileName: str,
        repository: ArtifactsRepository | None = None,
    ) -> None:
        self.received = {
            "maze": maze,
            "config": config,
            "reward_system": rewardSystem,
            "statistics": statistics,
            "profile_name": profileName,
            "repository": repository,
        }
        self.specificData = None

    def initializeSpecificData(self, data: dict[str, Any]) -> None:
        self.specificData = data


class _LegacyCtorNoRepository:
    def __init__(
        self,
        maze: Any,
        config: Any,
        rewardSystem: Any,
        statistics: Any,
        profileName: str,
    ) -> None:
        self.received = {
            "maze": maze,
            "config": config,
            "reward_system": rewardSystem,
            "statistics": statistics,
            "profile_name": profileName,
        }
        self.specificData = None

    def initializeSpecificData(self, data: dict[str, Any]) -> None:
        self.specificData = data


class _FutureConfig:
    def __init__(self, alpha: float = 0.7, beta: int = 2) -> None:
        self.alpha = alpha
        self.beta = beta


class _RunnerDummyTools:
    def getOptimalPathInfo(self, _start: tuple[int, int], _end: tuple[int, int], output: str = "path") -> list[tuple[int, int]]:
        return [(0, 0), (0, 1)]

    def calculateNextPosition(self, _pos: tuple[int, int], _action: int) -> tuple[int, int]:
        return (0, 1)


class _RunnerDummyStats:
    def __init__(self) -> None:
        self.timesRevisitedSquares = 0
        self.nonRepeatingStepsTaken = 0
        self.totalSteps = 0
        self._visited: dict[tuple[int, int], int] = {}

    def getVisitedPositions(self) -> dict[tuple[int, int], int]:
        return self._visited

    def updateLastVisited(self, _pos: tuple[int, int]) -> None:
        return

    def updateVisitedPositions(self, pos: tuple[int, int]) -> None:
        self._visited[pos] = self._visited.get(pos, 0) + 1


class _RunnerDummyRewardSystem:
    def getReward(self, *_args: Any, **_kwargs: Any) -> float:
        return 1.0


class _RunnerDummyQ:
    def __init__(self) -> None:
        self.totalSteps = 0
        self.saved = False

    def chooseAction(self, _state: Any, _stats: Any) -> int:
        return 0

    def updateQValue(self, _state: Any, _action: int, _reward: float, _newState: Any) -> None:
        return

    def saveQTable(self) -> None:
        self.saved = True


class _RunnerDummyRepo:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def saveMazeEpisode(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls.append("save_maze_episode")

    def updateStepsFromHeatmap(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls.append("update_steps_from_heatmap")

    def incrementTimesHitWall(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls.append("increment_times_hit_wall")

    def appendReward(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls.append("append_reward")


class _RunnerDummyMaze:
    start = (0, 0)
    end = (0, 1)
    width = 2
    height = 1
    grid = [[0, 0]]

    def isValidPosition(self, _profileName: str, row: int, col: int) -> bool:
        return row == 0 and col in (0, 1)

    def getStart(self) -> tuple[int, int]:
        return self.start


class _VizDummyRepo:
    def __init__(self) -> None:
        self.persistCalls = 0

    def loadQTable(self, _profile: str) -> dict[Any, Any]:
        return {}

    def ensureMazeFile(self, _profile: str) -> None:
        return

    def loadMazeData(self, _profile: str) -> dict[str, Any]:
        return {"highest": {"reward": float("-inf")}, "lowest": {"reward": float("inf")}}

    def saveQTable(self, _profile: str, _qTable: dict[Any, Any]) -> None:
        self.persistCalls += 1

    def saveMazeEpisode(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1

    def updateStepsFromHeatmap(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1

    def incrementTimesHitWall(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1

    def appendReward(self, *_args: Any, **_kwargs: Any) -> None:
        self.persistCalls += 1


class TypingRegressionTests(unittest.TestCase):
    def testProfileFromDictUsesDefaultsOnMissingOrNoneData(self) -> None:
        botProfileCls: Any = BotProfile
        profile: Any = botProfileCls.fromDict(None, defaultName="fallback")
        self.assertEqual(profile.name, "fallback")
        self.assertEqual(profile.botType, "QLearningBot")
        self.assertIsInstance(profile.config, QLearningConfig)
        self.assertIsInstance(profile.rewardConfig, RewardConfig)
        self.assertIsInstance(profile.statistics, BotStatistics)
        self.assertEqual(profile.botSpecificData, {})

    def testProfileFromDictIgnoresUnknownConfigKeys(self) -> None:
        botProfileCls: Any = BotProfile
        profile: Any = botProfileCls.fromDict(
            {
                "name": "p1",
                "config": {
                    "learning_rate": 0.25,
                    "discount_factor": 0.8,
                    "use_position_in_state": False,
                    "unexpected": "ignored",
                },
            }
        )
        self.assertEqual(profile.config.learningRate, 0.25)
        self.assertEqual(profile.config.discountFactor, 0.8)
        self.assertFalse(profile.config.usePositionInState)
        self.assertFalse(hasattr(profile.config, "unexpected"))

    def testProfileFromDictDefaultsToQlearningForUnknownBotType(self) -> None:
        botProfileCls: Any = BotProfile
        profile: Any = botProfileCls.fromDict(
            {
                "name": "p-unknown",
                "bot_type": "FutureUnknownBot",
                "config": {"learning_rate": 0.33, "discount_factor": 0.77},
            }
        )
        self.assertEqual(profile.botType, "FutureUnknownBot")
        self.assertIsInstance(profile.config, QLearningConfig)
        self.assertEqual(profile.config.learningRate, 0.33)
        self.assertEqual(profile.config.discountFactor, 0.77)

    def testProfileFromDictUsesRegisteredConfigClassForBotType(self) -> None:
        botConfigs["FutureBot"] = {"class": _FutureConfig}
        self.addCleanup(lambda: botConfigs.pop("FutureBot", None))
        botProfileCls: Any = BotProfile
        profile: Any = botProfileCls.fromDict(
            {
                "name": "p-future",
                "bot_type": "FutureBot",
                "config": {"alpha": 1.5, "beta": 4, "ignored": 123},
            }
        )
        self.assertEqual(profile.botType, "FutureBot")
        self.assertIsInstance(profile.config, _FutureConfig)
        self.assertEqual(profile.config.alpha, 1.5)
        self.assertEqual(profile.config.beta, 4)
        self.assertFalse(hasattr(profile.config, "ignored"))

    def testProfileFromDictToleratesNonDictRewardAndStatistics(self) -> None:
        botProfileCls: Any = BotProfile
        profile: Any = botProfileCls.fromDict(
            {
                "name": "p2",
                "reward_config": "not-a-dict",
                "statistics": ["not", "a", "dict"],
            }
        )
        self.assertIsInstance(profile.rewardConfig, RewardConfig)
        self.assertIsInstance(profile.statistics, BotStatistics)

    def testProfileFromDictKeepsUnknownStatisticsFieldsFromDict(self) -> None:
        botProfileCls: Any = BotProfile
        profile: Any = botProfileCls.fromDict(
            {
                "name": "p3",
                "statistics": {
                    "total_steps": 123,
                    "custom_counter": "legacy-value",
                },
            }
        )
        self.assertEqual(profile.statistics.totalSteps, 123)
        self.assertEqual(getattr(profile.statistics, "custom_counter"), "legacy-value")

    def testBotFactoryRejectsUnknownBotType(self) -> None:
        factory: Any = BotFactory(_DummyMaze())
        with self.assertRaisesRegex(
            ValueError,
            r"Unknown bot type: MissingBot\. Registered bot types: <none>",
        ):
            factory.createBot(
                "MissingBot",
                "profile",
                QLearningConfig(),
                RewardConfig(),
                BotStatistics(),
                {},
            )

    def testBotFactoryRegistryHelpers(self) -> None:
        factory: Any = BotFactory(_DummyMaze())
        self.assertFalse(factory.isRegistered("QLearningBot"))
        factory.registerBot("ZBot", _LegacyCtorNoRepository)
        factory.registerBot("ABot", _LegacyCtorNoRepository)
        self.assertTrue(factory.isRegistered("ZBot"))
        self.assertEqual(factory.listRegisteredBotTypes(), ["ABot", "ZBot"])

    def testBotFactoryPassesRepositoryWhenCtorSupportsIt(self) -> None:
        repo = ArtifactsRepository(baseDir=os.path.join(ROOT, "profiles"))
        factory: Any = BotFactory(_DummyMaze(), repository=repo)
        factory.registerBot("RepoCtor", _CtorWithRepository)

        bot: _CtorWithRepository = cast(
            _CtorWithRepository,
            factory.createBot(
            "RepoCtor",
            "profileA",
            QLearningConfig(),
            RewardConfig(),
            BotStatistics(),
            {"k": "v"},
            ),
        )
        self.assertIs(bot.received["repository"], repo)
        self.assertEqual(bot.specificData, {"k": "v"})

    def testBotFactoryFallsBackForLegacyConstructor(self) -> None:
        factory: Any = BotFactory(_DummyMaze())
        factory.registerBot("LegacyCtor", _LegacyCtorNoRepository)

        bot: _LegacyCtorNoRepository = cast(
            _LegacyCtorNoRepository,
            factory.createBot(
            "LegacyCtor",
            "profileB",
            QLearningConfig(),
            RewardConfig(),
            BotStatistics(),
            {"legacy": 1},
            ),
        )
        self.assertEqual(bot.received["profile_name"], "profileB")
        self.assertEqual(bot.specificData, {"legacy": 1})

    def testQlearningStateToKeyAcceptsMixedRuntimeTypes(self) -> None:
        q = QLearning(QLearningConfig())
        state = ("state-id", ("wall", 2), {"direction": "N"})
        qAny: Any = q
        key: tuple[Any, ...] = cast(tuple[Any, ...], qAny.stateToKey(state))
        self.assertEqual(key, state)

    def testRunnerUsesExplicitStartStepEndLifecycle(self) -> None:
        class _DummyBot:
            def __init__(self) -> None:
                self.tools = _RunnerDummyTools()
                self.maze = _RunnerDummyMaze()
                self.statistics = _RunnerDummyStats()
                self.rewardSystem = _RunnerDummyRewardSystem()
                self.qLearning = _RunnerDummyQ()
                self.repo = _RunnerDummyRepo()
                self.profileName = "runner-test"
                self.position = (0, 0)
                self.state = ("s",)
                self.totalReward = 0.0
                self.episodeCounter = 0
                self.hooks: list[str] = []

            def calculateState(self, position: tuple[int, int] | None = None) -> tuple[Any, ...]:
                p = self.position if position is None else position
                return (p,)

            def onEpisodeStart(self, mode: str) -> None:
                self.hooks.append(f"start:{mode}")

            def onEpisodeStep(self, mode: str, _stepIndex: int) -> None:
                self.hooks.append(f"step:{mode}")

            def onEpisodeEnd(self, mode: str, outcome: str) -> None:
                self.hooks.append(f"end:{mode}:{outcome}")

        bot = _DummyBot()
        runner = QLearningEpisodeRunner(bot)
        runner.runEpisode()

        self.assertGreaterEqual(len(bot.hooks), 3)
        self.assertEqual(bot.hooks[0], "start:training")
        self.assertIn("step:training", bot.hooks)
        self.assertTrue(bot.hooks[-1].startswith("end:training:"))
        self.assertEqual(bot.episodeCounter, 1)
        self.assertTrue(bot.qLearning.saved)
        self.assertIn("append_reward", bot.repo.calls)

    def testRepositoryModelArtifactBytesRoundTrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = ArtifactsRepository(baseDir=td)
            payload = b"\x01\x02future-model"
            repo.saveModelArtifactBytes("p1", "weights.bin", payload)
            loaded = repo.loadModelArtifactBytes("p1", "weights.bin")
            self.assertEqual(loaded, payload)

    def testRepositoryModelArtifactRejectsPathTraversalName(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = ArtifactsRepository(baseDir=td)
            with self.assertRaisesRegex(ValueError, r"Invalid artifact name"):
                repo.saveModelArtifactBytes("p1", "../weights.bin", b"x")

    def testProfileManagerRoundTripQlearningProfile(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manager = ProfileManager(td)
            src = BotProfile(
                name="roundtrip_a",
                botType="QLearningBot",
                config=QLearningConfig(learningRate=0.2, discountFactor=0.85, usePositionInState=False),
                rewardConfig=RewardConfig(),
                statistics=BotStatistics(),
                botSpecificData={"tag": "x"},
            )
            manager.saveProfile(src)
            loaded = manager.loadProfile("roundtrip_a")
            self.assertEqual(loaded.name, "roundtrip_a")
            self.assertEqual(loaded.botType, "QLearningBot")
            self.assertIsInstance(loaded.config, QLearningConfig)
            self.assertEqual(loaded.config.learningRate, 0.2)
            self.assertEqual(loaded.config.discountFactor, 0.85)
            self.assertFalse(loaded.config.usePositionInState)
            self.assertEqual(loaded.botSpecificData.get("tag"), "x")

    def testVisualizationStepDoesNotPersistTrainingArtifacts(self) -> None:
        maze = _RunnerDummyMaze()
        rewardSystem = RewardSystem(maze, RewardConfig())
        stats = BotStatistics()
        repo = _VizDummyRepo()
        bot = QLearningBot(
            maze=maze,
            config=QLearningConfig(),
            rewardSystem=rewardSystem,
            statistics=stats,
            profileName="viz_test",
            repository=cast(Any, repo),
        )
        before = repo.persistCalls
        finished = bot.stepVisualization(maxSteps=3)
        self.assertIsInstance(finished, bool)
        self.assertEqual(repo.persistCalls, before)


if __name__ == "__main__":
    unittest.main()
