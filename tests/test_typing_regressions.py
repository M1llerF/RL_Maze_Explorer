import os
import sys
import unittest
from typing import Any, cast


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from BotConfigs import QLearningConfig
from BotFactory import BotFactory
from BotProfile import BotProfile
from BotStatistics import BotStatistics
from QLearningBot import QLearning
from RewardSystem import RewardConfig
from services.repository import ArtifactsRepository


class _DummyMaze:
    def __init__(self) -> None:
        self.width = 2
        self.height = 2
        self.grid = [[0, 0], [0, 0]]
        self.end = (1, 1)

    def is_valid_position(self, _profile_name: str, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width


class _CtorWithRepository:
    def __init__(
        self,
        maze: Any,
        config: Any,
        reward_system: Any,
        statistics: Any,
        profile_name: str,
        repository: ArtifactsRepository | None = None,
    ) -> None:
        self.received = {
            "maze": maze,
            "config": config,
            "reward_system": reward_system,
            "statistics": statistics,
            "profile_name": profile_name,
            "repository": repository,
        }
        self.specific_data = None

    def initialize_specific_data(self, data: dict[str, Any]) -> None:
        self.specific_data = data


class _LegacyCtorNoRepository:
    def __init__(
        self,
        maze: Any,
        config: Any,
        reward_system: Any,
        statistics: Any,
        profile_name: str,
    ) -> None:
        self.received = {
            "maze": maze,
            "config": config,
            "reward_system": reward_system,
            "statistics": statistics,
            "profile_name": profile_name,
        }
        self.specific_data = None

    def initialize_specific_data(self, data: dict[str, Any]) -> None:
        self.specific_data = data


class TypingRegressionTests(unittest.TestCase):
    def test_profile_from_dict_uses_defaults_on_missing_or_none_data(self) -> None:
        bot_profile_cls: Any = BotProfile
        profile: Any = bot_profile_cls.from_dict(None, default_name="fallback")
        self.assertEqual(profile.name, "fallback")
        self.assertEqual(profile.bot_type, "QLearningBot")
        self.assertIsInstance(profile.config, QLearningConfig)
        self.assertIsInstance(profile.reward_config, RewardConfig)
        self.assertIsInstance(profile.statistics, BotStatistics)
        self.assertEqual(profile.bot_specific_data, {})

    def test_profile_from_dict_ignores_unknown_config_keys(self) -> None:
        bot_profile_cls: Any = BotProfile
        profile: Any = bot_profile_cls.from_dict(
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
        self.assertEqual(profile.config.learning_rate, 0.25)
        self.assertEqual(profile.config.discount_factor, 0.8)
        self.assertFalse(profile.config.use_position_in_state)
        self.assertFalse(hasattr(profile.config, "unexpected"))

    def test_profile_from_dict_tolerates_non_dict_reward_and_statistics(self) -> None:
        bot_profile_cls: Any = BotProfile
        profile: Any = bot_profile_cls.from_dict(
            {
                "name": "p2",
                "reward_config": "not-a-dict",
                "statistics": ["not", "a", "dict"],
            }
        )
        self.assertIsInstance(profile.reward_config, RewardConfig)
        self.assertIsInstance(profile.statistics, BotStatistics)

    def test_profile_from_dict_keeps_unknown_statistics_fields_from_dict(self) -> None:
        bot_profile_cls: Any = BotProfile
        profile: Any = bot_profile_cls.from_dict(
            {
                "name": "p3",
                "statistics": {
                    "total_steps": 123,
                    "custom_counter": "legacy-value",
                },
            }
        )
        self.assertEqual(profile.statistics.total_steps, 123)
        self.assertEqual(getattr(profile.statistics, "custom_counter"), "legacy-value")

    def test_bot_factory_rejects_unknown_bot_type(self) -> None:
        factory: Any = BotFactory(_DummyMaze())
        with self.assertRaisesRegex(ValueError, r"Unknown bot type: MissingBot"):
            factory.create_bot(
                "MissingBot",
                "profile",
                QLearningConfig(),
                RewardConfig(),
                BotStatistics(),
                {},
            )

    def test_bot_factory_passes_repository_when_ctor_supports_it(self) -> None:
        repo = ArtifactsRepository(base_dir=os.path.join(ROOT, "profiles"))
        factory: Any = BotFactory(_DummyMaze(), repository=repo)
        factory.register_bot("RepoCtor", _CtorWithRepository)

        bot: _CtorWithRepository = cast(
            _CtorWithRepository,
            factory.create_bot(
            "RepoCtor",
            "profileA",
            QLearningConfig(),
            RewardConfig(),
            BotStatistics(),
            {"k": "v"},
            ),
        )
        self.assertIs(bot.received["repository"], repo)
        self.assertEqual(bot.specific_data, {"k": "v"})

    def test_bot_factory_falls_back_for_legacy_constructor(self) -> None:
        factory: Any = BotFactory(_DummyMaze())
        factory.register_bot("LegacyCtor", _LegacyCtorNoRepository)

        bot: _LegacyCtorNoRepository = cast(
            _LegacyCtorNoRepository,
            factory.create_bot(
            "LegacyCtor",
            "profileB",
            QLearningConfig(),
            RewardConfig(),
            BotStatistics(),
            {"legacy": 1},
            ),
        )
        self.assertEqual(bot.received["profile_name"], "profileB")
        self.assertEqual(bot.specific_data, {"legacy": 1})

    def test_qlearning_state_to_key_accepts_mixed_runtime_types(self) -> None:
        q = QLearning(QLearningConfig())
        state = ("state-id", ("wall", 2), {"direction": "N"})
        q_any: Any = q
        key: tuple[Any, ...] = cast(tuple[Any, ...], q_any.state_to_key(state))
        self.assertEqual(key, state)


if __name__ == "__main__":
    unittest.main()
