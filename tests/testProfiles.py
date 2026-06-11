from __future__ import annotations

from typing import Any

import pytest

from botConfigs import botConfigs
from botFactory import BotFactory
from botProfile import BotProfile, ProfileManager
from botStatistics import BotStatistics
from defaultProfiles import (
    DEFAULT_DQN_PROFILE_NAME,
    DEFAULT_QLEARNING_PROFILE_NAME,
    ensure_default_profiles,
)
from bots.dqnlearning import DQNConfig
from bots.qlearning import QLearningConfig
from rewardSystem import RewardConfig
from services.repository import ArtifactsRepository


class _DummyMaze:
    def __init__(self) -> None:
        self.width = 2
        self.height = 2
        self.grid = [[0, 0], [0, 0]]
        self.end = (1, 1)

    def isValidPosition(self, _profile_name: str | None, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width


class _ContextCaptureBot:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.specific_data: dict[str, Any] | None = None

    def initializeSpecificData(self, data: dict[str, Any]) -> None:
        self.specific_data = data


class _FutureConfig:
    def __init__(self, alpha: float = 0.7, beta: int = 2) -> None:
        self.alpha = alpha
        self.beta = beta


def test_profile_from_dict_uses_registered_config_class() -> None:
    botConfigs["FutureBot"] = {"class": _FutureConfig}
    try:
        profile = BotProfile.fromDict(
            {
                "name": "future_profile",
                "bot_type": "FutureBot",
                "config": {"alpha": 1.5, "beta": 4, "ignored": 123},
            }
        )
    finally:
        botConfigs.pop("FutureBot", None)

    assert profile.botType == "FutureBot"
    assert isinstance(profile.config, _FutureConfig)
    assert profile.config.alpha == 1.5
    assert profile.config.beta == 4
    assert not hasattr(profile.config, "ignored")


def test_bot_factory_rejects_unknown_bot_type() -> None:
    factory = BotFactory(_DummyMaze())

    with pytest.raises(ValueError, match=r"Unknown bot type: MissingBot\. Registered bot types: <none>"):
        factory.createBot(
            "MissingBot",
            "profile",
            QLearningConfig(),
            RewardConfig(),
            BotStatistics(),
            {},
        )


def test_bot_factory_passes_single_context_and_repository(tmp_path) -> None:
    repo = ArtifactsRepository(baseDir=str(tmp_path))
    factory = BotFactory(_DummyMaze(), repository=repo)
    factory.registerBot("ContextCaptureBot", _ContextCaptureBot)

    bot = factory.createBot(
        "ContextCaptureBot",
        "profileA",
        QLearningConfig(),
        RewardConfig(),
        BotStatistics(),
        {"tag": "x"},
    )

    assert bot.ctx.profileName == "profileA"
    assert isinstance(bot.ctx.config, QLearningConfig)
    assert bot.ctx.repository is repo
    assert bot.specific_data == {"tag": "x"}


def test_profile_manager_round_trips_qlearning_profile(tmp_path) -> None:
    manager = ProfileManager(str(tmp_path))
    source = BotProfile(
        name="roundtrip_q",
        botType="QLearningBot",
        config=QLearningConfig(learningRate=0.2, discountFactor=0.85, usePositionInState=False),
        rewardConfig=RewardConfig(),
        statistics=BotStatistics(),
        botSpecificData={"tag": "x"},
    )

    manager.saveProfile(source)
    loaded = manager.loadProfile("roundtrip_q")

    assert loaded.name == "roundtrip_q"
    assert loaded.botType == "QLearningBot"
    assert isinstance(loaded.config, QLearningConfig)
    assert loaded.config.learningRate == 0.2
    assert loaded.config.discountFactor == 0.85
    assert loaded.config.usePositionInState is False
    assert loaded.botSpecificData == {"tag": "x"}


def test_profile_manager_round_trips_dqn_profile(tmp_path) -> None:
    manager = ProfileManager(str(tmp_path))
    config = DQNConfig.from_profile_dict(
        {
            "learningRate": 5e-4,
            "discountFactor": 0.99,
            "replayCapacity": 12345,
            "useRichEncoding": 1,
            "epsilonDecaySteps": 54321,
        }
    )
    source = BotProfile(
        name="roundtrip_dqn",
        botType="DQNBot",
        config=config,
        rewardConfig=RewardConfig(),
        statistics=BotStatistics(),
        botSpecificData={"note": "dqn"},
    )

    manager.saveProfile(source)
    loaded = manager.loadProfile("roundtrip_dqn")

    assert loaded.botType == "DQNBot"
    assert isinstance(loaded.config, DQNConfig)
    assert loaded.config.replayCapacity == 12345
    assert loaded.config.epsilonDecaySteps == 54321
    assert loaded.config.useRichEncoding is True


def test_profile_manager_round_trips_reward_modifiers(tmp_path) -> None:
    manager = ProfileManager(str(tmp_path))
    reward_config = RewardConfig()
    reward_config.rewardModifiers["per_move_penalty"] = "-0.5"
    reward_config.rewardModifiers["goal_reached"] = "1234"
    source = BotProfile(
        name="roundtrip_reward",
        botType="DQNBot",
        config=DQNConfig(),
        rewardConfig=reward_config,
        statistics=BotStatistics(),
        botSpecificData={},
    )

    manager.saveProfile(source)
    loaded = manager.loadProfile("roundtrip_reward")

    assert loaded.rewardConfig.rewardModifiers["per_move_penalty"] == "-0.5"
    assert loaded.rewardConfig.rewardModifiers["goal_reached"] == "1234"


def test_ensure_default_profiles_creates_reviewer_presets(tmp_path) -> None:
    manager = ProfileManager(str(tmp_path))
    repository = ArtifactsRepository(baseDir=str(tmp_path))

    ensure_default_profiles(manager, repository)

    names = set(manager.listProfiles())
    assert DEFAULT_DQN_PROFILE_NAME in names
    assert DEFAULT_QLEARNING_PROFILE_NAME in names

    dqn_profile = manager.loadProfile(DEFAULT_DQN_PROFILE_NAME)
    q_profile = manager.loadProfile(DEFAULT_QLEARNING_PROFILE_NAME)

    assert isinstance(dqn_profile.config, DQNConfig)
    assert dqn_profile.config.useMacroActions is True
    assert dqn_profile.config.macroOptionSet == "astar"
    assert dqn_profile.rewardConfig.usePotentialShaping is True
    assert dqn_profile.rewardConfig.progressScale == 14.0

    assert isinstance(q_profile.config, QLearningConfig)
    assert q_profile.config.useMacroActions is True
    assert q_profile.rewardConfig.rewardModifiers["move_in_optimal_path"] == "30"


def test_ensure_default_profiles_does_not_overwrite_existing_profiles(tmp_path) -> None:
    manager = ProfileManager(str(tmp_path))
    repository = ArtifactsRepository(baseDir=str(tmp_path))
    existing = BotProfile(
        name=DEFAULT_DQN_PROFILE_NAME,
        botType="DQNBot",
        config=DQNConfig(learningRate=9e-4),
        rewardConfig=RewardConfig(),
        statistics=BotStatistics(),
        botSpecificData={"edited": True},
    )
    manager.saveProfile(existing)

    ensure_default_profiles(manager, repository)

    loaded = manager.loadProfile(DEFAULT_DQN_PROFILE_NAME)
    assert loaded.config.learningRate == 9e-4
    assert loaded.botSpecificData == {"edited": True}
