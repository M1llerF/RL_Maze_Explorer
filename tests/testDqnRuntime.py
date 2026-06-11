from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bots.common.decision import DecisionInput
from bots.dqnlearning import (
    CheckpointIO,
    CheckpointMeta,
    DQNCheckpointService,
    DQNConfig,
    DqnAgent,
    StateEncoder,
    UniformReplayStore,
    WarmupCoordinator,
)
from bots.dqnlearning.types import EncodedState, Transition
from services.episodeResult import EpisodeResult
from services.research.metrics import episode_to_row
from services.repository import ArtifactsRepository


class _WarmupActionSpace:
    numActions = 4

    def localId(self, semantic_id: int | None) -> int | None:
        if semantic_id is None:
            return None
        return int(semantic_id)


class _StubPlanner:
    def __init__(self, action: int | None) -> None:
        self.action = action
        self.positions: list[tuple[int, int]] = []

    def nextAction(self, position: tuple[int, int]) -> int | None:
        self.positions.append(position)
        return self.action

    def reset(self) -> None:
        return

    def onWallDiscovered(self) -> None:
        return


class _CheckpointBot:
    def __init__(
        self,
        *,
        repo: ArtifactsRepository,
        profile_name: str,
        agent: DqnAgent,
        config: DQNConfig,
    ) -> None:
        self.repo = repo
        self.profileName = profile_name
        self.agent = agent
        self.config = config
        self.episodeCounter = 0
        self.primitiveBootstrapAgent = None
        self.usesHierarchicalPolicy = False
        self.usesMacroOnlyPolicy = False

    def _primitiveActionSpace(self) -> Any:
        class _Space:
            numActions = 4

        return _Space()


def test_state_encoder_is_stable_and_config_scoped() -> None:
    base_config = DQNConfig.from_profile_dict({"useRichEncoding": False})
    encoder = StateEncoder(base_config, mazeHeight=8, mazeWidth=8)
    observation = ((1, 1), (1, 2, 3, 4), (0, 0), -1, (), (), (1, 1, 0, 1))

    first = encoder.encode(observation)
    second = encoder.encode(observation)

    assert first.values.dtype == np.float32
    assert first.validActionMask.shape == (4,)
    assert np.array_equal(first.values, second.values)

    fingerprint_a = encoder.inferSchema(observation).encoderConfigFingerprint
    rich_config = DQNConfig.from_profile_dict({"useRichEncoding": True})
    fingerprint_b = StateEncoder(rich_config, mazeHeight=8, mazeWidth=8).inferSchema(
        observation
    ).encoderConfigFingerprint

    assert fingerprint_a != fingerprint_b


def test_replay_store_sample_shapes_and_dtypes() -> None:
    replay = UniformReplayStore(capacity=2, rngSeed=7)
    transitions = [
        Transition(
            np.array([1, 2], dtype=np.float32),
            0,
            1.0,
            np.array([1, 3], dtype=np.float32),
            False,
            np.ones(4, dtype=bool),
            np.ones(4, dtype=bool),
        ),
        Transition(
            np.array([2, 2], dtype=np.float32),
            1,
            0.0,
            np.array([2, 3], dtype=np.float32),
            False,
            np.ones(4, dtype=bool),
            np.ones(4, dtype=bool),
        ),
        Transition(
            np.array([3, 2], dtype=np.float32),
            2,
            -1.0,
            np.array([3, 3], dtype=np.float32),
            True,
            np.ones(4, dtype=bool),
            np.ones(4, dtype=bool),
        ),
    ]

    for transition in transitions:
        replay.push(transition)

    batch = replay.sample(2)

    assert len(replay) == 2
    assert batch.states.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert batch.rewards.dtype == np.float32
    assert batch.dones.dtype == np.float32
    assert batch.validActionMasks.dtype == np.bool_
    assert batch.nextValidActionMasks.shape == (2, 4)


def test_warmup_coordinator_prefers_planner_action_during_warmup() -> None:
    config = DQNConfig.from_profile_dict({})
    agent = DqnAgent(config, inputDim=3, replay=UniformReplayStore(4, rngSeed=1))
    planner = _StubPlanner(action=1)
    coordinator = WarmupCoordinator(planner)
    decision = DecisionInput(
        state=EncodedState(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([False, True, False, False], dtype=bool),
        ),
        actionSpace=_WarmupActionSpace(),
        level="flat",
    )

    class _Bot:
        def __init__(self) -> None:
            self.agent = agent
            self.position = (2, 3)
            self.usesMacroOnlyPolicy = False
            self.isWarmingUp = True

    choice = coordinator.selectAction(_Bot(), decision, training=True)

    assert choice is not None
    assert choice.local_id == 1
    assert planner.positions == [(2, 3)]
    assert agent.plannerActionCount == 1


def test_agent_respects_valid_action_mask_and_training_cadence() -> None:
    config = DQNConfig(batchSize=2, replayWarmupSteps=2, trainFrequency=1, targetUpdateFrequency=2)
    agent = DqnAgent(config, inputDim=3, replay=UniformReplayStore(32, rngSeed=2))
    state = EncodedState(
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([False, True, False, False], dtype=bool),
    )
    decision = DecisionInput(state=state, actionSpace=_WarmupActionSpace(), level="flat")
    transition = Transition(
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        1,
        1.0,
        np.array([0.2, 0.2, 0.3], dtype=np.float32),
        False,
        np.array([False, True, False, False], dtype=bool),
        np.array([False, True, False, False], dtype=bool),
    )

    assert agent.chooseAction(decision, training=False).local_id == 1

    agent.storeTransition(transition)
    assert agent.trainStep() is False

    agent.storeTransition(transition)
    assert agent.trainStep() is True
    assert agent.trainingUpdates == 1

    agent.onEnvironmentStep()
    agent.onEnvironmentStep()
    assert agent.globalStep == 2


def test_checkpoint_io_rejects_mismatched_meta(tmp_path) -> None:
    repo = ArtifactsRepository(baseDir=str(tmp_path))
    checkpoint = CheckpointIO(repo, "profile", torch.device("cpu"))
    meta = CheckpointMeta(2, "abc", 10)

    checkpoint.save(
        policyStateDict={"x": 1},
        targetStateDict={"x": 1},
        optimizerStateDict={"x": 1},
        globalStep=5,
        episodeCount=2,
        meta=meta,
        replayState={"capacity": 4, "size": 0, "next_index": 0, "entries": []},
    )

    assert checkpoint.load(meta) is not None
    assert checkpoint.load(CheckpointMeta(3, "abc", 10)) is None


def test_checkpoint_service_restores_training_state(tmp_path) -> None:
    repo = ArtifactsRepository(baseDir=str(tmp_path))
    config = DQNConfig(batchSize=2, replayWarmupSteps=2, checkpointFrequency=1)
    agent = DqnAgent(config, inputDim=3, replay=UniformReplayStore(8, rngSeed=3))
    meta = CheckpointMeta(2, "fp", 3, actionDim=4, policyMode="flat")
    service = DQNCheckpointService(
        CheckpointIO(repo, "profile1", agent.device),
        CheckpointIO(repo, "profile1", agent.device, artifactName="dqn_model_primtive.pt"),
        meta,
        meta,
        profileName="profile1",
    )
    source_bot = _CheckpointBot(repo=repo, profile_name="profile1", agent=agent, config=config)
    source_bot.episodeCounter = 7
    agent.globalStep = 11
    agent.setManualEpsilon(0.25)

    service.save(source_bot)

    restored_agent = DqnAgent(config, inputDim=3, replay=UniformReplayStore(8, rngSeed=4))
    restored_bot = _CheckpointBot(
        repo=repo,
        profile_name="profile1",
        agent=restored_agent,
        config=config,
    )

    assert service.load(restored_bot) is True
    assert restored_bot.episodeCounter == 7
    assert restored_agent.globalStep == 11
    assert restored_agent.manualEpsilonOverride == 0.25


def test_episode_metrics_record_macro_duration_fields() -> None:
    class _Condition:
        condition_id = "fixed_small"
        mode = "fixed"
        width = 5
        height = 5

    class _Spec:
        profile_name = "profile"
        agent_type = "DQN_LSTM_NAIVE_MACROS"
        bot_type = "DQNBot"
        condition = _Condition()
        seed_label = "s0"
        seed = 123

    result = EpisodeResult(
        profile_name="profile",
        mode="training",
        outcome="goal_reached",
        success=True,
        total_reward=10.0,
        steps=12,
        optimal_steps=6,
        times_hit_wall=0,
        heatmap_data={},
        maze=None,
        decisions=3,
        option_selections=2,
        option_steps=10,
    )

    row = episode_to_row(
        result=result,
        spec=_Spec(),
        episode_num=1,
        global_episode=1,
        phase="training",
        bot=object(),
        elapsed_seconds=0.1,
        experiment_name="exp",
        maze_id="maze0",
    )

    assert row["decisions"] == 3
    assert row["mean_action_duration"] == 4.0
    assert row["option_selections"] == 2
    assert row["option_steps"] == 10
    assert row["option_step_fraction"] == 0.8333
