from __future__ import annotations

import dataclasses
import os
import shutil
from typing import Any

from botFactory import BotFactory
from botProfile import BotProfile, ProfileManager
from botStatistics import BotStatistics
from rewardSystem import RewardConfig
from services.repository import ArtifactsRepository
from services.research.plan import ConditionSpec, ProfileSpec


SHARED_REWARD_MODIFIERS: dict[str, str] = {
    "goal_reached": "1000",
    "hit_wall": "-100",
    "revisit_optimal_path": "0",
    "revisit_non_optimal_path": "0",
    "new_tile_visited": "0",
    "move_in_optimal_path": "0",
    "see_goal_new_location": "0",
    "see_goal_revisit": "0",
    "per_move_penalty": "-1",
    "enemy_contact": "-250",
    "death_by_enemy": "-1000",
    "enemy_killed": "500",
}


def build_shared_reward_config() -> RewardConfig:
    rc = RewardConfig()
    rc.rewardModifiers = dict(SHARED_REWARD_MODIFIERS)
    rc.usePotentialShaping = True
    rc.progressScale = 2.0
    return rc


def _dqn_budget_params(
    condition: ConditionSpec,
    *,
    replay_warmup_policy: str,
) -> tuple[int, int]:
    """Compute (replayWarmupSteps, epsilonDecaySteps) scaled to the condition's training budget."""
    area = condition.width * condition.height
    approx_optimal = int(1.5 * (condition.width + condition.height))
    step_limit_est = max(200, 12 * approx_optimal + int(0.5 * area))
    total_steps_est = condition.training_episodes * step_limit_est

    if replay_warmup_policy == "disabled":
        warmup = 0
    elif area <= 100:
        warmup = 0
    else:
        warmup = 2000

    decay = min(100000, max(10000, total_steps_est // 2))
    return warmup, decay


def _lstm_memory_params(condition: ConditionSpec) -> tuple[int, int]:
    """Scale recurrent memory with maze size while keeping small-maze baselines stable."""
    area = int(condition.width) * int(condition.height)
    if area >= 625:  # 25x25 and larger
        return 32, 256
    if area >= 225:  # 15x15 class
        return 16, 192
    return 8, 128


def build_dqn_flat_config(
    condition: ConditionSpec,
    *,
    replay_warmup_policy: str,
) -> Any:
    from bots.dqnlearning.config import DQNConfig

    warmup_steps, epsilon_decay_steps = _dqn_budget_params(
        condition,
        replay_warmup_policy=replay_warmup_policy,
    )
    return DQNConfig(
        learningRate=0.0001,
        discountFactor=0.99,
        epsilonStart=1.0,
        epsilonEnd=0.05,
        epsilonDecaySteps=epsilon_decay_steps,
        replayCapacity=50000,
        replayWarmupSteps=warmup_steps,
        batchSize=64,
        trainFrequency=4,
        targetUpdateFrequency=2000,
        maxStepsPerEpisode=5000,
        hiddenSize=256,
        useRichEncoding=False,
        useSharedComparisonState=False,
        usePositionInState=True,
        useEntityObservation=True,
        useEnemyObservation=False,
        useAttackActions=False,
        autoAttackAdjacentEnemy=False,
        pushCooldownSteps=3,
        useMacroActions=False,
        useMacroOnlyPolicy=False,
        dualRecordPrimitivePolicy=False,
        useHierarchicalPolicy=False,
        neuralMapWidth=31,
        neuralMapHeight=31,
        neuralMapPoolSize=8,
        immediateReversalPenalty=0.0,
        repeatVisitPenaltyScale=0.0,
        noProgressPenalty=-150.0,
        noProgressPatienceFactor=2.0,
        minNoProgressSteps=30,
        maxNoProgressSteps=1000,
        rewardClipMin=None,
        rewardClipMax=None,
        stateEncodingVersion=7,
        mapEmbedDim=128,
        checkpointFrequency=10,
        evaluationFrequency=10,
        stepLimitStepCoeff=12,
        stepLimitAreaCoeff=0.5,
        stepLimitMax=5000,
        stepLimitMin=200,
        stepLimitPenalty=-150.0,
        diagnosticsFrequency=0,
    )


def build_dqn_local_lstm_config(
    condition: ConditionSpec,
    *,
    replay_warmup_policy: str,
) -> Any:
    cfg = build_dqn_flat_config(condition, replay_warmup_policy=replay_warmup_policy)
    from bots.dqnlearning.config import DQNConfig

    d = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    lstm_sequence_length, lstm_hidden_size = _lstm_memory_params(condition)
    d["useLstmPolicy"] = True
    d["lstmSequenceLength"] = lstm_sequence_length
    d["lstmHiddenSize"] = lstm_hidden_size
    return DQNConfig(**d)


def build_dqn_lstm_macro_config(
    condition: ConditionSpec,
    *,
    replay_warmup_policy: str,
    macro_option_set: str,
) -> Any:
    cfg = build_dqn_local_lstm_config(condition, replay_warmup_policy=replay_warmup_policy)
    from bots.dqnlearning.config import DQNConfig

    d = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    d.update(
        {
            "useMacroActions": True,
            "macroOptionSet": str(macro_option_set),
            "useHierarchicalPolicy": False,
            "useMacroOnlyPolicy": False,
        }
    )
    return DQNConfig(**d)


def build_config_for_agent(
    agent_type: str,
    condition: ConditionSpec,
    *,
    replay_warmup_policy: str,
) -> Any:
    if agent_type == "DQN_LSTM_PRIMITIVE":
        return build_dqn_local_lstm_config(condition, replay_warmup_policy=replay_warmup_policy)
    if agent_type == "DQN_LSTM_NAIVE_MACROS":
        return build_dqn_lstm_macro_config(
            condition,
            replay_warmup_policy=replay_warmup_policy,
            macro_option_set="naive",
        )
    if agent_type in ("DQN_LSTM_ASTAR_MACROS", "DQN_LSTM_ORACLE_MACROS"):
        return build_dqn_lstm_macro_config(
            condition,
            replay_warmup_policy=replay_warmup_policy,
            macro_option_set="astar",
        )
    if agent_type == "DQN_LSTM_MOMENTUM_MACROS":
        return build_dqn_lstm_macro_config(
            condition,
            replay_warmup_policy=replay_warmup_policy,
            macro_option_set="momentum",
        )
    raise ValueError(f"Unsupported research-mode agent type: {agent_type}")


def create_profile(
    spec: ProfileSpec,
    profile_dir: str,
    *,
    replay_warmup_policy: str,
) -> BotProfile:
    """Create and save a fresh BotProfile for a research run."""
    config = build_config_for_agent(
        spec.agent_type,
        spec.condition,
        replay_warmup_policy=replay_warmup_policy,
    )
    reward_config = build_shared_reward_config()
    profile = BotProfile(
        name=spec.profile_name,
        botType=spec.bot_type,
        config=config,
        rewardConfig=reward_config,
        statistics=BotStatistics(),
        botSpecificData={},
    )
    manager = ProfileManager(profile_dir)
    manager.saveProfile(profile)
    repo = ArtifactsRepository(baseDir=profile_dir)
    repo.ensureProfileArtifacts(spec.profile_name)
    return profile


def delete_profile(profile_name: str, profile_dir: str) -> None:
    target = os.path.join(profile_dir, profile_name)
    if os.path.exists(target):
        shutil.rmtree(target)


def build_bot(spec: ProfileSpec, maze: Any, profile_dir: str) -> Any:
    """Instantiate and return a fully initialized bot for research use."""
    from bots import discoverBotClasses

    repo = ArtifactsRepository(baseDir=profile_dir)
    factory = BotFactory(maze=maze, repository=repo)
    for bot_type, bot_class in discoverBotClasses().items():
        factory.registerBot(bot_type, bot_class)

    manager = ProfileManager(profile_dir)
    profile = manager.loadProfile(spec.profile_name)

    return factory.createBot(
        botType=profile.botType,
        profileName=spec.profile_name,
        config=profile.config,
        rewardConfig=profile.rewardConfig,
        statistics=profile.statistics,
        botSpecificData=profile.botSpecificData,
        loadCheckpoint=True,
    )
