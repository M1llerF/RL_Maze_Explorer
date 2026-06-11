from __future__ import annotations

from botProfile import BotProfile, ProfileManager
from botStatistics import BotStatistics
from bots.dqnlearning import DQNConfig
from bots.qlearning import QLearningConfig
from rewardSystem import RewardConfig
from services.repository import ArtifactsRepository


DEFAULT_DQN_PROFILE_NAME = "DQN_Default"
DEFAULT_QLEARNING_PROFILE_NAME = "QLearning_Default"


def _build_fast_reward_config() -> RewardConfig:
    reward_config = RewardConfig()
    reward_config.rewardModifiers = {
        # Reviewer preset: deliberately over-shaped to make competent behavior appear quickly.
        "goal_reached": "750",
        "hit_wall": "-20",
        "revisit_optimal_path": "8",
        "revisit_non_optimal_path": "-6",
        "new_tile_visited": "12",
        "move_in_optimal_path": "30",
        "see_goal_new_location": "90",
        "see_goal_revisit": "20",
        "per_move_penalty": "-0.25",
        "enemy_contact": "-150",
        "death_by_enemy": "-500",
        "enemy_killed": "250",
    }
    reward_config.usePotentialShaping = True
    reward_config.progressScale = 14.0
    return reward_config


def build_default_dqn_profile() -> BotProfile:
    config = DQNConfig(
        learningRate=3e-4,
        discountFactor=0.97,
        epsilonStart=0.35,
        epsilonEnd=0.01,
        epsilonDecaySteps=2500,
        replayCapacity=12000,
        replayWarmupSteps=256,
        batchSize=32,
        trainFrequency=1,
        targetUpdateFrequency=250,
        maxStepsPerEpisode=250,
        hiddenSize=96,
        useRichEncoding=False,
        useSharedComparisonState=True,
        usePositionInState=True,
        useEntityObservation=True,
        useEnemyObservation=False,
        useAttackActions=False,
        autoAttackAdjacentEnemy=False,
        pushCooldownSteps=2,
        useMacroActions=True,
        macroOptionSet="astar",
        useMacroOnlyPolicy=False,
        dualRecordPrimitivePolicy=False,
        useHierarchicalPolicy=False,
        useLstmPolicy=False,
        neuralMapWidth=15,
        neuralMapHeight=15,
        neuralMapPoolSize=4,
        immediateReversalPenalty=0.0,
        oscillationPenalty=-10.0,
        repeatVisitPenaltyScale=0.0,
        noProgressPenalty=-40.0,
        noProgressPatienceFactor=1.0,
        minNoProgressSteps=18,
        maxNoProgressSteps=150,
        diagnosticsFrequency=0,
        rewardClipMin=-25.0,
        rewardClipMax=25.0,
        mapEmbedDim=32,
        checkpointFrequency=25,
        evaluationFrequency=25,
        stepLimitStepCoeff=6,
        stepLimitAreaCoeff=0.2,
        stepLimitMax=250,
        stepLimitMin=60,
        stepLimitPenalty=-40.0,
    )
    return BotProfile(
        name=DEFAULT_DQN_PROFILE_NAME,
        botType="DQNBot",
        config=config,
        rewardConfig=_build_fast_reward_config(),
        statistics=BotStatistics(),
        botSpecificData={},
    )


def build_default_qlearning_profile() -> BotProfile:
    config = QLearningConfig(
        learningRate=0.35,
        discountFactor=0.95,
        epsilonStart=0.25,
        epsilonEnd=0.01,
        epsilonDecaySteps=1200,
        maxStepsPerEpisode=250,
        usePositionInState=True,
        useMacroActions=True,
        useMacroOnlyPolicy=False,
        useEntityObservation=True,
        useEnemyObservation=False,
        useAttackActions=False,
        autoAttackAdjacentEnemy=False,
        pushCooldownSteps=2,
        immediateReversalPenalty=0.0,
        repeatVisitPenaltyScale=0.0,
        noProgressPenalty=-40.0,
        noProgressPatienceFactor=1.0,
        minNoProgressSteps=18,
        maxNoProgressSteps=150,
        rewardClipMin=-25.0,
        rewardClipMax=25.0,
        stepLimitStepCoeff=6,
        stepLimitAreaCoeff=0.2,
        stepLimitMax=250,
        stepLimitMin=60,
        stepLimitPenalty=-40.0,
    )
    return BotProfile(
        name=DEFAULT_QLEARNING_PROFILE_NAME,
        botType="QLearningBot",
        config=config,
        rewardConfig=_build_fast_reward_config(),
        statistics=BotStatistics(),
        botSpecificData={},
    )


def ensure_default_profiles(
    profile_manager: ProfileManager,
    repository: ArtifactsRepository,
) -> None:
    existing = set(profile_manager.listProfiles())
    for profile in (build_default_dqn_profile(), build_default_qlearning_profile()):
        if profile.name in existing:
            repository.ensureProfileArtifacts(profile.name)
            continue
        profile_manager.saveProfile(profile)
        repository.ensureProfileArtifacts(profile.name)
