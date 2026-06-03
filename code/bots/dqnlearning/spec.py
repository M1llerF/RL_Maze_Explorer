from __future__ import annotations

from typing import Any

from .config import DQNConfig

# UI display strings and reward defaults for the DQN bot.
# Separated from config.py so the algorithm module stays free of presentation concerns.
BOT_SPEC: dict[str, Any] = {
    "type": "DQNBot",
    "class": DQNConfig,
    "params": {
        "Learning Rate": "learningRate",
        "Discount Factor": "discountFactor",
        "Epsilon Start": "epsilonStart",
        "Epsilon End": "epsilonEnd",
        "Epsilon Decay Steps": "epsilonDecaySteps",
        "Replay Capacity": "replayCapacity",
        "Batch Size": "batchSize",
        "Train Frequency": "trainFrequency",
        "Target Update Frequency": "targetUpdateFrequency",
        "Hidden Size": "hiddenSize",
        "Max Steps Per Episode": "maxStepsPerEpisode",
        "Use Rich Encoding (0/1)": "useRichEncoding",
        "Use Position In State (0/1)": "usePositionInState",
        "Neural Map Width": "neuralMapWidth",
        "Neural Map Height": "neuralMapHeight",
        "Neural Map Pool Size": "neuralMapPoolSize",
        "Immediate Reversal Penalty": "immediateReversalPenalty",
        "Repeat Visit Penalty Scale": "repeatVisitPenaltyScale",
        "No Progress Penalty": "noProgressPenalty",
        "No Progress Patience Factor": "noProgressPatienceFactor",
        "Min No Progress Steps": "minNoProgressSteps",
        "Max No Progress Steps": "maxNoProgressSteps",
        "Reward Clip Min": "rewardClipMin",
        "Reward Clip Max": "rewardClipMax",
        "Map Embed Dim": "mapEmbedDim",
        "Checkpoint Frequency": "checkpointFrequency",
        "Diagnostics Frequency": "diagnosticsFrequency",
    },
    "rewards": {
        "goal_reached": 1000,
        "hit_wall": -100,
        "revisit_optimal_path": -10,
        "revisit_non_optimal_path": -15,
        "move_in_optimal_path": 5,
        "see_goal_new_location": 50,
        "see_goal_revisit": 5,
        "per_move_penalty": -1,
    },
}
