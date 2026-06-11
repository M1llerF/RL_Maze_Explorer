from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from bots.common.decision import ActionChoice, DecisionInput
from .config import DQNConfig
from .model import DqnModel, RecurrentDqnModel
from .replay import ReplayStore, UniformReplayStore
from .types import DqnTrainingBatch, EncodedState, Transition


@dataclass(frozen=True)
class DqnDiagnostics:
    globalStep: int
    replaySize: int
    epsilon: float
    trainingUpdates: int
    plannerActionCount: int
    randomFallbackCount: int
    warmupCompletionStep: int | None
    lastLoss: float | None


class DqnAgent:
    """
    Double DQN agent. The policy network selects actions and the target network evaluates them.
    Decoupling selection from evaluation reduces Q value overestimation bias common in plain DQN.
    The target network is periodically hard synced from policy network weights.
    """

    def __init__(
        self,
        config: DQNConfig,
        inputDim: int,
        replay: ReplayStore | None = None,
        numActions: int = 4,
        flatDim: int | None = None,
        mapShape: tuple[int, int, int] | None = None,
    ) -> None:
        self.config = config
        self.inputDim = int(inputDim)
        self.numActions = int(numActions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolvedFlatDim = int(flatDim) if flatDim is not None else self.inputDim
        if getattr(self.config, "useLstmPolicy", False):
            sequenceLength = int(getattr(self.config, "lstmSequenceLength", 1))
            if self.inputDim % max(1, sequenceLength) != 0:
                raise ValueError(
                    f"LSTM policy inputDim={self.inputDim} is not divisible by sequenceLength={sequenceLength}"
                )
            stepDim = max(1, int(self.inputDim) // max(1, sequenceLength))
            self.policy = RecurrentDqnModel(
                stepDim=stepDim,
                sequenceLength=sequenceLength,
                hiddenSize=self.config.hiddenSize,
                lstmHiddenSize=int(getattr(self.config, "lstmHiddenSize", self.config.hiddenSize)),
                numActions=self.numActions,
            ).to(self.device)
            self.target = RecurrentDqnModel(
                stepDim=stepDim,
                sequenceLength=sequenceLength,
                hiddenSize=self.config.hiddenSize,
                lstmHiddenSize=int(getattr(self.config, "lstmHiddenSize", self.config.hiddenSize)),
                numActions=self.numActions,
            ).to(self.device)
        else:
            mapEmbedDim = int(getattr(config, "mapEmbedDim", 0)) if mapShape is not None else 0
            self.policy = DqnModel(
                flatDim=resolvedFlatDim,
                hiddenSize=self.config.hiddenSize,
                numActions=self.numActions,
                mapShape=mapShape,
                mapEmbedDim=mapEmbedDim,
            ).to(self.device)
            self.target = DqnModel(
                flatDim=resolvedFlatDim,
                hiddenSize=self.config.hiddenSize,
                numActions=self.numActions,
                mapShape=mapShape,
                mapEmbedDim=mapEmbedDim,
            ).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = Adam(self.policy.parameters(), lr=self.config.learningRate)
        self.replay = replay or UniformReplayStore(self.config.replayCapacity)
        self.globalStep = 0
        self.episodeCount = 0
        self.trainingUpdates = 0
        self.plannerActionCount = 0
        self.randomFallbackCount = 0
        self.warmupCompletionStep: int | None = None
        self.lastLoss: float | None = None
        self.manualEpsilonOverride: float | None = None

    def chooseAction(self, decision: DecisionInput, training: bool) -> ActionChoice:
        encodedState = decision.state
        if not isinstance(encodedState, EncodedState):
            raise TypeError("DqnAgent expects DecisionInput.state to be an EncodedState")
        validActions = np.flatnonzero(encodedState.validActionMask).tolist()
        if not validActions:
            validActions = list(range(self.numActions))
        if training and random.random() < self._epsilon():
            return ActionChoice(local_id=int(random.choice(validActions)))
        with torch.inference_mode():
            stateTensor = torch.as_tensor(encodedState.values, dtype=torch.float32, device=self.device).unsqueeze(0)
            qValues = self.policy(stateTensor)
            mask = torch.as_tensor(encodedState.validActionMask, dtype=torch.bool, device=self.device).unsqueeze(0)
            qValues = qValues.masked_fill(~mask, torch.finfo(qValues.dtype).min)
            return ActionChoice(local_id=int(torch.argmax(qValues, dim=1).item()))

    def selectWarmupAction(self, decision: DecisionInput, plannerAction: int | None) -> ActionChoice:
        encodedState = decision.state
        if not isinstance(encodedState, EncodedState):
            raise TypeError("DqnAgent expects DecisionInput.state to be an EncodedState")
        valid = encodedState.validActionMask
        if plannerAction is not None and 0 <= plannerAction < self.numActions and bool(valid[plannerAction]):
            self.plannerActionCount += 1
            return ActionChoice(local_id=int(plannerAction))
        validActions = np.flatnonzero(valid).tolist()
        if not validActions:
            validActions = list(range(self.numActions))
        self.randomFallbackCount += 1
        return ActionChoice(local_id=int(random.choice(validActions)))

    def storeTransition(self, transition: Transition) -> None:
        self.replay.push(transition)

    def trainStep(self) -> bool:
        if self.globalStep % self.config.trainFrequency != 0:
            return False
        if len(self.replay) < self.config.batchSize:
            return False
        if self.warmupCompletionStep is None:
            self.warmupCompletionStep = self.globalStep
        batch = self.replay.sample(self.config.batchSize)
        loss = self._optimize(batch)
        self.lastLoss = float(loss)
        self.trainingUpdates += 1
        return True

    def onEnvironmentStep(self) -> None:
        self.globalStep += 1
        if self.globalStep % self.config.targetUpdateFrequency == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def diagnostics(self) -> DqnDiagnostics:
        return DqnDiagnostics(
            globalStep=self.globalStep,
            replaySize=len(self.replay),
            epsilon=self._epsilon(),
            trainingUpdates=self.trainingUpdates,
            plannerActionCount=self.plannerActionCount,
            randomFallbackCount=self.randomFallbackCount,
            warmupCompletionStep=self.warmupCompletionStep,
            lastLoss=self.lastLoss,
        )

    def _epsilon(self) -> float:
        # Linear decay rather than exponential. Linear gives a predictable exploration
        # budget across a known step horizon without requiring a separate decay rate.
        if self.manualEpsilonOverride is not None:
            return float(self.manualEpsilonOverride)
        decaySteps = max(1, int(self.config.epsilonDecaySteps))
        progress = min(1.0, float(self.globalStep) / float(decaySteps))
        return max(
            float(self.config.epsilonEnd),
            float(self.config.epsilonStart)
            + (float(self.config.epsilonEnd) - float(self.config.epsilonStart)) * progress,
        )

    def setManualEpsilon(self, value: float | None) -> None:
        if value is None:
            self.manualEpsilonOverride = None
            return
        epsilon = float(value)
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Manual epsilon must be between 0.0 and 1.0.")
        self.manualEpsilonOverride = epsilon

    def _optimize(self, batch: DqnTrainingBatch) -> float:
        # Double DQN update. Policy net picks the greedy next action and target net scores it.
        # Separating selection from evaluation reduces overestimation bias.
        # bootstrapDiscounts carries per transition gamma powers for macro action SMDP support.
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        nextStates = torch.as_tensor(batch.nextStates, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        nextMasks = torch.as_tensor(batch.nextValidActionMasks, dtype=torch.bool, device=self.device)
        bootstrapDiscounts = torch.as_tensor(batch.bootstrapDiscounts, dtype=torch.float32, device=self.device)
        currentQ = self.policy(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            nextPolicyQ = self.policy(nextStates).masked_fill(~nextMasks, torch.finfo(torch.float32).min)
            nextActions = torch.argmax(nextPolicyQ, dim=1, keepdim=True)
            nextTargetQ = self.target(nextStates).gather(1, nextActions).squeeze(1)
            targetQ = rewards + bootstrapDiscounts * nextTargetQ * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(currentQ, targetQ, reduction="mean")
        self.optimizer.zero_grad()
        torch.autograd.backward(loss)
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.detach().cpu().item())
