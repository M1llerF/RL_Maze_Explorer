from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .config import DQNConfig
from .model import DqnModel
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
    def __init__(
        self,
        config: DQNConfig,
        inputDim: int,
        replay: ReplayStore | None = None,
        numActions: int = 4,
    ) -> None:
        self.config = config
        self.inputDim = int(inputDim)
        self.numActions = int(numActions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DqnModel(self.inputDim, self.config.hiddenSize, self.numActions).to(self.device)
        self.target = DqnModel(self.inputDim, self.config.hiddenSize, self.numActions).to(self.device)
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

    def chooseAction(self, encodedState: EncodedState, training: bool) -> int:
        validActions = np.flatnonzero(encodedState.validActionMask).tolist()
        if not validActions:
            validActions = [0, 1, 2, 3]
        if training and random.random() < self._epsilon():
            return int(random.choice(validActions))
        with torch.no_grad():
            stateTensor = torch.as_tensor(encodedState.values, dtype=torch.float32, device=self.device).unsqueeze(0)
            qValues = self.policy(stateTensor)
            mask = torch.as_tensor(encodedState.validActionMask, dtype=torch.bool, device=self.device).unsqueeze(0)
            qValues = qValues.masked_fill(~mask, torch.finfo(qValues.dtype).min)
            return int(torch.argmax(qValues, dim=1).item())

    def selectWarmupAction(self, encodedState: EncodedState, plannerAction: int | None) -> int:
        valid = encodedState.validActionMask
        if plannerAction is not None and 0 <= plannerAction <= 3 and bool(valid[plannerAction]):
            self.plannerActionCount += 1
            return int(plannerAction)
        raise RuntimeError("Warmup planner action is invalid for the current action mask.")

    def storeTransition(self, transition: Transition) -> None:
        self.replay.push(transition)

    def trainStep(self) -> bool:
        if self.globalStep % self.config.trainFrequency != 0:
            return False
        if len(self.replay) < max(self.config.batchSize, self.config.replayWarmupSteps):
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
        decaySteps = max(1, int(self.config.epsilonDecaySteps))
        progress = min(1.0, float(self.globalStep) / float(decaySteps))
        return max(
            float(self.config.epsilonEnd),
            float(self.config.epsilonStart)
            + (float(self.config.epsilonEnd) - float(self.config.epsilonStart)) * progress,
        )

    def _optimize(self, batch: DqnTrainingBatch) -> float:
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device).clamp(
            float(self.config.rewardClipMin),
            float(self.config.rewardClipMax),
        )
        nextStates = torch.as_tensor(batch.nextStates, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        nextMasks = torch.as_tensor(batch.nextValidActionMasks, dtype=torch.bool, device=self.device)
        currentQ = self.policy(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            nextQ = self.target(nextStates).masked_fill(~nextMasks, torch.finfo(torch.float32).min)
            maxNextQ = torch.max(nextQ, dim=1).values
            targetQ = rewards + float(self.config.discountFactor) * maxNextQ * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(currentQ, targetQ, reduction="mean")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.detach().cpu().item())
