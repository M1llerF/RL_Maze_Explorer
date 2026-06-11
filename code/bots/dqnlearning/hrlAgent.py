from __future__ import annotations

from dataclasses import dataclass

from bots.common.decision import ActionChoice, DecisionInput

from .agent import DqnAgent, DqnDiagnostics
from .config import DQNConfig
from .types import Transition


@dataclass(frozen=True)
class HierarchicalDqnDiagnostics:
    highLevel: DqnDiagnostics
    lowLevel: DqnDiagnostics

    @property
    def epsilon(self) -> float:
        return float(self.highLevel.epsilon)


class HierarchicalDqnAgent:
    """
    Two level DQN. The high level agent selects macro options over the full action horizon.
    The low level agent selects primitive actions conditioned on the current active option.
    The low level input appends a one hot encoding of the active option to the state vector.
    """

    def __init__(
        self,
        config: DQNConfig,
        *,
        inputDim: int,
        optionCount: int,
        primitiveCount: int,
        flatDim: int | None = None,
        mapShape: tuple[int, int, int] | None = None,
    ) -> None:
        self.highLevel = DqnAgent(
            config,
            inputDim=inputDim,
            numActions=optionCount,
            flatDim=flatDim,
            mapShape=mapShape,
        )
        self.lowLevel = DqnAgent(
            config,
            inputDim=inputDim + optionCount,
            numActions=primitiveCount,
            flatDim=(inputDim + optionCount),
            mapShape=None,
        )
        self.device = self.highLevel.device
        self.replay = self.highLevel.replay
        self.numActions = int(optionCount)

    @property
    def globalStep(self) -> int:
        return int(self.highLevel.globalStep)

    @globalStep.setter
    def globalStep(self, value: int) -> None:
        self.highLevel.globalStep = int(value)
        self.lowLevel.globalStep = int(value)

    @property
    def policy(self):  # noqa: ANN201 - torch module
        return self.highLevel.policy

    @property
    def target(self):  # noqa: ANN201 - torch module
        return self.highLevel.target

    @property
    def optimizer(self):  # noqa: ANN201 - torch optimizer
        return self.highLevel.optimizer

    def chooseHighLevelAction(self, decision: DecisionInput, training: bool) -> ActionChoice:
        return self.highLevel.chooseAction(decision, training=training)

    def chooseLowLevelAction(self, decision: DecisionInput, training: bool) -> ActionChoice:
        return self.lowLevel.chooseAction(decision, training=training)

    def storeHighTransition(self, transition: Transition) -> None:
        self.highLevel.storeTransition(transition)

    def storeLowTransition(self, transition: Transition) -> None:
        self.lowLevel.storeTransition(transition)

    def trainStep(self) -> bool:
        highTrained = self.highLevel.trainStep()
        lowTrained = self.lowLevel.trainStep()
        return bool(highTrained or lowTrained)

    def onEnvironmentStep(self) -> None:
        self.highLevel.onEnvironmentStep()
        self.lowLevel.onEnvironmentStep()

    def diagnostics(self) -> HierarchicalDqnDiagnostics:
        return HierarchicalDqnDiagnostics(
            highLevel=self.highLevel.diagnostics(),
            lowLevel=self.lowLevel.diagnostics(),
        )

    def setManualEpsilon(self, value: float | None) -> None:
        self.highLevel.setManualEpsilon(value)
        self.lowLevel.setManualEpsilon(value)
