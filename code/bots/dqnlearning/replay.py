from __future__ import annotations

import random
from typing import Protocol

import numpy as np

from .types import DqnTrainingBatch, Transition


class ReplayStore(Protocol):
    def push(self, transition: Transition) -> None:
        ...

    def sample(self, batchSize: int) -> DqnTrainingBatch:
        ...

    def __len__(self) -> int:
        ...


class UniformReplayStore:
    def __init__(self, capacity: int, rngSeed: int | None = None) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.buffer: list[Transition | None] = [None] * self.capacity
        self.nextIndex = 0
        self.size = 0
        self._rng = random.Random(rngSeed)

    def push(self, transition: Transition) -> None:
        self.buffer[self.nextIndex] = transition
        self.nextIndex = (self.nextIndex + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batchSize: int) -> DqnTrainingBatch:
        batchSize = int(batchSize)
        if batchSize <= 0:
            raise ValueError(f"batchSize must be > 0, got {batchSize}")
        if self.size < batchSize:
            raise ValueError(f"not enough samples: size={self.size}, batchSize={batchSize}")
        indices = self._rng.sample(range(self.size), batchSize)
        samples = [self.buffer[i] for i in indices]
        if any(sample is None for sample in samples):
            raise RuntimeError("replay buffer contains uninitialized entries")
        transitions = [sample for sample in samples if sample is not None]
        return DqnTrainingBatch(
            states=np.stack([t.state for t in transitions]).astype(np.float32, copy=False),
            actions=np.asarray([t.action for t in transitions], dtype=np.int64),
            rewards=np.asarray([t.reward for t in transitions], dtype=np.float32),
            nextStates=np.stack([t.nextState for t in transitions]).astype(np.float32, copy=False),
            dones=np.asarray([t.done for t in transitions], dtype=np.float32),
            validActionMasks=np.stack([t.validActionMask for t in transitions]).astype(np.bool_, copy=False),
            nextValidActionMasks=np.stack([t.nextValidActionMask for t in transitions]).astype(np.bool_, copy=False),
        )

    def __len__(self) -> int:
        return self.size
