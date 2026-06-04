from __future__ import annotations

import random
from typing import Any, Protocol, cast

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
            bootstrapDiscounts=np.asarray([t.bootstrapDiscount for t in transitions], dtype=np.float32),
        )

    def __len__(self) -> int:
        return self.size

    def to_serializable(self) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        for i in range(self.size):
            t = self.buffer[i]
            if t is None:
                continue
            entries.append(
                {
                    "state": np.asarray(t.state, dtype=np.float32).tolist(),
                    "action": int(t.action),
                    "reward": float(t.reward),
                    "next_state": np.asarray(t.nextState, dtype=np.float32).tolist(),
                    "done": bool(t.done),
                    "valid_action_mask": np.asarray(t.validActionMask, dtype=np.bool_).tolist(),
                    "next_valid_action_mask": np.asarray(t.nextValidActionMask, dtype=np.bool_).tolist(),
                    "bootstrap_discount": float(t.bootstrapDiscount),
                }
            )
        return {
            "capacity": int(self.capacity),
            "size": int(self.size),
            "next_index": int(self.nextIndex),
            "entries": entries,
        }

    def load_serializable(self, data: dict[str, Any]) -> None:
        capacity = int(data.get("capacity", self.capacity))
        size = int(data.get("size", 0))
        nextIndex = int(data.get("next_index", 0))
        rawEntries = data.get("entries", [])
        if not isinstance(rawEntries, list):
            raise ValueError("replay entries payload is invalid")
        entries: list[Any] = cast(list[Any], rawEntries)
        if capacity != self.capacity:
            raise ValueError(f"replay capacity mismatch: checkpoint={capacity}, runtime={self.capacity}")
        if size < 0 or size > self.capacity:
            raise ValueError("replay size payload is invalid")
        if nextIndex < 0 or nextIndex >= self.capacity:
            raise ValueError("replay nextIndex payload is invalid")

        self.buffer = [None] * self.capacity
        self.size = 0
        self.nextIndex = 0
        for idx, raw_item in enumerate(entries[: self.capacity]):
            if not isinstance(raw_item, dict):
                raise ValueError("replay entry payload is invalid")
            item = cast(dict[str, Any], raw_item)
            transition = Transition(
                state=np.asarray(item["state"], dtype=np.float32),
                action=int(item["action"]),
                reward=float(item["reward"]),
                nextState=np.asarray(item["next_state"], dtype=np.float32),
                done=bool(item["done"]),
                validActionMask=np.asarray(item["valid_action_mask"], dtype=np.bool_),
                nextValidActionMask=np.asarray(item["next_valid_action_mask"], dtype=np.bool_),
                bootstrapDiscount=float(item.get("bootstrap_discount", 1.0)),
            )
            self.buffer[idx] = transition
            self.size += 1
        self.size = min(self.size, size)
        self.nextIndex = nextIndex
