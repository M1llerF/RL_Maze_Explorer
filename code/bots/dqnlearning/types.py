from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_float32_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"values must be rank-1, got shape={arr.shape}")
    return arr


def _as_action_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.bool_)
    if arr.shape != (4,):
        raise ValueError(f"action mask must have shape (4,), got shape={arr.shape}")
    if not np.any(arr):
        arr = np.ones((4,), dtype=np.bool_)
    return arr


@dataclass(frozen=True)
class EncodedState:
    
    values: np.ndarray
    validActionMask: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _as_float32_vector(self.values))
        object.__setattr__(self, "validActionMask", _as_action_mask(self.validActionMask))


@dataclass(frozen=True)
class Transition:
    """Single replay transition with explicit masks for s and s'."""

    state: np.ndarray
    action: int
    reward: float
    nextState: np.ndarray
    done: bool
    validActionMask: np.ndarray
    nextValidActionMask: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _as_float32_vector(self.state))
        object.__setattr__(self, "nextState", _as_float32_vector(self.nextState))
        object.__setattr__(self, "validActionMask", _as_action_mask(self.validActionMask))
        object.__setattr__(self, "nextValidActionMask", _as_action_mask(self.nextValidActionMask))
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "done", bool(self.done))
        if not (0 <= self.action <= 3):
            raise ValueError(f"action must be in [0, 3], got {self.action}")
        if self.state.shape != self.nextState.shape:
            raise ValueError(
                f"state and nextState must match shape, got {self.state.shape} vs {self.nextState.shape}"
            )


@dataclass
class DqnTrainingBatch:
    """Batch container returned by replay sampling."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    nextStates: np.ndarray
    dones: np.ndarray
    validActionMasks: np.ndarray
    nextValidActionMasks: np.ndarray

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.nextStates = np.asarray(self.nextStates, dtype=np.float32)
        self.dones = np.asarray(self.dones, dtype=np.float32)
        self.validActionMasks = np.asarray(self.validActionMasks, dtype=np.bool_)
        self.nextValidActionMasks = np.asarray(self.nextValidActionMasks, dtype=np.bool_)
        b = self.actions.shape[0]
        if self.states.ndim != 2 or self.nextStates.ndim != 2:
            raise ValueError("states and nextStates must be rank-2")
        if self.states.shape != self.nextStates.shape:
            raise ValueError("states and nextStates shapes must match")
        if self.rewards.shape != (b,) or self.dones.shape != (b,):
            raise ValueError("rewards and dones must have shape [B]")
        if self.validActionMasks.shape != (b, 4) or self.nextValidActionMasks.shape != (b, 4):
            raise ValueError("action masks must have shape [B, 4]")

    @property
    def batchSize(self) -> int:
        return int(self.actions.shape[0])
