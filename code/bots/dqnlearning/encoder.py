from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

import numpy as np

from .config import DQNConfig
from .types import EncodedState


@dataclass(frozen=True)
class EncoderSchemaMeta:
    stateSchemaVersion: int
    inputDim: int
    encoderConfigFingerprint: str


class StateEncoder:
    def __init__(self, config: DQNConfig, mazeHeight: int, mazeWidth: int) -> None:
        self.config = config
        self.mazeHeight = max(1, int(mazeHeight))
        self.mazeWidth = max(1, int(mazeWidth))

    def encode(self, observation: tuple[Any, ...]) -> EncodedState:
        positionIndex, wallDistances = observation[:2]
        previousDelta = (0, 0)
        lastAction = -1
        localObservation: tuple[tuple[float, float, float], ...] = ()
        neuralMap: tuple[float, ...] = ()
        validActions = (1, 1, 1, 1)
        if len(observation) >= 7:
            previousDelta = cast(tuple[int, int], observation[2])
            lastAction = int(observation[3])
            localObservation = cast(tuple[tuple[float, float, float], ...], observation[4])
            neuralMap = cast(tuple[float, ...], observation[5])
            validActions = cast(tuple[int, int, int, int], observation[6])

        encoded: list[float] = []
        if bool(getattr(self.config, "usePositionInState", True)):
            row, col = cast(tuple[int, int], positionIndex)
            encoded.extend([float(row) / self.mazeHeight, float(col) / self.mazeWidth])

        maxDistance = float(max(1, self.mazeWidth, self.mazeHeight))
        for distance in cast(tuple[int, int, int, int], wallDistances):
            encoded.append(float(distance) / maxDistance)

        encoded.extend(float(delta) for delta in cast(tuple[int, int], previousDelta))
        for actionIndex in range(4):
            encoded.append(1.0 if int(lastAction) == actionIndex else 0.0)
        if bool(getattr(self.config, "useRichEncoding", False)):
            for cellFeatures in localObservation:
                encoded.extend(float(v) for v in cellFeatures)
            encoded.extend(float(value) for value in neuralMap)

        mask = np.asarray([int(v) > 0 for v in validActions], dtype=np.bool_)
        return EncodedState(values=np.asarray(encoded, dtype=np.float32), validActionMask=mask)

    def inferSchema(self, sampleObservation: tuple[Any, ...]) -> EncoderSchemaMeta:
        inputDim = int(self.encode(sampleObservation).values.shape[0])
        return EncoderSchemaMeta(
            stateSchemaVersion=int(self.config.stateEncodingVersion),
            inputDim=inputDim,
            encoderConfigFingerprint=self._fingerprint(),
        )

    def _fingerprint(self) -> str:
        payload = {
            "stateEncodingVersion": int(self.config.stateEncodingVersion),
            "useRichEncoding": bool(getattr(self.config, "useRichEncoding", False)),
            "usePositionInState": bool(getattr(self.config, "usePositionInState", True)),
            "mazeHeight": int(self.mazeHeight),
            "mazeWidth": int(self.mazeWidth),
            "neuralMapHeight": int(getattr(self.config, "neuralMapHeight", 31)),
            "neuralMapWidth": int(getattr(self.config, "neuralMapWidth", 31)),
            "neuralMapPoolSize": int(getattr(self.config, "neuralMapPoolSize", 8)),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
