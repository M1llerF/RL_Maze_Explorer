from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

import numpy as np

from .config import DQNConfig
from .model import MAP_CHANNELS
from .types import EncodedState

@dataclass(frozen=True)
class EncoderSchemaMeta:
    stateSchemaVersion: int
    inputDim: int
    encoderConfigFingerprint: str
    # flatDim: floats before the appended map (= inputDim when no map is used).
    # mapShape: (C, H, W) in channel-first order when the map is included, else None.
    # Both are used by DqnAgent to construct a model that knows how to split its input.
    flatDim: int = 0
    mapShape: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        # Ensure flatDim has a sensible default when constructed without it
        # (e.g. existing code that only passes the first three positional args).
        if self.flatDim == 0:
            object.__setattr__(self, "flatDim", self.inputDim)


class StateEncoder:
    def __init__(self, config: DQNConfig, mazeHeight: int, mazeWidth: int) -> None:
        self.config = config
        self.mazeHeight = max(1, int(mazeHeight))
        self.mazeWidth = max(1, int(mazeWidth))

    def encode(self, observation: tuple[Any, ...]) -> EncodedState:
        if getattr(self.config, "useSharedComparisonState", False) and self._isComparisonObservation(observation):
            return self._encodeComparisonObservation(observation)

        positionIndex, wallDistances = observation[:2]
        previousDelta: tuple[int, int] = (0, 0)
        lastAction: int = -1
        localObservation: tuple[tuple[float, float, float], ...] = ()
        neuralMap: tuple[float, ...] | np.ndarray = ()
        validActions: tuple[int, ...] = (1, 1, 1, 1)
        goalInfo: tuple[float, float, float] = (0.0, 0.0, 0.0)
        entityFeatures: tuple[float, ...] = ()

        if len(observation) >= 7:
            previousDelta = cast(tuple[int, int], observation[2])
            lastAction = int(observation[3])
            localObservation = cast(tuple[tuple[float, float, float], ...], observation[4])
            neuralMap = cast(tuple[float, ...] | np.ndarray, observation[5])
            validActions = cast(tuple[int, ...], observation[6])
        if len(observation) >= 8:
            entityFeatures = tuple(float(v) for v in cast(tuple[float, ...], observation[7]))

        mask = np.asarray([int(v) > 0 for v in validActions], dtype=np.bool_)
        version = int(getattr(self.config, "stateEncodingVersion", 2))
        useRich = getattr(self.config, "useRichEncoding", False)

        if version >= 3:
            features = self._encodeV3Features(
                cast(tuple[int, int], positionIndex),
                wallDistances,
                previousDelta,
                lastAction,
                localObservation,
                entityFeatures=entityFeatures,
                neuralMap=neuralMap,
                useRichEncoding=useRich,
            )
        else:
            features = self._encodeV2Features(
                positionIndex, wallDistances, previousDelta, lastAction,
                localObservation, neuralMap, useRich, entityFeatures,
            )

        return EncodedState(values=np.asarray(features, dtype=np.float32), validActionMask=mask)

    def _isComparisonObservation(self, observation: tuple[Any, ...]) -> bool:
        if len(observation) not in (5, 6):
            return False
        maybeMask = observation[-1]
        if not isinstance(maybeMask, tuple):
            return False
        return all(isinstance(v, (bool, int, np.bool_)) for v in maybeMask)

    def _encodeComparisonObservation(self, observation: tuple[Any, ...]) -> EncodedState:
        if len(observation) == 5:
            positionIndex, wallDistances, goalDirection, entityFeatures, validActions = observation
            enemyFeatures: tuple[float, ...] = ()
        else:
            positionIndex, wallDistances, goalDirection, entityFeatures, enemyFeatures, validActions = observation
        mask = np.asarray([int(v) > 0 for v in cast(tuple[int, ...], validActions)], dtype=np.bool_)
        features = self._encodeComparisonFeatures(
            cast(tuple[int, int], positionIndex),
            cast(tuple[int, int, int, int], wallDistances),
            cast(tuple[int, int, int, int], goalDirection),
            tuple(float(v) for v in cast(tuple[float, ...], entityFeatures)),
            tuple(float(v) for v in cast(tuple[Any, ...], enemyFeatures)),
        )
        return EncodedState(values=np.asarray(features, dtype=np.float32), validActionMask=mask)

    def inferSchema(self, sampleObservation: tuple[Any, ...]) -> EncoderSchemaMeta:
        encoded = self.encode(sampleObservation)
        inputDim = int(encoded.values.shape[0])
        version = int(getattr(self.config, "stateEncodingVersion", 2))
        useRich = getattr(self.config, "useRichEncoding", False)

        if version >= 3 and useRich:
            mapH_actual = int(getattr(self.config, "neuralMapHeight", 31))
            mapW_actual = int(getattr(self.config, "neuralMapWidth", 31))
            mapShape: tuple[int, int, int] | None
            expectedMapDim = MAP_CHANNELS * mapH_actual * mapW_actual
            if inputDim > expectedMapDim and (inputDim - expectedMapDim) > 0:
                mapShape = (MAP_CHANNELS, mapH_actual, mapW_actual)
                flatDim = inputDim - expectedMapDim
            else:
                # Map is absent or dimensions don't match — fall back to MLP-only.
                mapShape = None
                flatDim = inputDim
        else:
            mapShape = None
            flatDim = inputDim

        return EncoderSchemaMeta(
            stateSchemaVersion=int(self.config.stateEncodingVersion),
            inputDim=inputDim,
            encoderConfigFingerprint=self._fingerprint(),
            flatDim=flatDim,
            mapShape=mapShape,
        )

    def encodeNeuralMap(
        self,
        position: tuple[int, int],
        knownMap: np.ndarray,
        visitedMap: np.ndarray,
    ) -> np.ndarray:
        """
        Build the ego-centric spatial feature array for the neural map branch.

        Output shape: (mapH * mapW * 7,) float32, zero-padded when the maze is
        smaller than the configured window.

        Channels: [knownOpen, knownWall, visited, isCurrent, seenGoal, frontier, reserved]
        """
        mapH = int(self.config.neuralMapHeight)
        mapW = int(self.config.neuralMapWidth)
        mazeH = self.mazeHeight
        mazeW = self.mazeWidth
        actualH = min(mapH, mazeH)
        actualW = min(mapW, mazeW)
        originRow = max(0, min(position[0] - mapH // 2, mazeH - actualH))
        originCol = max(0, min(position[1] - mapW // 2, mazeW - actualW))

        result = np.zeros((mapH, mapW, 7), dtype=np.float32)
        s_r = slice(0, actualH)
        s_c = slice(0, actualW)
        m_r = slice(originRow, originRow + actualH)
        m_c = slice(originCol, originCol + actualW)

        result[s_r, s_c, 0] = knownMap[m_r, m_c, 0]   # knownOpen
        result[s_r, s_c, 1] = knownMap[m_r, m_c, 1]   # knownWall
        result[s_r, s_c, 2] = visitedMap[m_r, m_c]    # visited
        result[s_r, s_c, 4] = knownMap[m_r, m_c, 2]   # seenGoal

        r_rel = position[0] - originRow
        c_rel = position[1] - originCol
        if 0 <= r_rel < actualH and 0 <= c_rel < actualW:
            result[r_rel, c_rel, 3] = 1.0              # isCurrent

        result[s_r, s_c, 5] = (                        # frontier = open & unvisited
            result[s_r, s_c, 0] * (1.0 - result[s_r, s_c, 2])
        )
        return result.ravel()

    def _observationScale(self) -> float:
        # Use a config-defined scale so encoded magnitudes remain stable across
        # different maze sizes.
        return float(max(
            1,
            int(getattr(self.config, "neuralMapHeight", 31)),
            int(getattr(self.config, "neuralMapWidth", 31)),
        ))

    def _encodeComparisonFeatures(
        self,
        positionIndex: tuple[int, int],
        wallDistances: tuple[int, int, int, int],
        goalDirection: tuple[int, int, int, int],
        entityFeatures: tuple[float, ...],
        enemyFeatures: tuple[float, ...],
    ) -> list[float]:
        encoded: list[float] = []
        if getattr(self.config, "usePositionInState", True):
            row, col = positionIndex
            positionScale = self._observationScale()
            encoded.extend([
                float(np.clip(float(row) / positionScale, 0.0, 1.0)),
                float(np.clip(float(col) / positionScale, 0.0, 1.0)),
            ])
        distanceScale = self._observationScale()
        for distance in wallDistances:
            encoded.append(float(np.clip(float(distance) / distanceScale, 0.0, 1.0)))
        encoded.extend(float(v) for v in goalDirection)
        encoded.extend(float(v) for v in entityFeatures)
        if len(enemyFeatures) >= 2:
            encoded.append(float(enemyFeatures[0]) / 4.0)
            encoded.append(float(enemyFeatures[1]) / 3.0)
        return encoded

    # ------------------------------------------------------------------
    # v3: first-person encoding
    # ------------------------------------------------------------------

    def _encodeV3Features(
        self,
        positionIndex: tuple[int, int],
        wallDistances: Any,
        previousDelta: tuple[int, int],
        lastAction: int,
        localObservation: tuple[tuple[float, float, float], ...],
        *,
        entityFeatures: tuple[float, ...] = (),
        neuralMap: tuple[float, ...] | np.ndarray = (),
        useRichEncoding: bool = False,
    ) -> list[float] | np.ndarray:
        """
        First-person state encoding. Flat core (24 floats before optional
        entity features):

          2  absolute position: (row_norm, col_norm)
          4  wall distances (line-of-sight, normalised)
          2  previous-move delta
          4  last action one-hot
         12  local observation: 4 rays × (wall_dist, goal_visible, visited_visible)

        When useRichEncoding is True the neural map is appended as a flat
        HWC array (MAP_CHANNELS × H × W floats) so the model's CNN can
        process it.  The split point is always at flat position 25.
        """
        maxDist = self._observationScale()

        flat: list[float] = []
        if getattr(self.config, "usePositionInState", True):
            row, col = positionIndex
            flat.extend([
                float(np.clip(float(row) / float(max(1, self.mazeHeight - 1)), 0.0, 1.0)),
                float(np.clip(float(col) / float(max(1, self.mazeWidth - 1)), 0.0, 1.0)),
            ])
        for d in cast(tuple[int, int, int, int], wallDistances):
            flat.append(float(d) / maxDist)
        flat.extend(float(delta) for delta in previousDelta)
        for i in range(4):
            flat.append(1.0 if lastAction == i else 0.0)
        rays = list(localObservation)[:4]
        for ray in rays:
            flat.extend(float(v) for v in ray)
        for _ in range(4 - len(rays)):
            flat.extend([0.0, 0.0, 0.0])
        flat.extend(float(v) for v in entityFeatures)

        if useRichEncoding and len(neuralMap) > 0:
            # Fast path: np.concatenate avoids a second Python-level iteration
            # over the (potentially large) map array.
            flat_arr = np.array(flat, dtype=np.float32)
            map_arr = neuralMap if isinstance(neuralMap, np.ndarray) else np.asarray(neuralMap, dtype=np.float32)
            return np.concatenate([flat_arr, map_arr])

        return flat

    # ------------------------------------------------------------------
    # v2: legacy encoding (backward compatible)
    # ------------------------------------------------------------------

    def _encodeV2Features(
        self,
        positionIndex: Any,
        wallDistances: Any,
        previousDelta: tuple[int, int],
        lastAction: int,
        localObservation: tuple[tuple[float, float, float], ...],
        neuralMap: tuple[float, ...] | np.ndarray,
        useRichEncoding: bool,
        entityFeatures: tuple[float, ...],
    ) -> list[float]:
        encoded: list[float] = []

        if getattr(self.config, "usePositionInState", True):
            row, col = cast(tuple[int, int], positionIndex)
            positionScale = self._observationScale()
            encoded.extend([
                float(np.clip(float(row) / positionScale, 0.0, 1.0)),
                float(np.clip(float(col) / positionScale, 0.0, 1.0)),
            ])

        maxDist = self._observationScale()
        for distance in cast(tuple[int, int, int, int], wallDistances):
            encoded.append(float(np.clip(float(distance) / maxDist, 0.0, 1.0)))

        encoded.extend(float(delta) for delta in previousDelta)

        for actionIndex in range(4):
            encoded.append(1.0 if lastAction == actionIndex else 0.0)

        if useRichEncoding:
            for cellFeatures in localObservation:
                encoded.extend(float(v) for v in cellFeatures)
            encoded.extend(float(value) for value in neuralMap)
        encoded.extend(float(v) for v in entityFeatures)

        return encoded

    # ------------------------------------------------------------------
    # Fingerprint (checkpoint schema invalidation)
    # ------------------------------------------------------------------

    def _fingerprint(self) -> str:
        payload = {
            "stateEncodingVersion": int(self.config.stateEncodingVersion),
            "useRichEncoding": getattr(self.config, "useRichEncoding", False),
            "useSharedComparisonState": getattr(self.config, "useSharedComparisonState", False),
            "useLstmPolicy": getattr(self.config, "useLstmPolicy", False),
            "lstmSequenceLength": int(getattr(self.config, "lstmSequenceLength", 1)),
            "lstmHiddenSize": int(getattr(self.config, "lstmHiddenSize", 0)),
            "usePositionInState": getattr(self.config, "usePositionInState", True),
            "useEntityObservation": getattr(self.config, "useEntityObservation", False),
            "useEnemyObservation": getattr(self.config, "useEnemyObservation", False),
            "useAttackActions": getattr(self.config, "useAttackActions", False),
            "macroOptionSet": str(getattr(self.config, "macroOptionSet", "naive")),
            "neuralMapHeight": int(getattr(self.config, "neuralMapHeight", 31)),
            "neuralMapWidth": int(getattr(self.config, "neuralMapWidth", 31)),
            "neuralMapPoolSize": int(getattr(self.config, "neuralMapPoolSize", 8)),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
