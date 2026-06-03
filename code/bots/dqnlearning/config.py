from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ConfigValidationError(ValueError):
    pass


def _as_int(value: Any, fieldName: str, minValue: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{fieldName} must be an integer, got {value!r}") from exc
    if minValue is not None and parsed < minValue:
        raise ConfigValidationError(f"{fieldName} must be >= {minValue}, got {parsed}")
    return parsed


def _as_float(value: Any, fieldName: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{fieldName} must be a float, got {value!r}") from exc


def _as_bool(value: Any, fieldName: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if int(value) in (0, 1):
            return bool(int(value))
        raise ConfigValidationError(f"{fieldName} must be 0/1 or boolean, got {value!r}")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("0", "false", "no", "off"):
            return False
        if normalized in ("1", "true", "yes", "on"):
            return True
        raise ConfigValidationError(f"{fieldName} must be 0/1 or boolean-like string, got {value!r}")
    raise ConfigValidationError(f"{fieldName} must be boolean-compatible, got {value!r}")


@dataclass(frozen=True)
class DQNConfig:
    learningRate: float = 1e-4
    discountFactor: float = 0.99
    epsilonStart: float = 1.0
    epsilonEnd: float = 0.05
    epsilonDecaySteps: int = 100000
    replayCapacity: int = 50000
    replayWarmupSteps: int = 10000
    batchSize: int = 64
    trainFrequency: int = 4
    targetUpdateFrequency: int = 2000
    maxStepsPerEpisode: int = 5000
    hiddenSize: int = 256
    useRichEncoding: bool = False
    usePositionInState: bool = True
    neuralMapWidth: int = 31
    neuralMapHeight: int = 31
    neuralMapPoolSize: int = 8
    immediateReversalPenalty: float = -5.0
    repeatVisitPenaltyScale: float = 0.0
    noProgressPenalty: float = -100.0
    noProgressPatienceFactor: float = 3.0
    minNoProgressSteps: int = 50
    maxNoProgressSteps: int = 1000
    diagnosticsFrequency: int = 0
    rewardClipMin: float = -10.0
    rewardClipMax: float = 10.0
    stateEncodingVersion: int = 4
    mapEmbedDim: int = 128
    checkpointFrequency: int = 10
    # Step-budget formula: min(stepLimitMax, max(stepLimitMin, stepLimitStepCoeff * optimalLen + stepLimitAreaCoeff * w * h))
    stepLimitStepCoeff: int = 12
    stepLimitAreaCoeff: float = 0.5
    stepLimitMax: int = 5000
    stepLimitMin: int = 200
    stepLimitPenalty: float = -100.0
    @classmethod
    def from_profile_dict(cls, profile: dict[str, Any]) -> DQNConfig:
        if not isinstance(profile, dict):
            raise TypeError(f"profile must be a dict, got {type(profile).__name__}")

        raw = dict(profile)
        rewardClip = raw.get("rewardClip")
        if rewardClip is not None and ("rewardClipMin" not in raw or "rewardClipMax" not in raw):
            if not isinstance(rewardClip, (list, tuple)) or len(rewardClip) != 2:
                raise ConfigValidationError("rewardClip must be a 2-item list/tuple")
            raw["rewardClipMin"], raw["rewardClipMax"] = rewardClip[0], rewardClip[1]

        config = cls(
            learningRate=_as_float(raw.get("learningRate", cls.learningRate), "learningRate"),
            discountFactor=_as_float(raw.get("discountFactor", cls.discountFactor), "discountFactor"),
            epsilonStart=_as_float(raw.get("epsilonStart", cls.epsilonStart), "epsilonStart"),
            epsilonEnd=_as_float(raw.get("epsilonEnd", cls.epsilonEnd), "epsilonEnd"),
            epsilonDecaySteps=_as_int(raw.get("epsilonDecaySteps", cls.epsilonDecaySteps), "epsilonDecaySteps", 1),
            replayCapacity=_as_int(raw.get("replayCapacity", cls.replayCapacity), "replayCapacity", 1),
            # Warmup transition target is shared behavior, not a per-profile knob.
            replayWarmupSteps=cls.replayWarmupSteps,
            batchSize=_as_int(raw.get("batchSize", cls.batchSize), "batchSize", 1),
            trainFrequency=_as_int(raw.get("trainFrequency", cls.trainFrequency), "trainFrequency", 1),
            targetUpdateFrequency=_as_int(raw.get("targetUpdateFrequency", cls.targetUpdateFrequency), "targetUpdateFrequency", 1),
            maxStepsPerEpisode=_as_int(raw.get("maxStepsPerEpisode", cls.maxStepsPerEpisode), "maxStepsPerEpisode", 1),
            hiddenSize=_as_int(raw.get("hiddenSize", cls.hiddenSize), "hiddenSize", 1),
            useRichEncoding=_as_bool(raw.get("useRichEncoding", cls.useRichEncoding), "useRichEncoding"),
            usePositionInState=_as_bool(raw.get("usePositionInState", cls.usePositionInState), "usePositionInState"),
            neuralMapWidth=_as_int(raw.get("neuralMapWidth", cls.neuralMapWidth), "neuralMapWidth", 1),
            neuralMapHeight=_as_int(raw.get("neuralMapHeight", cls.neuralMapHeight), "neuralMapHeight", 1),
            neuralMapPoolSize=_as_int(raw.get("neuralMapPoolSize", cls.neuralMapPoolSize), "neuralMapPoolSize", 1),
            immediateReversalPenalty=_as_float(raw.get("immediateReversalPenalty", cls.immediateReversalPenalty), "immediateReversalPenalty"),
            repeatVisitPenaltyScale=_as_float(raw.get("repeatVisitPenaltyScale", cls.repeatVisitPenaltyScale), "repeatVisitPenaltyScale"),
            noProgressPenalty=_as_float(raw.get("noProgressPenalty", cls.noProgressPenalty), "noProgressPenalty"),
            noProgressPatienceFactor=_as_float(raw.get("noProgressPatienceFactor", cls.noProgressPatienceFactor), "noProgressPatienceFactor"),
            minNoProgressSteps=_as_int(raw.get("minNoProgressSteps", cls.minNoProgressSteps), "minNoProgressSteps", 1),
            maxNoProgressSteps=_as_int(raw.get("maxNoProgressSteps", cls.maxNoProgressSteps), "maxNoProgressSteps", 1),
            diagnosticsFrequency=_as_int(raw.get("diagnosticsFrequency", cls.diagnosticsFrequency), "diagnosticsFrequency", 0),
            rewardClipMin=_as_float(raw.get("rewardClipMin", cls.rewardClipMin), "rewardClipMin"),
            rewardClipMax=_as_float(raw.get("rewardClipMax", cls.rewardClipMax), "rewardClipMax"),
            # Promote older profiles to the current encoder schema so
            # size-dependent legacy checkpoints/warmup stores are invalidated.
            stateEncodingVersion=max(
                cls.stateEncodingVersion,
                _as_int(raw.get("stateEncodingVersion", cls.stateEncodingVersion), "stateEncodingVersion", 1),
            ),
            mapEmbedDim=_as_int(raw.get("mapEmbedDim", cls.mapEmbedDim), "mapEmbedDim", 0),
            checkpointFrequency=_as_int(raw.get("checkpointFrequency", cls.checkpointFrequency), "checkpointFrequency", 0),
            stepLimitStepCoeff=_as_int(raw.get("stepLimitStepCoeff", cls.stepLimitStepCoeff), "stepLimitStepCoeff", 1),
            stepLimitAreaCoeff=_as_float(raw.get("stepLimitAreaCoeff", cls.stepLimitAreaCoeff), "stepLimitAreaCoeff"),
            stepLimitMax=_as_int(raw.get("stepLimitMax", cls.stepLimitMax), "stepLimitMax", 1),
            stepLimitMin=_as_int(raw.get("stepLimitMin", cls.stepLimitMin), "stepLimitMin", 1),
            stepLimitPenalty=_as_float(raw.get("stepLimitPenalty", cls.stepLimitPenalty), "stepLimitPenalty"),
        )
        config._validate_ranges()
        return config

    def _validate_ranges(self) -> None:
        if not (0.0 <= self.discountFactor <= 1.0):
            raise ConfigValidationError("discountFactor must be in [0.0, 1.0]")
        if not (0.0 <= self.epsilonStart <= 1.0):
            raise ConfigValidationError("epsilonStart must be in [0.0, 1.0]")
        if not (0.0 <= self.epsilonEnd <= 1.0):
            raise ConfigValidationError("epsilonEnd must be in [0.0, 1.0]")
        if self.epsilonEnd > self.epsilonStart:
            raise ConfigValidationError("epsilonEnd must be <= epsilonStart")
        if self.learningRate <= 0.0:
            raise ConfigValidationError("learningRate must be > 0")
        if self.rewardClipMin > self.rewardClipMax:
            raise ConfigValidationError("rewardClipMin must be <= rewardClipMax")
        if self.noProgressPatienceFactor < 0.0:
            raise ConfigValidationError("noProgressPatienceFactor must be >= 0")
        if self.maxNoProgressSteps < self.minNoProgressSteps:
            raise ConfigValidationError("maxNoProgressSteps must be >= minNoProgressSteps")


