from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


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


def _as_optional_float(value: Any, fieldName: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return _as_float(value, fieldName)


def _as_penalty_float(value: Any, fieldName: str) -> float:
    return -abs(_as_float(value, fieldName))


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


def _as_string(value: Any, fieldName: str) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_macro_option_set(value: Any, fieldName: str) -> str:
    if isinstance(value, bool):
        return "naive"
    if isinstance(value, (int, float)):
        parsed = int(value)
        if float(value) == float(parsed):
            if parsed in (0, 1):
                return "naive"
            if parsed == 2:
                return "momentum"
            if parsed == 3:
                return "astar"
        raise ConfigValidationError(f"{fieldName} must be 1 (naive), 2 (momentum), 3 (astar), or a matching name")
    normalized = _as_string(value, fieldName).lower()
    if normalized in ("", "0", "0.0", "1", "1.0", "naive"):
        return "naive"
    if normalized in ("2", "2.0", "momentum"):
        return "momentum"
    if normalized in ("3", "3.0", "astar"):
        return "astar"
    raise ConfigValidationError(f"{fieldName} must be 1/'naive', 2/'momentum', or 3/'astar'")


@dataclass(frozen=True)
class DQNConfig:
    """
    Frozen configuration for a DQN training session. All fields are validated at construction time.
    Build from a saved profile by passing its dictionary to the class method that accepts one.
    Macro action modes are mutually exclusive. Enabling hierarchical or macro only policy
    automatically sets useMacroActions even if the caller omits it.
    """

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
    useSharedComparisonState: bool = True
    usePositionInState: bool = True
    useEntityObservation: bool = True
    useEnemyObservation: bool = False
    useAttackActions: bool = False
    autoAttackAdjacentEnemy: bool = False
    pushCooldownSteps: int = 3
    useMacroActions: bool = False
    macroOptionSet: str = "naive"
    useMacroOnlyPolicy: bool = False
    dualRecordPrimitivePolicy: bool = False
    useHierarchicalPolicy: bool = False
    useLstmPolicy: bool = False
    lstmSequenceLength: int = 8
    lstmHiddenSize: int = 128
    neuralMapWidth: int = 31
    neuralMapHeight: int = 31
    neuralMapPoolSize: int = 8
    immediateReversalPenalty: float = -5.0
    oscillationPenalty: float = -50.0
    repeatVisitPenaltyScale: float = 0.0
    noProgressPenalty: float = -150.0
    noProgressPatienceFactor: float = 2.0
    minNoProgressSteps: int = 30
    maxNoProgressSteps: int = 1000
    diagnosticsFrequency: int = 0
    rewardClipMin: float | None = None
    rewardClipMax: float | None = None
    stateEncodingVersion: int = 7
    mapEmbedDim: int = 128
    checkpointFrequency: int = 10
    evaluationFrequency: int = 10
    # Step-budget formula: min(stepLimitMax, max(stepLimitMin, stepLimitStepCoeff * optimalLen + stepLimitAreaCoeff * w * h))
    stepLimitStepCoeff: int = 12
    stepLimitAreaCoeff: float = 0.5
    stepLimitMax: int = 5000
    stepLimitMin: int = 200
    stepLimitPenalty: float = -150.0

    def __post_init__(self) -> None:
        impliedMacroActions = bool(self.useMacroActions or self.useMacroOnlyPolicy or self.useHierarchicalPolicy)
        if impliedMacroActions != bool(self.useMacroActions):
            object.__setattr__(self, "useMacroActions", impliedMacroActions)
        self._validate_ranges()

    @classmethod
    def from_profile_dict(cls, profile: Any) -> DQNConfig:
        if not isinstance(profile, dict):
            raise TypeError(f"profile must be a dict, got {type(profile).__name__}")

        raw: dict[str, Any] = dict(cast(dict[str, Any], profile))
        rewardClip: Any = raw.get("rewardClip")
        if rewardClip is not None and ("rewardClipMin" not in raw or "rewardClipMax" not in raw):
            rewardClipSeq: list[Any] = cast(list[Any], rewardClip) if isinstance(rewardClip, (list, tuple)) else []
            if len(rewardClipSeq) != 2:
                raise ConfigValidationError("rewardClip must be a 2-item list/tuple")
            raw["rewardClipMin"], raw["rewardClipMax"] = rewardClipSeq[0], rewardClipSeq[1]

        useMacroOnlyPolicy = _as_bool(raw.get("useMacroOnlyPolicy", cls.useMacroOnlyPolicy), "useMacroOnlyPolicy")
        useHierarchicalPolicy = _as_bool(
            raw.get("useHierarchicalPolicy", cls.useHierarchicalPolicy),
            "useHierarchicalPolicy",
        )
        config = cls(
            learningRate=_as_float(raw.get("learningRate", cls.learningRate), "learningRate"),
            discountFactor=_as_float(raw.get("discountFactor", cls.discountFactor), "discountFactor"),
            epsilonStart=_as_float(raw.get("epsilonStart", cls.epsilonStart), "epsilonStart"),
            epsilonEnd=_as_float(raw.get("epsilonEnd", cls.epsilonEnd), "epsilonEnd"),
            epsilonDecaySteps=_as_int(raw.get("epsilonDecaySteps", cls.epsilonDecaySteps), "epsilonDecaySteps", 1),
            replayCapacity=_as_int(raw.get("replayCapacity", cls.replayCapacity), "replayCapacity", 1),
            replayWarmupSteps=_as_int(raw.get("replayWarmupSteps", cls.replayWarmupSteps), "replayWarmupSteps", 0),
            batchSize=_as_int(raw.get("batchSize", cls.batchSize), "batchSize", 1),
            trainFrequency=_as_int(raw.get("trainFrequency", cls.trainFrequency), "trainFrequency", 1),
            targetUpdateFrequency=_as_int(raw.get("targetUpdateFrequency", cls.targetUpdateFrequency), "targetUpdateFrequency", 1),
            maxStepsPerEpisode=_as_int(raw.get("maxStepsPerEpisode", cls.maxStepsPerEpisode), "maxStepsPerEpisode", 1),
            hiddenSize=_as_int(raw.get("hiddenSize", cls.hiddenSize), "hiddenSize", 1),
            useRichEncoding=_as_bool(raw.get("useRichEncoding", cls.useRichEncoding), "useRichEncoding"),
            useSharedComparisonState=_as_bool(
                raw.get("useSharedComparisonState", cls.useSharedComparisonState),
                "useSharedComparisonState",
            ),
            usePositionInState=_as_bool(raw.get("usePositionInState", cls.usePositionInState), "usePositionInState"),
            useEntityObservation=_as_bool(raw.get("useEntityObservation", cls.useEntityObservation), "useEntityObservation"),
            useEnemyObservation=_as_bool(raw.get("useEnemyObservation", cls.useEnemyObservation), "useEnemyObservation"),
            useAttackActions=_as_bool(raw.get("useAttackActions", cls.useAttackActions), "useAttackActions"),
            autoAttackAdjacentEnemy=_as_bool(
                raw.get("autoAttackAdjacentEnemy", cls.autoAttackAdjacentEnemy),
                "autoAttackAdjacentEnemy",
            ),
            pushCooldownSteps=_as_int(raw.get("pushCooldownSteps", cls.pushCooldownSteps), "pushCooldownSteps", 1),
            useMacroActions=_as_bool(raw.get("useMacroActions", cls.useMacroActions), "useMacroActions")
            or useMacroOnlyPolicy
            or useHierarchicalPolicy,
            macroOptionSet=_as_macro_option_set(raw.get("macroOptionSet", 1), "macroOptionSet"),
            useMacroOnlyPolicy=useMacroOnlyPolicy,
            dualRecordPrimitivePolicy=_as_bool(
                raw.get("dualRecordPrimitivePolicy", cls.dualRecordPrimitivePolicy),
                "dualRecordPrimitivePolicy",
            ),
            useHierarchicalPolicy=useHierarchicalPolicy,
            useLstmPolicy=_as_bool(raw.get("useLstmPolicy", cls.useLstmPolicy), "useLstmPolicy"),
            lstmSequenceLength=_as_int(raw.get("lstmSequenceLength", cls.lstmSequenceLength), "lstmSequenceLength", 1),
            lstmHiddenSize=_as_int(raw.get("lstmHiddenSize", cls.lstmHiddenSize), "lstmHiddenSize", 1),
            neuralMapWidth=_as_int(raw.get("neuralMapWidth", cls.neuralMapWidth), "neuralMapWidth", 1),
            neuralMapHeight=_as_int(raw.get("neuralMapHeight", cls.neuralMapHeight), "neuralMapHeight", 1),
            neuralMapPoolSize=_as_int(raw.get("neuralMapPoolSize", cls.neuralMapPoolSize), "neuralMapPoolSize", 1),
            immediateReversalPenalty=_as_penalty_float(raw.get("immediateReversalPenalty", cls.immediateReversalPenalty), "immediateReversalPenalty"),
            oscillationPenalty=_as_penalty_float(raw.get("oscillationPenalty", cls.oscillationPenalty), "oscillationPenalty"),
            repeatVisitPenaltyScale=_as_penalty_float(raw.get("repeatVisitPenaltyScale", cls.repeatVisitPenaltyScale), "repeatVisitPenaltyScale"),
            noProgressPenalty=_as_penalty_float(raw.get("noProgressPenalty", cls.noProgressPenalty), "noProgressPenalty"),
            noProgressPatienceFactor=_as_float(raw.get("noProgressPatienceFactor", cls.noProgressPatienceFactor), "noProgressPatienceFactor"),
            minNoProgressSteps=_as_int(raw.get("minNoProgressSteps", cls.minNoProgressSteps), "minNoProgressSteps", 1),
            maxNoProgressSteps=_as_int(raw.get("maxNoProgressSteps", cls.maxNoProgressSteps), "maxNoProgressSteps", 1),
            diagnosticsFrequency=_as_int(raw.get("diagnosticsFrequency", cls.diagnosticsFrequency), "diagnosticsFrequency", 0),
            rewardClipMin=_as_optional_float(raw.get("rewardClipMin", cls.rewardClipMin), "rewardClipMin"),
            rewardClipMax=_as_optional_float(raw.get("rewardClipMax", cls.rewardClipMax), "rewardClipMax"),
            # Promote older profiles to the current encoder schema so
            # size-dependent legacy checkpoints/warmup stores are invalidated.
            stateEncodingVersion=max(
                cls.stateEncodingVersion,
                _as_int(raw.get("stateEncodingVersion", cls.stateEncodingVersion), "stateEncodingVersion", 1),
            ),
            mapEmbedDim=_as_int(raw.get("mapEmbedDim", cls.mapEmbedDim), "mapEmbedDim", 0),
            checkpointFrequency=_as_int(raw.get("checkpointFrequency", cls.checkpointFrequency), "checkpointFrequency", 0),
            evaluationFrequency=_as_int(raw.get("evaluationFrequency", cls.evaluationFrequency), "evaluationFrequency", 0),
            stepLimitStepCoeff=_as_int(raw.get("stepLimitStepCoeff", cls.stepLimitStepCoeff), "stepLimitStepCoeff", 1),
            stepLimitAreaCoeff=_as_float(raw.get("stepLimitAreaCoeff", cls.stepLimitAreaCoeff), "stepLimitAreaCoeff"),
            stepLimitMax=_as_int(raw.get("stepLimitMax", cls.stepLimitMax), "stepLimitMax", 1),
            stepLimitMin=_as_int(raw.get("stepLimitMin", cls.stepLimitMin), "stepLimitMin", 1),
            stepLimitPenalty=_as_float(raw.get("stepLimitPenalty", cls.stepLimitPenalty), "stepLimitPenalty"),
        )
        return config

    def _validate_ranges(self) -> None:
        # Mutual-exclusion checks first so they surface with clear messages
        # before any secondary validation that could produce a misleading error.
        if self.useHierarchicalPolicy and self.useMacroOnlyPolicy:
            raise ConfigValidationError("useHierarchicalPolicy and useMacroOnlyPolicy are mutually exclusive")
        if self.dualRecordPrimitivePolicy and self.useHierarchicalPolicy:
            raise ConfigValidationError("dualRecordPrimitivePolicy is not supported with hierarchical policy")
        if self.useLstmPolicy and self.useHierarchicalPolicy:
            raise ConfigValidationError("useLstmPolicy is not supported with hierarchical policy")
        # Dependency checks
        if self.useMacroOnlyPolicy and not self.useMacroActions:
            raise ConfigValidationError("useMacroOnlyPolicy requires useMacroActions to be enabled")
        if self.useHierarchicalPolicy and not self.useMacroActions:
            raise ConfigValidationError("useHierarchicalPolicy requires useMacroActions to be enabled")
        if self.dualRecordPrimitivePolicy and not self.useMacroOnlyPolicy:
            raise ConfigValidationError("dualRecordPrimitivePolicy requires useMacroOnlyPolicy to be enabled")
        if self.macroOptionSet not in {"naive", "astar", "momentum"}:
            raise ConfigValidationError("macroOptionSet must resolve to 'naive', 'astar', or 'momentum'")
        # Range checks
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
        if (self.rewardClipMin is None) != (self.rewardClipMax is None):
            raise ConfigValidationError("rewardClipMin and rewardClipMax must both be set or both be blank")
        if self.rewardClipMin is not None and self.rewardClipMax is not None and self.rewardClipMin > self.rewardClipMax:
            raise ConfigValidationError("rewardClipMin must be <= rewardClipMax")
        if self.noProgressPatienceFactor < 0.0:
            raise ConfigValidationError("noProgressPatienceFactor must be >= 0")
        if self.maxNoProgressSteps < self.minNoProgressSteps:
            raise ConfigValidationError("maxNoProgressSteps must be >= minNoProgressSteps")
