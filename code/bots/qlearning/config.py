from __future__ import annotations

from typing import Any


class ConfigValidationError(ValueError):
    pass


def _as_float(value: Any, fieldName: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{fieldName} must be a float, got {value!r}") from exc


def _as_int(value: Any, fieldName: str, minValue: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{fieldName} must be an integer, got {value!r}") from exc
    if minValue is not None and parsed < minValue:
        raise ConfigValidationError(f"{fieldName} must be >= {minValue}, got {parsed}")
    return parsed


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


class QLearningConfig:
    def __init__(
        self,
        learningRate: float = 0.1,
        discountFactor: float = 0.9,
        epsilonStart: float = 1.0,
        epsilonEnd: float = 0.05,
        epsilonDecaySteps: int = 100000,
        maxStepsPerEpisode: int = 5000,
        usePositionInState: bool = True,
        useMacroActions: bool = False,
        useMacroOnlyPolicy: bool = False,
        useEntityObservation: bool = True,
        useEnemyObservation: bool = False,
        useAttackActions: bool = False,
        autoAttackAdjacentEnemy: bool = False,
        pushCooldownSteps: int = 3,
        immediateReversalPenalty: float = -12.0,
        repeatVisitPenaltyScale: float = -2.0,
        noProgressPenalty: float = -150.0,
        noProgressPatienceFactor: float = 2.0,
        minNoProgressSteps: int = 30,
        maxNoProgressSteps: int = 1000,
        rewardClipMin: float | None = None,
        rewardClipMax: float | None = None,
        stepLimitStepCoeff: int = 12,
        stepLimitAreaCoeff: float = 0.5,
        stepLimitMax: int = 5000,
        stepLimitMin: int = 200,
        stepLimitPenalty: float = -150.0,
    ) -> None:
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecaySteps = max(1, int(epsilonDecaySteps))
        self.maxStepsPerEpisode = max(1, int(maxStepsPerEpisode))
        self.usePositionInState = usePositionInState
        self.useMacroActions = bool(useMacroActions or useMacroOnlyPolicy)
        self.useMacroOnlyPolicy = bool(useMacroOnlyPolicy)
        self.useEntityObservation = bool(useEntityObservation)
        self.useEnemyObservation = bool(useEnemyObservation)
        self.useAttackActions = bool(useAttackActions)
        self.autoAttackAdjacentEnemy = bool(autoAttackAdjacentEnemy)
        self.pushCooldownSteps = max(1, int(pushCooldownSteps))
        self.immediateReversalPenalty = -abs(float(immediateReversalPenalty))
        self.repeatVisitPenaltyScale = -abs(float(repeatVisitPenaltyScale))
        self.noProgressPenalty = -abs(float(noProgressPenalty))
        self.noProgressPatienceFactor = max(0.0, float(noProgressPatienceFactor))
        self.minNoProgressSteps = max(1, int(minNoProgressSteps))
        self.maxNoProgressSteps = max(self.minNoProgressSteps, int(maxNoProgressSteps))
        self.rewardClipMin = rewardClipMin
        self.rewardClipMax = rewardClipMax
        self.stepLimitStepCoeff = max(1, int(stepLimitStepCoeff))
        self.stepLimitAreaCoeff = float(stepLimitAreaCoeff)
        self.stepLimitMax = max(1, int(stepLimitMax))
        self.stepLimitMin = max(1, int(stepLimitMin))
        self.stepLimitPenalty = float(stepLimitPenalty)
        self._validate_ranges()

    @classmethod
    def from_profile_dict(cls, profile: Any) -> QLearningConfig:
        if not isinstance(profile, dict):
            raise TypeError(f"profile must be a dict, got {type(profile).__name__}")
        raw = dict(profile)
        return cls(
            learningRate=_as_float(raw.get("learningRate", 0.1), "learningRate"),
            discountFactor=_as_float(raw.get("discountFactor", 0.9), "discountFactor"),
            epsilonStart=_as_float(raw.get("epsilonStart", 1.0), "epsilonStart"),
            epsilonEnd=_as_float(raw.get("epsilonEnd", 0.05), "epsilonEnd"),
            epsilonDecaySteps=_as_int(raw.get("epsilonDecaySteps", 100000), "epsilonDecaySteps", 1),
            maxStepsPerEpisode=_as_int(raw.get("maxStepsPerEpisode", 5000), "maxStepsPerEpisode", 1),
            usePositionInState=_as_bool(raw.get("usePositionInState", True), "usePositionInState"),
            useMacroActions=_as_bool(raw.get("useMacroActions", False), "useMacroActions"),
            useMacroOnlyPolicy=_as_bool(raw.get("useMacroOnlyPolicy", False), "useMacroOnlyPolicy"),
            useEntityObservation=_as_bool(raw.get("useEntityObservation", True), "useEntityObservation"),
            useEnemyObservation=_as_bool(raw.get("useEnemyObservation", False), "useEnemyObservation"),
            useAttackActions=_as_bool(raw.get("useAttackActions", False), "useAttackActions"),
            autoAttackAdjacentEnemy=_as_bool(
                raw.get("autoAttackAdjacentEnemy", False),
                "autoAttackAdjacentEnemy",
            ),
            pushCooldownSteps=_as_int(raw.get("pushCooldownSteps", 3), "pushCooldownSteps", 1),
            immediateReversalPenalty=_as_penalty_float(
                raw.get("immediateReversalPenalty", -12.0),
                "immediateReversalPenalty",
            ),
            repeatVisitPenaltyScale=_as_penalty_float(
                raw.get("repeatVisitPenaltyScale", -2.0),
                "repeatVisitPenaltyScale",
            ),
            noProgressPenalty=_as_penalty_float(raw.get("noProgressPenalty", -150.0), "noProgressPenalty"),
            noProgressPatienceFactor=_as_float(
                raw.get("noProgressPatienceFactor", 2.0),
                "noProgressPatienceFactor",
            ),
            minNoProgressSteps=_as_int(raw.get("minNoProgressSteps", 30), "minNoProgressSteps", 1),
            maxNoProgressSteps=_as_int(raw.get("maxNoProgressSteps", 1000), "maxNoProgressSteps", 1),
            rewardClipMin=_as_optional_float(raw.get("rewardClipMin", None), "rewardClipMin"),
            rewardClipMax=_as_optional_float(raw.get("rewardClipMax", None), "rewardClipMax"),
            stepLimitStepCoeff=_as_int(raw.get("stepLimitStepCoeff", 12), "stepLimitStepCoeff", 1),
            stepLimitAreaCoeff=_as_float(raw.get("stepLimitAreaCoeff", 0.5), "stepLimitAreaCoeff"),
            stepLimitMax=_as_int(raw.get("stepLimitMax", 5000), "stepLimitMax", 1),
            stepLimitMin=_as_int(raw.get("stepLimitMin", 200), "stepLimitMin", 1),
            stepLimitPenalty=_as_float(raw.get("stepLimitPenalty", -150.0), "stepLimitPenalty"),
        )

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
        if (self.rewardClipMin is None) != (self.rewardClipMax is None):
            raise ConfigValidationError("rewardClipMin and rewardClipMax must both be set or both be blank")
        if self.rewardClipMin is not None and self.rewardClipMax is not None and self.rewardClipMin > self.rewardClipMax:
            raise ConfigValidationError("rewardClipMin must be <= rewardClipMax")
        if self.maxNoProgressSteps < self.minNoProgressSteps:
            raise ConfigValidationError("maxNoProgressSteps must be >= minNoProgressSteps")

    def customize(self) -> None:
        self.learningRate = self._getFloatInput("Enter learning rate (default 0.1): ", self.learningRate)
        self.discountFactor = self._getFloatInput("Enter discount factor (default 0.9): ", self.discountFactor)

    @staticmethod
    def _getFloatInput(prompt: str, default: float) -> float:
        try:
            return float(input(prompt) or default)
        except ValueError:
            print(f"Invalid input. Using default value {default}.")
            return default


BOT_SPEC: dict[str, Any] = {
    "type": "QLearningBot",
    "class": QLearningConfig,
    "params": {
        "Learning Rate": "learningRate",
        "Discount Factor": "discountFactor",
        "Epsilon Start": "epsilonStart",
        "Epsilon End": "epsilonEnd",
        "Epsilon Decay Steps": "epsilonDecaySteps",
        "Max Steps Per Episode": "maxStepsPerEpisode",
        "Use Macro Actions (0/1)": "useMacroActions",
        "Use Macro-Only Policy (0/1)": "useMacroOnlyPolicy",
        "Use Entity Observation (0/1)": "useEntityObservation",
        "Use Enemy Observation (0/1)": "useEnemyObservation",
        "Use Attack Actions (0/1)": "useAttackActions",
        "Auto Attack Adjacent Enemy (0/1)": "autoAttackAdjacentEnemy",
        "Push Cooldown Steps": "pushCooldownSteps",
        "Immediate Reversal Penalty": "immediateReversalPenalty",
        "Repeat Visit Penalty Scale": "repeatVisitPenaltyScale",
        "No Progress Penalty": "noProgressPenalty",
        "No Progress Patience Factor": "noProgressPatienceFactor",
        "Min No Progress Steps": "minNoProgressSteps",
        "Max No Progress Steps": "maxNoProgressSteps",
        "Reward Clip Min": "rewardClipMin",
        "Reward Clip Max": "rewardClipMax",
        "Step Limit Step Coeff": "stepLimitStepCoeff",
        "Step Limit Area Coeff": "stepLimitAreaCoeff",
        "Step Limit Max": "stepLimitMax",
        "Step Limit Min": "stepLimitMin",
        "Step Limit Penalty": "stepLimitPenalty",
    },
    "tabs": [
        {"key": "general", "label": "General"},
        {"key": "rewards", "label": "Rewards"},
    ],
    "paramTabs": {
        "learningRate": "general",
        "discountFactor": "general",
        "epsilonStart": "general",
        "epsilonEnd": "general",
        "epsilonDecaySteps": "general",
        "maxStepsPerEpisode": "general",
        "useMacroActions": "general",
        "useMacroOnlyPolicy": "general",
        "useEntityObservation": "general",
        "useEnemyObservation": "general",
        "useAttackActions": "general",
        "autoAttackAdjacentEnemy": "general",
        "pushCooldownSteps": "general",
        "stepLimitStepCoeff": "general",
        "stepLimitAreaCoeff": "general",
        "stepLimitMax": "general",
        "stepLimitMin": "general",
        "rewardClipMin": "rewards",
        "rewardClipMax": "rewards",
        "immediateReversalPenalty": "rewards",
        "repeatVisitPenaltyScale": "rewards",
        "noProgressPenalty": "rewards",
        "noProgressPatienceFactor": "rewards",
        "minNoProgressSteps": "rewards",
        "maxNoProgressSteps": "rewards",
        "stepLimitPenalty": "rewards",
    },
    "paramHelp": {
        "learningRate": "How aggressively the Q-table updates after each step. Higher learns faster but can become unstable.",
        "discountFactor": "How much the bot values future reward versus immediate reward. Values closer to 1 favor long-term planning.",
        "epsilonStart": "Initial exploration rate used at the start of training.",
        "epsilonEnd": "Lowest exploration rate allowed after epsilon decay finishes.",
        "epsilonDecaySteps": "Number of environment steps used to decay epsilon from start to end.",
        "maxStepsPerEpisode": "Hard cap on steps before an episode is terminated.",
        "useMacroActions": "Enable higher-level actions made of multiple primitive moves. Useful for more structured exploration.",
        "useMacroOnlyPolicy": "Restrict the policy to macro-actions only. This removes direct primitive action choices.",
        "useEntityObservation": "Include extra observed environment features in the state representation.",
        "useEnemyObservation": "Append a compact enemy-scan slot (direction bucket and distance bucket) to the Q-table key. Keep disabled unless enemies are present to avoid unnecessary state explosion.",
        "useAttackActions": "Enable directional attack actions (attack_up/down/left/right). Adds 4 new actions that kill adjacent enemies in 1 hit. Incompatible with existing Q-tables.",
        "autoAttackAdjacentEnemy": "Cheat toggle: when enabled and attack actions are available, the bot forcibly picks the matching attack action whenever a live enemy is adjacent. Useful for proving combat wiring and bootstrapping combat learning.",
        "pushCooldownSteps": "Cooldown (in steps) between pushes. Bot cannot push-displace an enemy until this many steps have passed since the last push.",
        "immediateReversalPenalty": "Penalty added when the bot immediately undoes its previous move on the next step.",
        "repeatVisitPenaltyScale": "Penalty added when the bot steps onto a cell it already visited earlier in the same episode.",
        "noProgressPenalty": "Extra penalty applied when the bot hits the no-progress limit.",
        "noProgressPatienceFactor": "Multiplier used to compute the allowed no-progress window in steps from the optimal path length.",
        "minNoProgressSteps": "Minimum allowed no-progress window, measured in consecutive steps.",
        "maxNoProgressSteps": "Maximum allowed no-progress window, measured in consecutive steps.",
        "rewardClipMin": "Optional lower bound for reward clipping. Leave both clip fields blank to disable clipping.",
        "rewardClipMax": "Optional upper bound for reward clipping. Leave both clip fields blank to disable clipping.",
        "stepLimitStepCoeff": "Optimal-path multiplier used when computing the shared episode step budget.",
        "stepLimitAreaCoeff": "Maze-area multiplier used when computing the shared episode step budget.",
        "stepLimitMax": "Upper bound for the dynamic episode step budget before maxStepsPerEpisode is applied.",
        "stepLimitMin": "Lower bound for the dynamic episode step budget.",
        "stepLimitPenalty": "Penalty applied when the shared episode step budget is exhausted.",
    },
    "negativeParams": [
        "immediateReversalPenalty",
        "repeatVisitPenaltyScale",
        "noProgressPenalty",
    ],
    "rewards": {
        "goal_reached": 1000,
        "hit_wall": -100,
        "revisit_optimal_path": -10,
        "revisit_non_optimal_path": -15,
        "new_tile_visited": 2,
        "move_in_optimal_path": 5,
        "see_goal_new_location": 50,
        "see_goal_revisit": 5,
        "per_move_penalty": -1,
        "enemy_contact": -250,
        "death_by_enemy": -1000,
        "enemy_killed": 500,
    },
    "rewardLabels": {
        "goal_reached": "Goal Reached Reward",
        "hit_wall": "Hit Wall Penalty",
        "revisit_optimal_path": "Revisit Optimal Path Penalty",
        "revisit_non_optimal_path": "Revisit Non-Optimal Path Penalty",
        "new_tile_visited": "New Tile Visited Reward",
        "move_in_optimal_path": "Move In Optimal Path Reward",
        "see_goal_new_location": "See Goal New Location Reward",
        "see_goal_revisit": "See Goal Revisit Reward",
        "per_move_penalty": "Per Move Penalty",
        "enemy_contact": "Enemy Contact Penalty",
        "death_by_enemy": "Death By Enemy Penalty",
        "enemy_killed": "Enemy Killed Reward",
    },
    "rewardSections": [
        {"title": "Value Bounds", "keys": ["rewardClipMin", "rewardClipMax"]},
        {
            "title": "Penalties And Progress Rules",
            "keys": [
                "immediateReversalPenalty",
                "repeatVisitPenaltyScale",
                "noProgressPenalty",
                "noProgressPatienceFactor",
                "minNoProgressSteps",
                "maxNoProgressSteps",
                "stepLimitPenalty",
                "hit_wall",
                "revisit_optimal_path",
                "revisit_non_optimal_path",
                "per_move_penalty",
                "enemy_contact",
                "death_by_enemy",
            ],
        },
        {
            "title": "Exploration",
            "keys": ["new_tile_visited"],
        },
        {
            "title": "Positive Rewards",
            "keys": [
                "goal_reached",
                "move_in_optimal_path",
                "see_goal_new_location",
                "see_goal_revisit",
            ],
        },
        {
            "title": "Combat",
            "keys": ["enemy_killed"],
        },
    ],
    "rewardHelp": {
        "goal_reached": "Reward granted when the bot reaches the maze goal.",
        "hit_wall": "Penalty applied when the bot tries to move into a wall.",
        "revisit_optimal_path": "Reward or penalty for revisiting a tile that lies on the optimal path.",
        "revisit_non_optimal_path": "Reward or penalty for revisiting a tile outside the optimal path.",
        "new_tile_visited": "Reward granted the first time the bot steps onto a tile it has not visited yet in the current episode.",
        "move_in_optimal_path": "Reward for moving along the optimal path toward the goal.",
        "see_goal_new_location": "Reward when the bot first spots the goal from a new position.",
        "see_goal_revisit": "Reward when the bot spots the goal again from a previously used viewpoint.",
        "per_move_penalty": "Small step cost applied every move to encourage shorter solutions.",
        "enemy_contact": "Penalty applied when an enemy catches the bot during the enemy turn.",
        "death_by_enemy": "Additional terminal penalty applied when the episode ends because an enemy caught the bot.",
        "enemy_killed": "Reward granted each time the bot kills an enemy (via attack or push-into-wall).",
    },
    "negativeRewards": [
        "hit_wall",
        "revisit_optimal_path",
        "revisit_non_optimal_path",
        "per_move_penalty",
        "enemy_contact",
        "death_by_enemy",
    ],
}
