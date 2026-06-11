from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from services.research.schema import normalize_export_keys


class PlanValidationError(ValueError):
    pass


LSTM_MACRO_PLAN_AGENTS: tuple[str, ...] = (
    "DQN_LSTM_PRIMITIVE",
    "DQN_LSTM_NAIVE_MACROS",
    "DQN_LSTM_ASTAR_MACROS",
    "DQN_LSTM_ORACLE_MACROS",   # legacy alias for ASTAR — matches existing run data
    "DQN_LSTM_MOMENTUM_MACROS",
)


@dataclass
class ConditionSpec:
    condition_id: str
    mode: str
    width: int
    height: int
    training_episodes: int
    evaluation_episodes: int
    maze_id: str = ""
    maze_path: str = ""
    pool_size: int = 0
    evaluation_episodes_per_pool_maze: int = 5
    maze_id_prefix: str = ""
    maze_paths: list[str] = field(default_factory=list)
    training_maze_paths: list[str] = field(default_factory=list)
    evaluation_maze_paths: list[str] = field(default_factory=list)


@dataclass
class ProfileSpec:
    profile_name: str
    agent_type: str
    bot_type: str
    condition: ConditionSpec
    seed_label: str
    seed: int


@dataclass
class ExperimentPlan:
    experiment_name: str
    branch: str
    seeds: list[int]
    seed_labels: list[str]
    agents: list[str]
    conditions: dict[str, ConditionSpec]
    dqn_replay_warmup_policy: str
    raw: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str) -> ExperimentPlan:
        with open(path, "r", encoding="utf-8") as f:
            data = normalize_export_keys(json.load(f))
        ExperimentPlan._validate(data)
        conditions: dict[str, ConditionSpec] = {}
        for cid, cdata in data["conditions"].items():
            conditions[cid] = _parse_condition(cid, cdata)
        return ExperimentPlan(
            experiment_name=data["experiment_name"],
            branch=data["branch"],
            seeds=list(data["seeds"]),
            seed_labels=list(data["seed_labels"]),
            agents=list(data["agents"]),
            conditions=conditions,
            dqn_replay_warmup_policy=str(data.get("dqn_replay_warmup_policy", "disabled")),
            raw=data,
        )

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        if "experiment_name" not in data:
            raise PlanValidationError("Plan missing 'experiment_name'")
        if "branch" not in data:
            raise PlanValidationError("Plan missing 'branch'")
        seeds = data.get("seeds")
        seed_labels = data.get("seed_labels")
        if not seeds or not isinstance(seeds, list):
            raise PlanValidationError("Plan missing or invalid 'seeds'")
        if not seed_labels or not isinstance(seed_labels, list):
            raise PlanValidationError("Plan missing or invalid 'seed_labels'")
        if len(seeds) != len(seed_labels):
            raise PlanValidationError(
                f"'seeds' and 'seed_labels' must have the same length "
                f"(got {len(seeds)} vs {len(seed_labels)})"
            )
        agents = data.get("agents")
        if not agents or not isinstance(agents, list):
            raise PlanValidationError("Plan missing or invalid 'agents'")
        unknown_agents = [str(agent) for agent in agents if str(agent) not in LSTM_MACRO_PLAN_AGENTS]
        if unknown_agents:
            raise PlanValidationError(
                "Research mode only supports lstm_macro_plan agents: "
                f"{list(LSTM_MACRO_PLAN_AGENTS)}; got unsupported {unknown_agents}"
            )
        dqn_replay_warmup_policy = str(data.get("dqn_replay_warmup_policy", "disabled"))
        if dqn_replay_warmup_policy not in {"disabled", "budget_scaled"}:
            raise PlanValidationError(
                "Plan field 'dqn_replay_warmup_policy' must be 'disabled' or 'budget_scaled'"
            )
        conditions = data.get("conditions")
        if not conditions or not isinstance(conditions, dict):
            raise PlanValidationError("Plan missing or invalid 'conditions'")
        for cid, cdata in conditions.items():
            _validate_condition(cid, cdata)

    def expand_profiles(
        self,
        agents_filter: list[str] | None = None,
        conditions_filter: list[str] | None = None,
    ) -> list[ProfileSpec]:
        profiles: list[ProfileSpec] = []
        agent_list = [a for a in self.agents if not agents_filter or a in agents_filter]
        cond_list = [c for c in self.conditions if not conditions_filter or c in conditions_filter]
        for cid in cond_list:
            cond = self.conditions[cid]
            for agent in agent_list:
                bot_type = _agent_to_bot_type(agent)
                for seed_label, seed in zip(self.seed_labels, self.seeds):
                    profile_name = f"{agent}_{cid}_{seed_label}"
                    profiles.append(ProfileSpec(
                        profile_name=profile_name,
                        agent_type=agent,
                        bot_type=bot_type,
                        condition=cond,
                        seed_label=seed_label,
                        seed=seed,
                    ))
        return profiles

    def total_training_episodes(
        self,
        agents_filter: list[str] | None = None,
        conditions_filter: list[str] | None = None,
    ) -> int:
        profiles = self.expand_profiles(agents_filter, conditions_filter)
        return sum(p.condition.training_episodes for p in profiles)

    def total_evaluation_episodes(
        self,
        agents_filter: list[str] | None = None,
        conditions_filter: list[str] | None = None,
    ) -> int:
        profiles = self.expand_profiles(agents_filter, conditions_filter)
        return sum(p.condition.evaluation_episodes for p in profiles)

    def agent_to_bot_type(self, agent: str) -> str:
        return _agent_to_bot_type(agent)


def _parse_condition(cid: str, data: dict[str, Any]) -> ConditionSpec:
    return ConditionSpec(
        condition_id=cid,
        mode=str(data["mode"]),
        width=int(data["width"]),
        height=int(data["height"]),
        training_episodes=int(data["training_episodes"]),
        evaluation_episodes=int(data["evaluation_episodes"]),
        maze_id=str(data.get("maze_id", "")),
        maze_path=str(data.get("maze_path", "")),
        pool_size=int(data.get("pool_size", 0)),
        evaluation_episodes_per_pool_maze=int(data.get("evaluation_episodes_per_pool_maze", 5)),
        maze_id_prefix=str(data.get("maze_id_prefix", "")),
        maze_paths=[str(path) for path in data.get("maze_paths", [])],
        training_maze_paths=[str(path) for path in data.get("training_maze_paths", [])],
        evaluation_maze_paths=[str(path) for path in data.get("evaluation_maze_paths", [])],
    )


def _validate_condition(cid: str, data: dict[str, Any]) -> None:
    for required in ("mode", "width", "height", "training_episodes", "evaluation_episodes"):
        if required not in data:
            raise PlanValidationError(f"Condition '{cid}' missing field '{required}'")
    mode = data["mode"]
    if mode not in ("fixed", "pool", "random"):
        raise PlanValidationError(f"Condition '{cid}' has invalid mode '{mode}'")
    for int_field in ("width", "height", "training_episodes", "evaluation_episodes"):
        v = data[int_field]
        if not isinstance(v, int) or v <= 0:
            raise PlanValidationError(f"Condition '{cid}' field '{int_field}' must be a positive integer")
    if mode == "fixed":
        maze_path = data.get("maze_path")
        if not isinstance(maze_path, str) or not maze_path.strip():
            raise PlanValidationError(f"Fixed condition '{cid}' missing non-empty 'maze_path'")
    if mode == "pool":
        if "pool_size" not in data:
            raise PlanValidationError(f"Pool condition '{cid}' missing 'pool_size'")
        if "maze_id_prefix" not in data:
            raise PlanValidationError(f"Pool condition '{cid}' missing 'maze_id_prefix'")
        if not isinstance(data["pool_size"], int) or data["pool_size"] <= 0:
            raise PlanValidationError(f"Pool condition '{cid}' 'pool_size' must be a positive integer")
        maze_paths = data.get("maze_paths")
        train_paths = data.get("training_maze_paths")
        eval_paths = data.get("evaluation_maze_paths")
        has_legacy_paths = isinstance(maze_paths, list) and bool(maze_paths)
        has_split_paths = isinstance(train_paths, list) and bool(train_paths) and isinstance(eval_paths, list) and bool(eval_paths)
        if not has_legacy_paths and not has_split_paths:
            raise PlanValidationError(
                f"Pool condition '{cid}' requires non-empty 'maze_paths' or both "
                "'training_maze_paths' and 'evaluation_maze_paths'"
            )
        if has_legacy_paths and len(maze_paths) != int(data["pool_size"]):
            raise PlanValidationError(
                f"Pool condition '{cid}' maze_paths length must equal pool_size "
                f"(got {len(maze_paths)} vs {int(data['pool_size'])})"
            )
        if has_split_paths and len(train_paths) != int(data["pool_size"]):
            raise PlanValidationError(
                f"Pool condition '{cid}' training_maze_paths length must equal pool_size "
                f"(got {len(train_paths)} vs {int(data['pool_size'])})"
            )
        per_pool = data.get("evaluation_episodes_per_pool_maze", 5)
        if not isinstance(per_pool, int) or per_pool <= 0:
            raise PlanValidationError(
                f"Pool condition '{cid}' 'evaluation_episodes_per_pool_maze' must be a positive integer"
            )
        # Split pool plans may intentionally evaluate on a held-out subset whose
        # cardinality differs from the training pool_size. In that case the
        # evaluation episode budget is derived from the evaluation split size.
        eval_pool_size = len(eval_paths) if has_split_paths else int(data["pool_size"])
        expected_eval = eval_pool_size * per_pool
        if int(data["evaluation_episodes"]) != expected_eval:
            raise PlanValidationError(
                f"Pool condition '{cid}' evaluation_episodes must equal "
                f"evaluation pool size * evaluation_episodes_per_pool_maze ({expected_eval})"
            )


_AGENT_BOT_MAP: dict[str, str] = {
    "DQN_LSTM_PRIMITIVE": "DQNBot",
    "DQN_LSTM_NAIVE_MACROS": "DQNBot",
    "DQN_LSTM_ASTAR_MACROS": "DQNBot",
    "DQN_LSTM_ORACLE_MACROS": "DQNBot",
    "DQN_LSTM_MOMENTUM_MACROS": "DQNBot",
}


def _agent_to_bot_type(agent: str) -> str:
    bt = _AGENT_BOT_MAP.get(agent)
    if bt is None:
        raise PlanValidationError(f"Unknown agent type '{agent}'. Known: {list(_AGENT_BOT_MAP)}")
    return bt
