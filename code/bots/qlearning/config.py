from __future__ import annotations

from typing import Any


class QLearningConfig:
    def __init__(
        self,
        learningRate: float = 0.1,
        discountFactor: float = 0.9,
        usePositionInState: bool = True,
        useMacroActions: bool = False,
        useMacroOnlyPolicy: bool = False,
        useEntityObservation: bool = True,
        useEnemyObservation: bool = False,
        useAttackActions: bool = False,
        autoAttackAdjacentEnemy: bool = False,
        pushCooldownSteps: int = 3,
        rewardClipMin: float | None = None,
        rewardClipMax: float | None = None,
    ) -> None:
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.usePositionInState = usePositionInState
        self.useMacroActions = bool(useMacroActions or useMacroOnlyPolicy)
        self.useMacroOnlyPolicy = bool(useMacroOnlyPolicy)
        self.useEntityObservation = bool(useEntityObservation)
        self.useEnemyObservation = bool(useEnemyObservation)
        self.useAttackActions = bool(useAttackActions)
        self.autoAttackAdjacentEnemy = bool(autoAttackAdjacentEnemy)
        self.pushCooldownSteps = max(1, int(pushCooldownSteps))
        self.rewardClipMin = rewardClipMin
        self.rewardClipMax = rewardClipMax

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
        "Use Macro Actions (0/1)": "useMacroActions",
        "Use Macro-Only Policy (0/1)": "useMacroOnlyPolicy",
        "Use Entity Observation (0/1)": "useEntityObservation",
        "Use Enemy Observation (0/1)": "useEnemyObservation",
        "Use Attack Actions (0/1)": "useAttackActions",
        "Auto Attack Adjacent Enemy (0/1)": "autoAttackAdjacentEnemy",
        "Push Cooldown Steps": "pushCooldownSteps",
        "Reward Clip Min": "rewardClipMin",
        "Reward Clip Max": "rewardClipMax",
    },
    "tabs": [
        {"key": "general", "label": "General"},
        {"key": "rewards", "label": "Rewards"},
    ],
    "paramTabs": {
        "learningRate": "general",
        "discountFactor": "general",
        "useMacroActions": "general",
        "useMacroOnlyPolicy": "general",
        "useEntityObservation": "general",
        "useEnemyObservation": "general",
        "useAttackActions": "general",
        "autoAttackAdjacentEnemy": "general",
        "pushCooldownSteps": "general",
        "rewardClipMin": "rewards",
        "rewardClipMax": "rewards",
    },
    "paramHelp": {
        "learningRate": "How aggressively the Q-table updates after each step. Higher learns faster but can become unstable.",
        "discountFactor": "How much the bot values future reward versus immediate reward. Values closer to 1 favor long-term planning.",
        "useMacroActions": "Enable higher-level actions made of multiple primitive moves. Useful for more structured exploration.",
        "useMacroOnlyPolicy": "Restrict the policy to macro-actions only. This removes direct primitive action choices.",
        "useEntityObservation": "Include extra observed environment features in the state representation.",
        "useEnemyObservation": "Append a compact enemy-scan slot (direction bucket and distance bucket) to the Q-table key. Keep disabled unless enemies are present to avoid unnecessary state explosion.",
        "useAttackActions": "Enable directional attack actions (attack_up/down/left/right). Adds 4 new actions that kill adjacent enemies in 1 hit. Incompatible with existing Q-tables.",
        "autoAttackAdjacentEnemy": "Cheat toggle: when enabled and attack actions are available, the bot forcibly picks the matching attack action whenever a live enemy is adjacent. Useful for proving combat wiring and bootstrapping combat learning.",
        "pushCooldownSteps": "Cooldown (in steps) between pushes. Bot cannot push-displace an enemy until this many steps have passed since the last push.",
        "rewardClipMin": "Optional lower bound for reward clipping. Leave both clip fields blank to disable clipping.",
        "rewardClipMax": "Optional upper bound for reward clipping. Leave both clip fields blank to disable clipping.",
    },
    "negativeParams": [],
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
            "title": "Penalties",
            "keys": [
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
