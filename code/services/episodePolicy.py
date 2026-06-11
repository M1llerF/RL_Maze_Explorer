from __future__ import annotations

from typing import Any


def compute_step_limit(config: Any, maze: Any, optimalLength: int) -> int:
    # Step budget scales with both the optimal path length and maze area so large mazes
    # give the agent proportionally more steps. A hard cap from maxStepsPerEpisode applies
    # as an absolute ceiling regardless of maze size.
    stepCoeff = max(1, int(getattr(config, "stepLimitStepCoeff", 12)))
    areaCoeff = float(getattr(config, "stepLimitAreaCoeff", 0.5))
    limitMin = max(1, int(getattr(config, "stepLimitMin", 200)))
    limitMax = max(limitMin, int(getattr(config, "stepLimitMax", 5000)))
    areaBonus = int(areaCoeff * int(maze.width) * int(maze.height))
    if optimalLength > 0:
        dynamicLimit = min(limitMax, max(limitMin, stepCoeff * int(optimalLength) + areaBonus))
    else:
        dynamicLimit = max(limitMin, areaBonus)
    hardCap = max(1, int(getattr(config, "maxStepsPerEpisode", limitMax)))
    return min(dynamicLimit, hardCap)


def compute_no_progress_patience(config: Any, optimalLength: int, stepLimit: int) -> int:
    # Patience scales with optimal path length so short mazes terminate stalled agents
    # sooner. A minimum floor prevents the window from collapsing on trivial mazes.
    patienceFactor = max(0.0, float(getattr(config, "noProgressPatienceFactor", 2.0)))
    minSteps = max(1, int(getattr(config, "minNoProgressSteps", 30)))
    maxSteps = max(minSteps, int(getattr(config, "maxNoProgressSteps", max(stepLimit, minSteps))))
    dynamicPatience = int(max(1, int(optimalLength)) * patienceFactor)
    return min(int(stepLimit), max(minSteps, min(maxSteps, dynamicPatience)))
