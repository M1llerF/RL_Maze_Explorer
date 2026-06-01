from __future__ import annotations

import inspect
from typing import Any, cast

from bots import discoverBotSpecs
from bots.qlearning import QLearningConfig


botConfigs: dict[str, dict[str, Any]] = discoverBotSpecs()


def getConfigClassForBotType(botType: str | None) -> type[Any]:
    """Return the config class registered for a bot type, defaulting to QLearningConfig."""
    if botType and botType in botConfigs:
        configCls = botConfigs[botType].get("class")
        if isinstance(configCls, type):
            return configCls
    return QLearningConfig


def buildConfigForBotType(botType: str | None, rawConfig: Any) -> Any:
    """Build a bot config instance from raw profile payload data."""
    configCls = getConfigClassForBotType(botType)
    if isinstance(rawConfig, configCls):
        return rawConfig

    cfgDict = cast(dict[str, Any], rawConfig) if isinstance(rawConfig, dict) else {}
    if isinstance(cfgDict, dict):
        fromProfile = getattr(configCls, "from_profile_dict", None)
        if callable(fromProfile):
            try:
                return fromProfile(cfgDict)
            except Exception:
                pass

    try:
        signature = inspect.signature(configCls)
        allowed: set[str] = {
            name
            for name in signature.parameters
            if name != "self"
            and signature.parameters[name].kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
    except (TypeError, ValueError):
        allowed = set()

    normalized: dict[str, Any] = {}
    for k, v in cfgDict.items():
        key = str(k)
        normalized[key] = v
        if "_" in key:
            parts = [p for p in key.split("_") if p]
            if parts:
                camel = parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])
                normalized.setdefault(camel, v)

    kwargs = {k: v for k, v in normalized.items() if k in allowed and v is not None}
    try:
        return configCls(**kwargs)
    except TypeError:
        return configCls()
