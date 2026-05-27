
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Mapping
from typing import Any, cast


def discoverBotSpecs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for moduleInfo in pkgutil.iter_modules(__path__):
        pkgName = moduleInfo.name
        try:
            module = importlib.import_module(f"{__name__}.{pkgName}")
        except Exception:
            continue
        botSpec = getattr(module, "BOT_SPEC", None)
        if isinstance(botSpec, Mapping):
            typedSpec = cast(Mapping[str, Any], botSpec)
            botType = typedSpec.get("type")
            if isinstance(botType, str) and botType:
                entry: dict[str, Any] = dict(typedSpec)
                entry.pop("type", None)
                specs[botType] = entry
    return specs


def discoverBotClasses() -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for moduleInfo in pkgutil.iter_modules(__path__):
        pkgName = moduleInfo.name
        try:
            module = importlib.import_module(f"{__name__}.{pkgName}")
        except Exception:
            continue

        botType = getattr(module, "BOT_TYPE", None)
        botClass = getattr(module, "BOT_CLASS", None)
        if isinstance(botType, str) and botType and botClass is not None:
            classes[botType] = botClass
            continue

        botSpec = getattr(module, "BOT_SPEC", None)
        if isinstance(botSpec, Mapping):
            typedSpec = cast(Mapping[str, Any], botSpec)
            specType = typedSpec.get("type")
            specClass = getattr(module, str(specType), None) if isinstance(specType, str) else None
            if isinstance(specType, str) and specType and specClass is not None:
                classes[specType] = specClass
    return classes
