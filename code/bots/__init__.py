
from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast


# Explicit registry: adding a new bot package requires adding it here.
# This makes registration auditable and prevents silent discovery failures.
_REGISTERED_BOT_PACKAGES: list[str] = [
    "qlearning",
    "dqnlearning",
]

_SPEC_REQUIRED_KEYS: frozenset[str] = frozenset({"type", "class", "tabs", "params"})


def _validateBotSpec(pkgName: str, spec: Any) -> bool:
    """Validate a BOT_SPEC dict; print [BotDiscovery] warnings on violations.

    Returns True only when the spec is fully valid.
    """
    if not isinstance(spec, Mapping):
        print(f"[BotDiscovery] '{pkgName}': BOT_SPEC is not a mapping — skipping")
        return False

    s: Mapping[str, Any] = cast(Mapping[str, Any], spec)

    missing = _SPEC_REQUIRED_KEYS - set(s.keys())
    if missing:
        print(f"[BotDiscovery] '{pkgName}': BOT_SPEC missing required keys {sorted(missing)} — skipping")
        return False

    botType = s.get("type")
    if not isinstance(botType, str) or not botType:
        print(f"[BotDiscovery] '{pkgName}': BOT_SPEC 'type' must be a non-empty string — skipping")
        return False

    configCls = s.get("class")
    if not isinstance(configCls, type):
        print(f"[BotDiscovery] '{pkgName}': BOT_SPEC 'class' must be a type — skipping")
        return False

    tabs = s.get("tabs")
    if not isinstance(tabs, list) or not tabs:
        print(f"[BotDiscovery] '{pkgName}' ({botType}): BOT_SPEC 'tabs' must be a non-empty list — skipping")
        return False
    for i, tab in enumerate(cast(list[Any], tabs)):
        if not isinstance(tab, dict) or not isinstance(tab.get("key"), str) or not isinstance(tab.get("label"), str):
            print(f"[BotDiscovery] '{pkgName}' ({botType}): tabs[{i}] must have string 'key' and 'label' — skipping")
            return False

    params = s.get("params")
    if not isinstance(params, dict):
        print(f"[BotDiscovery] '{pkgName}' ({botType}): BOT_SPEC 'params' must be a dict — skipping")
        return False

    return True


def discoverBotSpecs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for pkgName in _REGISTERED_BOT_PACKAGES:
        try:
            module = importlib.import_module(f"{__name__}.{pkgName}")
        except Exception as e:
            print(f"[BotDiscovery] Failed to import bot package '{pkgName}': {e}")
            continue
        botSpec = getattr(module, "BOT_SPEC", None)
        if not _validateBotSpec(pkgName, botSpec):
            continue
        typedSpec = cast(Mapping[str, Any], botSpec)
        botType = str(typedSpec["type"])
        entry: dict[str, Any] = dict(typedSpec)
        entry.pop("type", None)
        specs[botType] = entry
    return specs


def discoverBotClasses() -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for pkgName in _REGISTERED_BOT_PACKAGES:
        try:
            module = importlib.import_module(f"{__name__}.{pkgName}")
        except Exception as e:
            print(f"[BotDiscovery] Failed to import bot package '{pkgName}': {e}")
            continue

        botType = getattr(module, "BOT_TYPE", None)
        botClass = getattr(module, "BOT_CLASS", None)

        # Validate BOT_TYPE and BOT_CLASS together
        if botType is not None or botClass is not None:
            if not isinstance(botType, str) or not botType:
                print(f"[BotDiscovery] '{pkgName}': BOT_TYPE must be a non-empty string when BOT_CLASS is set — skipping")
                continue
            if botClass is None:
                print(f"[BotDiscovery] '{pkgName}': BOT_CLASS missing while BOT_TYPE='{botType}' is set — skipping")
                continue
            # Cross-check against BOT_SPEC type if present
            botSpec = getattr(module, "BOT_SPEC", None)
            if isinstance(botSpec, Mapping):
                typedBotSpec: Mapping[str, Any] = cast(Mapping[str, Any], botSpec)
                specType = typedBotSpec.get("type")
                if isinstance(specType, str) and specType and specType != botType:
                    print(
                        f"[BotDiscovery] '{pkgName}': BOT_TYPE '{botType}' does not match "
                        f"BOT_SPEC type '{specType}' — skipping"
                    )
                    continue
            classes[botType] = botClass

    return classes
