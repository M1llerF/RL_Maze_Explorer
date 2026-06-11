from __future__ import annotations

import re
from typing import Any, Mapping


_CANONICAL_INTERNAL_KEYS: dict[str, str] = {
    "r_m_observed": "R_m_observed",
    "rm_observed": "R_m_observed",
    "r_m_observed_mean": "R_m_observed_mean",
    "rm_observed_mean": "R_m_observed_mean",
    "r_m_observed_std": "R_m_observed_std",
    "rm_observed_std": "R_m_observed_std",
    "train_r_m_observed": "train_R_m_observed",
    "train_rm_observed": "train_R_m_observed",
    "eval_r_m_observed": "eval_R_m_observed",
    "eval_rm_observed": "eval_R_m_observed",
    "train_r_m_observed_mean": "train_R_m_observed_mean",
    "train_rm_observed_mean": "train_R_m_observed_mean",
    "train_r_m_observed_std": "train_R_m_observed_std",
    "train_rm_observed_std": "train_R_m_observed_std",
    "eval_r_m_observed_mean": "eval_R_m_observed_mean",
    "eval_rm_observed_mean": "eval_R_m_observed_mean",
    "eval_r_m_observed_std": "eval_R_m_observed_std",
    "eval_rm_observed_std": "eval_R_m_observed_std",
    "r_m_proxy": "R_m_proxy",
    "rm_proxy": "R_m_proxy",
    "break_even_r_m_definition": "break_even_R_m_definition",
    "break_even_rm_definition": "break_even_R_m_definition",
    "compression_ratio_c": "compression_ratio_C",
    "step_overhead_o": "step_overhead_O",
}


def _is_stable_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9_]+", name))


def snake_to_camel(name: str) -> str:
    if _is_stable_identifier(str(name)):
        return str(name)
    parts = str(name).split("_")
    if not parts:
        return str(name)
    first = parts[0]
    head = first[:1].lower() + first[1:] if first else first
    tail = "".join(part[:1].upper() + part[1:] if part else "" for part in parts[1:])
    return head + tail


def camel_to_snake(name: str) -> str:
    if _is_stable_identifier(str(name)):
        return str(name)
    value = str(name).replace("-", "_")
    value = re.sub(r"(.)([A-Z][a-z0-9]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return _CANONICAL_INTERNAL_KEYS.get(value.lower(), value.lower())


def normalize_export_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {camel_to_snake(key): normalize_export_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_export_keys(item) for item in value]
    return value


def camelize_export_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {snake_to_camel(str(key)): camelize_export_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [camelize_export_keys(item) for item in value]
    return value


def export_value(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in data:
        return data[key]
    return data.get(snake_to_camel(key), default)
