from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML config; `overrides` is a nested dict merged on top (used by tests)."""
    with open(path or DEFAULT_CONFIG, encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(cfg_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(cfg_path)
    return p if p.is_absolute() else ROOT / p
