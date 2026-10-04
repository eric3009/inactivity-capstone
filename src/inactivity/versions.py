"""Library versions recorded with the model, and a check that the serving environment matches them.

A pickled scikit-learn or LightGBM model is only guaranteed to load faithfully under the version it was trained with,
so the bundle stores the versions and the API warns (and reports it in /model-info) when they differ.
"""
from __future__ import annotations

import platform
from importlib import metadata

TRACKED = ("scikit-learn", "lightgbm", "pandas", "numpy")


def current_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in TRACKED:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def mismatches(saved: dict | None, current: dict | None = None) -> list[str]:
    """Human-readable differences between the training and the current environment (empty list if they match)."""
    if not saved:
        return ["the bundle does not record library versions"]
    current = current or current_versions()
    diffs = []
    for pkg in TRACKED:
        if saved.get(pkg) != current.get(pkg):
            diffs.append(f"{pkg}: trained with {saved.get(pkg)}, running {current.get(pkg)}")
    return diffs
