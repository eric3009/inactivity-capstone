"""Chronological splits with a label-maturity embargo (no random splits)."""
from __future__ import annotations

import pandas as pd

BLOCKS = ["train", "val", "cal", "test"]


def make_snapshots(start, data_max, horizon: int, freq: str = "ME") -> list[pd.Timestamp]:
    """Snapshots whose label window closes on or before the last observed date."""
    last = pd.Timestamp(data_max).normalize() - pd.Timedelta(days=horizon)
    return list(pd.date_range(pd.Timestamp(start), last, freq=freq))


def assign_blocks(snapshots, val_start, cal_start, test_start, horizon: int) -> tuple[dict, list]:
    """Assign snapshots to train/val/cal/test; drop any whose label window overlaps the next block.

    A snapshot s in block b is kept only if s + horizon <= start of block b+1, so no label used
    to fit or tune on an earlier block depends on behaviour observed after the later block begins.
    """
    starts = [pd.Timestamp(val_start), pd.Timestamp(cal_start), pd.Timestamp(test_start)]
    blocks: dict[str, list] = {b: [] for b in BLOCKS}
    dropped: list = []
    for s in snapshots:
        s = pd.Timestamp(s)
        b = sum(s >= st for st in starts)           # 0=train .. 3=test
        if b < 3 and s + pd.Timedelta(days=horizon) > starts[b]:
            dropped.append(s)
            continue
        blocks[BLOCKS[b]].append(s)
    return blocks, dropped


def validate_embargo(blocks: dict, horizon: int) -> None:
    """Raise if any earlier block's label window reaches into a later block."""
    for i, name in enumerate(BLOCKS[:-1]):
        later = [s for b in BLOCKS[i + 1:] for s in blocks[b]]
        if blocks[name] and later:
            if max(blocks[name]) + pd.Timedelta(days=horizon) > min(later):
                raise AssertionError(f"Embargo violated between '{name}' and a later block")


def summarise(blocks: dict, dropped: list) -> dict:
    return {
        "blocks": {b: [str(s.date()) for s in v] for b, v in blocks.items()},
        "dropped_by_embargo": [str(s.date()) for s in dropped],
    }
