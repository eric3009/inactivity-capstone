import pandas as pd
import pytest

from inactivity.splits import BLOCKS, assign_blocks, make_snapshots, validate_embargo


@pytest.mark.parametrize("horizon", [30, 60, 90, 120, 180])
def test_no_label_window_crosses_into_a_later_block(horizon):
    snaps = make_snapshots("2010-02-28", "2011-12-09", horizon)
    blocks, dropped = assign_blocks(snaps, "2010-08-31", "2011-01-31", "2011-06-30", horizon)
    validate_embargo(blocks, horizon)
    for i, name in enumerate(BLOCKS[:-1]):
        later = [s for b in BLOCKS[i + 1:] for s in blocks[b]]
        if not later:      # e.g. a 180-day horizon leaves no matured test snapshot in this data
            continue
        for s in blocks[name]:
            assert s + pd.Timedelta(days=horizon) <= min(later)


def test_every_snapshot_is_kept_or_dropped_exactly_once():
    snaps = make_snapshots("2010-02-28", "2011-12-09", 90)
    blocks, dropped = assign_blocks(snaps, "2010-08-31", "2011-01-31", "2011-06-30", 90)
    kept = [s for b in BLOCKS for s in blocks[b]]
    assert sorted(kept + dropped) == sorted(snaps)
    assert len(set(kept)) == len(kept)


def test_snapshots_have_matured_labels():
    last = pd.Timestamp("2011-12-09")
    for s in make_snapshots("2010-02-28", last, 90):
        assert s + pd.Timedelta(days=90) <= last


def test_validate_embargo_catches_a_violation():
    bad = {"train": [pd.Timestamp("2010-05-31")], "val": [pd.Timestamp("2010-06-30")], "cal": [], "test": []}
    with pytest.raises(AssertionError):
        validate_embargo(bad, 90)


def test_configured_blocks_are_chronological():
    snaps = make_snapshots("2010-02-28", "2011-12-09", 90)
    blocks, _ = assign_blocks(snaps, "2010-08-31", "2011-01-31", "2011-06-30", 90)
    ends = [max(blocks[b]) for b in BLOCKS]
    assert ends == sorted(ends)
