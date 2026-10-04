import warnings

import pytest

warnings.filterwarnings("ignore")

from inactivity.config import load_config  # noqa: E402
from inactivity.data import clean  # noqa: E402
from inactivity.synthetic import generate_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return generate_raw(seed=3)


@pytest.fixture(scope="session")
def tx(raw):
    df, _ = clean(raw)
    return df


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory):
    """Train a tiny model on synthetic data once; reused by the API tests."""
    from inactivity.pipeline import run
    root = tmp_path_factory.mktemp("proj")
    cfg = load_config(overrides={
        "modelling": {"lgbm_trials": 2, "bootstrap_reps": 0},
        "tracking": {"mlflow": False},
        "data": {"interim_dir": str(root / "interim")},
        "paths": {"artifacts_dir": str(root / "artifacts"), "reports_dir": str(root / "reports")},
    })
    run(cfg, synthetic=True, sensitivity=False)
    return root / "artifacts" / "model_bundle.joblib"
