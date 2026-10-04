"""Baselines, candidate models, benchmark and probability calibration."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ baselines
def rule_score(X: pd.DataFrame) -> np.ndarray:
    """Fixed recency rule: the longer since the last purchase, the higher the risk."""
    return X["recency_days"].to_numpy(dtype=float)


# ------------------------------------------------------------------ logistic regression
def fit_logreg(Xtr, ytr, Xva, yva, Cs, seed=42):
    rows, best, best_ll = [], None, np.inf
    for C in Cs:
        pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=C, max_iter=3000, random_state=seed))
        pipe.fit(Xtr, ytr)
        ll = log_loss(yva, pipe.predict_proba(Xva)[:, 1], labels=[0, 1])
        rows.append({"C": C, "val_logloss": ll})
        if ll < best_ll:
            best, best_ll = pipe, ll
    return best, pd.DataFrame(rows)


# ------------------------------------------------------------------ LightGBM
def _lgbm_space(rng: np.random.Generator) -> dict:
    return {
        "num_leaves": int(rng.choice([7, 15, 31, 63])),
        "learning_rate": float(rng.choice([0.02, 0.05, 0.1])),
        "min_child_samples": int(rng.choice([20, 50, 100])),
        "subsample": float(rng.choice([0.7, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
        "reg_lambda": float(rng.choice([0.0, 1.0, 10.0])),
    }


def _fit_lgbm_random(Xtr, ytr, Xva, yva, trials=12, early_stopping_rounds=50, seed=42):
    """Fallback random search (used only if Optuna is unavailable); configurations are ranked on validation log loss with early stopping."""
    rng = np.random.default_rng(seed)
    best, best_ll, rows = None, np.inf, []
    for t in range(trials):
        params = _lgbm_space(rng)
        model = lgb.LGBMClassifier(n_estimators=800, subsample_freq=1, random_state=seed,
                                   verbose=-1, n_jobs=2, **params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        ll = log_loss(yva, model.predict_proba(Xva)[:, 1], labels=[0, 1])
        rows.append({**params, "best_iteration": int(model.best_iteration_ or 0), "val_logloss": ll})
        if ll < best_ll:
            best, best_ll = model, ll
    return best, pd.DataFrame(rows).sort_values("val_logloss").reset_index(drop=True)



def _tpe_params(trial) -> dict:
    """Log-aware 6-dimensional search space (Week 4: learning rate, leaves, leaf size, subsampling, L2)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
    }


def fit_lgbm(Xtr, ytr, Xva, yva, trials=60, early_stopping_rounds=50, seed=42):
    """Optuna TPE search on validation log loss; n_estimators is fixed high and early stopping picks the real size.

    Returns (best model refitted with its best parameters, trial table sorted by validation log loss).
    Falls back to random search if Optuna is not installed.
    """
    try:
        import optuna
    except Exception:
        return _fit_lgbm_random(Xtr, ytr, Xva, yva, trials, early_stopping_rounds, seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def build(params):
        return lgb.LGBMClassifier(n_estimators=2000, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=2, **params)

    def fit_one(params):
        m = build(params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        return m

    def objective(trial):
        m = fit_one(_tpe_params(trial))
        trial.set_user_attr("best_iteration", int(m.best_iteration_ or 0))
        return log_loss(yva, m.predict_proba(Xva)[:, 1], labels=[0, 1])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = fit_one(study.best_params)
    rows = [{**t.params, "best_iteration": t.user_attrs.get("best_iteration"), "val_logloss": t.value}
            for t in study.trials if t.value is not None]
    return best, pd.DataFrame(rows).sort_values("val_logloss").reset_index(drop=True)


# ------------------------------------------------------------------ the rest of the course model ladder
def _imputer():
    return SimpleImputer(strategy="median", add_indicator=True)


def fit_tree(Xtr, ytr, Xva, yva, seed=42):
    """Single depth-limited decision tree: readable, high-variance reference point for the ensembles."""
    from sklearn.tree import DecisionTreeClassifier
    best, best_ll = None, np.inf
    for leaf in (50, 100, 200):
        pipe = make_pipeline(_imputer(), DecisionTreeClassifier(max_depth=4, min_samples_leaf=leaf, random_state=seed))
        pipe.fit(Xtr, ytr)
        ll = log_loss(yva, np.clip(pipe.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1])
        if ll < best_ll:
            best, best_ll = pipe, ll
    return best


def fit_rf(Xtr, ytr, Xva, yva, seed=42):
    """Random forest (bagging + feature decorrelation); leaf size chosen on validation log loss."""
    from sklearn.ensemble import RandomForestClassifier
    best, best_ll = None, np.inf
    for leaf in (10, 20, 50):
        pipe = make_pipeline(_imputer(), RandomForestClassifier(n_estimators=400, min_samples_leaf=leaf, max_features="sqrt",
                                                                n_jobs=2, random_state=seed))
        pipe.fit(Xtr, ytr)
        ll = log_loss(yva, np.clip(pipe.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1])
        if ll < best_ll:
            best, best_ll = pipe, ll
    return best


def fit_mlp(Xtr, ytr, Xva, yva, seed=42):
    """Neural-network challenger (small MLP on scaled inputs), as the course asks for a documented challenger."""
    from sklearn.neural_network import MLPClassifier
    best, best_ll = None, np.inf
    for hidden in ((32,), (64, 32)):
        for alpha in (1e-3, 1e-1):
            pipe = make_pipeline(_imputer(), StandardScaler(),
                                 MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=400, random_state=seed))
            pipe.fit(Xtr, ytr)
            ll = log_loss(yva, np.clip(pipe.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1])
            if ll < best_ll:
                best, best_ll = pipe, ll
    return best


# ------------------------------------------------------------------ BG/NBD benchmark
def bgnbd_scores(fit_frame: pd.DataFrame, evals: dict[str, pd.DataFrame], horizon: int):
    """Probabilistic benchmark: BG/NBD ranked by expected purchases over the horizon.

    The model is unsupervised (no labels), and is fitted on *all* customers observed at one
    snapshot, active or not, because fitting only on recently active customers hides the
    dropout process and the optimiser degenerates. Choose a snapshot before the test period.
    Returns {name: score} where a higher score = more likely inactive (negative expected
    transactions over the horizon), or None if the optional dependency or the fit fails.
    """
    try:
        from lifetimes import BetaGeoFitter
    except Exception:
        return None

    def rfm(df):
        x = (df["n_purchase_days"] - 1).clip(lower=0).to_numpy(float)
        T = np.maximum(df["tenure_days"].to_numpy(float), 1.0) / 7.0
        tx = np.clip((df["tenure_days"] - df["recency_days"]).to_numpy(float), 0, None) / 7.0
        return x, np.minimum(tx, T), T

    x, tx, T = rfm(fit_frame)
    bgf = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pen in (0.0, 0.01, 0.1):
            try:
                bgf = BetaGeoFitter(penalizer_coef=pen).fit(x, tx, T)
                break
            except Exception:
                continue
        if bgf is None:
            print("[warn] BG/NBD did not converge; benchmark skipped")
            return None
        out = {}
        for name, df in evals.items():
            xe, txe, Te = rfm(df)
            exp = bgf.conditional_expected_number_of_purchases_up_to_time(horizon / 7.0, xe, txe, Te)
            score = -np.asarray(exp, dtype=float)
            bad = ~np.isfinite(score)
            if bad.any():   # hyp2f1 can overflow for very frequent customers; use a neutral (median) score
                print(f"[info] BG/NBD undefined for {int(bad.sum())}/{len(score)} rows in '{name}'; median score used")
                score[bad] = np.nanmedian(score[~bad]) if (~bad).any() else 0.0
            out[name] = score
    return out


# ------------------------------------------------------------------ calibration
class Calibrator:
    """Platt (sigmoid on logit) or isotonic mapping from raw to calibrated probability."""

    def __init__(self, kind: str):
        self.kind = kind
        self.model = None

    def fit(self, p, y):
        if self.kind == "platt":
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.kind == "platt":
            return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return self.model.predict(p)


def choose_calibrator(p_cal, y_cal, groups, seed=42) -> tuple[Calibrator, dict]:
    """Pick Platt vs isotonic by grouped (customer) cross-validated Brier score on the calibration period."""
    p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal)
    n_splits = int(min(5, len(np.unique(groups))))
    scores = {}
    for kind in ("platt", "isotonic"):
        briers = []
        for tr, te in GroupKFold(n_splits=n_splits).split(p_cal, y_cal, groups):
            if len(np.unique(y_cal[tr])) < 2:
                continue
            cal = Calibrator(kind).fit(p_cal[tr], y_cal[tr])
            briers.append(np.mean((cal.transform(p_cal[te]) - y_cal[te]) ** 2))
        scores[kind] = float(np.mean(briers)) if briers else np.inf
    kind = min(scores, key=scores.get)
    return Calibrator(kind).fit(p_cal, y_cal), {"cv_brier": scores, "chosen": kind}
