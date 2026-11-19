"""Statistical tests for the EDA: which inputs differ between customers who lapsed and customers who stayed?

One snapshot per customer (the last training snapshot) is used so the rows are independent; the monthly snapshots repeat the
same customers and would overstate significance. Each feature gets a Mann-Whitney U test (no normality assumption) with an
effect size (the chance a lapser scores higher than a stayer, 0.5 = no difference) and a Holm correction for the 21 tests.
UK against non-UK customers is an A/B-style comparison of lapse rates (two-proportion z-test and chi-square test).

    python -m inactivity stats
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .blocks import load_blocks
from .config import resolve
from .features import FEATURES


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def feature_tests(df: pd.DataFrame, features: list[str], label: str = "y_inactive") -> pd.DataFrame:
    rows = []
    y = df[label].to_numpy()
    for f in features:
        x = df[f].astype(float)
        a, b = x[y == 1].dropna(), x[y == 0].dropna()          # lapsed, stayed
        if len(a) < 5 or len(b) < 5 or x.nunique() < 2:
            rows.append({"feature": f, "n_lapsed": len(a), "n_stayed": len(b), "median_lapsed": np.nan, "median_stayed": np.nan,
                         "u": np.nan, "p": 1.0, "prob_lapser_higher": np.nan})
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": f, "n_lapsed": int(len(a)), "n_stayed": int(len(b)), "median_lapsed": float(a.median()),
                     "median_stayed": float(b.median()), "u": float(u), "p": float(p), "prob_lapser_higher": float(u / (len(a) * len(b)))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    out["effect"] = (out["prob_lapser_higher"] - 0.5).abs() * 2           # 0 = none, 1 = complete separation
    out["size"] = pd.cut(out["effect"], [-np.inf, 0.1, 0.3, 0.5, np.inf], labels=["negligible", "small", "medium", "large"])
    out["significant"] = out["p_holm"] < 0.05
    return out.sort_values("effect", ascending=False).reset_index(drop=True)


def uk_comparison(df: pd.DataFrame, label: str = "y_inactive") -> dict:
    """A/B-style comparison: lapse rate for UK against non-UK customers."""
    uk, non = df[df["is_uk"] == 1][label], df[df["is_uk"] == 0][label]
    n1, n2, p1, p2 = len(uk), len(non), uk.mean(), non.mean()
    pooled = (uk.sum() + non.sum()) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_z = float(2 * (1 - stats.norm.cdf(abs(z))))
    table = np.array([[uk.sum(), n1 - uk.sum()], [non.sum(), n2 - non.sum()]])
    chi2, p_chi, _, _ = stats.chi2_contingency(table, correction=False)
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    cramers_v = float(np.sqrt(chi2 / (n1 + n2)))
    return {"uk_n": int(n1), "non_uk_n": int(n2), "uk_lapse_rate": float(p1), "non_uk_lapse_rate": float(p2), "difference": float(p1 - p2),
            "ci95": [float(p1 - p2 - 1.96 * se_diff), float(p1 - p2 + 1.96 * se_diff)], "z": float(z), "p_value": p_z,
            "chi2": float(chi2), "chi2_p": float(p_chi), "cramers_v": cramers_v}


def run_stats(cfg: dict, tx: pd.DataFrame, art_dir: Path | None = None, reports_dir: Path | None = None) -> dict:
    rep = reports_dir or resolve(cfg["paths"]["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    bundle, parts, blocks = load_blocks(cfg, tx, art_dir)
    last = max(blocks["train"])
    one = parts["train"][parts["train"]["snapshot_date"] == last]
    one = one.drop_duplicates("customer_id")
    table = feature_tests(one, bundle["features"])
    ab = uk_comparison(one)
    result = {"snapshot": str(last.date()), "customers": int(len(one)), "lapse_rate": float(one["y_inactive"].mean()),
              "tests": int(len(table)), "significant_after_holm": int(table["significant"].sum()),
              "features": table.assign(size=table["size"].astype(str)).to_dict("records"), "uk_vs_non_uk": ab}
    (rep / "hypothesis_tests.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf8")
    table.to_csv(rep / "hypothesis_tests.csv", index=False)
    return result


def summary(res: dict, top: int = 8) -> str:
    lines = [f"Mann-Whitney U tests, lapsed against stayed: snapshot {res['snapshot']}, {res['customers']:,} independent customers, "
             f"{res['lapse_rate']:.0%} lapsed. {res['significant_after_holm']} of {res['tests']} inputs differ after Holm correction."]
    for r in res["features"][:top]:
        lines.append(f"  {r['feature']:<26} P(lapser higher) {r['prob_lapser_higher']:.2f}  effect {r['size']:<10} Holm p {r['p_holm']:.1e}")
    ab = res["uk_vs_non_uk"]
    lines.append(f"UK vs non-UK lapse rate: {ab['uk_lapse_rate']:.0%} vs {ab['non_uk_lapse_rate']:.0%} (difference {ab['difference']:+.1%}, "
                 f"95% CI {ab['ci95'][0]:+.1%} to {ab['ci95'][1]:+.1%}, p = {ab['p_value']:.3f}, Cramer's V {ab['cramers_v']:.2f})")
    return "\n".join(lines)
