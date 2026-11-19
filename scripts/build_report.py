"""Build the final report (Word) from the pipeline outputs.

    python scripts/make_figures.py && python scripts/build_report.py [output.docx]
"""
import json
import re
import sys
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).parent))
from mdtodocx import NAVY, add_markdown, new_document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
m = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf8"))
t = json.loads((ROOT / "artifacts" / "trust.json").read_text(encoding="utf8"))
q = json.loads((ROOT / "reports" / "data_quality.json").read_text(encoding="utf8"))
_dr = json.loads((ROOT / "reports" / "drift_check.json").read_text(encoding="utf8"))
drift_score_psi = f"{_dr['score_psi']:.2f}"
drift_slice_cur = f"{_dr['slice']['current_auc']:.2f}"
drift_slice_base = f"{_dr['slice']['baseline_auc']:.2f}"
M, CI, SZ, SP = m["metrics"], m["ci"], m["sizes"], m["splits"]
vm, ch, th = m["val_metrics"], m["customer_holdout"], m["thresholds"]
VC = m["val_comparison"]
seg = {(s["dimension"], s["group"]): s for s in t["segments"]}
sens = m["sensitivity"]
gains = t["gains"]


def c(model, key="roc_auc"):
    lo, hi = CI[f"{model}|{key}"]
    return f"{M[model][key]:.3f} [{lo:.3f}, {hi:.3f}]"


def dd(pair, key="roc_auc"):
    lo, hi = CI[f"{pair}|{key}"]
    return f"{lo:+.3f} to {hi:+.3f}"


def n(x):
    return f"{x:,}"


def pc(x, d=0):
    return f"{x * 100:.{d}f}%"


def lift(k):
    return f"{M[k]['lift_top']:.2f}"


per = ", ".join(f"{s['snapshot']} {s['roc_auc']:.3f}" for s in m["per_snapshot"])
sens_rows = "\n".join(f"| {k} | {v['sizes']['prevalence']:.3f} | {v['metrics']['roc_auc']:.3f} | {v['metrics']['pr_auc']:.3f} |" for k, v in sens.items())
NICE = {"n_purchase_days": "number of purchase days", "median_interval_days": "usual days between orders", "rev_90_log": "spend in the last 90 days",
        "interval_cv": "irregularity of order timing", "rev_180_log": "spend in the last 180 days", "recency_to_interval": "time since last order vs usual",
        "n_products_180": "distinct products bought", "recency_days": "days since last order", "tenure_days": "days since first order"}
perm = [(NICE.get(r["feature"], r["feature"].replace("_", " ")), r["auc_drop"]) for r in t["importance"][:5]]
tech = (DOCS / "technique_justification.md").read_text(encoding="utf8")
tech_used = "{widths=3.4,5.0,2.5,5.7}" + chr(10) + tech[tech.index("## 2. Techniques used"):tech.index("## 3. Techniques deliberately")]
tech_not = "{widths=3.4,10.6,3.0}" + chr(10) + tech[tech.index("## 3. Techniques deliberately"):tech.index("## 4. Where the project departs")]
tech_dep = tech[tech.index("## 4. Where the project departs"):tech.index("## 5. What this adds up to")]
tech_end = tech[tech.index("## 5. What this adds up to"):tech.index("## References")]
def shift(md):
    return re.sub(r"^## \d+\. ", "### ", md, flags=re.M)


d = new_document(landscape=False)

# ------------------------------------------------------------------ title page
for _ in range(5):
    d.add_paragraph()
p = d.add_paragraph()
r = p.add_run("DSA 8401 Applied Machine Learning | Capstone project report")
r.font.size = Pt(12)
r.font.color.rgb = RGBColor(0x5D, 0x67, 0x68)
p = d.add_paragraph()
r = p.add_run("Which customers are about to go quiet, and how far can we trust the answer?")
r.font.size = Pt(26)
r.bold = True
r.font.color.rgb = NAVY
p = d.add_paragraph()
r = p.add_run("Explainable customer inactivity prediction for a wholesale retailer, from raw data to a deployed scoring service")
r.font.size = Pt(14)
for _ in range(4):
    d.add_paragraph()
for line in ("Eric Muriuki Kiama (Registration number 120890)", "MSc Data Science and Analytics, Strathmore University",
             "Lecturers: Prof. Javier Serrano and Emmanuel Bundi", "October 2026"):
    p = d.add_paragraph()
    p.add_run(line).font.size = Pt(12)
d.add_paragraph()
p = d.add_paragraph()
r = p.add_run("Code, tests, trained model and web UI: the `inactivity-capstone` repository (start with README.md).")
r.font.size = Pt(10)
r.font.color.rgb = RGBColor(0x5D, 0x67, 0x68)
d.add_page_break()

# ------------------------------------------------------------------ body
body = f"""
# Summary

> **Problem.** A wholesale customer never cancels; they just stop ordering. The common fix, a fixed rule such as "no order in six months", fires late and mislabels customers with long order cycles. **This project** predicts, for every customer who ordered in the last 180 days, the probability of placing no order in the next 90 days, explains the score in plain language, and serves it through a tested API and a web interface.

**Data.** The public UCI Online Retail II dataset (CC BY 4.0): {n(q['rows_raw'])} raw transaction lines from a UK online retailer between December 2009 and December 2011. After cleaning, {n(q['rows_clean'])} lines for {n(q['customers'])} customers remain. {pc(q['missing_customer_id_share'], 1)} of raw rows had no customer ID.

**Approach.** Month-end snapshots with 21 features built only from data available at each date; strictly chronological train, validation, calibration and test periods with a gap so no label depends on the future; a ladder of models from a simple rule up to a tuned gradient-boosted model and a neural-network challenger; probability calibration; bootstrap confidence intervals that resample whole customers.

**What I found.**
- The served LightGBM model reaches ROC-AUC {M['lgbm_cal']['roc_auc']:.3f} on a later period it never saw, against {M['rule_recency']['roc_auc']:.3f} for the "longest since last order" rule. That gain ({dd('lgbm_cal-rule_recency')}) is clear.
- It is **not clearly better** than a logistic regression, a random forest or the classic BG/NBD model, and a tuned neural network did not beat it. Most of the achievable signal is simple.
- Following the ranking and calling the top 30% of customers reaches {pc(gains['lgbm_cal'][30])} of those who go quiet, against {pc(gains['rule_recency'][30])} with the rule and 30% at random. Put per 1,000 customers who go quiet: the model reaches about {(gains['lgbm_cal'][30])*1000:.0f} of them, the rule about {(gains['rule_recency'][30])*1000:.0f}, random calling 300, so the model finds roughly {(gains['lgbm_cal'][30]-gains['rule_recency'][30])*1000:.0f} more than the rule from the same number of calls.
- The probabilities themselves run high because the calibration period had a different season's inactivity rate ({pc(SZ['cal']['prevalence'])} against {pc(SZ['test']['prevalence'])}). The ranking held up on the test period, with weaker results for small buyers (below); the percentage is best read as a priority score.
- The model ranks regular, high-spend customers well (ROC-AUC about 0.75) and small or one-off buyers poorly (about 0.63 to 0.65).

**System.** A FastAPI service with a calibrated probability, risk band and reason codes for each customer; a seven-page web interface (overview, interactive data exploration, model evidence, a retention worklist, customer pages with a live what-if, a try-it page, and an about page, each in light, dark or system theme); a Dockerfile and compose file; 63 automated tests. The Docker image has been built and run once on the development machine.

\\pagebreak

# 1. Problem framing and relevance

## 1.1 The business problem

A wholesaler's customers are on no contract. They order when they need stock, at intervals that differ by customer and product. There is no cancellation event, so "churn" has to be inferred from silence. By the time silence is obvious the relationship is often gone, and the sales team has no principled way to decide whom to call first with limited weekly capacity.

## 1.2 What the system has to do

1. Identify, among customers who are currently active, those most likely to place **no order in the next 90 days**.
2. Say **why**, in terms a salesperson can act on, without implying causes the model cannot know.
3. Be **honest about its own reliability**: evaluated on a later period than it was trained on, compared with simple alternatives, with uncertainty and known weaknesses stated.
4. Be **deployable**: a scripted, reproducible pipeline and a containerised API that a client can send data to and get predictions back.

## 1.3 Definitions

- **Scoring date (snapshot):** a month-end. Everything the model sees is dated on or before it.
- **Eligible customer:** one with at least one purchase in the previous 180 days. This removes customers who have already lapsed, which would otherwise inflate the score because "long since last order" would trivially predict "no order next".
- **Target:** 1 if the customer places no qualifying order in the 90 days after the scoring date. A qualifying order is a non-cancelled invoice with positive quantity and price for a real product code.
- **Horizon:** 90 days. The data span about 24 months, and every snapshot needs a fully observed outcome window plus a gap between periods equal to that window. A 180-day horizon leaves too few snapshots to train, validate, calibrate and test, so 90 days is the longest that works; 60 and 120 days are reported as sensitivity checks.

## 1.4 Place in my wider work

This is the first stage of my MSc thesis, which concerns two linked risks for a distributor: customers going inactive and customers paying late on trade credit. The capstone isolates the inactivity half on public data, so the whole engineering chain could be built, evaluated and deployed without waiting for company data approvals.

# 2. Data

## 2.1 Source

UCI Machine Learning Repository, Online Retail II (CC BY 4.0). It records every transaction of a UK-registered online retailer between 1 December 2009 and 9 December 2011; many customers are wholesalers. It is downloaded at run time and not redistributed in the repository.

## 2.2 The data is raw, and cleaning is deliberate

{{widths=5.2,3.0,8.8}}
| Issue found | Count | Treatment and reason |
|---|---|---|
| Raw transaction lines | {n(q['rows_raw'])} | Starting point |
| Exact duplicate lines | {n(q['duplicate_rows_dropped'])} | Dropped; duplicates inflate order counts |
| Lines with no customer ID | {n(q['missing_customer_id_rows'])} ({pc(q['missing_customer_id_share'], 1)}) | Dropped; no customer to score. This can bias the population towards registered and repeat buyers |
| Non-product lines (postage, fees, adjustments) | {n(q['non_product_rows_dropped'])} | Dropped by a product-code rule; they are not purchases |
| Cancellation and return lines | {n(q['cancellation_return_lines_kept'])} | Kept as returns (a feature); never counted as purchases |
| Other anomalous lines (zero price, negative quantity without a cancellation prefix) | {n(q['other_anomalous_lines_dropped'])} | Dropped |
| Extreme quantities (above the 99.9th percentile, {q['quantity_p99_9']:.0f} units) | {n(q['extreme_quantity_lines'])} | **Kept**; in wholesale, large orders can be the signal, so spend features are log-scaled instead of trimmed |
| **Clean lines / customers** | **{n(q['rows_clean'])} / {n(q['customers'])}** | {q['date_min']} to {q['date_max']} |

# 3. Approach

## 3.1 Snapshots and features

For each month-end from February 2010, the pipeline builds one row per eligible customer from transactions dated on or before that day, then looks forward 90 days to set the label. The same feature code serves training and the API, so there is no training-serving skew; a test asserts that features computed from raw lines match the batch path.

![Features use only the past; the label uses only the future.](figures/fig_snapshot.png){{width=15.5}}

{{widths=3.2,8.0,5.8}}
| Group | Features (21 in total) | Why they matter |
|---|---|---|
| Recency and tenure | days since last order, days since first order, number of purchase days | How recently and how long the customer has bought |
| Order frequency | orders in the last 30, 90, 180 and 365 days; orders in the previous 90-day window; change in count | Slowing down is the classic early warning |
| Spend | spend over 90, 180 and 365 days (log scale); change in spend; average order value; items per order | Value and direction of the relationship |
| Rhythm | usual days between orders, irregularity of order timing, time since last order relative to the usual interval | Customers have different natural cycles, which a single fixed threshold ignores |
| Breadth and behaviour | distinct products, share of value returned, UK flag | Product range, returns and country |

## 3.2 Validation: strictly chronological, with an embargo

Random splits would let future behaviour of the same customers train the model. Instead, snapshots are assigned to four consecutive periods: training, validation (model selection), calibration (probability mapping only) and an out-of-time test whose dates were fixed in advance and on which nothing was chosen. A snapshot is **dropped** unless its 90-day outcome window ends before the next period begins, so no label used earlier depends on behaviour from a later period. {len(SP['dropped_by_embargo'])} of 19 monthly snapshots are dropped for this reason, which is the honest price of leakage-free evaluation on short history.

![Monthly snapshots by period; dashed circles are dropped because their outcome was not yet known.](figures/fig_timeline.png){{width=16}}

| Period | Snapshots | Rows | Customers | Share who went quiet |
|---|---|---|---|---|
| Training | {len(SP['blocks']['train'])} | {n(SZ['train']['rows'])} | {n(SZ['train']['customers'])} | {pc(SZ['train']['prevalence'])} |
| Validation | {len(SP['blocks']['val'])} | {n(SZ['val']['rows'])} | {n(SZ['val']['customers'])} | {pc(SZ['val']['prevalence'])} |
| Calibration | {len(SP['blocks']['cal'])} | {n(SZ['cal']['rows'])} | {n(SZ['cal']['customers'])} | {pc(SZ['cal']['prevalence'])} |
| Test | {len(SP['blocks']['test'])} | {n(SZ['test']['rows'])} | {n(SZ['test']['customers'])} | {pc(SZ['test']['prevalence'])} |

Because the same customer appears in several months, rows are correlated. Every confidence interval therefore resamples **whole customers**, and a separate check retrains with every test-period customer removed (section 5.4).

## 3.3 Models and tuning

Each model was trained on the training period, compared on the validation period (all of them, in the table in section 5.1), and then scored on the test period. The test period's dates were fixed in advance and nothing was chosen on it, but its results were visible during development and some models (Optuna tuning, the neural-network challenger) were added after earlier runs, so it is a clean test of the chosen design rather than a single-use hold-out:

- **Baselines:** class prevalence, and the rule "the longer since the last order, the riskier".
- **Decision tree (depth 4)**, **regularised logistic regression**, **random forest**, and a **neural-network challenger** (a small multilayer perceptron), forming the model ladder.
- **BG/NBD**, the classic probabilistic model of non-contractual purchasing, fitted without labels on all customers at the last calibration-period snapshot and ranked by expected purchases.
- **LightGBM**, the served model, tuned with a 60-trial Optuna TPE search over six log-scaled hyperparameters, choosing the number of trees by early stopping on validation log loss.

## 3.4 Calibration and risk bands

The API returns probabilities, so rank quality is not enough. A Platt or isotonic mapping is fitted on the separate calibration period, chosen by customer-grouped cross-validated Brier score (Platt won). Risk bands are capacity-based, since the dataset has no cost data for a cost-optimal threshold: the top 10% of calibration-period scores are High (probability at least {th['high']:.2f}) and the next 20% Medium (at least {th['medium']:.2f}).

## 3.5 Evaluation

ROC-AUC is the primary metric because the target is close to balanced ({pc(SZ['test']['prevalence'])} in test); PR-AUC, top-decile lift, Brier score, calibration slope and intercept, and expected calibration error are also reported. Differences between models are judged on **paired** customer-clustered bootstrap intervals, not on point estimates.

## 3.6 Explanations

Per-customer reasons come from TreeSHAP contributions. Only inputs that raise a customer's own predicted risk are listed, and the wording is direction-neutral ("Days since last purchase: 161"), so a reason can never contradict the model or imply a cause. Global importance uses permutation importance on the test period, which is safer than training-time gain importance.

# 4. The system

![End-to-end architecture, from raw download to the served model.](figures/fig_architecture.png){{width=16.5}}

{{widths=4.2,12.8}}
| Component | What it does |
|---|---|
| Pipeline (`python -m inactivity all`) | Downloads and cleans the data (schema and file-hash checks), builds snapshots, trains and tunes, calibrates, evaluates, writes the model bundle, reports and UI data. `predict` scores every customer at any date without retraining. Scripted, deterministic and runnable from scratch with pinned versions (`requirements.lock.txt`) |
| Model bundle | One versioned file: the model, calibrator, feature list, risk thresholds, test metrics and the library versions it was built with (the API warns if the serving environment differs) |
| FastAPI service | `POST /score` (features in), `POST /score/transactions` (raw order lines in, features computed by the training code), `GET /worklist`, `GET /trust`, `GET /eda`, `GET /health`, `GET /model-info`. A customer who has not bought in 180 days returns "Not eligible" instead of a misleading low-risk score |
| Web interface | Served by the API, seven pages: an **overview** with twelve KPIs, **explore data** (interactive EDA: activity over time, value concentration, distributions, lapse rate by month, a per-feature explorer and a correlation matrix), **model evidence** (the trust evidence in plain language), a **retention worklist** (filters, scatter, "top N calls" back-test, CSV export), a **customer** page (reasons, timeline, live what-if), **try it** and **about**. See the screenshots below |
| Container | Dockerfile and compose file with only serving dependencies. **Built and run** with Docker Desktop (Linux containers): the 656 MB image builds, the container reports healthy, serves the interface and API, and the smoke test scores a customer through it |
| Tests | 63 automated tests: cleaning rules, no-leakage checks, split embargo, metric sanity, API behaviour, and training-serving parity |
| Tracking and monitoring | MLflow records each run. A drift alert (`python -m inactivity drift`) compares a window with a stored reference and returns an exit code (0 ok, 1 warning, 2 alert) |

![The web interface, light theme: the overview page with its twelve KPIs.](figures/ui_overview.png){{width=16}}

![The explore page: interactive EDA with measure and date-range controls (top of page shown).](figures/ui_explore.png){{width=16}}

![A customer page: reasons, comparison with other customers, timeline, and a live what-if that re-scores through the API.](figures/ui_customer.png){{width=13}}

## 4.1 The drift alert

A system that cannot tell you when it is wrong is not production-ready, so the service ships with a drift gate. It compares a window of scored customers with a **stored reference** (bin edges and baselines saved at training time) and checks three things: the distribution of each input (Population Stability Index), the distribution of the predicted risk, and one **domain slice metric**, the ROC-AUC for customers with a single purchase day, which is the model's weakest group. It writes a JSON decision and an Evidently HTML report, and returns an exit code (0 ok, 1 warning, 2 alert) that a scheduler can act on. The thresholds, the owner and the response for each status are in a one-page runbook (`docs/drift_runbook.md`).

On the out-of-time test window the gate returns **ALERT**: 8 of 21 inputs are in major shift, mainly tenure-type features that grow with calendar time because the data starts in December 2009. The score distribution (PSI {drift_score_psi}) and the slice metric (AUC {drift_slice_cur} against a baseline of {drift_slice_base}) stay within their thresholds, and an unshifted control returns 0. So the alert fires for a known, documented reason, which is what the runbook is for.

# 5. Results

All figures in this section are measured on the out-of-time test period: {len(SP['blocks']['test'])} monthly snapshots (June to August 2011), {n(SZ['test']['rows'])} customer-snapshots, {n(SZ['test']['customers'])} customers.

## 5.1 Does the model beat the alternatives?

![ROC-AUC on the test period with 95% intervals (resampling whole customers).](figures/fig_models.png){{width=15.5}}

{{widths=5.4,4.5,4.5,2.6}}
| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Top-decile lift |
|---|---|---|---|
| Prevalence only | 0.500 | {M['prevalence']['pr_auc']:.3f} | 1.00 |
| Fixed recency rule | {c('rule_recency')} | {c('rule_recency', 'pr_auc')} | {lift('rule_recency')} |
| Decision tree (depth 4) | {c('tree_depth4')} | {c('tree_depth4', 'pr_auc')} | {lift('tree_depth4')} |
| Logistic regression (calibrated) | {c('logreg_cal')} | {c('logreg_cal', 'pr_auc')} | {lift('logreg_cal')} |
| Neural network (MLP challenger) | {c('mlp_challenger')} | {c('mlp_challenger', 'pr_auc')} | {lift('mlp_challenger')} |
| Random forest | {c('random_forest')} | {c('random_forest', 'pr_auc')} | {lift('random_forest')} |
| Support vector machine (RBF kernel) | {c('svm_rbf')} | {c('svm_rbf', 'pr_auc')} | {lift('svm_rbf')} |
| BG/NBD benchmark | {c('bgnbd')} | {c('bgnbd', 'pr_auc')} | {lift('bgnbd')} |
| **LightGBM (calibrated), served** | **{c('lgbm_cal')}** | **{c('lgbm_cal', 'pr_auc')}** | **{lift('lgbm_cal')}** |

**How LightGBM came to be the served model.** It was chosen by design, not because it won on validation: it handles missing values natively, captures interactions, and has exact TreeSHAP for the reason codes. Every model was also scored on the validation period (earlier than the test period), and there the top three were within {VC['logreg_cal']['roc_auc'] - VC['random_forest']['roc_auc']:.3f} ROC-AUC of each other, with logistic regression marginally ahead on ROC-AUC and LightGBM ahead on top-decile lift:

{{widths=5.4,4.5,4.5,4.5}}
| Model | Validation ROC-AUC | Validation PR-AUC | Validation top-decile lift |
|---|---|---|---|
| Fixed recency rule | {VC['rule_recency']['roc_auc']:.3f} | {VC['rule_recency']['pr_auc']:.3f} | {VC['rule_recency']['lift_top']:.2f} |
| Decision tree (depth 4) | {VC['tree_depth4']['roc_auc']:.3f} | {VC['tree_depth4']['pr_auc']:.3f} | {VC['tree_depth4']['lift_top']:.2f} |
| Logistic regression | {VC['logreg_cal']['roc_auc']:.3f} | {VC['logreg_cal']['pr_auc']:.3f} | {VC['logreg_cal']['lift_top']:.2f} |
| Neural network (MLP) | {VC['mlp_challenger']['roc_auc']:.3f} | {VC['mlp_challenger']['pr_auc']:.3f} | {VC['mlp_challenger']['lift_top']:.2f} |
| Random forest | {VC['random_forest']['roc_auc']:.3f} | {VC['random_forest']['pr_auc']:.3f} | {VC['random_forest']['lift_top']:.2f} |
| Support vector machine (RBF) | {VC['svm_rbf']['roc_auc']:.3f} | {VC['svm_rbf']['pr_auc']:.3f} | {VC['svm_rbf']['lift_top']:.2f} |
| BG/NBD benchmark | {VC['bgnbd']['roc_auc']:.3f} | {VC['bgnbd']['pr_auc']:.3f} | {VC['bgnbd']['lift_top']:.2f} |
| LightGBM, served | {VC['lgbm_cal']['roc_auc']:.3f} | {VC['lgbm_cal']['pr_auc']:.3f} | {VC['lgbm_cal']['lift_top']:.2f} |

Validation and test agree on the main message: everything beats the recency rule by a wide margin, and the better models are close to each other. A reader who prefers the simplest model near the top could reasonably serve logistic regression instead; LightGBM was kept for the explanations and because it was ahead on the test period, though the difference is small.

- **Clearly better than the rule** a business would use today: paired ROC-AUC gain {dd('lgbm_cal-rule_recency')}. Also clearly better than a single tree ({dd('lgbm_cal-tree_depth4')}).
- **Not clearly better than the other sensible models.** Against logistic regression the ROC-AUC gain is {dd('lgbm_cal-logreg_cal')} while the PR-AUC and lift gaps span zero; against the random forest it is {dd('lgbm_cal-random_forest')} with a PR-AUC gap that spans zero; against BG/NBD the gap spans zero.
- **The tuned neural network did not beat LightGBM** (ROC-AUC gap {dd('lgbm_cal-mlp_challenger')}, PR-AUC gap {dd('lgbm_cal-mlp_challenger', 'pr_auc')}). This is the evidence for not using deep learning here. **The RBF support vector machine was weaker still** (ROC-AUC gap {dd('lgbm_cal-svm_rbf')}); it also gives no probabilities or reason codes.

## 5.2 If I can only call some customers, how many lapsing ones do I reach?

![Share of customers who go quiet that are reached, contacting highest risk first.](figures/fig_gains.png){{width=10.5}}

| Contact the top | Model reaches | Rule reaches | Random reaches |
|---|---|---|---|
| 10% | {pc(gains['lgbm_cal'][10])} | {pc(gains['rule_recency'][10])} | 10% |
| 20% | {pc(gains['lgbm_cal'][20])} | {pc(gains['rule_recency'][20])} | 20% |
| 30% | {pc(gains['lgbm_cal'][30])} | {pc(gains['rule_recency'][30])} | 30% |
| 50% | {pc(gains['lgbm_cal'][50])} | {pc(gains['rule_recency'][50])} | 50% |

**What this means in practice.** Per 1,000 customers who will go quiet, a call budget covering the top 30% of the ranked list reaches about {(gains['lgbm_cal'][30])*1000:.0f} of them with the model, about {(gains['rule_recency'][30])*1000:.0f} with the recency rule and 300 by calling at random. The model therefore finds roughly {(gains['lgbm_cal'][30]-gains['rule_recency'][30])*1000:.0f} more of the customers worth contacting than the rule would, from the same number of calls. Without margin and contact-cost data this cannot be turned into money; it is a measure of how much better the call list is, not of revenue retained.

## 5.3 Are the percentages honest?

![Left: observed against predicted rates. Right: the inactivity rate differs across periods.](figures/fig_calibration.png){{width=16}}

No. The inactivity rate was {pc(SZ['cal']['prevalence'])} in the calibration period but {pc(SZ['test']['prevalence'])} in the test period, so a mapping fitted on one overstates risk on the other (calibration intercept {M['lgbm_cal']['cal_intercept']:.2f}; ECE {M['lgbm_cal']['ece']:.3f} against {M['lgbm_raw']['ece']:.3f} for the uncalibrated model). The raw model sat slightly above the diagonal, and calibration overcorrected it. **The ranking is unaffected.** The fix, before anyone relies on a number, is to recalibrate on recent data; until then the percentage is a priority score and the risk bands inherit the same shift.

## 5.4 Stability and honesty checks

- **Across test months**, ROC-AUC is stable: {per}.
- **Optimism check.** LightGBM scored {vm['roc_auc']:.3f} ROC-AUC on the validation period used to tune it and {M['lgbm_raw']['roc_auc']:.3f} on the test period. The test score is higher, so there is no sign of overfitting the validation set; it also shows that skill moves by several points between periods, so small differences should not be over-read.
- **Unseen-customer check (group leakage).** Retraining with every test-period customer removed from training and validation (only {n(ch['train_rows'])} training rows left) still gives {ch['roc_auc']:.3f} ROC-AUC on those customers, within {M['lgbm_raw']['roc_auc'] - ch['roc_auc']:.3f} of the served model. Part of that gap reflects less data, not only unfamiliar customers.

## 5.5 Does it work equally well for every kind of customer?

![ROC-AUC within customer groups on the test period (brass: below 0.70).](figures/fig_segments.png){{width=14}}

No. Ranking is weakest for small and one-off buyers (ROC-AUC {seg[('Order history', 'One purchase day')]['roc_auc']:.2f} for one-purchase-day customers and {seg[('Spend tier (365 days)', 'Lower third')]['roc_auc']:.2f} for the lowest spend third) and strongest for regular, high-spend customers ({seg[('Order history', '5+ purchase days')]['roc_auc']:.2f} and {seg[('Spend tier (365 days)', 'Upper third')]['roc_auc']:.2f}). Predicted risk is above the actual rate in every group. For small and one-off buyers the model mostly separates them from regular customers rather than ranking within the group.

## 5.5b Customer segments (unsupervised view)

K-Means on eight behavioural inputs (standardised, fitted on the training period only, k chosen by silhouette; PCA is used only to draw the picture) splits customers into three types. The segments are descriptive: they do not feed the model.

![Left: the three K-Means segments in two PCA dimensions. Right: how often each segment lapses, how often the model predicted it would, and how well the model ranks within it.](figures/fig_clusters.png){{width=16}}

{{widths=3.6,2.4,2.4,2.4,5.2}}
| Segment | Share of customers | Went quiet | Model predicted | Model ROC-AUC within segment |
|---|---|---|---|---|
| Core regulars | 4% | 2% | 10% | 0.85 |
| Occasional buyers | 52% | 30% | 40% | 0.73 |
| Fading | 44% | 64% | 74% | 0.63 |

The picture agrees with the segment results in section 5.5. The model is strongest for the small group of core regulars and for occasional buyers, and weakest for the large "fading" group of customers with one or two orders and a long silence, where it also overstates risk. The segments make the weak spot concrete: it is not a few odd customers but nearly half of the active base.

## 5.5c Statistical tests, feature selection and anomalies

**Tests.** On one snapshot per customer (2,652 customers, 50% lapsed), Mann-Whitney U tests with Holm correction find that 17 of 21 inputs differ between customers who lapsed and those who stayed. Spend, order count and purchase days have the largest effects (a lapser scores higher than a stayer about 27% of the time, a medium effect). Revenue trend, recency-to-interval, interval variability and the UK flag show no effect. UK and non-UK lapse rates are 50% and 53% (difference -3.2%, 95% CI -10.7% to +4.2%, p = 0.40, Cramer's V 0.02): no evidence that country matters.

**Feature selection.** Filter (mutual information), wrapper (RFE) and embedded (L1, LightGBM gain) selection were compared at 5, 8 and 12 inputs, chosen on training and judged on validation with the served hyperparameters. Five inputs reach 0.720 to 0.726 validation ROC-AUC against 0.723 for all 21; the differences are within noise. All 21 inputs are kept because dropping them gains nothing, but a 5-input model would serve about as well.

**Anomalies.** An Isolation Forest fitted on the training period flags 2.0% of training rows and 12.6% of test rows: test-period customers are more often large accounts than training-period customers, a row-level sign of drift. Flagged customers lapse less (16% against 48%), are predicted at 25% against 58%, and are ranked better by the model (ROC-AUC 0.88 against 0.74), so the model is not failing on them. The top anomalies are very large wholesale accounts.

## 5.6 What does the model pay attention to?

Permutation importance on the test period (ROC-AUC lost when one input is shuffled) is small for every input because the inputs are correlated: {", ".join(f"{k} ({v:.3f})" for k, v in perm)}. Customers' own order rhythm and purchase history matter most; "days since last order" alone, the old rule, is only part of the story. These are associations, not causes.

## 5.7 Sensitivity

{{widths=7.5,3.2,3.0,3.0}}
| Variant (LightGBM, 20 search trials) | Inactivity rate | ROC-AUC | PR-AUC |
|---|---|---|---|
{sens_rows}

The 60, 90 and 120-day horizons give similar skill. The segment-aware definition (inactive only if the gap exceeds three times the customer's own median order interval) looks much easier, but that label is built partly from recency and the order interval, which are also model inputs. The apparent gain is circular and is not better prediction.

# 6. Technique justification

Every plausible technique was either trained and compared on the same out-of-time test period, or excluded for a reason tied to this data and to the course. Slide references use the form W4 s26 (Week 4 lecture, slide 26) and S8 s51 (Session 8, slide 51).

{shift(tech_used)}

{shift(tech_not)}

{shift(tech_dep)}

{shift(tech_end)}

# 7. Limitations and next steps

## 7.1 Limitations

- **Short, left-censored history.** About 24 months of data starting in December 2009: training uses early 2010 and testing mid 2011. Tenure-type features grow mechanically with calendar time, and the drift report flags them.
- **Overlapping monthly snapshots** give correlated rows. The clustered bootstrap handles this, but the effective sample size is smaller than the row count.
- **One UK online retailer.** Results do not transfer to other businesses without re-testing.
- **Probabilities are not well calibrated across seasons**, and the risk bands inherit that.
- **The test period is not a single-use hold-out.** Its dates were fixed in advance and nothing was chosen on it, but its results were visible during development.
- **Weak ranking for small and one-off buyers.**
- **No cost data**, so there is no cost-optimal threshold and no estimate of money saved. The "customers I can call" figures are a back-test of ranking quality, not a promise of retained revenue.
- **Customers without an ID** ({pc(q['missing_customer_id_share'], 1)} of raw rows) are dropped, which can bias the population.
- **Reasons are associations.** Nothing here shows that contacting a flagged customer changes the outcome; that would need a controlled pilot.
- **Docker was verified once, on one machine** (Docker Desktop, Linux containers, 4 October 2026): build, health check and smoke test pass. It has not been tried on another operating system or in CI.

## 7.2 What I would do next

1. Recalibrate on a rolling recent window and monitor calibration alongside ranking.
2. Cap history windows and drop tenure-type features, or add monotonic constraints so risk cannot fall as time since the last order grows.
3. Add margin and contact-cost data to set a cost-based threshold, and rank by expected value rather than probability alone.
4. Build a separate approach for small and one-off buyers, where the current model is weak.
5. Pilot with a holdout group of customers to measure whether outreach actually retains them.
6. Combine with trade-credit delinquency risk, as in my thesis, for a dual-risk view.

# 8. Conclusion

The project delivers what the brief asks: a reproducible pipeline, an evaluated model, a justified choice of techniques, and a deployed scoring service with an interface. The more useful result is the honest one. A gradient-boosted model clearly beats the rule a business would use today, and a simple ranking of customers reaches somewhat more lapsing customers than the rule when capacity is limited (about {(gains['lgbm_cal'][30]-gains['rule_recency'][30])*1000:.0f} more per 1,000 who go quiet at a 30% call budget). But it barely beats simpler models, its probabilities need recalibrating, and it is weakest exactly where small customers are. Framing the problem, preventing leakage and evaluating against honest baselines mattered more than the choice of algorithm.

# Appendix A. Reproducing the results

```
pip install -r requirements.txt
python -m inactivity all          # download, clean, train, evaluate, write artifacts and reports
python -m pytest -q               # 63 tests, no download needed
python -m inactivity serve        # API and web UI on http://localhost:8000
python scripts/make_figures.py    # redraw the report figures
python scripts/build_report.py    # rebuild this report
```

If the download is blocked, place `online_retail_II.xlsx` (from the UCI archive) in `data/raw/` and rerun.

# Appendix B. Repository map

{{widths=4.5,12.5}}
| Path | Contents |
|---|---|
| `src/inactivity/` | pipeline, features, splits, models, evaluation, explanations, monitoring, API and web UI |
| `tests/` | 63 automated tests |
| `artifacts/` | trained model bundle and the data files behind the UI |
| `reports/` | metrics, plots, drift and data-quality outputs from the real-data run |
| `docs/` | design decisions, technique justification, figures, this report |
| `Dockerfile`, `docker-compose.yml` | container build and run |
| `AI_USE.md` | disclosure of AI assistance |

# Appendix C. Use of AI assistance

Claude Code (Anthropic) was used to draft and test the code, documentation and this report. The detailed disclosure, including what I directed, how the work was checked and its limits, is in `AI_USE.md` in the repository.

# References

- Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). "Counting your customers" the easy way: An alternative to the Pareto/NBD model. Marketing Science, 24(2), 275-284.
- Schmittlein, D. C., Morrison, D. G., & Colombo, R. (1987). Counting your customers: Who are they and what will they do next? Management Science, 33(1), 1-24.
- UCI Machine Learning Repository. Online Retail II dataset (CC BY 4.0). https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
- Dietterich, T. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS.
- Huyen, C. (2022). Designing Machine Learning Systems. O'Reilly.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML.
- Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. KDD.
- Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. NeurIPS.
- DSA 8401 Applied Machine Learning lecture notes, Weeks 1 to 6 and Sessions 7 to 9 (Strathmore University).
"""

add_markdown(d, body, DOCS, table_font=9.5)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS / "Final_Report.docx"
d.save(out)
print("saved", out)
