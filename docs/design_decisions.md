# Design decisions and technique justification

The full, course-referenced technique justification (what was used, what was tested and rejected, and where the project departs from course advice) is in [`technique_justification.md`](technique_justification.md). This file records the framing and serving decisions.

This document records why each technique was chosen, why plausible alternatives were left out, and which
choices are compromises forced by the data. It supports the "technique justification" part of the report.

## 1. Problem framing

**Target.** For a customer who bought in the last 180 days (the *eligible* population), predict whether they make
**no qualifying purchase in the next 90 days**. A "qualifying purchase" is a non-cancelled invoice line with
positive quantity and price for a real product code.

**Why not the thesis' 180-day horizon.** The dataset spans about 24 months (Dec 2009 to Dec 2011). Every snapshot
needs a fully observed label window, and a chronological split needs an *embargo* between blocks equal to the
label length. With 180 days the embargo leaves too few snapshots to train, validate, calibrate and test. The
90-day horizon is the longest that leaves a usable block in each role; 60 and 120 days are reported as
sensitivity analyses.

**Why eligibility matters.** Without it, customers who stopped buying a year ago are trivially "inactive" and recency
alone would give a flattering AUC. Restricting to recently active customers makes the task the useful one: spotting
the *transition* to inactivity.

**Why monthly snapshots.** Quarterly snapshots (as in the thesis) would give only about six matured snapshots. Monthly
snapshots give 19 but make windows overlap heavily, so rows of the same customer are correlated. This is handled by
(a) never putting the same time period in two blocks, (b) a customer-clustered bootstrap for every confidence
interval, and (c) reporting the limitation openly.

## 2. Validation: no random splits

Snapshots are assigned to four consecutive blocks: train, validation (model selection), calibration (probability
mapping only) and out-of-time test. A snapshot is **dropped** unless its label window closes before the next block
starts (`splits.assign_blocks`; verified by `tests/test_splits.py`). This costs six of the 19 snapshots, which is the
honest price of leakage-free evaluation on short history. Separate validation and calibration blocks mean calibration
never reuses the data used to pick the model.

## 3. Techniques used and why

| Choice | Reason |
|---|---|
| Fixed-recency rule baseline | The rule the business would use today; any model must beat it. |
| Regularised logistic regression | Transparent, strong on tabular data with few strong signals, and a calibration reference. |
| LightGBM | Captures non-linear recency/frequency interactions; built-in TreeSHAP (`pred_contrib`) gives explanations without a heavy extra dependency in the API image. Tuned with a 60-trial Optuna TPE search on **validation log loss** (a proper scoring rule), not AUC. |
| BG/NBD benchmark | Classic probabilistic model for non-contractual purchasing; shows whether a hand-crafted supervised model adds anything over the standard customer-base model. Used for ranking only. It is fitted without labels on **all** customers (active or not) at the last calibration snapshot, before the test period; fitting on recently active customers only made the optimiser degenerate. |
| Platt vs isotonic calibration | Chosen by customer-grouped cross-validated Brier score on the calibration block, because the API shows probabilities, not just ranks. |
| Customer-clustered bootstrap | Snapshots of the same customer are not independent; resampling whole customers gives honest intervals and paired model comparisons. |
| SHAP-based reason codes | Faithful by construction: only features that raise the model's own risk score are reported, and the wording is direction-neutral so it cannot contradict the model. |
| PSI drift report | Cheap, interpretable monitoring of train-versus-test input shift. |

## 4. Techniques deliberately not used

| Not used | Why |
|---|---|
| Deep networks (MLP, LSTM, Transformer) | Tabular data of modest size (thousands of customers, tens of features); boosted trees are usually as good and far easier to explain and serve. A sequence model would need far more history than 24 months provides. |
| LLMs / RAG / fine-tuning | No text signal in the target; free-text product descriptions add little to predicting purchase timing. |
| SMOTE / oversampling | The target is not extremely imbalanced, and synthetic rows distort both calibration and the temporal dependence between snapshots. |
| Unsupervised clustering as the model | Labels can be derived directly from the data, so a supervised model is more direct. Segments are handled by the segment-aware label sensitivity analysis instead. |
| Survival models (Cox, discrete-time) | A reasonable alternative, but a binary horizon target matches how Sales would use the score, and BG/NBD already covers the probabilistic-timing viewpoint. Noted as future work. |
| Accuracy as a metric | Misleading under prevalence shift; ROC-AUC, PR-AUC, lift, Brier score and calibration slope/intercept are used instead. |
| Random train/test split | Would leak future behaviour of the same customers into training. |

## 5. Serving design

- **One feature code path.** `features.compute_features` is used by batch training and by `POST /score/transactions`;
  `tests/test_features.py::test_api_path_matches_batch_path` asserts they agree.
- **Two endpoints on purpose.** `/score` takes precomputed features (cheap, for a batch system that already has
  them); `/score/transactions` takes raw lines (convenient, and removes training/serving skew).
- **Explicit "Not eligible" state.** A dormant customer returns no probability rather than a misleading low-risk one.
- **Model bundle.** One versioned joblib file holds the model, calibrator, feature list, risk thresholds and test
  metrics, so the container needs no training code or data.
- **Risk bands** are capacity-based (top 10% of calibration-period scores = High, next 20% = Medium), because a
  sales team can only follow up a fixed number of customers.

## 6. Known limitations

- Only about 24 months of history, so the training period is early 2010 and the test period is mid-2011; the
  business and the seasonal mix may differ. The drift report quantifies part of this.
- Monthly snapshots overlap, so effective sample size is smaller than the row count (handled by clustered intervals).
- The dataset is one UK online retailer; results do not transfer to other businesses or to the African trade-credit
  setting of the thesis without re-validation.
- Customers with missing IDs (a notable share of raw rows; the exact share is in `reports/data_quality.json`) are dropped, which can bias the population towards
  registered or repeat buyers.
- Features such as tenure, purchase-day count and 365-day order count are left-censored by the Dec 2009 start of the data and drift upward over time (PSI report); calibration fitted on the spring-2011 block overstates risk on the summer-2011 test block because the inactivity rate differs between seasons.
- BG/NBD scores can be numerically undefined for very frequent customers; those rows receive a neutral score and
  the count is printed by the pipeline.
