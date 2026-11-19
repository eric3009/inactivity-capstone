# Drift runbook: customer inactivity model

**Owner:** Eric Muriuki Kiama (model owner, accountable for responding). **Escalation:** the project supervisors (in a production setting, the sales and credit managers who use the scores).
**Check:** `python -m inactivity drift` after each month-end scoring run. **Exit code:** 0 OK, 1 WARNING, 2 ALERT, so a scheduler or CI job fails on any non-zero code.
**Outputs:** `reports/drift_check.json` (decision and reasons), `reports/drift_report.html` (Evidently report), console summary.

## What is watched

| Check | How | Warning | Alert |
|---|---|---|---|
| Input drift | PSI of each of the 21 inputs against bin edges stored from the training window (`artifacts/drift_reference.json`) | any input PSI >= 0.10 | PSI >= 0.25 on **3 or more** inputs |
| Score drift | PSI of the calibrated probability against the validation window | PSI >= 0.10 | PSI >= 0.25 |
| **Slice metric** | ROC-AUC for customers with a single purchase day (the model's weakest group) against the stored baseline | AUC down 0.03 | AUC down 0.06 |
| Window size | rows in the checked window | fewer than 300: reported as a warning, not assessed | none |

The slice metric needs outcomes, which mature 90 days after scoring, so it runs on older snapshots; the input and score checks run immediately. Thresholds live in `configs/config.yaml` (`monitoring:`); changing one needs the owner's sign-off and a note in the change log.

## What to do

| Status | Response | Time limit |
|---|---|---|
| OK | File the JSON; no action | none |
| WARNING | Open the Evidently report and find which inputs moved; check the data feed for a changed field, join or filter; note the finding | 5 working days |
| ALERT | **Stop quoting probabilities**: use only the ranking and risk bands. Find the cause, then act: score shift means recalibrate on the latest matured window; input shift means retrain on a recent window; a broken or changed field means fix the pipeline first | 2 working days |
| After any fix | Rebuild the reference with `python -m inactivity drift --init`, run the 58 tests, record the model version and the decision | same day |

## Known and expected drift (read before reacting)

- `tenure_days`, `n_purchase_days` and `inv_count_365` grow with calendar time because the data starts in December 2009, so every customer's history lengthens. They are the first to trip on any later window.
- The out-of-time test window (June to August 2011) currently returns **ALERT** for this reason (8 inputs in major shift) while the score distribution (PSI 0.07) and the slice metric stay within thresholds. The recorded response was to keep the model, treat the percentage as a priority score, and recalibrate. The durable fix, listed in the report, is to cap history windows.
- An unshifted control (`python -m inactivity drift --control`) must return 0. If it does not, the gate is broken, not the data.

**Limits:** PSI depends on sample size and its 0.10 and 0.25 cut-offs are conventions; labels lag 90 days, so performance drift is seen late; the slice check covers one group.
