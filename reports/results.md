# Results (out-of-time test period)

Horizon: 90 days | label: `y_inactive`

## Split sizes

| Block | Snapshots | Rows | Customers | Inactivity rate |
|---|---|---|---|---|
| train | 4 | 8910 | 2684 | 0.451 |
| val | 3 | 8917 | 3557 | 0.438 |
| cal | 3 | 9985 | 3745 | 0.580 |
| test | 3 | 8256 | 3118 | 0.440 |

Snapshots dropped by the label-maturity embargo: 2010-06-30, 2010-07-31, 2010-11-30, 2010-12-31, 2011-04-30, 2011-05-31

## Model comparison

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Top-decile lift | Brier | Cal. slope | Cal. intercept | ECE |
|---|---|---|---|---|---|---|---|
| prevalence | 0.500 | 0.440 | 1.000 | 0.246 | n/a | n/a | 0.030 |
| rule_recency | 0.662 [0.647, 0.677] | 0.595 [0.574, 0.618] | 1.550 [1.476, 1.635] | n/a | n/a | n/a | n/a |
| logreg_raw | 0.760 [0.746, 0.775] | 0.686 [0.663, 0.710] | 1.795 [1.713, 1.895] | 0.221 | 0.619 | 0.496 | 0.133 |
| logreg_cal | 0.760 [0.746, 0.775] | 0.686 [0.663, 0.710] | 1.795 [1.713, 1.895] | 0.203 | 0.810 | -0.310 | 0.064 |
| lgbm_raw | 0.771 [0.757, 0.785] | 0.693 [0.670, 0.715] | 1.795 [1.699, 1.871] | 0.198 | 1.235 | 0.410 | 0.068 |
| lgbm_cal | 0.771 [0.757, 0.785] | 0.693 [0.670, 0.715] | 1.795 [1.699, 1.871] | 0.202 | 1.025 | -0.508 | 0.096 |
| bgnbd | 0.765 [0.751, 0.779] | 0.696 [0.673, 0.719] | 1.820 [1.736, 1.906] | n/a | n/a | n/a | n/a |
| tree_depth4 | 0.731 [0.716, 0.746] | 0.628 [0.608, 0.650] | 1.709 [1.562, 1.749] | n/a | n/a | n/a | n/a |
| random_forest | 0.766 [0.753, 0.781] | 0.689 [0.665, 0.712] | 1.800 [1.717, 1.900] | n/a | n/a | n/a | n/a |
| mlp_challenger | 0.759 [0.744, 0.773] | 0.662 [0.637, 0.684] | 1.619 [1.529, 1.711] | n/a | n/a | n/a | n/a |

## Model selection on the validation period (before the test period was used)

| Model | Validation ROC-AUC | Validation PR-AUC | Validation top-decile lift |
|---|---|---|---|
| logreg_cal | 0.726 | 0.631 | 1.544 |
| lgbm_cal | 0.723 | 0.627 | 1.593 |
| random_forest | 0.719 | 0.627 | 1.565 |
| mlp_challenger | 0.708 | 0.594 | 1.439 |
| bgnbd | 0.703 | 0.605 | 1.552 |
| tree_depth4 | 0.701 | 0.597 | 1.439 |
| rule_recency | 0.609 | 0.542 | 1.429 |

## Honesty checks

- **How the test period was used:** its dates were fixed in advance and no model, hyper-parameter, calibrator or threshold was chosen on it. Its results were, however, visible during development, and some models (Optuna tuning, the neural-network challenger) were added after earlier runs, so treat it as a clean test of the chosen design rather than a single-use hold-out.
- **Optimism gap (selection set vs test period):** LightGBM validation ROC-AUC 0.723 vs test 0.771; validation PR-AUC 0.627 vs test 0.693.
- **Customer-held-out retrain (group-leakage check):** with every test-period customer removed from training and validation (3,620 training rows left), test ROC-AUC 0.753 and PR-AUC 0.684, against 0.771 and 0.693 for the served model.

## Paired differences (95% customer-clustered bootstrap CI)

| Comparison | ROC-AUC diff | PR-AUC diff | Lift diff |
|---|---|---|---|
| lgbm_cal-bgnbd | [-0.002, +0.013] | [-0.019, +0.014] | [-0.113, +0.056] |
| lgbm_cal-logreg_cal | [+0.005, +0.016] | [-0.008, +0.019] | [-0.092, +0.064] |
| lgbm_cal-mlp_challenger | [+0.004, +0.019] | [+0.012, +0.048] | [+0.052, +0.277] |
| lgbm_cal-random_forest | [+0.001, +0.007] | [-0.000, +0.008] | [-0.055, +0.025] |
| lgbm_cal-rule_recency | [+0.094, +0.123] | [+0.078, +0.118] | [+0.140, +0.337] |
| lgbm_cal-tree_depth4 | [+0.030, +0.047] | [+0.053, +0.078] | [+0.057, +0.216] |

## Sensitivity (LightGBM, calibrated; 20 search trials, no bootstrap)

| Variant | Test rows | Inactivity rate | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|
| horizon 60d (fixed label) | 11175 | 0.535 | 0.757 | 0.747 | 0.204 |
| horizon 90d (fixed label) | 8256 | 0.440 | 0.771 | 0.693 | 0.202 |
| horizon 120d (fixed label) | 5488 | 0.378 | 0.781 | 0.654 | 0.197 |
| horizon 90d (segment-aware label) | 8256 | 0.149 | 0.897 | 0.567 | 0.096 |

Sensitivity rows use different label definitions, so their metrics are not directly comparable with each other as a ranking of models. The segment-aware label is defined partly from recency and the customer's median order interval, which are also model inputs, so its high AUC is partly circular and must not be read as better predictive skill.

## Data quality

```json
{
  "rows_raw": 1067371,
  "duplicate_rows_dropped": 34337,
  "missing_customer_id_rows": 235151,
  "missing_customer_id_share": 0.22763142355430702,
  "non_product_rows_dropped": 3662,
  "cancellation_return_lines_kept": 17586,
  "other_anomalous_lines_dropped": 60,
  "quantity_p99_9": 576.0,
  "extreme_quantity_lines": 745,
  "rows_clean": 794161,
  "customers": 5875,
  "date_min": "2009-12-01",
  "date_max": "2011-12-09",
  "source_url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
  "source_bytes": 45622278,
  "source_sha256": "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980",
  "hash_matches_expected": true,
  "source": "uci-online-retail-ii"
}
```