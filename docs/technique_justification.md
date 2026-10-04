# Technique justification

DSA 8401 Applied Machine Learning | Capstone | Eric Muriuki Kiama (120890)

Slide references use the form W4 s26 (Week 4 lecture, slide 26) and S8 s51 (Session 8, slide 51), from the course notes.

## 1. Approach

Two principles from the course drove every choice. First, "start with the simplest thing that could possibly work, then justify every added gram of ML complexity" (W1 s5). Second, "model choice is an evaluation question, not a fashion question" (S8 s51). So each plausible technique was either **trained and compared on the same out-of-time test period**, or **excluded with a reason tied to this data and to course guidance**. Where the course recommends something the project does not do, section 4 says so.

The problem fixes the shape of the toolbox: one row per customer per month-end, 21 hand-built features, about 9,000 training rows, a balanced target (44% of active customers placed no order in the next 90 days), customers who repeat over time, and a need for probabilities, reasons and a deployed API.

## 2. Techniques used, and what the evidence showed

| Technique | Why it was chosen | Course reference | Evidence on this data (ROC-AUC on validation, then on test) |
|---|---|---|---|
| Prevalence dummy and recency rule ("longest since last order is riskiest") | A baseline sets the floor; report lift over it, never a score alone | W1 s5, s18, s25; S8 s12 | Dummy 0.500. Rule 0.609 on validation, 0.662 on test. The model beats the rule by 0.09 to 0.12 |
| Regularised logistic regression | Transparent, sample-efficient, "often 90% of the value"; a model must beat it before it can claim to work | W3 s5 to s7; W1 s25 | Validation 0.726 (the highest), test 0.760 |
| Decision tree, depth 4 | Readable but high variance; the reference point that motivates ensembles | W4 s14 to s15 | Validation 0.701, test 0.731. Clearly beaten (gap +0.030 to +0.047) |
| Random forest | Bagging plus feature decorrelation cuts variance | W4 s25 to s26 | Validation 0.719, test 0.766. Near tie with LightGBM (gap +0.001 to +0.007) |
| **LightGBM (the served model)** | Gradient-boosted trees are the tabular champion; native handling of missing values, early stopping, and built-in TreeSHAP for reasons. Chosen by design, not as the validation winner (see section 5) | W4 s2, s34 to s36, s41; W6 s8 | Validation 0.723 (logistic regression 0.726, random forest 0.719); **test 0.771**, the best, by small margins |
| Neural-network challenger (MLP) | The course requires a documented challenger: "beat it, or report that you did not" | W6 s9, s54 | Validation 0.708, test 0.759. LightGBM ahead by 0.004 to 0.019 ROC-AUC and 0.012 to 0.048 PR-AUC |
| BG/NBD probabilistic model | The classic benchmark for non-contractual purchasing (Fader et al., 2005, from my thesis literature); not a course topic | Thesis literature | Validation 0.703, test 0.765. Statistically tied with LightGBM |
| Optuna TPE search, 60 trials, six log-scaled dimensions, early stopping, tuned on validation log loss | Random beats grid, TPE beats random once trials are costly; budget at least 10 times the dimensions; fix the tree count high and let early stopping choose it | W4 s49 to s58 | Tuning bought almost nothing here: the best validation log loss (0.6022) was reached on the very first trial and none of the other 59 improved on it. The course illustration (W4 s61) had tuning matter more; here the features carry most of the signal. Trials are logged in `reports/lgbm_search.csv` and MLflow |
| Chronological split with label-maturity exclusion; features end at the scoring date; imputers and scalers inside pipelines | Random splits leak the future; observations whose label has not matured by the boundary "must be excluded, not assumed" | W2 s31 to s34; W1 s21 to s22 | Six of 19 monthly snapshots are dropped by the embargo. Automated tests check that features ignore the future and that no label window crosses a block |
| Customer-held-out retrain | Group leakage: the same customer must not sit on both sides of a boundary | W2 s31 (cause 5), s32 | With every test-period customer removed from training, 0.753 against 0.771. Familiarity with a customer is not what drives the score, though the training set shrank from 8,910 to 3,620 rows, so this also reflects less data |
| ROC-AUC primary, with PR-AUC, lift, Brier score, calibration slope and intercept | The course rule: balanced classes use ROC-AUC, rare classes PR-AUC. Inactivity is 44%, so ROC-AUC leads and PR-AUC is reported | W3 s13, s19 to s22 | See the model comparison in the README and the UI |
| Customer-clustered bootstrap intervals and paired differences | A result without dispersion is incomplete; "+0.004 AUC inside the noise band is not a result" | W3 s25, s28 | Used to call ties honestly (BG/NBD, random forest, and logistic regression on PR-AUC) |
| Platt or isotonic calibration on a separate later period | Probabilities must be checked and repaired on held-out data, never on training data | W3 s22 | Did not help on test: the calibration period had 58% inactivity against 44% in test, so risk is overstated (intercept -0.51), and the calibrated model has worse expected calibration error (0.096 against 0.068 raw) and Brier score (0.202 against 0.198) than the raw one. Calibrating on a period closer to the test period is the obvious next step |
| TreeSHAP reason codes; permutation importance on the test period | Impurity or gain importance favours inputs with many distinct values; permutation importance on held-out data is the safer global view | W2 s28; W4 s27 | Top inputs are the number of purchase days, the usual order interval and recent spend, each worth about 0.01 ROC-AUC or less on its own because the inputs are correlated |
| Per-segment evaluation | A good overall score can hide a failing group | W3 s36 | Ranking is weak for one-off and low-spend customers (0.63 to 0.65) and strong for regular, high-spend ones (about 0.75) |
| Population Stability Index drift report | Train and serving data drift; monitoring closes the loop | W1 s12 to s13, s27 | Flags tenure-type inputs, which grow mechanically because the data starts in Dec 2009 |
| FastAPI, Docker, MLflow, tests | "The model is rarely why projects fail. The system around it is." | W1 s12, s13, s27 | Built. The Docker image builds and the container runs healthy; the smoke test passes through it (2026-10-04) |

## 3. Techniques deliberately not used

| Technique | Why not | Course reference |
|---|---|---|
| SMOTE, resampling, class weights | The target is not imbalanced (44%). The course orders remedies as threshold moving, then weights, then SMOTE, only "if ranking quality itself is poor"; resampling also distorts calibration, which this project needs | W3 s31 to s33 |
| Cost-optimal threshold t* = cFP / (cFP + cFN) | Needs costs for a missed lapse and a wasted call. The dataset has no margins or contact costs, so risk bands are capacity-based (top 10% High, next 20% Medium). With margin data this would be the first upgrade | W3 s16 |
| Support vector machines | Kernel SVMs scale poorly beyond tens of thousands of rows (fine at 9,000), give no native probabilities (Platt scaling on top of a margin) and no exact feature attributions for reason codes. Expected to land near regularised logistic regression, which is already in the ladder and was the validation leader | W3 s5 to s7 |
| XGBoost or CatBoost in place of LightGBM | The same gradient-boosted family, so a different library would not test a different idea. LightGBM was chosen for speed, native missing values and TreeSHAP support; XGBoost would be a reasonable swap and no evidence here says it would differ. Not run, so no claim is made | W4 s34 to s36 |
| Survival or discrete-time hazard models (Cox, Weibull) | The natural alternative for "when will they stop?". The business question here is a fixed-horizon yes or no, and the BG/NBD benchmark already models purchase timing probabilistically; a hazard model would add time-varying covariates and censoring handling that the monthly snapshot design sidesteps. Listed as future work | Thesis literature |
| Stacking or voting | Typical gain is +0.005 to +0.02 AUC for serving every base model; the top models here are already statistically tied, so the gain would sit inside the noise | W4 s44, s61 |
| Nested cross-validation | Replaced by the course's stated industry compromise: one temporal hold-out for tuning and one out-of-time test period whose dates were fixed in advance and on which nothing was chosen (its results were visible during development). The gap is reported: validation 0.723 against test 0.771. The test score is higher, so there is no sign of overfitting the validation set, but skill moves by several points between periods | W3 s27; W4 s57 to s58 |
| Sequence models (LSTM, GRU, Temporal Fusion Transformer) | The course's own decision table points to LightGBM on tabular covariates, and sequence models to regular series with a lookback chosen from the autocorrelation. Here each customer has a sparse, irregular list of orders over about two years, and there is no regular grid to forecast. "If your LSTM cannot beat seasonal naive, ship seasonal naive" applies equally to the recency rule | S8 s12, s18, s51 |
| Convolutional networks | Convolution encodes assumptions about images: locality, weight sharing, translation equivariance. There is no spatial structure in order histories | CNN deck s2, s9 |
| LLMs, retrieval, fine-tuning | There is no text signal in the target. The course lists core tabular scoring as "still the wrong tool" for LLMs and says to escalate only when a measurement says to | W1 s28; S9 s34 |
| Clustering or PCA as the model | Labels can be derived directly from the data, so a supervised model is more direct. The curse of dimensionality (W5 s25) is a concern at about 200 features, not 21. Customer segments are a sensible future feature (W5 s33) | W5 s5, s25, s31 |
| Monotonic constraints | Introduced for regulated credit decisions. Not required here, though constraining risk to rise with time since the last order could harden the model against the drift seen in tenure inputs. Considered and left as future work | W4 s40 to s41 |
| Cyclical or seasonal features | With about 24 months the model would see each season once or twice and memorise it. The calibration drift shows seasons matter, which is recorded as a limitation | W2 s26 |
| Target encoding, hashing, embeddings | There are no high-cardinality categoricals (only a UK flag) | W2 s20 to s22 |
| Removing or winsorising outliers | In transaction data outliers can be the signal; heavy tails are handled with log transforms instead | W2 s14, s18 |
| MICE or KNN imputation | The course advice for tree models is simple imputation with an indicator, or none because trees route missing values natively. Logistic regression uses median plus indicator; LightGBM uses missing values directly | W2 s10 to s12 |

## 4. Where the project departs from course advice

- **Horizon.** The prediction horizon is 90 days, not the 180 days in my thesis proposal. About 24 months of data cannot support 180-day labels plus a split with embargoes. 60 and 120 days are reported as sensitivity runs.
- **Overlapping monthly snapshots.** Monthly snapshots give more rows from short history but are correlated. The mitigation is the customer-clustered bootstrap, and no period appears in two blocks.
- **No nested cross-validation and no cost-based threshold**, for the reasons above.
- **Monitoring.** The course names Evidently AI (W1 s27); I implemented a Population Stability Index report directly instead.
- **Fairness.** The data has no protected attributes, so the audit is by country, spend tier and order history only. It found a real weakness for small and one-off buyers.
- **Deployment.** The Docker image was built and run once, on one machine (Docker Desktop, Linux containers); it has not been tried on another operating system or in CI.

## 5. What this adds up to

LightGBM was chosen as the served model by design, before the test period was scored, and not as the winner of a comparison: on the validation period logistic regression was marginally ahead on ROC-AUC (0.726 against 0.723) and LightGBM was ahead on top-decile lift (1.59 against 1.54), with the top three models within 0.007. The reasons for choosing it are early stopping, native missing-value handling, interactions without hand-built terms, and exact TreeSHAP reason codes that suit an API. It then also came out best on the test period. The honest reading is that its edge is small, and that serving the simpler logistic regression would also have been defensible. A calibrated logistic regression, the course's baseline (W3 s5), is within 0.005 to 0.016 ROC-AUC, and a random forest or the classic BG/NBD model are statistically close. The neural-network challenger did not beat the tree models, which is the evidence for leaving deep learning out. The decisions that mattered most were not the choice of algorithm but the framing, the leakage-safe split and the honest evaluation, which is where the course says projects are won (W1 s16, s18).

## References

Course material (DSA 8401, Strathmore University): Weeks 1 to 6 lecture decks, Session 7 (CNNs), Session 8 (sequence models), Session 9 (transformers and LLMs).

Works cited in those notes and used here:

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

From my thesis literature: Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). "Counting your customers" the easy way: An alternative to the Pareto/NBD model. Marketing Science, 24(2), 275-284.
