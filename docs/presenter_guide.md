# Presenter guide: how this project works, in plain words

Read this once, out loud, the night before. Everything here is true of the project as built.

## 1. The 60-second version

"A wholesaler's customers never cancel. They just stop ordering, and by the time that is obvious they are gone. I built a system that looks at every customer who bought in the last six months and gives each one a score: how likely are they to place no order in the next 90 days? It explains the score in plain language, and it serves it through an API and a web interface.

The honest result: a boosted-tree model beats the rule a business would use today ('longest since last order') by a clear margin, ROC-AUC 0.77 against 0.66. It only barely beats simpler models like logistic regression. The ranking is good enough to help a sales team decide who to call first. The percentages themselves run high, so read them as a priority score, not a literal probability."

If you remember only that paragraph, you can answer most questions from it.

## 2. The story in seven steps (and why each step exists)

**Step 1. The question.** Not "who churned?" (nobody cancels), but: *among customers who ordered in the last 180 days, who will place no order in the next 90 days?* Limiting to the last 180 days matters: customers already gone would make the problem trivially easy ("long since last order, so no order next").

**Step 2. The data.** UCI Online Retail II, public, a UK online wholesaler, Dec 2009 to Dec 2011. 1,067,371 raw lines. I removed duplicates (34,337), non-product codes like postage (3,662) and rows with no customer ID (22.8%, they cannot be tracked). That leaves 794,161 lines and 5,875 customers. Returns are kept as a feature but never counted as orders.

**Step 3. Snapshots.** I pretend to stand at the end of each month (a "snapshot") and ask: what did I know on that date, and what happened in the next 90 days? Features use only data up to the snapshot. The label looks only at the following 90 days. One customer therefore appears once per month, which gives about 36,000 rows from 5,875 customers.

**Step 4. 21 features.** Simple, explainable things: days since last order, orders in the last 30/90/180/365 days, spend over the same windows (log-scaled because spend is heavy-tailed), average order value, the customer's own usual gap between orders, how long since the last order *relative to that usual gap*, return rate, a UK flag.

**Step 5. Honest splitting.** Random splits would let the future leak into training (the same customer on both sides). Instead I split by time: train (Feb to May 2010), validation (Aug to Oct 2010), calibration (Jan to Mar 2011), test (Jun to Aug 2011). If a snapshot's 90-day outcome was not yet known when the next period began, I *drop* it rather than guess. That removed 6 of 19 snapshots. This is the "embargo".

**Step 6. Models.** A ladder from simple to complex, all scored on the same test months:

| Model | Test ROC-AUC |
|---|---|
| Recency rule (the business baseline) | 0.662 |
| Decision tree, depth 4 | 0.731 |
| Neural network (MLP) | 0.759 |
| Logistic regression | 0.760 |
| BG/NBD (classic customer-lifetime model) | 0.765 |
| Random forest | 0.766 |
| **LightGBM (served)** | **0.771** |

LightGBM was tuned with Optuna (60 trials), calibrated, and given TreeSHAP "reason codes". Uncertainty: 500 bootstrap resamples of *customers* (not rows), so the intervals are honest. LightGBM's 95% interval is 0.757 to 0.785.

**Step 7. The system.** One scoring code path is used in training and in the API (a test checks they agree). The API (FastAPI) takes features or raw order lines and returns a probability, a risk band (High, Medium, Low, or "Not eligible") and three reason codes. It runs in Docker. A seven-page web interface sits on top.

## 3. Words you must be able to explain in one sentence

- **ROC-AUC:** the chance that a randomly chosen customer who went quiet gets a higher score than one who stayed. 0.5 is a coin flip, 1.0 is perfect. Ours is 0.77.
- **PR-AUC (0.69):** how well the top of the ranking is filled with real lapsers; compare it with the base rate (0.44), not with 1.
- **Top-decile lift (1.8x):** the top 10% of scores contain 1.8 times as many lapsers as a random 10%.
- **Calibration:** whether "70%" really means 70 in 100 customers go quiet. Ours does not quite: it runs high.
- **Leakage:** the model accidentally seeing information from the future. Everything in step 5 exists to prevent it.
- **Embargo:** dropping snapshots whose outcome was not yet known at the period boundary.
- **Bootstrap (customer-clustered):** re-running the evaluation on many resampled sets of customers to see how much the numbers wobble. Customers, not rows, because one customer contributes many rows.
- **TreeSHAP reason codes:** for each customer, which features pushed the score up the most. They explain the *model*, not the real-world cause.
- **PSI (drift):** a measure of how much a feature's distribution moved between training and test.

## 4. Ten numbers worth memorising

1. 1,067,371 raw lines, 794,161 clean, 5,875 customers
2. 90 days ahead, 180 days activity window, 21 features
3. 6 of 19 monthly snapshots dropped by the embargo
4. Test: 3 months, 8,256 customer-snapshots, 3,118 customers, 44% went quiet
5. ROC-AUC 0.771 (LightGBM) against 0.662 (rule); gain 0.094 to 0.123
6. Against logistic regression: +0.005 to +0.016, "barely"
7. Call the top 30%: reach 48% of lapsers (model), 43% (rule), 30% (random). Per 1,000 lapsers that is 478, 428, 300.
8. Calibration period 58% inactive, test 44%, so probabilities run high
9. Weak for small buyers: ROC-AUC 0.63 to 0.65, against 0.75 for big, regular ones
10. 43 automated tests; Docker built and run once on one machine

## 5. Slide by slide (15 slides)

1. **Title.** Say the question: which customers are about to go quiet, and how far can we trust the answer?
2. **Customers don't cancel.** Wholesalers have no cancellation event. A fixed rule ("no order in 6 months") fires late. Goal: rank customers, explain why, and serve it to a sales team with limited calls. 44% of active customers go quiet in 90 days.
3. **The data is raw.** The cleaning numbers. Point out the 22.8% with no ID: a limit you accept openly.
4. **Built so the future cannot leak in.** The timeline picture: features up to the snapshot, label after it. Train, validate, calibrate, test are separate periods in time. 6 of 19 dropped.
5. **A ladder of models.** All scored on a period none of them saw. Key message: clearly beats the rule, barely beats simple models, a neural network did not win. Say it plainly: on the validation period logistic regression was marginally ahead, and I chose LightGBM by design (it handles missing values, captures interactions, and gives exact explanations).
6. **Following the ranking reaches more lapsing customers.** The gains curve. Top 30% reaches 48% of lapsers against 43% for the rule.
7. **What that means for a sales team.** 478 against 428 against 300 per 1,000 lapsers: about 50 more reached from the same calls. No cost data, so this is a better call list, not a money figure.
8. **The ranking is solid. The percentages run high.** Calibration chart. Cause: the calibration period happened to be a higher-inactivity season. Use the percentage as a priority score.
9. **It does not work equally for every customer.** The segment chart. Weak for one-off and low-spend buyers.
10. **How I chose the techniques.** What I tried and what I left out with a reason (SMOTE, sequence models, CNNs, LLMs, and so on).
11. **Honesty checks.** Validation 0.723 to test 0.771 (test higher, so no sign of overfitting the validation set); retrained with every test customer removed: 0.753; stable across the three test months; 6 snapshots dropped.
12. **A system, not just a model.** Pipeline to bundle to API to Docker; 43 tests; training and serving share the same feature code.
13. **The interface.** Seven pages. (Demo, section 6.)
14. **Limits and next steps.** Say the limits before anyone asks.
15. **Three takeaways.** Framing and leakage-safe evaluation mattered more than the algorithm; boosted trees clearly beat today's rule but only barely beat simple models; use the ranking, treat the percentages as a priority score.

## 6. A three-minute live demo (the app is at http://localhost:8000/)

1. **Overview.** "Twelve numbers at a glance. 2,768 customers scored at the last date, 199 High risk."
2. **Explore data.** Pick "Feature explorer" and choose *Spend in the last 365 days*: the lapse rate falls as spend rises. "This is the signal the model uses."
3. **Retention worklist.** Slide the "calls available" box to 100. "Of the top 100 by score, 79 really went quiet, against about 39 if I had picked at random."
4. **Click one customer.** Show the reasons and the timeline. Drag the "Days since last order" slider: the risk moves live because the page calls the API.
5. **Model evidence.** One sentence: "This is where I show how far to trust it."

If anything breaks, say "the same thing is in my slides" and carry on. Start the app before you begin: `docker compose up -d` from the project folder, or `python -m inactivity serve`.

## 7. Likely questions, with honest answers

**Why 90 days and not 180?** I only have about 24 months of data. A 180-day outcome plus a gap between periods leaves too few snapshots. 60 and 120 days give similar results (0.757 and 0.781).

**Why LightGBM if logistic regression is about as good?** Honest answer: the gap is small, and on validation logistic regression was marginally ahead (0.726 against 0.723). I chose LightGBM by design: it handles missing values, captures interactions without hand-building them, and has exact explanations. It also came out best on test. Serving logistic regression would also have been defensible.

**Is 0.77 good?** It is clearly better than the rule (0.66), and a rule like that is what a business would use. It is not high in absolute terms. It means useful ranking, not certainty.

**Why not a neural network or deep learning?** I did train one as a challenger and it did not beat the tree model (0.759 against 0.771). With about 9,000 training rows and 21 tabular features, boosted trees are the standard.

**Did you tune on the test set?** No. The test dates were fixed in advance and nothing was chosen on it. I will be upfront that its results were visible while I developed, and that some models were added after earlier runs. Tuning used the validation period.

**Why do the probabilities run high?** I calibrated on a period where 58% of customers went quiet, while the test period had 44%. The mapping inherited the wrong base rate. Ranking quality is unaffected. The fix would be to calibrate on a more recent window; I list it as next work.

**What is data leakage and how did you avoid it?** Leakage is the model seeing the future. Features use only data up to the snapshot, the splits are by time, and snapshots whose outcome was not yet known at a boundary are dropped. Automated tests check that features ignore the future and that no label window crosses a period.

**Why are returns kept?** As a feature (share of value returned), never as an order. Counting a return as a purchase would hide a lapse.

**How do you know the explanations are right?** They are TreeSHAP contributions: exact for tree models, and they say what pushed *this model's* score up. They are not causes. Nothing here shows that calling a customer changes whether they stay; that would need a controlled pilot.

**Is it fair?** There are no protected personal attributes in the data; these are businesses. I checked performance by country, spend tier and order history, and found a real weakness for small and one-off buyers.

**Would this work on another company?** It shows the method. The numbers are from one UK retailer, so it would need re-testing and recalibration.

**How much money does it save?** I cannot say: no margin or contact-cost data. I can say how many more at-risk customers are reached for the same calls (about 50 more per 1,000).

**Is the Docker part tested?** Built and run once on one machine (Docker Desktop, Linux containers): it builds, reports healthy, and a smoke test scores a customer. Not tried on another operating system or in CI.

**How can I reproduce it?** `python -m inactivity all` runs the whole pipeline from download to model bundle; results are deterministic (I re-ran it from a clean environment and the 58 metrics matched exactly). A lock file pins versions.

**Did AI write this?** Yes, Claude Code drafted much of the code, tests and documents. The disclosure is in `AI_USE.md`. Be ready to say what you directed (the problem, the data choice, the changes you asked for) and what you checked yourself. Fill in `AI_USE.md` before you submit, so that what you say matches what it says.

## 8. What not to claim

- Do not say the model "predicts churn". It estimates the chance of no order in 90 days.
- Do not say the probabilities are accurate. Say the ranking is reliable and the percentages are a priority score.
- Do not say LightGBM clearly beats the other sensible models. It does not.
- Do not say the test set was untouched or scored once.
- Do not claim money saved or customers retained.
- Do not say the Docker image works everywhere.

## 9. Where things live (if someone asks to see code)

| Question | File |
|---|---|
| How are customers' features built? | `src/inactivity/features.py` (21 features; the same code serves training and the API) |
| How are the splits and the embargo done? | `src/inactivity/splits.py`, settings in `configs/config.yaml` |
| What models were trained? | `src/inactivity/models.py` |
| How is it evaluated (bootstrap, calibration)? | `src/inactivity/evaluate.py` |
| What runs end to end? | `src/inactivity/pipeline.py`, started by `python -m inactivity all` |
| The API | `src/inactivity/service/app.py` |
| The web interface | `src/inactivity/service/static/` (plain JavaScript and SVG, no libraries) |
| Tests | `tests/` (43) |
| Results in numbers | `reports/results.md` |
