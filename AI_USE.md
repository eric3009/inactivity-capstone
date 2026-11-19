# AI use disclosure (DSA 8401)

## Tool and scope

- **Tool:** Claude Code (Anthropic), used interactively in the Claude desktop app, running the Claude Sonnet 5.5 model.
- **Used for:**
  - drafting the capstone proposal and choosing the validation design (chronological split with a label-maturity embargo);
  - writing the Python source in `src/inactivity/` (data pipeline, features, models, evaluation, explanations, drift alert,
    segments, hypothesis tests, feature selection, anomaly check), the unit and API tests in `tests/`, the Dockerfile and
    compose file, the GitHub Actions workflow, and the web interface served by the API;
  - writing the documentation (`README.md`, `docs/`), the final report and the slide deck, and building their figures;
  - debugging, and running the pipeline and the tests.
- **Not used for:** choosing the project topic (mine, from my MSc thesis work) or the dataset (I picked UCI Online Retail II
  from options presented).

## What I directed and decided

- The problem framing: predicting customer inactivity, with the dual-risk (inactivity and credit) thesis as later work.
- The decision to use a public dataset instead of confidential company data.
- The additions I asked for after the first version: the drift alert, the SVM and K-Means clustering, GitHub Actions CI,
  and the hypothesis tests, feature-selection comparison and Isolation Forest check.

## How the AI-written work was checked

- 63 automated tests (data cleaning rules, no-leakage checks, split embargo, metric sanity, drift gate, statistical tests,
  API behaviour and training/serving parity) run with `python -m pytest`.
- The pipeline was first run end to end on synthetic data, then on the real UCI data (October 2026); the statistical tests,
  feature selection and anomaly check were run on the real data in November 2026.
- The reported results come from those real-data runs and are saved in `reports/`. The README and report state the weak
  findings as well as the strong ones, for example that the model barely beats simpler models and that the probabilities
  run high because of a calibration-period shift.

## Limits of that checking

- The tests show that the code does what it is written to do. They do not prove the modelling choices are the best ones.
- The Docker image was built and run once on one machine, and the GitHub Actions workflow had not run when this was written.

## Data and privacy

- Only the public UCI Online Retail II dataset (CC BY 4.0) is used. No confidential, personal or company data is in this
  repository, its data, or its code.

## Author's statement (to be completed and signed by the author)

What I personally read, ran or re-derived, and what I can explain and defend:

I understand this project is individually assessed and I can explain and defend every part of it.

Name: Eric Muriuki Kiama (120890)   Date: 19 November 2026

Signature: <img src="docs/figures/signature.png" alt="Signature of Eric Muriuki Kiama" width="130">
