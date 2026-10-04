# AI use disclosure (DSA 8401)

> Fill in the bracketed parts yourself and check this against the course AI policy before submitting.

## Tool and scope

- **Tool:** Claude Code (Anthropic), model Claude Sonnet 5.5, used interactively in the desktop app.
- **Used for:** drafting the one-page proposal; designing the pipeline and validation scheme; writing the Python
  source (`src/inactivity/`), the unit and API tests (`tests/`), the Dockerfile and compose file, and the
  documentation (`README.md`, `docs/`); and debugging.
- **Not used for:** choosing the project topic (mine, from my MSc thesis work), choosing the dataset
  (I picked UCI Online Retail II from options presented), or interpreting the final results [confirm].

## What I directed and decided

- [The problem framing: inactivity prediction on public data, with the dual-risk thesis as later work.]
- [The decision to use a public dataset instead of confidential company data.]
- [Any parameter or design change I requested.]

## How the AI-written work was checked

- 41 automated tests (data cleaning rules, no-leakage checks, split embargo, metric sanity, API behaviour and
  training/serving parity) run with `python -m pytest`.
- The pipeline was first run end to end on synthetic data, then on the real data [confirm date].
- [What I personally read, ran or re-derived, for example reading every module, re-running the pipeline,
  inspecting the SHAP outputs.]

## Limits of that checking

- The tests show the code does what it is written to do; they do not prove the modelling choices are the best ones.
- Results depend on the real data run and my own review of them [confirm].

## Data and privacy

- Only the public UCI Online Retail II dataset (CC BY 4.0) is used. No confidential, personal or company data was
  shared with the AI tool.

## Declaration

I understand this project is individually assessed and I can explain and defend every part of it.

Name: Eric Muriuki Kiama (120890)  Date: [date]  Signature: [ ]
