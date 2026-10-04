import { S, el, pageHead, card, pct } from "../common.js";

export function render() {
  const page = el("div", "page");
  const T = S.trust, i = S.info;
  page.appendChild(pageHead("About", "Definitions and limits", "What the numbers mean, and what this tool should not be used for."));

  const c1 = card("What is predicted");
  c1.appendChild(el("p", "lede", `A customer is eligible at a month-end if they ordered in the last ${T.activity_window_days} days. The model estimates the probability that such a customer places no order in the following ${T.horizon_days} days.`));
  c1.appendChild(el("p", "muted", `High risk means a calibrated probability of at least ${pct(T.thresholds.high)}; Medium means ${pct(T.thresholds.medium)} to ${pct(T.thresholds.high)}. The cut-offs are set from call capacity, not from a convention.`));
  page.appendChild(c1);

  const c2 = card("Glossary");
  const dl = el("dl", "gloss");
  [["ROC-AUC", "Chance that a randomly chosen lapser is scored above a randomly chosen active customer. 0.5 is a coin flip, 1.0 is perfect."],
   ["PR-AUC", "Area under the precision-recall curve. Compare it to the base rate, not to 1."],
   ["Top-decile lift", "How many times more lapsers the top 10% of scores contain than a random 10%."],
   ["Brier score", "Mean squared error of the probabilities. Lower is better; always guessing the base rate scores about 0.25."],
   ["Calibration slope", "Whether stated probabilities are as extreme as reality. Below 1 means over-confident."],
   ["PSI", "Population stability index: how far a feature's distribution moved between train and test. Above 0.25 is a major shift."],
   ["Embargo", "Snapshots whose 90-day outcome was not yet known when the next data block began are dropped, so no label leaks across blocks."],
   ["Reason codes", "TreeSHAP contributions from the model for the individual customer. They explain the model, not causes in the world."]]
    .forEach(([a, b]) => { dl.appendChild(el("dt", "", a)); dl.appendChild(el("dd", "", b)); });
  c2.appendChild(dl);
  page.appendChild(c2);

  const c3 = card("Limits");
  const ul = el("ul", "reasons");
  ["Public data from one UK wholesaler, 2009 to 2011. It demonstrates the method; it does not describe your customers.",
   "Customers without an ID (about a fifth of the lines) cannot be tracked and are excluded.",
   "Predictions are for prioritising outreach. They are not a guarantee and are not a reason to treat any customer worse.",
   "Probabilities are over-confident for some groups; see Model evidence for the calibration and segment tables.",
   "The Docker image was built and run once, on one machine; it has not been tried in CI or on another operating system."].forEach((t) => ul.appendChild(el("li", "", t)));
  c3.appendChild(ul);
  page.appendChild(c3);

  const c4 = card("Build");
  c4.appendChild(el("p", "small muted", `Model ${i.model_version}. Trained ${i.trained_at}. Features: ${i.features.length}. Data source: ${i.source}.`));
  if (i.version_mismatch && i.version_mismatch.length) c4.appendChild(el("div", "err", "Library versions differ from training: " + i.version_mismatch.join("; ")));
  page.appendChild(c4);
  return page;
}
