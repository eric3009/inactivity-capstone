import { S, el, kpi, pageHead, card, f2, pct, gbp, num, bandClass, mean, legend } from "../common.js";
import { columnChart } from "../charts.js";

export function render() {
  const page = el("div", "page");
  const { work, trust, eda } = S;
  const rows = work.customers;
  const t = trust.thresholds;
  page.appendChild(pageHead("Overview", "Who is about to go quiet?",
    `Every customer who bought in the last ${trust.activity_window_days} days, scored for the chance of placing no order in the next ${trust.horizon_days} days. Snapshot ${work.snapshot_date}.`));

  const hi = rows.filter((r) => r.band === "High"), med = rows.filter((r) => r.band === "Medium");
  const atRisk = rows.filter((r) => r.band !== "Low");
  const spendAt = atRisk.reduce((a, r) => a + r.spend_180, 0), spendAll = rows.reduce((a, r) => a + r.spend_180, 0);
  const m = trust.metrics.lgbm_cal, ci = trust.ci;
  const realised = mean(rows.map((r) => r.inactive));
  const predicted = mean(rows.map((r) => r.p));

  const k = el("div", "kpis");
  kpi(k, "Customers scored", num(rows.length), `eligible at ${work.snapshot_date}`);
  kpi(k, "High risk", num(hi.length), `p at or above ${pct(t.high)}`, "bad");
  kpi(k, "Medium risk", num(med.length), `p ${pct(t.medium)} to ${pct(t.high)}`);
  kpi(k, "Spend exposed", pct(spendAt / spendAll), `${gbp(spendAt)} of ${gbp(spendAll)} in the last 180 days sits with High and Medium`, "bad");
  kpi(k, "Predicted lapse rate", pct(predicted), "mean predicted probability");
  kpi(k, "Actual lapse rate", pct(realised), "what happened in the next 90 days");
  kpi(k, "ROC-AUC", f2(m.roc_auc), `95% CI ${f2(ci["lgbm_cal|roc_auc"][0])} to ${f2(ci["lgbm_cal|roc_auc"][1])}`);
  kpi(k, "PR-AUC", f2(m.pr_auc), `base rate ${f2(m.prevalence)}`);
  kpi(k, "Top-decile lift", f2(m.lift_top) + "x", "vs picking customers at random");
  kpi(k, "Brier", f2(m.brier), "lower is better; 0.25 = no skill");
  kpi(k, "Training window", `${eda.snapshots.length} snapshots`, `${eda.rows_modelling.toLocaleString()} modelling rows`);
  kpi(k, "Customers in data", num(eda.quality.customers), `${eda.quality.date_min.slice(0, 10)} to ${eda.quality.date_max.slice(0, 10)}`);
  page.appendChild(k);

  const g = el("div", "grid2");
  const c1 = card("Distribution of predicted risk", "Number of customers per probability bin, coloured by band. Hover for counts.");
  const bins = 20, labels = [], cnt = { High: Array(bins).fill(0), Medium: Array(bins).fill(0), Low: Array(bins).fill(0) };
  for (let i = 0; i < bins; i++) labels.push(`${Math.round((i * 100) / bins)}-${Math.round(((i + 1) * 100) / bins)}%`);
  rows.forEach((r) => { cnt[r.band][Math.min(bins - 1, Math.floor(r.p * bins))]++; });
  c1.appendChild(columnChart({ half: true, labels, series: [
    { name: "Low", color: "var(--low)", values: cnt.Low }, { name: "Medium", color: "var(--med)", values: cnt.Medium }, { name: "High", color: "var(--high)", values: cnt.High }],
    label: "Histogram of predicted inactivity probability", xRotate: true }));
  c1.appendChild(legend([{ label: "Low", color: "var(--low)", square: true }, { label: "Medium", color: "var(--med)", square: true }, { label: "High", color: "var(--high)", square: true }]));
  g.appendChild(c1);

  const c2 = card("Top at-risk customers by spend", "Highest 180-day spend among High and Medium risk. Click a row for the customer page.");
  const top = atRisk.slice().sort((a, b) => b.spend_180 - a.spend_180).slice(0, 8);
  const wrap = el("div", "tablewrap"), tb = el("table");
  tb.innerHTML = "<thead><tr><th>Customer</th><th>Risk</th><th class='n'>Spend 180d</th><th class='n'>Days since</th></tr></thead>";
  const body = el("tbody");
  top.forEach((r) => {
    const tr = el("tr", "cust"); tr.tabIndex = 0;
    const go = () => (location.hash = "#/customer/" + encodeURIComponent(r.customer_id));
    tr.addEventListener("click", go); tr.addEventListener("keydown", (e) => e.key === "Enter" && go());
    tr.innerHTML = `<td>${r.customer_id}</td><td><span class="band ${bandClass(r.band)}">${r.band} ${pct(r.p)}</span></td><td class="n">${gbp(r.spend_180)}</td><td class="n">${r.recency_days}</td>`;
    body.appendChild(tr);
  });
  tb.appendChild(body); wrap.appendChild(tb); c2.appendChild(wrap);
  g.appendChild(c2);
  page.appendChild(g);

  const lc = el("div", "linkcards");
  [["#/explore", "Explore the data", "Activity over time, who generates the revenue, how each feature relates to lapsing."],
   ["#/model", "Model evidence", "Metrics with confidence intervals, calibration, drift and segment results."],
   ["#/worklist", "Retention worklist", "Filter, sort and export the ranked list; test a call capacity."],
   ["#/try", "Try it", "Score a customer from features or raw order lines."]].forEach(([h, b, s]) => {
    const a = el("a", "linkcard"); a.href = h; a.appendChild(el("b", "", b)); a.appendChild(el("span", "", s)); lc.appendChild(a);
  });
  page.appendChild(lc);
  return page;
}
