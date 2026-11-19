import { S, el, kpi, pageHead, card, seg, ctl, legend, f2, f3, pct, num } from "../common.js";
import { lineChart, barChart } from "../charts.js";

const NAMES = {
  prevalence: "Base rate (no model)", rule_recency: "Rule: days since last order", logreg_raw: "Logistic regression", logreg_cal: "Logistic regression, calibrated",
  tree_depth4: "Decision tree, depth 4", random_forest: "Random forest", mlp_challenger: "Neural net (MLP)", svm_rbf: "SVM (RBF kernel)", bgnbd: "BG/NBD probabilistic",
  lgbm_raw: "LightGBM, raw", lgbm_cal: "LightGBM, calibrated (chosen)",
};
const redraw = (host, fn) => { host.textContent = ""; host.appendChild(fn()); };

export function render() {
  const page = el("div", "page");
  const T = S.trust;
  page.appendChild(pageHead("Model evidence", "How far can this be trusted?",
    `Scored on ${num(T.test_rows)} customer-snapshots from three held-out months the model never saw. Intervals are 95% customer-clustered bootstrap.`));

  const m = T.metrics.lgbm_cal, ci = T.ci;
  const rule = T.metrics.rule_recency;
  const k = el("div", "kpis");
  kpi(k, "ROC-AUC", f3(m.roc_auc), `rule baseline ${f3(rule.roc_auc)}`);
  kpi(k, "PR-AUC", f3(m.pr_auc), `base rate ${f3(m.prevalence)}`);
  kpi(k, "Top-decile lift", f2(m.lift_top) + "x", `rule baseline ${f2(rule.lift_top)}x`);
  kpi(k, "Calibration slope", f2(m.cal_slope), "1.0 is perfect; below 1 means over-confident", m.cal_slope < 0.9 ? "bad" : "");
  kpi(k, "Expected calibration error", pct(m.ece, 1), "average gap, predicted vs observed");
  kpi(k, "Unseen-customer AUC", f3(T.customer_holdout.roc_auc), "customers never seen in training");
  page.appendChild(k);

  // ---- model comparison table with metric toggle
  const c1 = card("Model comparison", "Same test months, same customers. Pick a metric; the interval is the bootstrap 95% range where one was computed. The Validation column is the earlier period every model was compared on before the test months were scored.");
  const st = { metric: "roc_auc", sort: "value" };
  const host = el("div", "tablewrap");
  const MET = { roc_auc: ["ROC-AUC", f3], pr_auc: ["PR-AUC", f3], lift_top: ["Top-decile lift", f2], brier: ["Brier (lower is better)", f3] };
  const vcell = (key) => { const v = T.val_comparison && T.val_comparison[key] && T.val_comparison[key][st.metric]; return v === undefined || v === null ? "n/a" : MET[st.metric][1](v); };
  const draw1 = () => redraw(host, () => {
    const rows = Object.keys(NAMES).map((key) => ({ key, v: T.metrics[key] && T.metrics[key][st.metric], ci: ci[`${key}|${st.metric}`] })).filter((r) => r.v !== null && r.v !== undefined);
    rows.sort((a, b) => (st.metric === "brier" ? a.v - b.v : b.v - a.v));
    const lo = Math.min(...rows.map((r) => (r.ci ? r.ci[0] : r.v))), hi = Math.max(...rows.map((r) => (r.ci ? r.ci[1] : r.v)));
    const t = el("table");
    t.innerHTML = "<thead><tr><th>Model</th><th class='n'>Value</th><th class='n'>95% interval</th><th class='n'>Validation</th><th style='width:34%'>Range</th></tr></thead>";
    const b = el("tbody");
    rows.forEach((r) => {
      const tr = el("tr");
      const pos = (x) => ((x - lo) / (hi - lo || 1)) * 100;
      const chosen = r.key === "lgbm_cal";
      tr.innerHTML = `<td>${chosen ? "<b>" + NAMES[r.key] + "</b>" : NAMES[r.key]}</td><td class="n">${MET[st.metric][1](r.v)}</td><td class="n">${r.ci ? MET[st.metric][1](r.ci[0]) + " to " + MET[st.metric][1](r.ci[1]) : "n/a"}</td><td class="n">${vcell(r.key)}</td>
        <td><div style="position:relative;height:14px"><div style="position:absolute;left:${pos(r.ci ? r.ci[0] : r.v)}%;width:${Math.max(1, pos(r.ci ? r.ci[1] : r.v) - pos(r.ci ? r.ci[0] : r.v))}%;top:6px;height:3px;background:var(--muted)"></div>
        <div style="position:absolute;left:calc(${pos(r.v)}% - 4px);top:2px;width:9px;height:11px;background:${chosen ? "var(--signal)" : "var(--accent)"}"></div></div></td>`;
      b.appendChild(tr);
    });
    t.appendChild(b);
    return t;
  });
  const ctrl1 = el("div", "controls");
  ctrl1.appendChild(ctl("Metric", seg(Object.entries(MET).map(([v, a]) => [v, a[0]]), st.metric, (v) => { st.metric = v; draw1(); })));
  c1.appendChild(ctrl1);
  c1.appendChild(host);
  draw1();
  page.appendChild(c1);

  // ---- gains
  const g = el("div", "grid2");
  const c2 = card("Cumulative gains", "Call the top X% of customers by score: what share of all lapsers do you reach? Toggle the lines.");
  const G = T.gains;
  const lines = [["lgbm_cal", "LightGBM", "var(--signal)"], ["rule_recency", "Recency rule", "var(--accent)"], ["bgnbd", "BG/NBD", "var(--med)"]];
  const gs = lines.map(([k2, n, c]) => ({ name: n, color: c, data: G[k2], hidden: false }));
  gs.push({ name: "Random", color: "var(--muted)", dash: true, data: G.x });
  const gh = el("div");
  const draw2 = () => redraw(gh, () => lineChart({ half: true, labels: G.x.map((x) => Math.round(x * 100) + "%"), series: gs, yFmt: (v) => pct(v), yMin: 0, yMax: 1, xEvery: 20,
    yTitle: "share of lapsers reached", label: "Cumulative gains",
    tip: (i) => `<b>Top ${Math.round(G.x[i] * 100)}% by score</b><br>` + gs.filter((s) => !s.hidden).map((s) => `${s.name}: ${pct(s.data[i], 1)}`).join("<br>") }));
  c2.appendChild(legend(gs.map((s) => ({ label: s.name, color: s.color, dash: s.dash })), (i, on) => { gs[i].hidden = !on; draw2(); }));
  c2.appendChild(gh);
  draw2();
  g.appendChild(c2);

  // ---- reliability
  const c3 = card("Calibration: predicted vs observed", "When the model says 70%, do 70% lapse? Points on the diagonal are well calibrated.");
  const R = T.reliability;
  const rs = [{ key: "lgbm_cal", name: "Calibrated", color: "var(--signal)" }, { key: "lgbm_raw", name: "Raw", color: "var(--accent)" }];
  const rh = el("div");
  const rstate = { raw: false };
  const draw3 = () => redraw(rh, () => {
    const list = rstate.raw ? rs : [rs[0]];
    const base = R.lgbm_cal;
    const series = [{ name: "Perfect", color: "var(--muted)", dash: true, data: base.map((r) => r.predicted) }];
    list.forEach((s) => series.push({ name: s.name, color: s.color, data: R[s.key].map((r) => r.observed) }));
    return lineChart({ half: true, labels: base.map((r) => pct(r.predicted)), series, yFmt: (v) => pct(v), yMin: 0, yMax: 1, yTitle: "observed lapse rate", label: "Reliability",
      tip: (i) => `<b>Predicted ${pct(base[i].predicted, 0)}</b><br>` + list.map((s) => `${s.name}: observed ${pct(R[s.key][i].observed, 0)} (n=${num(R[s.key][i].n)})`).join("<br>") });
  });
  const ctrl3 = el("div", "controls");
  ctrl3.appendChild(ctl("Show", seg([["c", "Calibrated"], ["b", "Calibrated and raw"]], "c", (v) => { rstate.raw = v === "b"; draw3(); })));
  c3.appendChild(ctrl3);
  c3.appendChild(rh);
  draw3();
  g.appendChild(c3);
  page.appendChild(g);

  // ---- per snapshot stability
  const c4 = card("Stability across test months", "The test period is three month-ends. Pick a metric to see if it holds month to month.");
  const ps = T.per_snapshot, sk = { metric: "roc_auc" };
  const ph = el("div");
  const draw4 = () => redraw(ph, () => lineChart({ labels: ps.map((p) => p.snapshot), series: [{ name: MET[sk.metric][0], color: "var(--signal)", data: ps.map((p) => p[sk.metric]) }],
    yFmt: MET[sk.metric][1], yMin: sk.metric === "lift_top" ? 1 : 0, yMax: sk.metric === "lift_top" ? undefined : 1, label: "Metric per snapshot",
    tip: (i) => `<b>${ps[i].snapshot}</b><br>${MET[sk.metric][0]}: ${MET[sk.metric][1](ps[i][sk.metric])}<br>${num(ps[i].n)} customers, base rate ${pct(ps[i].prevalence)}` }));
  const ctrl4 = el("div", "controls");
  ctrl4.appendChild(ctl("Metric", seg(Object.entries(MET).map(([v, a]) => [v, a[0]]), sk.metric, (v) => { sk.metric = v; draw4(); })));
  c4.appendChild(ctrl4);
  c4.appendChild(ph);
  draw4();
  page.appendChild(c4);

  // ---- importance + drift
  const g2 = el("div", "grid2");
  const LAB = Object.fromEntries(Object.entries(S.eda.features).map(([f, d]) => [f, d.label]));
  const c5 = card("What drives the score", "Drop in test ROC-AUC when a feature is shuffled (permutation importance, mean over repeats).");
  c5.appendChild(barChart({ half: true, items: T.importance.slice(0, 10).map((r) => ({ label: LAB[r.feature] || r.feature, value: r.auc_drop, color: "var(--signal)",
    tip: `<b>${LAB[r.feature] || r.feature}</b><br>AUC drop ${f3(r.auc_drop)} (sd ${f3(r.auc_drop_std)})` })), fmt: f3, labelW: 210, label: "Permutation importance" }));
  g2.appendChild(c5);
  const c6 = card("Drift between train and test", "Population stability index. Above 0.25 is a major shift; much of it is structural (customers age, tenure grows).");
  c6.appendChild(barChart({ half: true, items: T.drift.slice(0, 10).map((r) => ({ label: LAB[r.feature] || r.feature, value: r.psi, color: r.psi > 0.25 ? "var(--high)" : r.psi > 0.1 ? "var(--med)" : "var(--low)",
    tip: `<b>${LAB[r.feature] || r.feature}</b><br>PSI ${f2(r.psi)} (${r.status})` })), fmt: f2, labelW: 210, label: "PSI per feature" }));
  g2.appendChild(c6);
  page.appendChild(g2);

  // ---- segments
  const c7 = card("Who is it weaker for?", "Performance by customer segment on the test months. Small groups are noisy.");
  const sw = el("div", "tablewrap"), t = el("table");
  t.innerHTML = "<thead><tr><th>Dimension</th><th>Group</th><th class='n'>Rows</th><th class='n'>Lapse rate</th><th class='n'>Predicted</th><th class='n'>ROC-AUC</th><th class='n'>Lift</th></tr></thead>";
  const tb = el("tbody");
  T.segments.forEach((r) => {
    const tr = el("tr");
    tr.innerHTML = `<td>${r.dimension}</td><td>${r.group}</td><td class="n">${num(r.n)}</td><td class="n">${pct(r.inactivity_rate)}</td><td class="n">${pct(r.mean_predicted)}</td><td class="n">${f3(r.roc_auc)}</td><td class="n">${f2(r.lift_top)}x</td>`;
    tb.appendChild(tr);
  });
  t.appendChild(tb); sw.appendChild(t); c7.appendChild(sw);
  page.appendChild(c7);

  // ---- sensitivity
  const c8 = card("Sensitivity to the label definition", "The same model family re-run with a different horizon or label rule.");
  const sw2 = el("div", "tablewrap"), t2 = el("table");
  t2.innerHTML = "<thead><tr><th>Variant</th><th class='n'>Rows</th><th class='n'>Base rate</th><th class='n'>ROC-AUC</th><th class='n'>PR-AUC</th><th class='n'>Lift</th></tr></thead>";
  const tb2 = el("tbody");
  Object.entries(T.sensitivity).forEach(([name, v]) => {
    const tr = el("tr");
    tr.innerHTML = `<td>${name}</td><td class="n">${num(v.sizes.rows)}</td><td class="n">${pct(v.sizes.prevalence)}</td><td class="n">${f3(v.metrics.roc_auc)}</td><td class="n">${f3(v.metrics.pr_auc)}</td><td class="n">${f2(v.metrics.lift_top)}x</td>`;
    tb2.appendChild(tr);
  });
  t2.appendChild(tb2); sw2.appendChild(t2); c8.appendChild(sw2);
  page.appendChild(c8);
  return page;
}
