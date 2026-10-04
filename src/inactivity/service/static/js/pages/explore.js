import { S, el, kpi, pageHead, card, seg, ctl, select, checkbox, legend, gbp, gbpShort, num, pct, monthLabel } from "../common.js";
import { lineChart, barChart, columnChart, heatmap } from "../charts.js";

const BLOCK_COLOR = { train: "var(--accent)", val: "var(--med)", cal: "var(--low)", test: "var(--signal)", dropped: "var(--bar)" };

function redraw(host, fn) {
  host.textContent = "";
  host.appendChild(fn());
}

export function render() {
  const page = el("div", "page");
  const eda = S.eda, q = eda.quality;
  page.appendChild(pageHead("Explore data", "What the data looks like",
    "Aggregates over the cleaned UCI Online Retail II transactions. Hover any chart for values; use the controls above each chart."));

  const k = el("div", "kpis");
  kpi(k, "Raw lines", num(q.rows_raw), "as published");
  kpi(k, "Duplicates dropped", num(q.duplicate_rows_dropped), "exact repeat rows");
  kpi(k, "No customer ID", pct(q.missing_customer_id_share, 1), "cannot be tracked, excluded", "bad");
  kpi(k, "Clean lines", num(q.rows_clean), `${num(q.customers)} customers`);
  kpi(k, "Return lines kept", num(q.cancellation_return_lines_kept), "used as negative revenue");
  kpi(k, "Revenue", gbpShort(eda.lorenz.revenue), gbp(eda.lorenz.revenue) + " lifetime, before returns");
  page.appendChild(k);

  // ---- activity over time
  const c1 = card("Activity over time", "Monthly totals across the whole period. Pick a measure and narrow the window.");
  const mo = eda.monthly;
  const MEAS = {
    revenue: ["Revenue", gbpShort, "var(--accent)"], orders: ["Orders", num, "var(--accent)"], customers: ["Active customers", num, "var(--accent)"],
    new_customers: ["New customers", num, "var(--low)"], returns: ["Returns value", gbpShort, "var(--high)"],
  };
  const st = { m: "revenue", from: 0, to: mo.length - 1 };
  const chart1 = el("div");
  const draw1 = () => redraw(chart1, () => {
    const slice = mo.slice(st.from, st.to + 1), [name, fmt, color] = MEAS[st.m];
    return lineChart({ labels: slice.map((r) => monthLabel(r.month)), series: [{ name, color, data: slice.map((r) => r[st.m]) }],
      yFmt: fmt, yMin: 0, xEvery: Math.max(1, Math.round(slice.length / 8)), label: name + " by month" });
  });
  const monthOpts = mo.map((r, i) => [String(i), monthLabel(r.month)]);
  const ctrl1 = el("div", "controls");
  ctrl1.appendChild(ctl("Measure", seg(Object.entries(MEAS).map(([v, a]) => [v, a[0]]), st.m, (v) => { st.m = v; draw1(); })));
  const fromSel = select(monthOpts, "0", (v) => { st.from = +v; if (st.to < st.from) { st.to = st.from; toSel.value = String(st.to); } draw1(); });
  const toSel = select(monthOpts, String(mo.length - 1), (v) => { st.to = +v; if (st.from > st.to) { st.from = st.to; fromSel.value = String(st.from); } draw1(); });
  ctrl1.appendChild(ctl("From month", fromSel));
  ctrl1.appendChild(ctl("To month", toSel));
  c1.appendChild(ctrl1);
  c1.appendChild(chart1);
  draw1();
  page.appendChild(c1);

  // ---- concentration + countries
  const g = el("div", "grid2");
  const L = eda.lorenz;
  const c2 = card("Value concentration", `The top 10% of customers bring ${pct(L.top10)} of revenue; the top 1% bring ${pct(L.top1)}. This is why the worklist weighs spend.`);
  c2.appendChild(lineChart({ half: true, labels: L.x.map((x) => Math.round(x * 100) + "%"), series: [
    { name: "Cumulative share of revenue", color: "var(--signal)", data: L.y }, { name: "Equal spend", color: "var(--muted)", dash: true, data: L.x }],
  yFmt: (v) => pct(v), yMin: 0, yMax: 1, xEvery: 20, yTitle: "revenue share, customers ranked low to high",
  tip: (i) => `<b>Bottom ${Math.round(L.x[i] * 100)}% of customers</b><br>${pct(L.y[i], 1)} of revenue`, label: "Lorenz curve" }));
  g.appendChild(c2);
  const c3 = card("Revenue by country", "Top ten countries by lifetime revenue.");
  const cs = eda.countries;
  c3.appendChild(barChart({ half: true, items: cs.map((c) => ({ label: c.country, value: c.revenue, color: c.country === "United Kingdom" ? "var(--accent)" : "var(--signal)",
    tip: `<b>${c.country}</b><br>${gbp(c.revenue)}<br>${num(c.customers)} customers, ${num(c.orders)} orders` })),
  fmt: gbp, max: Math.max(...cs.map((c) => c.revenue)) * 1.05, labelW: 120, label: "Revenue by country" }));
  g.appendChild(c3);
  page.appendChild(g);

  // ---- distributions
  const c4 = card("Distributions", "How customers and orders spread. Heavy right tails are why money and counts are log-transformed for the model.");
  const D = eda.distributions, dk = Object.keys(D), ds = { key: dk[0], log: false };
  const chart4 = el("div");
  const draw4 = () => redraw(chart4, () => {
    const d = D[ds.key];
    return columnChart({ labels: d.labels, series: [{ name: d.unit, color: "var(--accent)", values: d.counts }], log: ds.log, yFmt: num,
      tip: (i) => `<b>${d.labels[i]}</b><br>${num(d.counts[i])} ${d.unit} (${pct(d.counts[i] / d.total, 1)})`, label: d.title });
  });
  const ctrl4 = el("div", "controls");
  ctrl4.appendChild(ctl("Distribution", select(dk.map((x) => [x, D[x].title]), ds.key, (v) => { ds.key = v; draw4(); })));
  ctrl4.appendChild(ctl("Scale", checkbox("Log axis", false, (v) => { ds.log = v; draw4(); })));
  c4.appendChild(ctrl4);
  c4.appendChild(chart4);
  draw4();
  page.appendChild(c4);

  // ---- lapse by snapshot, coloured by block
  const c5 = card("Lapse rate by snapshot", "Share of eligible customers who then placed no order in 90 days, one bar per month-end. Colour is the data block; grey snapshots were dropped by the label-maturity embargo.");
  const sn = eda.snapshots, s5 = { seg: false };
  const chart5 = el("div");
  const draw5 = () => redraw(chart5, () => {
    const val = (s) => (s5.seg ? s.lapse_seg : s.lapse);
    const series = Object.keys(BLOCK_COLOR).map((b) => ({ name: b, color: BLOCK_COLOR[b], values: sn.map((s) => (s.block === b ? val(s) : 0)) }));
    return columnChart({ labels: sn.map((s) => monthLabel(s.date)), series, mode: "stack", yFmt: (v) => pct(v),
      tip: (i) => `<b>${monthLabel(sn[i].date)}</b> (${sn[i].block})<br>fixed-label lapse ${pct(sn[i].lapse, 1)}<br>segment-aware ${pct(sn[i].lapse_seg, 1)}<br>${num(sn[i].eligible)} eligible`,
      label: "Lapse rate by snapshot" });
  });
  const ctrl5 = el("div", "controls");
  ctrl5.appendChild(ctl("Label", seg([["f", "Fixed 90 days"], ["s", "Segment-aware"]], "f", (v) => { s5.seg = v === "s"; draw5(); })));
  c5.appendChild(ctrl5);
  c5.appendChild(chart5);
  c5.appendChild(legend(Object.entries(BLOCK_COLOR).map(([n, c]) => ({ label: n, color: c, square: true }))));
  draw5();
  page.appendChild(c5);

  // ---- feature explorer
  const F = eda.features, fk = Object.keys(F);
  const rank = fk.map((f) => {
    const d = F[f].sets.all;
    const rates = d.n0.map((n0, i) => (n0 + d.n1[i] >= 100 ? d.n1[i] / (n0 + d.n1[i]) : null)).filter((v) => v !== null);
    return { f, spread: rates.length > 1 ? Math.max(...rates) - Math.min(...rates) : 0 };
  }).sort((a, b) => b.spread - a.spread);

  const c6 = card("Feature explorer", "How the lapse rate changes across the range of each model input. Bars are customer-snapshots; the line is the share that lapsed.");
  const fs = { f: rank[0].f, set: "all" };
  const chart6 = el("div");
  const draw6 = () => redraw(chart6, () => {
    const d = F[fs.f], s = d.sets[fs.set];
    const tot = s.n0.map((v, i) => v + s.n1[i]);
    const rate = tot.map((t, i) => (t >= 30 ? s.n1[i] / t : null));
    const m = (v) => (v === null || v === undefined ? "n/a" : String(Math.round(v * 100) / 100));
    const wrap = el("div");
    wrap.appendChild(columnChart({ labels: d.bins, series: [{ name: "Stayed active", color: "var(--low)", values: s.n0 }, { name: "Lapsed", color: "var(--high)", values: s.n1 }],
      line: { name: "Lapse rate", color: "var(--text)", values: rate, fmt: pct }, mode: "stack", yFmt: num,
      tip: (i) => `<b>${d.label}: ${d.bins[i]}</b><br>${num(tot[i])} rows<br>lapse rate ${rate[i] === null ? "too few rows" : pct(rate[i], 1)}`, label: d.label }));
    wrap.appendChild(el("p", "small muted", `Median for customers who stayed: ${m(d.median_active)}. Median for customers who lapsed: ${m(d.median_lapsed)}.`));
    return wrap;
  });
  const ctrl6 = el("div", "controls");
  const featSel = select(rank.map((r) => [r.f, F[r.f].label]), fs.f, (v) => { fs.f = v; draw6(); });
  const pick = (f) => { fs.f = f; featSel.value = f; draw6(); c6.scrollIntoView({ behavior: "smooth", block: "start" }); };
  ctrl6.appendChild(ctl("Feature", featSel));
  ctrl6.appendChild(ctl("Period", select([["all", "All blocks"], ["train", "Train"], ["val", "Validation"], ["cal", "Calibration"], ["test", "Test"]], "all", (v) => { fs.set = v; draw6(); })));
  c6.appendChild(ctrl6);
  c6.appendChild(chart6);
  page.appendChild(c6);
  draw6();

  const c7 = card("Which features separate lapsers best?", "Spread between the highest and lowest lapse rate across bins with at least 100 rows. Click a row to open it in the explorer above.");
  const bars = barChart({ items: rank.map((r) => ({ label: F[r.f].label, value: r.spread, color: "var(--signal)", tip: `<b>${F[r.f].label}</b><br>spread ${pct(r.spread, 0)} points` })),
    fmt: (v) => pct(v), labelW: 210, label: "Feature lapse-rate spread" });
  bars.querySelectorAll("rect.hit").forEach((n, i) => { n.style.cursor = "pointer"; n.addEventListener("click", () => pick(rank[i].f)); });
  c7.appendChild(bars);
  page.appendChild(c7);

  // ---- correlation
  const C = eda.correlation;
  const c8 = card("Feature correlations", "Spearman rank correlation between the model inputs. Strong overlap is expected because the windows share orders; click a row label to explore that feature.");
  const hm = heatmap({ labels: C.labels, matrix: C.matrix, label: "Correlation matrix", onRow: (i) => pick(C.features[i]) });
  const sl = el("input");
  sl.type = "range"; sl.min = 0; sl.max = 0.95; sl.step = 0.05; sl.value = 0;
  const out = el("span", "small muted", " showing all");
  sl.addEventListener("input", () => { hm.setMin(+sl.value); out.textContent = +sl.value === 0 ? " showing all" : " only |rho| of " + (+sl.value).toFixed(2) + " or more"; });
  const sw = el("div");
  sw.appendChild(sl);
  sw.appendChild(out);
  const ctrl8 = el("div", "controls");
  ctrl8.appendChild(ctl("Hide weak links", sw));
  c8.appendChild(ctrl8);
  c8.appendChild(hm);
  page.appendChild(c8);
  return page;
}
