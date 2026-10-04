import { S, el, sv, kpi, pageHead, card, api, pct, gbp, num, bandClass, tipShow, tipHide } from "../common.js";
import { barChart } from "../charts.js";

const day = 86400000;
const pctRank = (arr, v) => arr.filter((x) => x <= v).length / arr.length;

function timeline(c, snap) {
  const w = 720, h = 150, L = 20, R = 20, B = 30;
  const past = c.past_orders.map(([d, v]) => ({ d: new Date(d).getTime(), v, past: true }));
  const fut = c.future_orders.map(([d, v]) => ({ d: new Date(d).getTime(), v, past: false }));
  const all = past.concat(fut);
  const s0 = new Date(snap).getTime();
  const lo = Math.min(...all.map((o) => o.d), s0 - 30 * day), hi = Math.max(...all.map((o) => o.d), s0 + 90 * day);
  const X = (t) => L + ((t - lo) / (hi - lo)) * (w - L - R);
  const vmax = Math.max(1, ...all.map((o) => Math.max(0, o.v)));
  const Hh = (v) => 4 + (Math.sqrt(Math.max(0, v)) / Math.sqrt(vmax)) * (h - B - 14);
  const svg = sv("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": "Order timeline", width: "100%" });
  svg.appendChild(sv("rect", { x: X(s0), y: 0, width: X(s0 + 90 * day) - X(s0), height: h - B, style: "fill:var(--bar);opacity:.5" }));
  svg.appendChild(sv("text", { x: X(s0) + 6, y: 14 }, "next 90 days (hindsight)"));
  svg.appendChild(sv("line", { x1: L, x2: w - R, y1: h - B, y2: h - B, class: "axis" }));
  svg.appendChild(sv("line", { x1: X(s0), x2: X(s0), y1: 0, y2: h - B, style: "stroke:var(--signal);stroke-width:2" }));
  svg.appendChild(sv("text", { x: X(s0) - 6, y: 14, "text-anchor": "end", class: "strong" }, "snapshot"));
  all.forEach((o) => {
    const r = sv("rect", { x: X(o.d) - 2, y: h - B - Hh(o.v), width: 4, height: Hh(o.v), style: `fill:${o.past ? "var(--accent)" : "var(--signal)"}` });
    r.addEventListener("mousemove", (e) => tipShow(`<b>${new Date(o.d).toISOString().slice(0, 10)}</b><br>order value ${gbp(o.v)}`, e.clientX, e.clientY));
    r.addEventListener("mouseleave", tipHide);
    svg.appendChild(r);
  });
  [lo, s0, hi].forEach((t, i) => svg.appendChild(sv("text", { x: X(t), y: h - 10, "text-anchor": i === 0 ? "start" : i === 2 ? "end" : "middle" }, new Date(t).toISOString().slice(0, 10))));
  return svg;
}

export function render(id) {
  const page = el("div", "page");
  const rows = S.work.customers;
  const c = rows.find((r) => String(r.customer_id) === String(id));
  if (!c) {
    page.appendChild(pageHead("Customer", "Customer not found", `No customer ${id} in this worklist.`));
    const a = el("a", "btn secondary", "Back to the worklist"); a.href = "#/worklist"; page.appendChild(a);
    return page;
  }
  page.appendChild(pageHead("Customer " + c.customer_id, `${c.band} risk of going quiet`, `Snapshot ${S.work.snapshot_date}. Next 90 days shown as hindsight.`));
  const back = el("a", "small", "← Back to the worklist"); back.href = "#/worklist"; page.appendChild(back);

  const k = el("div", "kpis");
  kpi(k, "Predicted risk", pct(c.p, 1), `${c.band} band`, c.band === "High" ? "bad" : c.band === "Low" ? "good" : "");
  kpi(k, "Rank by risk", "#" + (rows.slice().sort((a, b) => b.p - a.p).findIndex((r) => r.customer_id === c.customer_id) + 1), `of ${num(rows.length)}`);
  kpi(k, "Spend, 180 days", gbp(c.spend_180), `${pct(pctRank(rows.map((r) => r.spend_180), c.spend_180))} of customers spend this or less`);
  kpi(k, "Orders, 180 days", String(c.orders_180), "");
  kpi(k, "Days since last order", String(c.recency_days), "");
  kpi(k, "What happened", c.inactive ? "Went quiet" : "Ordered again", `${c.future_orders.length} orders in the next 90 days`, c.inactive ? "bad" : "good");
  page.appendChild(k);

  const g = el("div", "grid2");
  const c1 = card("Why the model says so", "Top reasons raising this customer's risk (TreeSHAP contribution).");
  const ul = el("ul", "reasons");
  (c.reasons.length ? c.reasons : ["No single feature stands out."]).forEach((r) => ul.appendChild(el("li", "", r.replace(" (raises inactivity risk)", ""))));
  c1.appendChild(ul);
  const meter = el("div", "meter");
  const fill = el("div", "fill"); fill.style.width = pct(c.p); meter.appendChild(fill);
  [S.work.thresholds.medium, S.work.thresholds.high].forEach((t) => { const tk = el("div", "tick"); tk.style.left = pct(t); meter.appendChild(tk); });
  c1.appendChild(meter);
  c1.appendChild(el("p", "small muted", `Ticks mark the Medium (${pct(S.work.thresholds.medium)}) and High (${pct(S.work.thresholds.high)}) thresholds.`));
  g.appendChild(c1);

  const c2 = card("How this customer compares", "Percentile among everyone on the worklist.");
  const f = c.features, all = rows.map((r) => r.features);
  const items = [["Days since last order", "recency_days"], ["Orders in last 90 days", "inv_count_90"], ["Orders in last 180 days", "inv_count_180"], ["Purchase days ever", "n_purchase_days"], ["Days since first order", "tenure_days"]]
    .filter(([, key]) => f[key] !== null && f[key] !== undefined)
    .map(([label, key]) => { const p = pctRank(all.map((x) => x[key] ?? 0), f[key]); return { label, value: p, color: "var(--accent)", tip: `<b>${label}</b><br>value ${Math.round(f[key])}<br>higher than ${pct(p)} of customers` }; });
  items.push({ label: "Predicted risk", value: pctRank(rows.map((r) => r.p), c.p), color: "var(--signal)", tip: "Percentile of predicted risk" });
  c2.appendChild(barChart({ half: true, items, fmt: (v) => Math.round(v * 100) + "th", max: 1, labelW: 170, label: "Percentiles" }));
  g.appendChild(c2);
  page.appendChild(g);

  const c3 = card("Order timeline", "Plain bars are orders before the snapshot, red bars are what the customer actually did next. Bar height is order value (square-root scale).");
  c3.appendChild(timeline(c, S.work.snapshot_date));
  page.appendChild(c3);

  // ---- what-if
  const c4 = card("What if?", "Move the sliders and the model re-scores this customer live. Everything else stays as it is.");
  const base = { recency_days: f.recency_days, inv_count_90: f.inv_count_90, inv_count_180: f.inv_count_180, rev_90: Math.expm1(f.rev_90_log), rev_180: Math.expm1(f.rev_180_log), median: f.median_interval_days };
  const st = { ...base };
  const out = el("div", "callout", "Scoring...");
  const defs = [["recency_days", "Days since last order", 0, 179, 1, (v) => v + " days"], ["inv_count_90", "Orders in last 90 days", 0, 20, 1, (v) => v], ["inv_count_180", "Orders in last 180 days", 0, 40, 1, (v) => v],
    ["rev_90", "Spend in last 90 days", 0, Math.max(5000, Math.round(base.rev_90 * 2)), 10, gbp], ["rev_180", "Spend in last 180 days", 0, Math.max(10000, Math.round(base.rev_180 * 2)), 10, gbp]];
  if (base.median !== null && base.median !== undefined) defs.push(["median", "Usual days between orders", 1, 180, 1, (v) => v + " days"]);
  let timer;
  const payload = () => {
    const nf = { ...f };
    nf.recency_days = st.recency_days; nf.inv_count_90 = st.inv_count_90; nf.inv_count_180 = Math.max(st.inv_count_180, st.inv_count_90);
    nf.inv_count_90_prev = nf.inv_count_180 - nf.inv_count_90; nf.freq_trend = nf.inv_count_90 - nf.inv_count_90_prev;
    nf.rev_90_log = Math.log1p(st.rev_90); nf.rev_180_log = Math.log1p(Math.max(st.rev_180, st.rev_90));
    nf.rev_trend = nf.rev_90_log - Math.log1p(Math.max(0, Math.max(st.rev_180, st.rev_90) - st.rev_90));
    if (st.median !== null && st.median !== undefined) { nf.median_interval_days = st.median; nf.recency_to_interval = st.recency_days / st.median; }
    return nf;
  };
  const score = async () => {
    try {
      const r = await api("/score", { customer_id: String(c.customer_id), snapshot_date: S.work.snapshot_date, features: payload() });
      out.textContent = "";
      if (!r.eligible) { out.textContent = "At this many days since the last order the customer would no longer count as active, so no prediction is made."; return; }
      const d = r.probability_inactive - c.p;
      out.appendChild(el("span", "big", pct(r.probability_inactive, 1)));
      out.appendChild(el("span", `band ${bandClass(r.risk_band)}`, " " + r.risk_band + " "));
      out.appendChild(el("p", "small muted", `${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)} points vs the actual risk of ${pct(c.p, 1)}.`));
    } catch (e) { out.textContent = "Could not score: " + e.message; }
  };
  const ctrls = el("div", "controls");
  defs.forEach(([key, label, min, max, step, fmt]) => {
    const r = el("input"); r.type = "range"; r.min = min; r.max = max; r.step = step; r.value = st[key];
    const o = el("span", "small muted", " " + fmt(st[key]));
    r.addEventListener("input", () => { st[key] = +r.value; o.textContent = " " + fmt(st[key]); clearTimeout(timer); timer = setTimeout(score, 120); });
    const w = el("div"); w.appendChild(r); w.appendChild(o);
    const cc = el("div", "ctl"); cc.appendChild(el("div", "lab", label)); cc.appendChild(w); ctrls.appendChild(cc);
    r.dataset.key = key;
  });
  const reset = el("button", "btn secondary", "Reset"); reset.type = "button";
  reset.addEventListener("click", () => { Object.assign(st, base); ctrls.querySelectorAll("input[type=range]").forEach((r) => { r.value = st[r.dataset.key]; r.dispatchEvent(new Event("input")); }); });
  c4.appendChild(ctrls); c4.appendChild(reset); c4.appendChild(out);
  page.appendChild(c4);
  score();
  return page;
}
