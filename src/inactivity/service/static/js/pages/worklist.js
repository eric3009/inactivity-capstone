import { S, el, kpi, pageHead, card, seg, ctl, select, legend, gbp, gbpShort, num, pct, bandClass, downloadCsv, mean } from "../common.js";
import { scatterChart } from "../charts.js";

const COLORS = { High: "var(--high)", Medium: "var(--med)", Low: "var(--low)" };

export function render() {
  const page = el("div", "page");
  const W = S.work, rows = W.customers.slice();
  page.appendChild(pageHead("Retention worklist", "Who to call first",
    `${num(rows.length)} eligible customers at ${W.snapshot_date}, with what actually happened afterwards shown for the back-test. Filter, sort, click a row to open the customer.`));

  const st = { band: "all", q: "", minSpend: 0, maxRec: 999, sort: "p", dir: -1, cap: 100, selected: null };
  const maxSpend = Math.max(...rows.map((r) => r.spend_180));

  // ---- controls
  const ctrls = el("div", "controls");
  ctrls.appendChild(ctl("Risk band", seg([["all", "All"], ["High", "High"], ["Medium", "Medium"], ["Low", "Low"]], "all", (v) => { st.band = v; update(); })));
  const search = el("input"); search.type = "search"; search.placeholder = "Customer ID";
  search.addEventListener("input", () => { st.q = search.value.trim(); update(); });
  ctrls.appendChild(ctl("Search", search));
  const ms = el("input"); ms.type = "range"; ms.min = 0; ms.max = Math.round(Math.min(maxSpend, 5000)); ms.step = 50; ms.value = 0;
  const msOut = el("span", "small muted", " any");
  ms.addEventListener("input", () => { st.minSpend = +ms.value; msOut.textContent = st.minSpend ? " " + gbp(st.minSpend) + "+" : " any"; update(); });
  const msw = el("div"); msw.appendChild(ms); msw.appendChild(msOut);
  ctrls.appendChild(ctl("Min spend, 180d", msw));
  const rc = el("input"); rc.type = "range"; rc.min = 10; rc.max = 179; rc.step = 1; rc.value = 179;
  const rcOut = el("span", "small muted", " any");
  rc.addEventListener("input", () => { st.maxRec = +rc.value; rcOut.textContent = st.maxRec >= 179 ? " any" : " up to " + st.maxRec + " days"; update(); });
  const rcw = el("div"); rcw.appendChild(rc); rcw.appendChild(rcOut);
  ctrls.appendChild(ctl("Days since last order", rcw));
  const exp = el("button", "btn secondary", "Export CSV"); exp.type = "button";
  ctrls.appendChild(ctl("Export", exp));
  page.appendChild(ctrls);

  const kp = el("div", "kpis");
  page.appendChild(kp);

  // ---- scatter
  const c1 = card("Risk against spend", "Each dot is a customer. Top-right is where retention effort pays most. Click a dot to find it in the table.");
  const sc = el("div");
  c1.appendChild(sc);
  c1.appendChild(legend(Object.entries(COLORS).map(([n, c]) => ({ label: n, color: c, square: true }))));
  page.appendChild(c1);

  // ---- capacity back-test
  const c2 = card("If you can only make N calls", "Pick the top N customers by score. The table shows how many of them really went quiet, and how that compares with picking at random.");
  const capHost = el("div");
  const capCtl = el("div", "controls");
  const cap = el("input"); cap.type = "range"; cap.min = 10; cap.max = Math.min(1000, rows.length); cap.step = 10; cap.value = st.cap;
  const capOut = el("span", "small muted", " " + st.cap + " calls");
  cap.addEventListener("input", () => { st.cap = +cap.value; capOut.textContent = " " + st.cap + " calls"; drawCap(); });
  const cw = el("div"); cw.appendChild(cap); cw.appendChild(capOut);
  capCtl.appendChild(ctl("Calls available", cw));
  c2.appendChild(capCtl); c2.appendChild(capHost);
  page.appendChild(c2);

  // ---- table
  const c3 = card("Customers");
  const wrap = el("div", "tablewrap"), t = el("table"), thead = el("thead"), tbody = el("tbody");
  const COLS = [["customer_id", "Customer", ""], ["p", "Risk", ""], ["spend_180", "Spend 180d", "n"], ["orders_180", "Orders 180d", "n"], ["recency_days", "Days since", "n"], ["inactive", "Went quiet?", ""]];
  const trh = el("tr");
  COLS.forEach(([key, label, cls]) => {
    const th = el("th", "sortable" + (cls ? " " + cls : ""), label);
    th.tabIndex = 0;
    const go = () => { if (st.sort === key) st.dir = -st.dir; else { st.sort = key; st.dir = key === "customer_id" ? 1 : -1; } update(); };
    th.addEventListener("click", go); th.addEventListener("keydown", (e) => e.key === "Enter" && go());
    th.dataset.key = key;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  const reasonsTh = el("th", "", "Main reason"); trh.appendChild(reasonsTh);
  t.appendChild(thead); t.appendChild(tbody); wrap.appendChild(t);
  const more = el("button", "btn secondary", "Show more"); more.type = "button";
  const count = el("p", "small muted");
  c3.appendChild(count); c3.appendChild(wrap); c3.appendChild(more);
  page.appendChild(c3);
  let shown = 50;
  more.addEventListener("click", () => { shown += 100; drawTable(); });

  let view = [];
  function filtered() {
    return rows.filter((r) => (st.band === "all" || r.band === st.band) && r.spend_180 >= st.minSpend && r.recency_days <= st.maxRec && (!st.q || String(r.customer_id).includes(st.q)));
  }
  function drawTable() {
    tbody.textContent = "";
    view.slice(0, shown).forEach((r) => {
      const tr = el("tr", "cust"); tr.tabIndex = 0; tr.dataset.id = r.customer_id;
      tr.setAttribute("aria-selected", String(r.customer_id === st.selected));
      const go = () => (location.hash = "#/customer/" + encodeURIComponent(r.customer_id));
      tr.addEventListener("click", go); tr.addEventListener("keydown", (e) => e.key === "Enter" && go());
      tr.innerHTML = `<td>${r.customer_id}</td><td><span class="band ${bandClass(r.band)}">${r.band} ${pct(r.p)}</span></td><td class="n">${gbp(r.spend_180)}</td><td class="n">${r.orders_180}</td><td class="n">${r.recency_days}</td>
        <td>${r.inactive ? "<span class='hindsight'>Went quiet</span>" : "Ordered again"}</td><td class="small">${(r.reasons[0] || "").replace(" (raises inactivity risk)", "")}</td>`;
      tbody.appendChild(tr);
    });
    more.hidden = shown >= view.length;
    count.textContent = `Showing ${Math.min(shown, view.length)} of ${num(view.length)} customers.`;
    thead.querySelectorAll("th.sortable").forEach((th) => { if (th.dataset.key === st.sort) th.setAttribute("aria-sort", st.dir > 0 ? "ascending" : "descending"); else th.removeAttribute("aria-sort"); });
  }
  function drawKpis() {
    kp.textContent = "";
    const hit = view.filter((r) => r.inactive).length;
    kpi(kp, "Customers shown", num(view.length), `of ${num(rows.length)}`);
    kpi(kp, "Spend in view", gbpShort(view.reduce((a, r) => a + r.spend_180, 0)), "last 180 days");
    kpi(kp, "Mean risk", view.length ? pct(mean(view.map((r) => r.p))) : "n/a", "predicted");
    kpi(kp, "Actually went quiet", view.length ? pct(hit / view.length) : "n/a", `${num(hit)} customers`, "bad");
  }
  let sCh = null;
  function drawScatter() {
    sc.textContent = "";
    if (!view.length) { sc.appendChild(el("p", "muted", "No customers match the filters.")); return; }
    sCh = scatterChart({ points: view.map((r) => ({ x: Math.max(1, r.spend_180), y: r.p, id: r.customer_id, color: COLORS[r.band] })), xLog: true, xFmt: gbp, yFmt: (v) => pct(v),
      xTitle: "spend in last 180 days (log scale)", yTitle: "predicted risk", yMin: 0, yMax: 1, selectedId: st.selected || undefined, label: "Risk against spend",
      tip: (p) => { const r = rows.find((x) => x.customer_id === p.id); return `<b>Customer ${r.customer_id}</b><br>${r.band} risk ${pct(r.p)}<br>${gbp(r.spend_180)} in 180 days<br>${r.recency_days} days since last order`; },
      onClick: (id) => { st.selected = id; sCh.setSelected(id); const i = view.findIndex((r) => r.customer_id === id); if (i >= shown) { shown = i + 20; } drawTable(); const row = tbody.querySelector(`tr[data-id="${CSS.escape(String(id))}"]`); if (row) row.scrollIntoView({ block: "center", behavior: "smooth" }); } });
    sc.appendChild(sCh);
  }
  function drawCap() {
    const ranked = rows.slice().sort((a, b) => b.p - a.p), n = Math.min(st.cap, ranked.length), top = ranked.slice(0, n);
    const base = mean(rows.map((r) => r.inactive)), caught = top.filter((r) => r.inactive).length;
    const lapsersAll = rows.filter((r) => r.inactive).length;
    const spendTop = top.reduce((a, r) => a + r.spend_180, 0), spendAll = rows.reduce((a, r) => a + r.spend_180, 0);
    capHost.textContent = "";
    const g = el("div", "cap");
    const box = (cls, num2, lab) => { const b = el("div", "box" + cls); b.appendChild(el("div", "num", num2)); b.appendChild(el("div", "small muted", lab)); g.appendChild(b); };
    box(" best", `${caught} of ${n}`, `top ${n} by score really went quiet (${pct(caught / n)})`);
    box("", `${Math.round(base * n)} of ${n}`, `expected from picking ${n} at random (${pct(base)})`);
    box("", pct(caught / lapsersAll, 1), `of all ${num(lapsersAll)} lapsers reached`);
    box("", pct(spendTop / spendAll, 1), `of 180-day spend (${gbp(spendTop)}) sits with these ${n}`);
    capHost.appendChild(g);
  }
  function update() {
    view = filtered().sort((a, b) => {
      const x = a[st.sort], y = b[st.sort];
      return (typeof x === "string" ? x.localeCompare(y, undefined, { numeric: true }) : x - y) * st.dir;
    });
    shown = 50;
    drawKpis(); drawScatter(); drawTable();
  }
  exp.addEventListener("click", () => downloadCsv(`worklist_${W.snapshot_date}.csv`, [["customer_id", "risk_probability", "band", "spend_180", "orders_180", "recency_days", "went_quiet"]]
    .concat(view.map((r) => [r.customer_id, r.p, r.band, r.spend_180, r.orders_180, r.recency_days, r.inactive]))));
  update(); drawCap();
  return page;
}
