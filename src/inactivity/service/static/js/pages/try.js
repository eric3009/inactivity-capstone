import { S, el, pageHead, card, api, ctl, select, pct, bandClass } from "../common.js";

const SAMPLE_LINES = `invoice,date,quantity,price
536001,2011-05-03,12,2.10
536001,2011-05-03,6,3.75
538210,2011-06-14,24,1.25
541022,2011-07-26,10,4.95
544310,2011-08-22,36,0.85`;

function showResult(out, r) {
  out.textContent = "";
  if (!r.eligible) { out.appendChild(el("div", "callout", r.note || "Not eligible: no purchase in the activity window.")); return; }
  const box = el("div", "callout");
  box.appendChild(el("span", "big", pct(r.probability_inactive, 1)));
  box.appendChild(el("span", `band ${bandClass(r.risk_band)}`, " " + r.risk_band + " "));
  box.appendChild(el("p", "small muted", `Chance of no order in the next ${r.horizon_days} days. Model ${r.model_version}.`));
  const ul = el("ul", "reasons");
  r.reason_codes.forEach((c) => ul.appendChild(el("li", "", `${c.text.replace(" (raises inactivity risk)", "")} (${c.contribution >= 0 ? "+" : ""}${c.contribution.toFixed(2)})`)));
  box.appendChild(ul);
  out.appendChild(box);
}

export function render() {
  const page = el("div", "page");
  page.appendChild(pageHead("Try it", "Score a customer", "Either replay a real customer from the test months, or paste raw order lines and let the service compute the features exactly as in training."));

  const g = el("div", "grid2");
  // ---- real example
  const c1 = card("Replay a real customer", "Pick a customer from the held-out months and compare the prediction with what happened.");
  const out1 = el("div");
  c1.appendChild(el("p", "muted small", "Loading examples..."));
  api("/demo/customers").then((list) => {
    c1.querySelector("p").remove();
    const s = select(list.map((d, i) => [String(i), `Customer ${d.customer_id} (${d.snapshot_date})`]), "0", () => {});
    const btn = el("button", "btn", "Score"); btn.type = "button";
    const go = async () => {
      const d = list[+s.value];
      try {
        const r = await api("/score", { customer_id: d.customer_id, snapshot_date: d.snapshot_date, features: d.features });
        showResult(out1, r);
        out1.appendChild(el("p", "small", `What actually happened: ${d.actual_inactive ? "no order in the next " + d.horizon_days + " days" : "the customer ordered again"}.`));
      } catch (e) { out1.textContent = "Could not score: " + e.message; }
    };
    btn.addEventListener("click", go);
    const row = el("div", "controls"); row.appendChild(ctl("Customer", s)); c1.appendChild(row); c1.appendChild(btn); c1.appendChild(out1);
  }).catch((e) => { c1.textContent = "Could not load examples: " + e.message; });
  g.appendChild(c1);

  // ---- raw lines
  const c2 = card("Paste order lines", "CSV with columns invoice, date, quantity, price. The snapshot date is when you are asking; lines after it are ignored.");
  const ta = el("textarea"); ta.rows = 9; ta.value = SAMPLE_LINES; ta.spellcheck = false; ta.setAttribute("aria-label", "Order lines CSV");
  const snap = el("input"); snap.type = "date"; snap.value = "2011-08-31";
  const btn2 = el("button", "btn", "Score these orders"); btn2.type = "button";
  const out2 = el("div");
  btn2.addEventListener("click", async () => {
    try {
      const lines = ta.value.trim().split(/\r?\n/).filter(Boolean);
      const head = lines.shift().split(",").map((x) => x.trim().toLowerCase());
      const ix = (n) => head.indexOf(n);
      if (["invoice", "date", "quantity", "price"].some((n) => ix(n) < 0)) throw new Error("header must contain invoice, date, quantity, price");
      const tx = lines.map((l) => { const p = l.split(","); return { invoice: p[ix("invoice")].trim(), invoice_date: p[ix("date")].trim() + "T12:00:00", quantity: +p[ix("quantity")], price: +p[ix("price")] }; });
      const r = await api("/score/transactions", { snapshot_date: snap.value, transactions: tx });
      showResult(out2, r);
    } catch (e) { out2.textContent = ""; out2.appendChild(el("div", "err", "Could not score: " + e.message)); }
  });
  const row2 = el("div", "controls"); row2.appendChild(ctl("Snapshot date", snap));
  c2.appendChild(ta); c2.appendChild(row2); c2.appendChild(btn2); c2.appendChild(out2);
  g.appendChild(c2);
  page.appendChild(g);
  return page;
}
