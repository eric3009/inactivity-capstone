// Shared state and small helpers used by every page.
const NS = "http://www.w3.org/2000/svg";

export const S = { info: null, trust: null, work: null, eda: null };
export const $ = (id) => document.getElementById(id);

export function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function sv(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

export async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
  return j;
}

export const f2 = (x) => (x === null || x === undefined ? "n/a" : x.toFixed(2));
export const f3 = (x) => (x === null || x === undefined ? "n/a" : x.toFixed(3));
export const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
export const gbp = (x) => "£" + Math.round(x).toLocaleString("en-GB");
export const gbpShort = (x) => (x >= 1e6 ? "£" + (x / 1e6).toFixed(1) + "m" : x >= 1e4 ? "£" + Math.round(x / 1e3) + "k" : gbp(x));
export const num = (x) => Math.round(x).toLocaleString("en-GB");
export const bandClass = (b) => (b === "Not eligible" ? "NotEligible" : b);

export const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
export function quantile(a, q) {
  const b = a.slice().sort((x, y) => x - y);
  if (!b.length) return 0;
  const pos = (b.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return b[lo] + (b[hi] - b[lo]) * (pos - lo);
}
export const median = (a) => quantile(a, 0.5);

export function kpi(parent, label, value, detail, tone) {
  const k = el("div", "kpi" + (tone ? " " + tone : ""));
  k.appendChild(el("div", "k", label));
  k.appendChild(el("div", "v", value));
  k.appendChild(el("div", "d", detail));
  parent.appendChild(k);
  return k;
}

export function pageHead(eyebrow, title, sub) {
  const h = el("div", "page-head");
  h.appendChild(el("div", "eyebrow", eyebrow));
  h.appendChild(el("h1", "", title));
  if (sub) h.appendChild(el("p", "sub", sub));
  return h;
}

export function card(title, note) {
  const c = el("section", "card");
  if (title) c.appendChild(el("h3", "", title));
  if (note) c.appendChild(el("p", "cap-note", note));
  return c;
}

/** Segmented control: options is [[value, label], ...]. */
export function seg(options, value, onChange) {
  const wrap = el("div", "segmented");
  wrap.setAttribute("role", "group");
  const btns = options.map(([v, label]) => {
    const b = el("button", "", label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(v === value));
    b.addEventListener("click", () => {
      btns.forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      onChange(v);
    });
    wrap.appendChild(b);
    return b;
  });
  return wrap;
}

export function ctl(label, node) {
  const d = el("div", "ctl");
  const field = node.tagName === "SELECT" || node.tagName === "INPUT";
  d.appendChild(el(field ? "label" : "div", field ? "" : "lab", label));
  d.appendChild(node);
  return d;
}

export function select(options, value, onChange) {
  const s = el("select");
  options.forEach(([v, label]) => {
    const o = el("option", "", label);
    o.value = v;
    if (v === value) o.selected = true;
    s.appendChild(o);
  });
  s.addEventListener("change", () => onChange(s.value));
  return s;
}

export function checkbox(label, checked, onChange) {
  const l = el("label", "check");
  const i = el("input");
  i.type = "checkbox";
  i.checked = checked;
  i.addEventListener("change", () => onChange(i.checked));
  l.appendChild(i);
  l.appendChild(document.createTextNode(label));
  return l;
}

/** Legend. Items: {label, color, dash?, square?}. With onToggle the entries become checkboxes. */
export function legend(items, onToggle) {
  const d = el("div", "legend");
  items.forEach((it, idx) => {
    const lab = el("label");
    if (onToggle) {
      const c = el("input");
      c.type = "checkbox";
      c.checked = it.on !== false;
      c.addEventListener("change", () => onToggle(idx, c.checked));
      lab.appendChild(c);
    }
    const sw = el("i", it.square ? "sq" : "");
    sw.style.borderTopColor = it.color;
    sw.style.color = it.color;
    if (it.dash) sw.style.borderTopStyle = "dashed";
    lab.appendChild(sw);
    lab.appendChild(document.createTextNode(it.label));
    d.appendChild(lab);
  });
  return d;
}

let tipEl = null;
export function tipShow(html, x, y) {
  if (!tipEl) {
    tipEl = el("div");
    tipEl.id = "tip";
    document.body.appendChild(tipEl);
  }
  tipEl.innerHTML = html;
  tipEl.style.opacity = "1";
  const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
  let left = x + 14, top = y + 14;
  if (left + w > window.innerWidth - 8) left = x - w - 14;
  if (top + h > window.innerHeight - 8) top = y - h - 14;
  tipEl.style.left = Math.max(4, left) + "px";
  tipEl.style.top = Math.max(4, top) + "px";
}
export function tipHide() {
  if (tipEl) tipEl.style.opacity = "0";
}
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function downloadCsv(filename, rows) {
  const q = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const blob = new Blob([rows.map((r) => r.map(q).join(",")).join("\n")], { type: "text/csv;charset=utf-8" });
  const a = el("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
}

export function monthLabel(iso) {
  const d = new Date(iso.length === 7 ? iso + "-01" : iso);
  return d.toLocaleString("en-GB", { month: "short", year: "2-digit" });
}
