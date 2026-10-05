// Small SVG chart toolkit: line, bars, stacked columns with a rate line, scatter, heatmap. No dependencies.
import { sv, tipShow, tipHide } from "./common.js";

const css = (n) => `var(--${n})`;

/** Draw width matched to the space the chart will occupy, so text is the same size on every chart and screen. */
export function chartW(half) {
  const host = document.getElementById("page");
  const vw = window.innerWidth;
  let inner;
  if (host && host.clientWidth > 0) {
    const cs = getComputedStyle(host);
    inner = host.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  } else {
    inner = vw >= 980 ? vw - 260 - 100 : vw - 32;
  }
  const w = half && vw >= 880 ? (inner - 18) / 2 - 42 : inner - 42;
  return Math.round(Math.max(280, w));
}
const autoH = (w, ratio = 0.3, lo = 250, hi = 440) => Math.round(Math.min(hi, Math.max(lo, w * ratio)));

export function niceTicks(min, max, n = 5) {
  if (max === min) return [min];
  const raw = (max - min) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(+v.toFixed(10));
  return out;
}

function frame(w, h, label) {
  const s = sv("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
  if (label) s.setAttribute("aria-label", label);
  fitWhenVisible(s, w, h);
  return s;
}

/** Text is measured by the browser, so once the chart is on screen widen the viewBox to include anything that overflows it. */
function fitWhenVisible(svg, w, h) {
  const ro = new ResizeObserver(() => {
    if (!svg.clientWidth) return;
    try {
      const b = svg.getBBox(), pad = 4;
      if (!b.width) return;
      ro.disconnect();
      const x0 = Math.min(0, b.x - pad), y0 = Math.min(0, b.y - pad);
      const x1 = Math.max(w, b.x + b.width + pad), y1 = Math.max(h, b.y + b.height + pad);
      if (x0 < 0 || y0 < 0 || x1 > w || y1 > h) svg.setAttribute("viewBox", `${x0} ${y0} ${x1 - x0} ${y1 - y0}`);
    } catch (e) { /* not rendered yet */ }
  });
  ro.observe(svg);
}

/** Lines over categorical x. series: [{name,color,data[],dash?,hidden?}] */
export function lineChart({ half = false, w = chartW(half), h = autoH(w), labels, series, yFmt = (v) => v, yMin, yMax, xEvery = 1, yTitle, tip, label, xRotate = false }) {
  const L = 62, R = 16, T = 14, B = xRotate ? 62 : 44;
  const pw = w - L - R, ph = h - T - B, n = labels.length;
  const s = frame(w, h, label);
  const all = series.filter((q) => !q.hidden).flatMap((q) => q.data).filter((v) => v !== null && v !== undefined);
  let lo = yMin !== undefined ? yMin : Math.min(0, ...all), hi = yMax !== undefined ? yMax : Math.max(...all) * 1.06;
  if (lo === hi) hi = lo + 1;
  if (yMax === undefined) { const tk = niceTicks(lo, hi, 5); if (tk.length > 1 && tk[tk.length - 1] < hi) hi = tk[tk.length - 1] + (tk[1] - tk[0]); }
  const X = (i) => L + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
  const Y = (v) => T + ph - ((v - lo) / (hi - lo)) * ph;
  niceTicks(lo, hi, 5).forEach((t) => {
    s.appendChild(sv("line", { x1: L, x2: w - R, y1: Y(t), y2: Y(t), class: "grid" }));
    s.appendChild(sv("text", { x: L - 8, y: Y(t) + 4, "text-anchor": "end" }, yFmt(t)));
  });
  s.appendChild(sv("line", { x1: L, x2: w - R, y1: T + ph, y2: T + ph, class: "axis" }));
  labels.forEach((l, i) => {
    if (i % xEvery) return;
    const t = sv("text", { x: X(i), y: T + ph + 16, "text-anchor": xRotate ? "end" : "middle" }, l);
    if (xRotate) t.setAttribute("transform", `rotate(-40 ${X(i)} ${T + ph + 16})`);
    s.appendChild(t);
  });
  if (yTitle) s.appendChild(sv("text", { x: 12, y: T + ph / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${T + ph / 2})` }, yTitle));
  series.forEach((q) => {
    if (q.hidden) return;
    let d = "", pen = false;
    q.data.forEach((v, i) => {
      if (v === null || v === undefined) { pen = false; return; }
      d += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1);
      pen = true;
    });
    const p = sv("path", { d, class: "line" + (q.dash ? " dash" : ""), style: `stroke:${q.color}` });
    s.appendChild(p);
  });
  const cross = sv("line", { y1: T, y2: T + ph, stroke: css("muted"), "stroke-width": 1, opacity: 0 });
  s.appendChild(cross);
  const dots = series.map((q) => { const c = sv("circle", { r: 4, style: `fill:${q.color}`, opacity: 0 }); s.appendChild(c); return c; });
  const hit = sv("rect", { x: L, y: T, width: pw, height: ph, class: "hit" });
  hit.addEventListener("mousemove", (e) => {
    const r = s.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * w;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - L) / pw) * (n - 1))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("opacity", 1);
    series.forEach((q, k) => {
      const v = q.data[i];
      if (q.hidden || v === null || v === undefined) { dots[k].setAttribute("opacity", 0); return; }
      dots[k].setAttribute("cx", X(i)); dots[k].setAttribute("cy", Y(v)); dots[k].setAttribute("opacity", 1);
    });
    tipShow(tip ? tip(i) : `<b>${labels[i]}</b><br>` + series.filter((q) => !q.hidden).map((q) => `${q.name}: ${yFmt(q.data[i])}`).join("<br>"), e.clientX, e.clientY);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); dots.forEach((d) => d.setAttribute("opacity", 0)); tipHide(); });
  s.appendChild(hit);
  return s;
}

/** Horizontal bars. items: [{label,value,color,tip}] */
export function barChart({ half = false, w = chartW(half), rowH = 28, items, fmt = (v) => v, max, labelW = 190, label, xMin = 0 }) {
  const R = 78, T = 6, h = T * 2 + rowH * items.length;
  const s = frame(w, h, label);
  const hi = max !== undefined ? max : Math.max(...items.map((i) => i.value)) * 1.05;
  const X = (v) => labelW + ((v - xMin) / (hi - xMin)) * (w - labelW - R);
  items.forEach((it, i) => {
    const y = T + i * rowH;
    const g = sv("g", {});
    g.appendChild(sv("text", { x: labelW - 10, y: y + rowH / 2 + 4, "text-anchor": "end" }, it.label));
    g.appendChild(sv("rect", { x: labelW, y: y + 5, width: Math.max(1, X(it.value) - labelW), height: rowH - 10, style: `fill:${it.color || css("accent")}` }));
    g.appendChild(sv("text", { x: X(it.value) + 6, y: y + rowH / 2 + 4, class: "strong" }, fmt(it.value)));
    const hit = sv("rect", { x: 0, y, width: w, height: rowH, class: "hit" });
    hit.addEventListener("mousemove", (e) => tipShow(it.tip || `<b>${it.label}</b><br>${fmt(it.value)}`, e.clientX, e.clientY));
    hit.addEventListener("mouseleave", tipHide);
    g.appendChild(hit);
    s.appendChild(g);
  });
  return s;
}

/**
 * Columns per bin. series: [{name,color,values[]}] drawn stacked (or grouped, or on a log axis).
 * line: {name,color,values[],fmt} drawn against a right-hand rate axis.
 */
export function columnChart({ half = false, w = chartW(half), h = autoH(w), labels, series, line, yFmt = (v) => v, mode = "stack", log = false, tip, label, xRotate = true }) {
  const L = 58, R = line ? 52 : 16, T = 14, B = xRotate ? 74 : 44;
  const pw = w - L - R, ph = h - T - B, n = labels.length, slot = pw / n;
  const s = frame(w, h, label);
  const tf = log ? (v) => Math.log10(1 + v) : (v) => v;
  const totals = labels.map((_, i) => series.reduce((a, q) => a + q.values[i], 0));
  const grouped = mode === "group" || log;
  const maxV = grouped ? Math.max(...series.flatMap((q) => q.values)) : Math.max(...totals);
  const hi = tf(maxV) * 1.08 || 1;
  const Y = (v) => T + ph - (tf(v) / hi) * ph;
  const ticks = log ? [0, 1, 10, 100, 1000, 10000, 100000].filter((t) => t <= maxV * 1.08) : niceTicks(0, maxV * 1.08, 5);
  ticks.forEach((t) => {
    s.appendChild(sv("line", { x1: L, x2: w - R, y1: Y(t), y2: Y(t), class: "grid" }));
    s.appendChild(sv("text", { x: L - 8, y: Y(t) + 4, "text-anchor": "end" }, yFmt(t)));
  });
  s.appendChild(sv("line", { x1: L, x2: w - R, y1: T + ph, y2: T + ph, class: "axis" }));
  labels.forEach((l, i) => {
    const x = L + slot * (i + 0.5);
    const t = sv("text", { x, y: T + ph + 14, "text-anchor": xRotate ? "end" : "middle" }, l);
    if (xRotate) t.setAttribute("transform", `rotate(-38 ${x} ${T + ph + 14})`);
    s.appendChild(t);
  });
  const bw = grouped ? (slot * 0.74) / series.length : slot * 0.7;
  labels.forEach((_, i) => {
    let base = 0;
    series.forEach((q, k) => {
      const v = q.values[i];
      if (grouped) {
        s.appendChild(sv("rect", { x: L + slot * i + slot * 0.13 + k * bw, y: Y(v), width: bw, height: Math.max(0, T + ph - Y(v)), style: `fill:${q.color}` }));
      } else {
        const y1 = Y(base + v), y0 = Y(base);
        s.appendChild(sv("rect", { x: L + slot * i + slot * 0.15, y: y1, width: bw, height: Math.max(0, y0 - y1), style: `fill:${q.color}` }));
        base += v;
      }
    });
  });
  if (line) {
    const lmax = Math.max(0.05, ...line.values.map((v) => (v === null ? 0 : v))) * 1.12;
    const LY = (v) => T + ph - (v / Math.min(1, lmax)) * ph;
    const top = Math.min(1, lmax);
    niceTicks(0, top, 5).forEach((t) => s.appendChild(sv("text", { x: w - R + 8, y: LY(t) + 4 }, Math.round(t * 100) + "%")));
    let d = "", pen = false;
    line.values.forEach((v, i) => {
      if (v === null) { pen = false; return; }
      d += (pen ? "L" : "M") + (L + slot * (i + 0.5)).toFixed(1) + " " + LY(v).toFixed(1);
      pen = true;
    });
    s.appendChild(sv("path", { d, class: "line", style: `stroke:${line.color}` }));
    line.values.forEach((v, i) => { if (v !== null) s.appendChild(sv("circle", { cx: L + slot * (i + 0.5), cy: LY(v), r: 3.5, style: `fill:${line.color}` })); });
  }
  labels.forEach((_, i) => {
    const hit = sv("rect", { x: L + slot * i, y: T, width: slot, height: ph, class: "hit" });
    hit.addEventListener("mousemove", (e) => tipShow(tip ? tip(i) : `<b>${labels[i]}</b><br>` + series.map((q) => `${q.name}: ${q.values[i]}`).join("<br>"), e.clientX, e.clientY));
    hit.addEventListener("mouseleave", tipHide);
    s.appendChild(hit);
  });
  return s;
}

/** Scatter with optional log x axis; click and hover per point. points: [{x,y,id,color?}] */
export function scatterChart({ half = false, w = chartW(half), h = autoH(w, 0.4, 340, 560), points, xLog = false, xFmt = (v) => v, yFmt = (v) => v, xTitle, yTitle, onClick, selectedId, tip, yMin = 0, yMax = 1, label }) {
  const L = 62, R = 16, T = 14, B = 46;
  const pw = w - L - R, ph = h - T - B;
  const s = frame(w, h, label);
  const xs = points.map((p) => p.x).filter((v) => v > 0 || !xLog);
  const xLo = xLog ? Math.max(1, Math.min(...xs)) : Math.min(...xs), xHi = Math.max(...xs);
  const tf = xLog ? (v) => Math.log10(Math.max(v, xLo)) : (v) => v;
  const X = (v) => L + ((tf(v) - tf(xLo)) / (tf(xHi) - tf(xLo) || 1)) * pw;
  const Y = (v) => T + ph - ((v - yMin) / (yMax - yMin)) * ph;
  niceTicks(yMin, yMax, 5).forEach((t) => {
    s.appendChild(sv("line", { x1: L, x2: w - R, y1: Y(t), y2: Y(t), class: "grid" }));
    s.appendChild(sv("text", { x: L - 8, y: Y(t) + 4, "text-anchor": "end" }, yFmt(t)));
  });
  const xt = xLog ? (() => { const a = []; for (let e = Math.floor(Math.log10(xLo)); e <= Math.ceil(Math.log10(xHi)); e++) [1, 3].forEach((m) => { const v = m * Math.pow(10, e); if (v >= xLo && v <= xHi) a.push(v); }); return a; })() : niceTicks(xLo, xHi, 6);
  xt.forEach((t) => {
    s.appendChild(sv("line", { x1: X(t), x2: X(t), y1: T, y2: T + ph, class: "grid" }));
    s.appendChild(sv("text", { x: X(t), y: T + ph + 16, "text-anchor": "middle" }, xFmt(t)));
  });
  s.appendChild(sv("line", { x1: L, x2: w - R, y1: T + ph, y2: T + ph, class: "axis" }));
  if (xTitle) s.appendChild(sv("text", { x: L + pw / 2, y: h - 6, "text-anchor": "middle" }, xTitle));
  if (yTitle) s.appendChild(sv("text", { x: 12, y: T + ph / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${T + ph / 2})` }, yTitle));
  const circles = new Map();
  points.forEach((p) => {
    const c = sv("circle", { cx: X(Math.max(p.x, xLo)), cy: Y(p.y), r: 3, class: "pt", style: `fill:${p.color || css("accent")}; fill-opacity:.55` });
    c.addEventListener("mousemove", (e) => tipShow(tip(p), e.clientX, e.clientY));
    c.addEventListener("mouseleave", tipHide);
    c.addEventListener("click", () => onClick && onClick(p.id));
    s.appendChild(c);
    circles.set(p.id, c);
  });
  let prev = null;
  s.setSelected = (id) => {
    if (prev) { prev.setAttribute("r", 3); prev.style.stroke = "none"; prev.style.fillOpacity = ".55"; }
    const c = circles.get(id);
    if (c) { c.setAttribute("r", 7); c.style.stroke = css("text"); c.style.strokeWidth = "2"; c.style.fillOpacity = "1"; c.parentNode.appendChild(c); prev = c; }
  };
  if (selectedId !== undefined) s.setSelected(selectedId);
  return s;
}

/** Correlation-style heatmap. Positive = oxide red, negative = pine; dimmed below minAbs. */
export function heatmap({ labels, matrix, onRow, label }) {
  const n = labels.length, labelW = 188, T = 8;
  const c = Math.max(21, Math.min(36, Math.floor((chartW(false) - labelW - 8) / n)));
  const w = labelW + n * c + 8, h = T + n * c + 8;
  const s = frame(w, h, label);
  const cells = [];
  matrix.forEach((row, i) => {
    const t = sv("text", { x: labelW - 8, y: T + i * c + c * 0.7, "text-anchor": "end", class: onRow ? "clickable" : "" }, labels[i]);
    if (onRow) t.addEventListener("click", () => onRow(i));
    s.appendChild(t);
    row.forEach((v, j) => {
      const r = sv("rect", { x: labelW + j * c, y: T + i * c, width: c - 1, height: c - 1, style: `fill:${v >= 0 ? css("signal") : css("low")}`, "fill-opacity": Math.abs(v).toFixed(2) });
      r.dataset.v = v;
      r.addEventListener("mousemove", (e) => tipShow(`<b>${labels[i]}</b><br>${labels[j]}<br>Spearman ${v.toFixed(2)}`, e.clientX, e.clientY));
      r.addEventListener("mouseleave", tipHide);
      s.appendChild(r);
      cells.push(r);
    });
  });
  s.setMin = (m) => cells.forEach((r) => { const v = Math.abs(+r.dataset.v); r.setAttribute("fill-opacity", (v >= m ? v : 0.05).toFixed(2)); });
  return s;
}
