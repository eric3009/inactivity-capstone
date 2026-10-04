// Router and boot: five data files load once, then every page renders from them.
import { S, $, el, api, seg } from "./common.js";
import * as overview from "./pages/overview.js";
import * as explore from "./pages/explore.js";
import * as model from "./pages/model.js";
import * as worklist from "./pages/worklist.js";
import * as customer from "./pages/customer.js";
import * as tryit from "./pages/try.js";
import * as about from "./pages/about.js";

const NAV = [["", "Overview"], ["explore", "Explore data"], ["model", "Model evidence"], ["worklist", "Retention worklist"], ["try", "Try it"], ["about", "About"]];
const PAGES = { "": overview, explore, model, worklist, customer, try: tryit, about };
const TITLES = { "": "Overview", explore: "Explore data", model: "Model evidence", worklist: "Retention worklist", customer: "Customer", try: "Try it", about: "About" };
const cache = {};

function buildNav() {
  const nav = $("nav");
  NAV.forEach(([k, label]) => {
    const a = el("a", "", label);
    a.href = "#/" + k;
    a.dataset.page = k;
    nav.appendChild(a);
  });
}

function buildTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || "system";
  const host = $("theme");
  host.appendChild(el("div", "lab", "Theme"));
  host.appendChild(seg([["light", "Light"], ["dark", "Dark"], ["system", "System"]], cur, (v) => {
    if (v === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", v);
    try { if (v === "system") localStorage.removeItem("theme"); else localStorage.setItem("theme", v); } catch (e) {}
  }));
}

function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const name = parts[0] || "", arg = parts[1];
  const mod = PAGES[name];
  if (!mod) { location.hash = "#/"; return; }
  document.querySelectorAll("#nav a").forEach((a) => {
    const on = a.dataset.page === name || (name === "customer" && a.dataset.page === "worklist");
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  const host = $("page");
  [...host.children].forEach((c) => (c.hidden = true));
  let node;
  if (name === "customer") {
    host.querySelectorAll(".page.cust-page").forEach((n) => n.remove());
    node = mod.render(decodeURIComponent(arg || ""));
    node.classList.add("cust-page");
    host.appendChild(node);
  } else {
    if (!cache[name]) { cache[name] = mod.render(); host.appendChild(cache[name]); }
    node = cache[name];
    node.hidden = false;
    if (mod.onShow) mod.onShow(node);
  }
  document.title = TITLES[name] + " \u00b7 Customer inactivity risk";
  window.scrollTo(0, 0);
}

async function boot() {
  buildNav();
  buildTheme();
  try {
    const [info, trust, work, eda] = await Promise.all([api("/model-info"), api("/trust"), api("/worklist"), api("/eda")]);
    Object.assign(S, { info, trust, work, eda });
    $("meta").textContent = `model ${info.model_version}\ndata ${info.source}\nhorizon ${info.horizon_days} days`;
    $("meta").style.whiteSpace = "pre-line";
  } catch (e) {
    $("page").textContent = "";
    const p = el("p", "err", "Could not load the data files: " + e.message + " Run `python -m inactivity demo` to create them.");
    $("page").appendChild(p);
    return;
  }
  $("page").textContent = "";
  window.addEventListener("hashchange", route);
  route();
}
boot();
