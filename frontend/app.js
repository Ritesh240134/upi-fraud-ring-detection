/* ==========================================================
   UPI Fraud Console
   Vanilla JS. No build step, no external dependencies - the
   charts are hand-rolled SVG so the console works offline.
   ========================================================== */

(() => {
"use strict";

// ---------- tiny helpers ----------

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  overview: null,
  metrics: null,
  rings: [],
  selectedRing: null,
  accounts: { offset: 0, level: "ALL", search: "", rows: [], total: 0 },
};

const VIEW_META = {
  overview: ["Overview", "Detection posture across the synthetic event stream"],
  rings:    ["Fraud rings", "Groups of accounts moving money together inside a short window"],
  accounts: ["Accounts", "Per-account fraud probability from the temporal GNN"],
  scoring:  ["Live scoring", "Score a transaction against the deployed model"],
  model:    ["Model", "Measured performance on the held-out test split"],
};

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtInt(value) {
  if (value === null || value === undefined) return "–";
  return Number(value).toLocaleString("en-IN");
}

function fmtNum(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return Number(value).toFixed(digits);
}

function fmtPct(value, digits = 1) {
  if (value === null || value === undefined) return "–";
  return (Number(value) * 100).toFixed(digits) + "%";
}

function fmtMoney(value) {
  if (value === null || value === undefined) return "–";
  return "₹" + Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: 2, maximumFractionDigits: 2,
  });
}

function fmtClock(value) {
  if (!value) return "–";
  const text = String(value);
  const match = text.match(/\d{2}:\d{2}:\d{2}/);
  return match ? match[0] : text.slice(0, 19);
}

function levelOf(score) {
  if (score === null || score === undefined) return "low";
  if (score >= 0.60) return "high";
  if (score >= 0.30) return "medium";
  return "low";
}

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

let toastTimer = null;
function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("is-show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("is-show"), 2600);
}

// ---------- SVG helpers ----------

function svgEl(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

function svgRoot(width, height) {
  const node = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    preserveAspectRatio: "xMidYMid meet",
  });
  node.style.display = "block";
  node.style.overflow = "visible";
  return node;
}

// ==========================================================
// NAVIGATION
// ==========================================================

function showView(name) {
  $$(".nav-item").forEach(button =>
    button.classList.toggle("is-active", button.dataset.view === name));

  $$(".view").forEach(section =>
    section.classList.toggle("is-active", section.dataset.view === name));

  const [title, subtitle] = VIEW_META[name] || ["", ""];
  $("#viewTitle").textContent = title;
  $("#viewSubtitle").textContent = subtitle;

  if (location.hash.slice(1) !== name) {
    history.replaceState(null, "", "#" + name);
  }
}

// ==========================================================
// OVERVIEW
// ==========================================================

function renderKpis(data) {
  const cards = [
    {
      label: "Accounts",
      value: fmtInt(data.accounts),
      sub: `${fmtInt(data.fraud_accounts)} fraudulent`,
      tone: "",
    },
    {
      label: "Events",
      value: fmtInt(data.events),
      sub: `${fmtInt(data.fraud_events)} fraud-flagged`,
      tone: "",
    },
    {
      label: "Rings detected",
      value: fmtInt(data.rings_detected),
      sub: `${fmtInt(data.accounts_in_rings)} accounts implicated`,
      tone: "is-danger",
    },
    {
      label: "Ring precision",
      value: data.ring_precision === null ? "–" : fmtPct(data.ring_precision),
      sub: `${fmtInt(data.true_fraud_in_rings)} of ${fmtInt(data.accounts_in_rings)} truly fraudulent`,
      tone: "is-success",
    },
    {
      label: "High-risk accounts",
      value: fmtInt(data.high_risk_accounts),
      sub: "scored above 0.60",
      tone: "is-warn",
    },
    {
      label: "Test ROC-AUC",
      value: fmtNum(data.test_roc_auc),
      sub: `F1 ${fmtNum(data.test_f1)} on held-out split`,
      tone: "",
    },
  ];

  $("#kpiGrid").innerHTML = cards.map(card => `
    <div class="kpi ${card.tone}">
      <div class="kpi-label">${esc(card.label)}</div>
      <div class="kpi-value">${esc(card.value)}</div>
      <div class="kpi-sub">${esc(card.sub)}</div>
    </div>`).join("");
}

function donut(fraction, caption) {
  const size = 132, stroke = 13, radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const filled = circumference * Math.max(0, Math.min(1, fraction || 0));

  const root = svgRoot(size, size);
  root.setAttribute("width", size);
  root.setAttribute("height", size);

  root.appendChild(svgEl("circle", {
    cx: size / 2, cy: size / 2, r: radius,
    fill: "none", stroke: "var(--surface-3)", "stroke-width": stroke,
  }));

  const arc = svgEl("circle", {
    cx: size / 2, cy: size / 2, r: radius,
    fill: "none", stroke: "var(--success)", "stroke-width": stroke,
    "stroke-linecap": "round",
    "stroke-dasharray": `${filled} ${circumference - filled}`,
    transform: `rotate(-90 ${size / 2} ${size / 2})`,
  });
  root.appendChild(arc);

  const value = svgEl("text", {
    x: size / 2, y: size / 2 - 2,
    "text-anchor": "middle", "dominant-baseline": "middle",
    fill: "var(--text)", "font-size": "23", "font-weight": "650",
  });
  value.textContent = fmtPct(fraction, 1);
  root.appendChild(value);

  const label = svgEl("text", {
    x: size / 2, y: size / 2 + 19,
    "text-anchor": "middle", fill: "var(--text-faint)", "font-size": "10.5",
  });
  label.textContent = caption;
  root.appendChild(label);

  return root;
}

function renderRingQuality(data) {
  const box = $("#ringQuality");
  box.innerHTML = "";

  const wrap = document.createElement("div");
  wrap.className = "donut-wrap";
  wrap.appendChild(donut(data.ring_precision, "precision"));
  box.appendChild(wrap);

  const stats = document.createElement("div");
  stats.className = "rq-stats";
  stats.innerHTML = `
    <div class="rq-item"><span class="rq-key">Rings detected</span>
      <span class="rq-val">${fmtInt(data.rings_detected)}</span></div>
    <div class="rq-item"><span class="rq-key">Accounts flagged</span>
      <span class="rq-val">${fmtInt(data.accounts_in_rings)}</span></div>
    <div class="rq-item"><span class="rq-key">Genuinely fraudulent</span>
      <span class="rq-val">${fmtInt(data.true_fraud_in_rings)}</span></div>
    <div class="rq-item"><span class="rq-key">Fraud coverage</span>
      <span class="rq-val">${fmtPct(data.ring_recall)}</span></div>`;
  box.appendChild(stats);
}

function renderBars(target, entries, danger = false) {
  const node = typeof target === "string" ? $(target) : target;
  const values = entries.map(entry => entry[1]);
  const max = Math.max(1, ...values);

  node.innerHTML = entries.map(([name, value]) => `
    <div class="bar-row">
      <div class="bar-name" title="${esc(name)}">${esc(name)}</div>
      <div class="bar-track">
        <div class="bar-fill ${danger ? "is-danger" : ""}" style="width:${(value / max) * 100}%"></div>
      </div>
      <div class="bar-value">${fmtInt(value)}</div>
    </div>`).join("");
}

function renderOverview(data) {
  renderKpis(data);
  renderRingQuality(data);

  const events = Object.entries(data.event_types)
    .sort((a, b) => b[1] - a[1]);
  renderBars("#eventMix", events);
  $("#eventTotal").textContent = fmtInt(data.events) + " events";

  const typologies = Object.entries(data.typologies)
    .sort((a, b) => b[1] - a[1]);

  if (typologies.length) {
    renderBars("#typologies", typologies, true);
  } else {
    $("#typologies").innerHTML =
      `<div class="empty"><p>No fraud typologies present.</p></div>`;
  }
}

function renderModelStatus(info) {
  const node = $("#modelStatus");
  const text = $(".status-text", node);

  node.classList.remove("is-live", "is-down");

  if (info && info.available) {
    node.classList.add("is-live");
    text.textContent = `model live · ${fmtInt(info.scored_accounts)} accounts`;
    node.title = `Threshold ${info.decision_threshold} · edge time ${info.uses_edge_time ? "on" : "off"}`;
  } else {
    node.classList.add("is-down");
    text.textContent = "rule layer only";
    node.title = info && info.error ? info.error : "Model artifacts unavailable";
  }
}

// ==========================================================
// RINGS
// ==========================================================

function renderRingTable(rings) {
  const body = $("#ringTable tbody");

  if (!rings.length) {
    body.innerHTML = `<tr><td colspan="5" class="muted" style="padding:24px">
      No rings detected. Run the detector.</td></tr>`;
    return;
  }

  body.innerHTML = rings.map(ring => {
    const pure = ring.true_fraud_accounts === ring.ring_size;
    return `
      <tr data-ring="${esc(ring.detected_ring_id)}">
        <td><span class="acct">${esc(ring.detected_ring_id.replace("GNN_RING_", "RING "))}</span></td>
        <td class="num">${fmtInt(ring.ring_size)}</td>
        <td class="num">${fmtNum(ring.time_span_minutes, 1)}m</td>
        <td class="num">${fmtNum(ring.ring_risk_score, 3)}</td>
        <td class="num">
          <span class="pill ${pure ? "pill-low" : "pill-medium"}">
            ${fmtInt(ring.true_fraud_accounts)}/${fmtInt(ring.ring_size)}
          </span>
        </td>
      </tr>`;
  }).join("");

  $$("#ringTable tbody tr").forEach(row => {
    row.addEventListener("click", () => selectRing(row.dataset.ring));
  });
}

function ringGraph(nodes, edges) {
  const size = 340, cx = size / 2, cy = 150, radius = 104;

  const root = svgRoot(size, 300);

  const positions = new Map();
  const count = nodes.length;

  nodes.forEach((node, index) => {
    const angle = (index / count) * Math.PI * 2 - Math.PI / 2;
    positions.set(node.account, {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    });
  });

  // arrow marker
  const defs = svgEl("defs");
  const marker = svgEl("marker", {
    id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5",
    markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse",
  });
  marker.appendChild(svgEl("path", {
    d: "M0 0 L10 5 L0 10 z", fill: "var(--accent)", "fill-opacity": ".5",
  }));
  defs.appendChild(marker);
  root.appendChild(defs);

  // edges as gentle arcs
  edges.forEach(edge => {
    const from = positions.get(edge.source);
    const to = positions.get(edge.target);
    if (!from || !to) return;

    const midX = (from.x + to.x) / 2;
    const midY = (from.y + to.y) / 2;
    const curveX = midX + (cx - midX) * 0.28;
    const curveY = midY + (cy - midY) * 0.28;

    const path = svgEl("path", {
      d: `M${from.x} ${from.y} Q${curveX} ${curveY} ${to.x} ${to.y}`,
      "stroke-width": 1.4,
      "marker-end": "url(#arrow)",
    });
    path.setAttribute("class", "edge-line" + (edge.is_fraud ? " is-fraud" : ""));

    const title = svgEl("title");
    title.textContent =
      `${edge.source} → ${edge.target}  ${fmtMoney(edge.amount)}  ${fmtClock(edge.timestamp)}`;
    path.appendChild(title);

    root.appendChild(path);
  });

  // nodes
  nodes.forEach(node => {
    const point = positions.get(node.account);
    const level = levelOf(node.score);
    const fill = level === "high" ? "var(--danger)"
               : level === "medium" ? "var(--warn)" : "var(--success)";

    const circle = svgEl("circle", {
      cx: point.x, cy: point.y,
      r: 7 + (node.score || 0) * 4,
      fill, "fill-opacity": ".9",
      stroke: "var(--surface)", "stroke-width": "2",
    });
    circle.setAttribute("class", "node-circle");

    const title = svgEl("title");
    title.textContent =
      `${node.account}\nscore ${fmtNum(node.score, 4)}\n` +
      `ground truth: ${node.true_label ? "fraud" : "normal"}`;
    circle.appendChild(title);

    root.appendChild(circle);

    const label = svgEl("text", {
      x: point.x,
      y: point.y + (point.y < cy ? -14 : 20),
      "text-anchor": "middle",
    });
    label.setAttribute("class", "node-label");
    label.textContent = node.account.replace("ACC_", "");
    root.appendChild(label);
  });

  return root;
}

function renderRingDetail(payload) {
  const { ring, nodes, edges, timeline } = payload;
  const box = $("#ringDetail");

  const truth = ring.matched_ground_truth_rings
    ? String(ring.matched_ground_truth_rings).split(",").filter(Boolean)
    : [];

  box.innerHTML = `
    <div class="ring-title">
      <h2>${esc(ring.detected_ring_id)}</h2>
      <span class="pill ${ring.true_fraud_accounts === ring.ring_size ? "pill-low" : "pill-medium"}">
        ${fmtInt(ring.true_fraud_accounts)} of ${fmtInt(ring.ring_size)} truly fraudulent
      </span>
    </div>
    <p class="muted">
      ${fmtInt(ring.ring_size)} accounts, ${fmtInt(ring.suspicious_edges)} suspicious transfers,
      inside a ${fmtNum(ring.time_span_minutes, 2)} minute window.
    </p>

    <div class="stat-row">
      <div class="stat"><div class="stat-key">Risk</div>
        <div class="stat-val">${fmtNum(ring.ring_risk_score, 3)}</div></div>
      <div class="stat"><div class="stat-key">Peak account</div>
        <div class="stat-val">${fmtNum(ring.max_account_score, 3)}</div></div>
      <div class="stat"><div class="stat-key">Value moved</div>
        <div class="stat-val">${fmtMoney(ring.total_amount)}</div></div>
      <div class="stat"><div class="stat-key">Window</div>
        <div class="stat-val">${fmtClock(ring.window_start)}</div></div>
    </div>

    ${truth.length ? `
      <div class="section-label">Matched ground-truth rings</div>
      <div class="member-grid">
        ${truth.map(id => `<span class="member"><span class="dot"></span>${esc(id)}</span>`).join("")}
      </div>` : ""}

    <div class="section-label">Money flow</div>
    <div class="graph-box" id="ringGraphBox"></div>
    <div class="legend">
      <div class="legend-item"><span class="legend-swatch" style="background:var(--danger)"></span>high risk</div>
      <div class="legend-item"><span class="legend-swatch" style="background:var(--warn)"></span>medium</div>
      <div class="legend-item"><span class="legend-swatch" style="background:var(--success)"></span>low</div>
      <div class="legend-item"><span class="legend-swatch" style="background:var(--danger);opacity:.55"></span>fraud-flagged transfer</div>
    </div>

    <div class="section-label">Accounts</div>
    <div class="member-grid">
      ${nodes.map(node => {
        const level = levelOf(node.score);
        const cls = level === "high" ? "is-high" : level === "medium" ? "is-med" : "";
        return `<span class="member ${cls} ${node.true_label ? "" : "is-fp"}"
                      title="score ${fmtNum(node.score, 4)} · ${node.true_label ? "fraud" : "normal in ground truth"}">
                  <span class="dot"></span>${esc(node.account)}
                </span>`;
      }).join("")}
    </div>

    <div class="section-label">Event timeline (${fmtInt(timeline.length)})</div>
    <div class="timeline">
      ${timeline.length ? timeline.map(row => `
        <div class="tl-row">
          <span class="tl-time">${fmtClock(row.timestamp)}</span>
          <span class="pill tl-type ${row.is_fraud ? "pill-high" : ""}">${esc(row.event_type)}</span>
          <span class="tl-flow">${esc(row.source)}${row.target ? " → " + esc(row.target) : ""}</span>
          <span class="tl-amt">${row.amount ? fmtMoney(row.amount) : ""}</span>
        </div>`).join("")
      : `<p class="muted">No events in this window.</p>`}
    </div>`;

  $("#ringGraphBox").appendChild(ringGraph(nodes, edges));
}

async function selectRing(ringId) {
  $$("#ringTable tbody tr").forEach(row =>
    row.classList.toggle("is-selected", row.dataset.ring === ringId));

  state.selectedRing = ringId;
  $("#ringDetail").innerHTML = `<div class="skeleton" style="height:320px"></div>`;

  try {
    renderRingDetail(await api(`/api/rings/${encodeURIComponent(ringId)}`));
  } catch (error) {
    $("#ringDetail").innerHTML =
      `<div class="empty"><p>Could not load ${esc(ringId)}: ${esc(error.message)}</p></div>`;
  }
}

// ==========================================================
// ACCOUNTS
// ==========================================================

function histogram(bins, threshold) {
  const width = 520, height = 190, padLeft = 34, padBottom = 26, padTop = 10;
  const plotWidth = width - padLeft - 10;
  const plotHeight = height - padBottom - padTop;

  const root = svgRoot(width, height);
  const max = Math.max(1, ...bins.map(bin => bin.count));

  // horizontal grid
  for (let step = 0; step <= 4; step++) {
    const y = padTop + plotHeight - (plotHeight * step) / 4;
    root.appendChild(svgEl("line", {
      x1: padLeft, x2: width - 10, y1: y, y2: y, class: "grid-line",
    }));
    const label = svgEl("text", { x: padLeft - 7, y: y + 3, "text-anchor": "end" });
    label.setAttribute("class", "axis-text");
    label.textContent = Math.round((max * step) / 4);
    root.appendChild(label);
  }

  const barWidth = plotWidth / bins.length;

  bins.forEach((bin, index) => {
    const barHeight = (bin.count / max) * plotHeight;
    const x = padLeft + index * barWidth;
    const y = padTop + plotHeight - barHeight;

    const mid = (bin.from + bin.to) / 2;
    const fill = mid >= 0.6 ? "var(--danger)"
               : mid >= 0.3 ? "var(--warn)" : "var(--success)";

    const rect = svgEl("rect", {
      x: x + 1, y, width: Math.max(1, barWidth - 2), height: barHeight,
      fill, "fill-opacity": ".82", rx: 2,
    });
    const title = svgEl("title");
    title.textContent = `${bin.from.toFixed(2)}–${bin.to.toFixed(2)}: ${bin.count} accounts`;
    rect.appendChild(title);
    root.appendChild(rect);
  });

  // x labels
  [0, 0.25, 0.5, 0.75, 1].forEach(value => {
    const label = svgEl("text", {
      x: padLeft + value * plotWidth,
      y: height - padBottom + 16,
      "text-anchor": "middle",
    });
    label.setAttribute("class", "axis-text");
    label.textContent = value.toFixed(2);
    root.appendChild(label);
  });

  // decision threshold marker
  if (threshold) {
    const x = padLeft + threshold * plotWidth;
    root.appendChild(svgEl("line", {
      x1: x, x2: x, y1: padTop, y2: padTop + plotHeight,
      stroke: "var(--accent)", "stroke-width": 1.4, "stroke-dasharray": "4 3",
    }));
    const tag = svgEl("text", { x: x + 5, y: padTop + 11 });
    tag.setAttribute("class", "axis-text");
    tag.setAttribute("fill", "var(--accent)");
    tag.textContent = `threshold ${threshold}`;
    root.appendChild(tag);
  }

  return root;
}

function accountRow(row) {
  const level = String(row.risk_level || "").toLowerCase();
  const pillClass = level === "high" ? "pill-high"
                  : level === "medium" ? "pill-medium" : "pill-low";

  return `
    <tr>
      <td><span class="acct">${esc(row.account)}</span></td>
      <td><span class="pill ${pillClass}">${esc(row.risk_level)}</span></td>
      <td class="num">${fmtNum(row.fraud_probability, 4)}</td>
      <td>
        <div class="meter is-${level}">
          <span style="width:${Math.max(2, (row.fraud_probability || 0) * 100)}%"></span>
        </div>
      </td>
      <td class="num">${row.true_label ? '<span class="pill pill-high">fraud</span>'
                                       : '<span class="pill">normal</span>'}</td>
    </tr>`;
}

async function loadAccounts(reset = false) {
  const filters = state.accounts;

  if (reset) {
    filters.offset = 0;
    filters.rows = [];
  }

  const params = new URLSearchParams({
    limit: "40",
    offset: String(filters.offset),
    level: filters.level,
    search: filters.search,
  });

  try {
    const data = await api("/api/accounts?" + params.toString());

    filters.total = data.total;
    filters.rows = filters.rows.concat(data.accounts);
    filters.offset += data.returned;

    $("#accountTable tbody").innerHTML =
      filters.rows.length
        ? filters.rows.map(accountRow).join("")
        : `<tr><td colspan="5" class="muted" style="padding:24px">No matching accounts.</td></tr>`;

    $("#accountCount").textContent =
      `showing ${fmtInt(filters.rows.length)} of ${fmtInt(filters.total)}`;

    $("#loadMore").style.display =
      filters.rows.length >= filters.total ? "none" : "inline-flex";

  } catch (error) {
    toast("Could not load accounts: " + error.message);
  }
}

async function loadAccountSummary() {
  try {
    const data = await api("/api/accounts-summary");

    const box = $("#histogram");
    box.innerHTML = "";
    box.appendChild(histogram(data.histogram, data.threshold));

    $("#histNote").textContent =
      `${fmtInt(data.flagged)} of ${fmtInt(data.total)} above threshold`;

  } catch (error) {
    $("#histogram").innerHTML = `<div class="empty"><p>${esc(error.message)}</p></div>`;
  }
}

async function lookupAccount(accountId) {
  const box = $("#lookupResult");
  box.innerHTML = `<div class="skeleton" style="height:74px"></div>`;

  try {
    const data = await api(`/accounts/${encodeURIComponent(accountId)}/risk`);
    const level = String(data.risk_level).toLowerCase();

    box.innerHTML = `
      <div class="stat-row" style="margin:0">
        <div class="stat"><div class="stat-key">Account</div>
          <div class="stat-val acct" style="font-size:14px">${esc(data.account_id)}</div></div>
        <div class="stat"><div class="stat-key">Probability</div>
          <div class="stat-val">${fmtNum(data.fraud_probability, 4)}</div></div>
        <div class="stat"><div class="stat-key">Level</div>
          <div class="stat-val"><span class="pill pill-${level}">${esc(data.risk_level)}</span></div></div>
        <div class="stat"><div class="stat-key">Flagged</div>
          <div class="stat-val">${data.flagged ? "yes" : "no"}</div></div>
      </div>`;

  } catch (error) {
    box.innerHTML = `<div class="status-note">${esc(error.message)}</div>`;
  }
}

// ==========================================================
// LIVE SCORING
// ==========================================================

function gauge(score) {
  const width = 168, height = 96, cx = width / 2, cy = 86, radius = 66;

  const root = svgRoot(width, height);
  root.setAttribute("width", width);
  root.setAttribute("height", height);

  const arcPath = (from, to) => {
    const start = Math.PI + Math.PI * from;
    const end = Math.PI + Math.PI * to;
    return `M${cx + radius * Math.cos(start)} ${cy + radius * Math.sin(start)}
            A${radius} ${radius} 0 0 1
            ${cx + radius * Math.cos(end)} ${cy + radius * Math.sin(end)}`;
  };

  root.appendChild(svgEl("path", {
    d: arcPath(0, 1), fill: "none",
    stroke: "var(--surface-3)", "stroke-width": 13, "stroke-linecap": "round",
  }));

  const value = Math.max(0, Math.min(1, score || 0));
  const color = value >= 0.7 ? "var(--danger)"
              : value >= 0.4 ? "var(--warn)" : "var(--success)";

  if (value > 0.001) {
    root.appendChild(svgEl("path", {
      d: arcPath(0, value), fill: "none",
      stroke: color, "stroke-width": 13, "stroke-linecap": "round",
    }));
  }

  const text = svgEl("text", {
    x: cx, y: cy - 12, "text-anchor": "middle",
    fill: "var(--text)", "font-size": "24", "font-weight": "680",
  });
  text.textContent = value.toFixed(3);
  root.appendChild(text);

  return root;
}

function renderScore(data) {
  const box = $("#scoreResult");

  const known = data.source_account_score !== null
             || data.target_account_score !== null;

  const noteText = data.model_status === "temporal_hetero_gnn"
    ? "Scored by the temporal heterogeneous GNN using the precomputed account risk table."
    : data.model_status === "rule_layer_unknown_accounts"
      ? "Neither account appears in the scored graph, so only the amount and event-type rules applied. The model is transductive - it cannot score an account it has never seen."
      : "The model is unavailable, so the rule layer answered alone.";

  box.innerHTML = `
    <div class="gauge-row">
      <div id="gaugeBox"></div>
      <div class="verdict">
        <span class="verdict-level is-${esc(data.risk_level)}">${esc(data.risk_level)}</span>
        <span class="muted">risk score ${fmtNum(data.risk_score, 4)}</span>
        <span class="verdict-note">${esc(noteText)}</span>
      </div>
    </div>

    <div class="breakdown">
      <div class="bd"><div class="bd-key">Source account</div>
        <div class="bd-val ${data.source_account_score === null ? "is-null" : ""}">
          ${data.source_account_score === null ? "unseen" : fmtNum(data.source_account_score, 4)}
        </div></div>
      <div class="bd"><div class="bd-key">Target account</div>
        <div class="bd-val ${data.target_account_score === null ? "is-null" : ""}">
          ${data.target_account_score === null ? "unseen" : fmtNum(data.target_account_score, 4)}
        </div></div>
      <div class="bd"><div class="bd-key">Rule layer</div>
        <div class="bd-val">${fmtNum(data.rule_layer_score, 4)}</div></div>
    </div>

    ${data.reasons && data.reasons.length ? `
      <div class="section-label">Reasons</div>
      <div class="chips">
        ${data.reasons.map(reason => `
          <span class="reason-chip ${reason.includes("gnn") || reason.includes("ring") ? "is-flag" : ""}">
            ${esc(reason)}
          </span>`).join("")}
      </div>` : ""}

    <div class="status-note">
      model_status <code>${esc(data.model_status)}</code>
      ${known ? "" : " &middot; fell back to rules"}
    </div>`;

  $("#gaugeBox").appendChild(gauge(data.risk_score));
}

async function submitScore(event) {
  event.preventDefault();

  const payload = {
    source_account: $("#srcAccount").value.trim(),
    target_account: $("#tgtAccount").value.trim(),
    amount: Number($("#amount").value),
    event_type: $("#eventType").value,
  };

  $("#scoreResult").innerHTML = `<div class="skeleton" style="height:150px"></div>`;

  try {
    const response = await fetch("/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error(await response.text());

    renderScore(await response.json());

  } catch (error) {
    $("#scoreResult").innerHTML =
      `<div class="empty"><p>Scoring failed: ${esc(error.message)}</p></div>`;
  }
}

// ==========================================================
// MODEL
// ==========================================================

const METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"];
const METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"];
const SERIES_COLORS = ["var(--text-faint)", "var(--warn)", "var(--accent)"];

function groupedBars(models) {
  const width = 700, height = 260;
  const padLeft = 38, padBottom = 42, padTop = 12;
  const plotWidth = width - padLeft - 12;
  const plotHeight = height - padBottom - padTop;

  const root = svgRoot(width, height);

  for (let step = 0; step <= 5; step++) {
    const value = step / 5;
    const y = padTop + plotHeight - value * plotHeight;

    root.appendChild(svgEl("line", {
      x1: padLeft, x2: width - 12, y1: y, y2: y, class: "grid-line",
    }));

    const label = svgEl("text", { x: padLeft - 7, y: y + 3, "text-anchor": "end" });
    label.setAttribute("class", "axis-text");
    label.textContent = value.toFixed(1);
    root.appendChild(label);
  }

  const groupWidth = plotWidth / METRIC_KEYS.length;
  const barWidth = Math.min(26, (groupWidth - 16) / models.length);

  METRIC_KEYS.forEach((key, groupIndex) => {
    const groupX = padLeft + groupIndex * groupWidth;

    models.forEach((model, seriesIndex) => {
      const value = model.test[key];
      if (value === null || value === undefined) return;

      const barHeight = value * plotHeight;
      const x = groupX + groupWidth / 2
              - (models.length * barWidth) / 2 + seriesIndex * barWidth;

      const rect = svgEl("rect", {
        x: x + 1.5, y: padTop + plotHeight - barHeight,
        width: barWidth - 3, height: barHeight,
        fill: SERIES_COLORS[seriesIndex] || "var(--accent)",
        rx: 3,
      });

      const title = svgEl("title");
      title.textContent = `${model.name} · ${METRIC_LABELS[groupIndex]}: ${fmtNum(value, 3)}`;
      rect.appendChild(title);

      root.appendChild(rect);
    });

    const label = svgEl("text", {
      x: groupX + groupWidth / 2, y: height - padBottom + 17,
      "text-anchor": "middle",
    });
    label.setAttribute("class", "axis-text");
    label.textContent = METRIC_LABELS[groupIndex];
    root.appendChild(label);
  });

  return root;
}

function renderMetrics(data) {
  const models = data.models;

  const chart = $("#metricChart");
  chart.innerHTML = "";

  if (!models.length) {
    chart.innerHTML = `<div class="empty"><p>No evaluation metrics found. Run the evaluation scripts.</p></div>`;
    return;
  }

  chart.appendChild(groupedBars(models));

  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = models.map((model, index) => `
    <div class="legend-item">
      <span class="legend-swatch" style="background:${SERIES_COLORS[index] || "var(--accent)"}"></span>
      ${esc(model.name)}
    </div>`).join("");
  chart.appendChild(legend);

  $("#testNote").textContent =
    `${fmtInt(data.test_accounts)} accounts, ${fmtInt(data.test_fraud_accounts)} fraudulent`;

  $("#metricCaveat").textContent = data.caveat;

  $("#metricTable tbody").innerHTML = models.map(model => `
    <tr>
      <td>
        <div>${esc(model.name)}</div>
        <div class="muted">${esc(model.subtitle)}</div>
      </td>
      <td class="num">${fmtNum(model.test.accuracy)}</td>
      <td class="num">${fmtNum(model.test.precision)}</td>
      <td class="num">${fmtNum(model.test.recall)}</td>
      <td class="num"><strong>${fmtNum(model.test.f1)}</strong></td>
      <td class="num">${fmtNum(model.test.roc_auc)}</td>
      <td class="num">${fmtNum(model.test.average_precision)}</td>
    </tr>`).join("");

  $("#confusionGrid").innerHTML = models.map(model => {
    const matrix = model.confusion_matrix || {};
    return `
      <article class="card">
        <div class="card-head">
          <h2>${esc(model.name)}</h2>
          <span class="muted">threshold ${fmtNum(model.decision_threshold, 2)}</span>
        </div>
        <div class="cm-grid">
          <div class="cm-axis"></div>
          <div class="cm-axis">pred normal</div>
          <div class="cm-axis">pred fraud</div>

          <div class="cm-axis">actual&nbsp;normal</div>
          <div class="cm-cell is-hit">
            <div class="cm-n">${fmtInt(matrix.true_negative)}</div><div class="cm-k">correct</div></div>
          <div class="cm-cell is-miss">
            <div class="cm-n">${fmtInt(matrix.false_positive)}</div><div class="cm-k">false alarm</div></div>

          <div class="cm-axis">actual&nbsp;fraud</div>
          <div class="cm-cell is-miss">
            <div class="cm-n">${fmtInt(matrix.false_negative)}</div><div class="cm-k">missed</div></div>
          <div class="cm-cell is-hit">
            <div class="cm-n">${fmtInt(matrix.true_positive)}</div><div class="cm-k">caught</div></div>
        </div>
      </article>`;
  }).join("");
}

// ==========================================================
// BOOT
// ==========================================================

async function loadAll() {
  try {
    const [overview, metrics, rings] = await Promise.all([
      api("/api/overview"),
      api("/api/metrics"),
      api("/api/rings"),
    ]);

    state.overview = overview;
    state.metrics = metrics;
    state.rings = rings.rings;

    renderOverview(overview);
    renderModelStatus(overview.model);
    renderMetrics(metrics);
    renderRingTable(state.rings);

    $("#navRingCount").textContent = rings.count;
    $("#ringSummary").textContent =
      `${fmtInt(overview.accounts_in_rings)} accounts · ${fmtPct(overview.ring_precision)} precision`;

    if (state.rings.length && !state.selectedRing) {
      selectRing(state.rings[0].detected_ring_id);
    }

    await Promise.all([loadAccountSummary(), loadAccounts(true)]);

  } catch (error) {
    toast("Could not reach the API: " + error.message);
    renderModelStatus(null);
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("upi-console-theme", theme); } catch (_) {}
}

function init() {
  // theme
  let theme = "dark";
  try { theme = localStorage.getItem("upi-console-theme") || "dark"; } catch (_) {}
  applyTheme(theme);

  $("#themeBtn").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark"
      ? "light" : "dark";
    applyTheme(next);
  });

  // navigation
  $$(".nav-item").forEach(button => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });

  const initial = location.hash.slice(1);
  showView(VIEW_META[initial] ? initial : "overview");

  // refresh
  $("#refreshBtn").addEventListener("click", async () => {
    state.selectedRing = null;
    await loadAll();
    toast("Data reloaded");
  });

  // accounts
  $$("#levelChips .chip").forEach(chip => {
    chip.addEventListener("click", () => {
      $$("#levelChips .chip").forEach(other =>
        other.classList.toggle("is-active", other === chip));
      state.accounts.level = chip.dataset.level;
      loadAccounts(true);
    });
  });

  let searchTimer = null;
  $("#accountSearch").addEventListener("input", event => {
    clearTimeout(searchTimer);
    const value = event.target.value.trim();
    searchTimer = setTimeout(() => {
      state.accounts.search = value;
      loadAccounts(true);
    }, 220);
  });

  $("#loadMore").addEventListener("click", () => loadAccounts(false));

  $("#lookupForm").addEventListener("submit", event => {
    event.preventDefault();
    const value = $("#lookupInput").value.trim();
    if (value) lookupAccount(value);
  });

  // scoring
  $("#scoreForm").addEventListener("submit", submitScore);

  $("#fillRing").addEventListener("click", () => {
    const ring = state.rings[0];
    if (!ring) return toast("No rings available");
    const members = String(ring.accounts).split(",");
    $("#srcAccount").value = members[0] || "";
    $("#tgtAccount").value = members[1] || "";
    $("#amount").value = 75000;
  });

  $("#fillUnknown").addEventListener("click", () => {
    $("#srcAccount").value = "ACC_NEW_0001";
    $("#tgtAccount").value = "ACC_NEW_0002";
    $("#amount").value = 120000;
  });

  loadAll();
}

document.addEventListener("DOMContentLoaded", init);

})();
