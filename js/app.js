/**
 * app.js — Refactored application controller (Sprints 1–3)
 */
import { parseFile, detectGame, getMatrix, GAME_CONFIG } from "./data-ingestion.js";
import { buildLSTMModel, buildSequences, trainLSTM } from "./lstm-model.js";
import { generatePredictions } from "./predictor.js";
import { scoreAndRank, savePick, loadSavedPicks, deletePick, clearAllPicks } from "./scorer.js";

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  records: [], matrix: [], game: "power655",
  freqModel: null, lstmModel: null, trainedMode: null,
  chart: null, lossChart: null,
};

const $ = id => document.getElementById(id);

// ── Boot ──────────────────────────────────────────────────────────────────────
setupGameCards();
setupDropZone();
setupExpertAccordion();
setupModeSelect();
$("generate-btn").addEventListener("click", onGenerate);
$("dismiss-disclaimer").addEventListener("click", () => $("disclaimer").style.display = "none");
$("saved-toggle-btn").addEventListener("click", toggleSavedPanel);
$("clear-saved-btn").addEventListener("click", () => { clearAllPicks(); renderSavedPanel(); });
refreshSavedCount();

// ── Game Cards (click = auto-load real data) ──────────────────────────────────
function setupGameCards() {
  document.querySelectorAll(".game-card").forEach(card => {
    card.addEventListener("click", () => {
      const game = card.dataset.game;
      const urls = {
        power655: { url: "./data/power655.jsonl", file: "power655.jsonl" },
        mega645:  { url: "./data/power645.jsonl", file: "power645.jsonl" },
      };
      loadFromUrl(urls[game].url, urls[game].file, game, card);
    });
  });
}

async function loadFromUrl(url, filename, game, card) {
  document.querySelectorAll(".game-card").forEach(c => c.classList.remove("active", "loading"));
  card.classList.add("loading");
  showStatus("loading", `Loading ${filename}…`);

  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const text = await resp.text();

    state.records = parseFile(text, filename);
    state.matrix  = getMatrix(state.records);
    state.game    = game;
    state.lstmModel = null;
    state.trainedMode = null;

    $("game-select") && ($("game-select").value = game);

    card.classList.remove("loading");
    card.classList.add("active");
    showStatus("success", `✓ ${state.records.length} draws loaded`);

    // Unlock step 2 + expert card
    $("step2-card").style.opacity = "1";
    $("step2-card").style.pointerEvents = "auto";
    $("expert-card").style.opacity = "1";
    $("expert-card").style.pointerEvents = "auto";
    $("generate-btn").disabled = false;

    $("empty-state").style.display = "none";

    await renderStats();
    renderLast10();
    renderGapTracker();
    renderFreqChart();

  } catch (err) {
    card.classList.remove("loading");
    showStatus("error", `✗ ${err.message}`);
  }
}

// ── Drop zone upload (fallback) ───────────────────────────────────────────────
function setupDropZone() {
  const zone  = $("drop-zone");
  const input = $("file-input");
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", e => handleUpload(e.target.files[0]));
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.style.borderColor = "var(--purple-l)"; });
  zone.addEventListener("dragleave", () => zone.style.borderColor = "");
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.style.borderColor = "";
    if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
  });
}

async function handleUpload(file) {
  if (!file) return;
  showStatus("loading", `Reading ${file.name}…`);
  const text = await new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = e => res(e.target.result);
    r.onerror = () => rej(new Error("File read error"));
    r.readAsText(file, "utf-8");
  });
  try {
    state.records = parseFile(text, file.name);
    state.matrix  = getMatrix(state.records);
    state.game    = detectGame(state.records);
    state.lstmModel = null;

    $("drop-zone").classList.add("has-file");
    $("drop-zone").querySelector(".drop-label").textContent = `${file.name} — ${state.records.length} draws`;
    showStatus("success", `✓ ${state.records.length} draws loaded`);

    $("step2-card").style.opacity = "1"; $("step2-card").style.pointerEvents = "auto";
    $("expert-card").style.opacity = "1"; $("expert-card").style.pointerEvents = "auto";
    $("generate-btn").disabled = false;
    $("empty-state").style.display = "none";

    await renderStats();
    renderLast10();
    renderGapTracker();
    renderFreqChart();
  } catch (err) {
    showStatus("error", `✗ ${err.message}`);
  }
}

// ── Status ────────────────────────────────────────────────────────────────────
function showStatus(type, msg) {
  const el = $("status-msg");
  el.textContent = msg;
  el.className = `status-msg status-${type}`;
  el.style.display = "block";
}

// ── Expert accordion ──────────────────────────────────────────────────────────
function setupExpertAccordion() {
  $("expert-toggle").addEventListener("click", () => {
    const body  = $("expert-body");
    const arrow = $("expert-arrow");
    const open  = body.style.display !== "none";
    body.style.display  = open ? "none" : "block";
    arrow.classList.toggle("open", !open);
    $("expert-toggle").setAttribute("aria-expanded", String(!open));
  });
}

function setupModeSelect() {
  $("mode-select").addEventListener("change", e => {
    $("lstm-options").style.display = e.target.value !== "frequency" ? "block" : "none";
  });
}

// ── Stats Panel ───────────────────────────────────────────────────────────────
async function renderStats() {
  const { FrequencyModel } = await import("./frequency-model.js");
  const cfg = GAME_CONFIG[state.game];
  const fm  = new FrequencyModel(cfg.max, cfg.pick);
  fm.fit(state.matrix);
  state.freqModel = fm;

  const topN  = fm.topNumbers(10);
  const maxC  = Math.max(...topN.map(x => x.count), 1);
  $("hot-balls").innerHTML = topN.map(({ ball, count }) =>
    `<span class="ball hot" title="${count} appearances">${ball}</span>`
  ).join("");

  const allBalls = Array.from({ length: cfg.max }, (_, i) => i + 1);
  const cold = [...allBalls].sort((a, b) => fm.counts[a] - fm.counts[b]).slice(0, 10);
  $("cold-balls").innerHTML = cold.map(b =>
    `<span class="ball cold" title="${fm.counts[b]} appearances">${b}</span>`
  ).join("");

  const gaps    = fm.gapAnalysis();
  const overdue = [...allBalls].sort((a, b) => gaps[b] - gaps[a]).slice(0, 10);
  $("overdue-balls").innerHTML = overdue.map(b =>
    `<span class="ball overdue" title="Gap: ${gaps[b]} draws">${b}</span>`
  ).join("");

  $("stats-panel").style.display = "block";
}

// ── Last 10 Draws ─────────────────────────────────────────────────────────────
function renderLast10() {
  if (!state.records.length) return;
  const last10 = state.records.slice(-10).reverse();
  const hotSet = state.freqModel ? new Set(state.freqModel.hotNumbers.slice(0, 18)) : new Set();
  const tbody  = $("last10-table").querySelector("tbody");

  tbody.innerHTML = last10.map((r, i) => {
    const balls = r.numbers.map(n =>
      `<span class="draw-ball${hotSet.has(n) ? " hot-ball" : ""}">${n}</span>`
    ).join("");
    return `<tr>
      <td class="draw-num">${state.records.length - i}</td>
      <td class="draw-date">${r.date ? r.date.toString().slice(0, 10) : "—"}</td>
      <td colspan="6">${balls}</td>
    </tr>`;
  }).join("");

  $("last10-section").style.display = "block";
}

// ── Gap Tracker ───────────────────────────────────────────────────────────────
function renderGapTracker() {
  if (!state.freqModel) return;
  const gaps    = state.freqModel.gapAnalysis();
  const allBalls = Array.from({ length: GAME_CONFIG[state.game].max }, (_, i) => i + 1);
  const top15   = [...allBalls].sort((a, b) => gaps[b] - gaps[a]).slice(0, 15);
  const maxGap  = Math.max(...top15.map(b => gaps[b]), 1);

  $("gap-grid").innerHTML = top15.map(b => {
    const g   = gaps[b];
    const pct = Math.round((g / maxGap) * 100);
    const cls = g >= 20 ? "gap-critical" : g >= 10 ? "gap-high" : "gap-normal";
    return `<div class="gap-item ${cls}">
      <div class="gap-ball-num">${b}</div>
      <div class="gap-bar-wrap">
        <div class="gap-label">${g} draw${g !== 1 ? "s" : ""} ago</div>
        <div class="gap-bar"><div class="gap-bar-fill" style="width:${pct}%"></div></div>
      </div>
    </div>`;
  }).join("");

  $("gap-section").style.display = "block";
}

// ── Frequency Chart ───────────────────────────────────────────────────────────
function renderFreqChart() {
  if (!state.freqModel) return;
  const cfg    = GAME_CONFIG[state.game];
  const labels = Array.from({ length: cfg.max }, (_, i) => i + 1);
  const data   = labels.map(b => state.freqModel.counts[b] || 0);
  const hotSet = new Set(state.freqModel.hotNumbers);
  const colors = labels.map(b => hotSet.has(b) ? "rgba(168,85,247,0.85)" : "rgba(6,182,212,0.5)");

  if (state.chart) state.chart.destroy();
  state.chart = new Chart($("freq-chart").getContext("2d"), {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 3, borderSkipped: false }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `Ball ${c.label}: ${c.raw} draws` } } },
      scales: {
        x: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } }
      },
      animation: { duration: 500 }
    }
  });
  $("chart-section").style.display = "block";
}

// ── Loss Chart ────────────────────────────────────────────────────────────────
function initLossChart() {
  if (state.lossChart) state.lossChart.destroy();
  state.lossChart = new Chart($("loss-chart").getContext("2d"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Train", data: [], borderColor: "#a855f7", backgroundColor: "rgba(168,85,247,0.1)", tension: 0.4, fill: true, pointRadius: 0 },
        { label: "Val",   data: [], borderColor: "#06b6d4", backgroundColor: "rgba(6,182,212,0.08)", tension: 0.4, fill: true, pointRadius: 0, borderDash: [4,3] }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { labels: { color: "#64748b", font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } }
      }
    }
  });
}

function updateLossChart(ep, total, logs, lH, vH) {
  state.lossChart.data.labels = lH.map((_, i) => i + 1);
  state.lossChart.data.datasets[0].data = lH;
  state.lossChart.data.datasets[1].data = vH;
  state.lossChart.update("none");
  $("epoch-progress").textContent = `Epoch ${ep}/${total} · loss: ${logs.loss.toFixed(5)} · val: ${logs.val_loss.toFixed(5)}`;
}

// ── Generate ──────────────────────────────────────────────────────────────────
async function onGenerate() {
  if (!state.matrix.length) return;
  const mode     = $("mode-select").value;
  const nTickets = parseInt($("ticket-count").value, 10);
  const strategy = $("strategy-select").value;
  const seqLen   = parseInt($("seq-len").value, 10);
  const epochs   = parseInt($("epochs").value, 10);
  const game     = state.game;

  $("generate-btn").disabled = true;
  $("generate-btn").textContent = "Working…";
  $("results-section").style.display = "none";

  try {
    let lstmModel = state.lstmModel;

    if (mode !== "frequency" && (!lstmModel || state.trainedMode !== mode)) {
      const { buildLSTMModel: buildM, buildSequences: buildS, trainLSTM: trainM } = await import("./lstm-model.js");
      const cfg = GAME_CONFIG[game];
      showStatus("loading", "Building LSTM sequences…");
      const { xs, ys } = buildS(state.matrix, seqLen, cfg.max);
      lstmModel = buildM(seqLen);
      $("training-section").style.display = "block";
      initLossChart();
      showStatus("loading", "Training LSTM… (~60s)");
      await trainM(lstmModel, xs, ys, { epochs, batchSize: 32 },
        (ep, tot, logs, lH, vH) => updateLossChart(ep, tot, logs, lH, vH));
      xs.dispose(); ys.dispose();
      state.lstmModel = lstmModel;
      state.trainedMode = mode;
      showStatus("success", "✓ Training complete");
    }

    showStatus("loading", "Generating tickets…");
    const { generatePredictions: genP } = await import("./predictor.js");
    const { tickets, freqModel } = await genP({ matrix: state.matrix, game, nTickets, mode, lstmModel: state.lstmModel, seqLen, strategy });
    state.freqModel = freqModel;

    const ranked = scoreAndRank(tickets, freqModel, state.matrix);
    renderTickets(ranked, game);
    showStatus("success", `✓ ${tickets.length} tickets generated`);

  } catch (err) {
    showStatus("error", `✗ ${err.message}`);
    console.error(err);
  } finally {
    $("generate-btn").disabled = false;
    $("generate-btn").textContent = "✨ Generate My Picks";
  }
}

// ── Ticket cache for safe onclick references ──────────────────────────────────
const _ticketCache = new Map();

// ── Render Tickets ────────────────────────────────────────────────────────────
function renderTickets(ranked, game) {
  const cfg    = GAME_CONFIG[game];
  const hotSet = state.freqModel ? new Set(state.freqModel.hotNumbers) : new Set();
  const gaps   = state.freqModel ? state.freqModel.gapAnalysis() : {};

  _ticketCache.clear();
  ranked.forEach(({ ticket, score }, i) => _ticketCache.set(i, { ticket, score, game }));

  $("tickets-container").innerHTML = ranked.map(({ ticket, score }, i) => {
    const tier    = score.tier;
    const delay   = i * 70;
    const balls   = ticket.map(n => {
      const cls = hotSet.has(n) ? "tball tball-hot" : gaps[n] >= 10 ? "tball tball-overdue" : "tball";
      return `<span class="${cls}" title="Ball ${n}">${n}</span>`;
    }).join("");
    const badgeCls = `ticket-badge badge-${tier}`;
    const scoreCls = tier;

    return `<div class="ticket-card tier-${tier}" style="animation-delay:${delay}ms">
      <div class="ticket-header">
        <span class="ticket-num">Ticket ${i + 1}</span>
        <span class="${badgeCls}">${score.label}</span>
      </div>
      <div class="ticket-balls">${balls}</div>
      <div class="score-row">
        <span class="score-num ${scoreCls}">${score.total}</span>
        <div class="score-bar"><div class="score-fill ${scoreCls}" style="width:${score.total}%"></div></div>
      </div>
      <div class="score-breakdown">
        <span title="Bayesian posterior rank">📊 Bayes ${score.breakdown.bayes}</span>
        <span title="Geometric gap Z-score">⏳ Gap ${score.breakdown.gap}</span>
        <span title="Pair co-occurrence lift">🔗 Lift ${score.breakdown.pair}</span>
        <span title="Number range spread">↔ Spread ${score.breakdown.spread}</span>
      </div>
      <div class="ticket-actions">
        <button class="copy-btn" onclick="copyTicket('${ticket.join(", ")}', this)">Copy</button>
        <button class="save-btn" id="save-${i}" onclick="saveTicketByIdx(${i}, 'save-${i}')">💾 Save</button>
      </div>
    </div>`;
  }).join("");

  $("results-section").style.display = "block";
  $("tickets-container").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Saved Picks Panel ─────────────────────────────────────────────────────────
function renderSavedPanel() {
  const picks = loadSavedPicks();
  refreshSavedCount();

  if (!picks.length) {
    $("saved-section").style.display = "none";
    return;
  }

  $("saved-container").innerHTML = picks.map(p => {
    const cfg   = GAME_CONFIG[p.game] || GAME_CONFIG["power655"];
    const balls = p.ticket.map(n => `<span class="tball">${n}</span>`).join("");
    const date  = new Date(p.savedAt).toLocaleDateString("vi-VN");
    const tier  = p.tier;
    return `<div class="ticket-card tier-${tier}">
      <div class="ticket-header">
        <span class="ticket-num">${cfg.name}</span>
        <span class="ticket-badge badge-${tier}">${p.label} · ${p.score}</span>
      </div>
      <div class="ticket-balls">${balls}</div>
      <div class="ticket-actions">
        <button class="copy-btn" onclick="copyTicket('${p.ticket.join(", ")}', this)">Copy</button>
        <button class="copy-btn delete-btn" onclick="doDeletePick(${p.id})">🗑 Remove</button>
      </div>
      <p class="saved-at">Saved ${date}</p>
    </div>`;
  }).join("");

  $("saved-section").style.display = "block";
}

function refreshSavedCount() {
  const n = loadSavedPicks().length;
  $("saved-count").textContent = n;
  $("saved-count").classList.toggle("visible", n > 0);
}

function toggleSavedPanel() {
  renderSavedPanel();
  const sec = $("saved-section");
  if (sec.style.display === "none" || !sec.style.display) {
    sec.style.display = "block";
    sec.scrollIntoView({ behavior: "smooth", block: "start" });
  } else {
    sec.style.display = "none";
  }
}

// ── Global helpers ────────────────────────────────────────────────────────────
window.copyTicket = (text, btn) => {
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = "Copied!";
    setTimeout(() => btn.textContent = "Copy", 1500);
  });
};

window.saveTicketByIdx = (idx, btnId) => {
  const cached = _ticketCache.get(idx);
  if (!cached) return;
  savePick(cached.ticket, cached.score, cached.game);
  const btn = $(btnId);
  if (btn) { btn.textContent = "✓ Saved"; btn.classList.add("saved"); btn.onclick = null; }
  refreshSavedCount();
};

window.doDeletePick = id => {
  deletePick(id);
  renderSavedPanel();
};
