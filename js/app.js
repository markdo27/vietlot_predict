/**
 * app.js — Refactored application controller (Sprints 1–3)
 */
import { parseFile, detectGame, getMatrix, GAME_CONFIG } from "./data-ingestion.js";
import { buildLSTMModel, buildSequences, trainLSTM } from "./lstm-model.js";
import { generatePredictions } from "./predictor.js";
import { scoreAndRank, savePick, loadSavedPicks, deletePick, clearAllPicks } from "./scorer.js";
import { generateBao, BAO_TYPES, C } from "./bao.js";
import { analyzeRecentDraws } from "./draw-analysis.js";

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  records: [], matrix: [], game: "power655",
  freqModel: null, lstmModel: null, trainedMode: null,
  chart: null, lossChart: null,
  baoType: "bao7",
  analysisN: 10,   // number of draws to analyze
};

const $ = id => document.getElementById(id);

// ── Boot ──────────────────────────────────────────────────────────────────────
setupGameCards();
setupDropZone();
setupExpertAccordion();
setupModeSelect();
setupBao();
setupDrawAnalysisTabs();
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
    showStatus("success", `✓ Đã tải ${state.records.length} kỳ`);

    // Unlock step 2 + expert card
    $("step2-card").style.opacity = "1";
    $("step2-card").style.pointerEvents = "auto";
    $("expert-card").style.opacity = "1";
    $("expert-card").style.pointerEvents = "auto";
    $("generate-btn").disabled = false;
    unlockBaoCard();

    $("empty-state").style.display = "none";

    await renderStats();
    renderLast10();
    renderDrawAnalysis();
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
    showStatus("success", `✓ Đã tải ${state.records.length} kỳ`);

    $("step2-card").style.opacity = "1"; $("step2-card").style.pointerEvents = "auto";
    $("expert-card").style.opacity = "1"; $("expert-card").style.pointerEvents = "auto";
    $("generate-btn").disabled = false;
    $("empty-state").style.display = "none";

    await renderStats();
    renderLast10();
    renderDrawAnalysis();
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

  // Hot = top by BLENDED weight (60% recent + 40% all-time)
  const topN = fm.topNumbers(10);
  $("hot-balls").innerHTML = topN.map(({ ball, count }) =>
    `<span class="ball hot" title="${count} all-time appearances">${ball}</span>`
  ).join("");

  // Cold = bottom of blended ranking
  const allBalls = Array.from({ length: cfg.max }, (_, i) => i + 1);
  const cold = [...allBalls].sort((a, b) => fm.blendedP(a) - fm.blendedP(b)).slice(0, 10);
  $("cold-balls").innerHTML = cold.map(b =>
    `<span class="ball cold" title="${fm.counts[b]} appearances">${b}</span>`
  ).join("");

  const gaps    = fm.gapAnalysis();
  const overdue = [...allBalls].sort((a, b) => gaps[b] - gaps[a]).slice(0, 10);
  $("overdue-balls").innerHTML = overdue.map(b =>
    `<span class="ball overdue" title="Gap: ${gaps[b]} draws">${b}</span>`
  ).join("");

  $("stats-panel").style.display = "block";
  renderTrending(fm, cfg.max);
}


// ── Trending Numbers Panel ─────────────────────────────────────────────────────
function renderTrending(fm, maxVal) {
  const allBalls = Array.from({ length: maxVal }, (_, i) => i + 1);

  // Sort by trend score (recent_rate - historical_rate)
  const sorted   = [...allBalls].sort((a, b) => fm.trend(b) - fm.trend(a));
  const rising   = sorted.slice(0, 12);                          // top 12 rising
  const fading   = sorted.slice(sorted.length - 12).reverse();  // bottom 12

  $("trending-up-balls").innerHTML = rising.map(b => {
    const t    = fm.trend(b);
    const pct  = (t * 100).toFixed(1);
    return `<span class="ball trending-up" title="+${pct}% above avg in last 50 draws">${b}</span>`;
  }).join("");

  $("trending-down-balls").innerHTML = fading.map(b => {
    const t    = fm.trend(b);
    const pct  = (Math.abs(t) * 100).toFixed(1);
    return `<span class="ball trending-down" title="${pct}% below avg in last 50 draws">${b}</span>`;
  }).join("");

  $("trending-section").style.display = "block";
}


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

// ── Draw Analysis Report ──────────────────────────────────────────────────────
function renderDrawAnalysis() {
  if (!state.records.length || !state.freqModel) return;
  const cfg = GAME_CONFIG[state.game];
  const n   = state.analysisN;
  const report = analyzeRecentDraws(state.records, state.freqModel, cfg, n);
  if (!report) return;

  const { meta, summary, enriched, drawSummaries, repeatPairs } = report;

  // ── Summary boxes ──────────────────────────────────────────────────────────
  const boxes = [
    { val: meta.analyzedDraws,              label: "Kỳ Phân Tích" },
    { val: meta.uniqueBalls,                label: "Số Riêng Biệt" },
    { val: `${meta.coverageRatio}%`,        label: "Độ Phủ Bảng Số" },
    { val: summary.hotAppeared,             label: "Số HOT Xuất Hiện" },
    { val: summary.coldAppeared,            label: "Số LẠNH Xuất Hiện" },
    { val: meta.avgAppearances,             label: "TB Lần/Số" },
  ];
  $("analysis-summary-grid").innerHTML = boxes.map(b =>
    `<div class="summary-box">
       <div class="summary-box-val">${b.val}</div>
       <div class="summary-box-label">${b.label}</div>
     </div>`
  ).join("");

  // ── Significance alert ─────────────────────────────────────────────────────
  const sigEl = $("analysis-sig-alert");
  if (summary.sigBalls.length) {
    const ballPills = summary.sigBalls.map(b => `<span class="sig-ball">${b}</span>`).join("");
    sigEl.innerHTML =
      `<div class="sig-alert">⚠️ <strong>Phát hiện ${summary.sigBalls.length} số có tần suất bất thường về mặt thống kê</strong>
       &nbsp;(vượt ngưỡng Bonferroni p<0.05/55):<div class="sig-ball-list">${ballPills}</div></div>`;
    sigEl.style.display = "block";
  } else {
    sigEl.style.display = "none";
  }

  // ── Tab 1: Per-ball cards ──────────────────────────────────────────────────
  const maxExpPct = (cfg.pick / cfg.max) * 100;  // e.g. 10.9% for 6/55
  const maxBarPct = Math.max(maxExpPct * 3.5, ...enriched.map(e => e.recentPct)); // scale

  $("ball-report-grid").innerHTML = enriched.map(e => {
    // Percentage bars (capped at maxBarPct for scale)
    const recentBarW   = Math.min(100, (e.recentPct  / maxBarPct) * 100).toFixed(1);
    const alltimeBarW  = Math.min(100, (e.allTimePct / maxBarPct) * 100).toFixed(1);
    const expBarW      = Math.min(100, (maxExpPct    / maxBarPct) * 100).toFixed(1);
    const expLineLeft  = expBarW;

    // Deviation badge
    const devSign = e.deviation > 1 ? "+" : "";
    const devCls  = e.deviation > 1 ? "dev-pos" : e.deviation < -1 ? "dev-neg" : "dev-neu";
    const devBadge = `<span class="dev-badge ${devCls}">${devSign}${e.deviation}%</span>`;

    // Gap pill
    const gapCls = e.gapNow >= 15 ? "gap-warn" : e.gapNow <= 2 ? "gap-ok" : "";
    const gapLabel = e.gapNow === 0 ? "Xuất hiện kỳ này" : `Vắng ${e.gapNow} kỳ`;

    // Draw-presence dots (10 dots)
    const dots = Array.from({ length: n }, (_, idx) => {
      const appeared = e.appearedIn.includes(idx);
      const kyNum    = meta.totalDraws - n + idx + 1;
      return `<div class="draw-dot ${appeared ? 'appeared' : 'absent'}" title="Kỳ ${kyNum}">${appeared ? '●' : '○'}</div>`;
    }).join("");

    // Stat pills
    const pills = [
      `<span class="stat-pill">Z tần suất: ${e.freqZ > 0 ? '+' : ''}${e.freqZ}</span>`,
      `<span class="stat-pill">Z khoảng cách: ${e.gapZ > 0 ? '+' : ''}${e.gapZ}</span>`,
      `<span class="stat-pill">Bayes: ${e.blendedP}%</span>`,
      e.isSig ? `<span class="stat-pill sig">⚡ Ý nghĩa thống kê</span>` : "",
      `<span class="stat-pill ${gapCls}">${gapLabel}</span>`,
    ].join("");

    // Best pair
    const pairHtml = e.bestPairBall
      ? `<div class="brc-pair">🔗 Hay đi cùng nhất: <span class="brc-pair-ball">${e.bestPairBall}</span> (lift ${e.bestPairLift}×)</div>`
      : "";

    return `<div class="ball-report-card type-${e.typeClass}">
      <div class="brc-header">
        <div class="brc-ball ${e.typeClass}">${e.ball}</div>
        <div class="brc-info">
          <div class="brc-type-label">${e.typeLabel}</div>
          <div class="brc-appearances">${e.appearances} lần <span>/ ${n} kỳ</span>${devBadge}</div>
          <div class="brc-trend-label">${e.trendLabel}</div>
        </div>
      </div>

      <div class="brc-pct-section">
        <div class="brc-pct-row">
          <span class="brc-pct-label">Tỷ lệ ${n} kỳ gần</span>
          <div class="brc-pct-bar-wrap">
            <div class="brc-pct-bar-fill bar-recent" style="width:${recentBarW}%"></div>
            <div class="brc-expected-line" style="left:${expLineLeft}%" title="Kỳ vọng ${maxExpPct.toFixed(1)}%"></div>
          </div>
          <span class="brc-pct-val val-recent">${e.recentPct}%</span>
        </div>
        <div class="brc-pct-row">
          <span class="brc-pct-label">Lịch sử toàn thời</span>
          <div class="brc-pct-bar-wrap">
            <div class="brc-pct-bar-fill bar-alltime" style="width:${alltimeBarW}%"></div>
            <div class="brc-expected-line" style="left:${expLineLeft}%"></div>
          </div>
          <span class="brc-pct-val val-alltime">${e.allTimePct}%</span>
        </div>
        <div class="brc-pct-row">
          <span class="brc-pct-label">Kỳ vọng lý thuyết</span>
          <div class="brc-pct-bar-wrap">
            <div class="brc-pct-bar-fill" style="width:${expBarW}%;background:rgba(255,255,255,0.15)"></div>
          </div>
          <span class="brc-pct-val val-expected">${e.expectedPct}%</span>
        </div>
      </div>

      <div class="brc-stat-pills">${pills}</div>

      <div style="margin-bottom:8px;">
        <div style="font-size:0.65rem;color:var(--text-muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.06em;font-weight:700">Xuất hiện trong ${n} kỳ (mới → cũ)</div>
        <div class="brc-draw-dots">${dots}</div>
      </div>

      ${pairHtml}

      <div class="brc-reasons">
        <div class="brc-reasons-title">💡 Lý do xuất hiện</div>
        ${e.reasons.map(r => `<div class="brc-reason-item">${r}</div>`).join("")}
      </div>
    </div>`;
  }).join("");

  // ── Tab 2: Per-draw table ──────────────────────────────────────────────────
  const hotSet  = new Set(state.freqModel.hotNumbers);
  const coldSet = new Set(state.freqModel.coldNumbers);
  const gaps    = state.freqModel.rawGaps ?? {};

  $("draw-by-draw-tbody").innerHTML = [...drawSummaries].reverse().map(ds => {
    const ballsHtml = ds.numbers.map(b => {
      let cls = "mb-neut";
      if (hotSet.has(b))       cls = "mb-hot";
      else if (coldSet.has(b)) cls = "mb-cold";
      else if ((gaps[b] ?? 0) >= Math.round(state.freqModel.muGap)) cls = "mb-over";
      return `<span class="mini-ball ${cls}" title="Số ${b}">${b}</span>`;
    }).join("");

    const charHtml = `<span class="char-badge ${ds.drawCharacter.cls}">${ds.drawCharacter.label}</span>`;

    const hcPills = [
      ds.hotCount   ? `<span class="hc-pill hp">🔥 ${ds.hotCount} HOT</span>`    : "",
      ds.coldCount  ? `<span class="hc-pill cp">🧊 ${ds.coldCount} Lạnh</span>`  : "",
      ds.risingCount ? `<span class="hc-pill rp">📈 ${ds.risingCount} Tăng</span>` : "",
    ].join("");

    const pairHtml = ds.topPair
      ? `<span class="mini-ball mb-hot">${ds.topPair.a}</span>+<span class="mini-ball mb-hot">${ds.topPair.b}</span> <span style="font-size:0.68rem;color:var(--amber)">${ds.topPair.lift.toFixed(1)}×</span>`
      : "—";

    return `<tr>
      <td><span class="draw-ky">#${ds.idx}</span></td>
      <td class="draw-date-cell">${ds.date}</td>
      <td>${ballsHtml}</td>
      <td>${charHtml}</td>
      <td><div class="draw-hc-pills">${hcPills || '<span style="color:var(--text-muted);font-size:0.75rem">—</span>'}</div></td>
      <td>${pairHtml}</td>
    </tr>`;
  }).join("");

  // ── Tab 3: Pair co-occurrence ──────────────────────────────────────────────
  const maxLift = Math.max(...repeatPairs.map(p => parseFloat(p.lift)), 1);

  if (repeatPairs.length === 0) {
    $("pair-analysis-content").innerHTML =
      `<div class="no-pair-data">🔍 Không có cặp số nào xuất hiện ≥ 2 lần trong ${n} kỳ này.</div>`;
  } else {
    $("pair-analysis-content").innerHTML =
      `<p style="font-size:0.78rem;color:var(--text-muted);margin-bottom:14px">
         Các cặp số xuất hiện cùng nhau ≥ 2 lần trong <strong style="color:var(--cyan-l)">${n} kỳ</strong> gần nhất.
         Cột <em>Lift</em> so sánh với kỳ vọng ngẫu nhiên (> 1 = hay đi cùng nhau hơn ngẫu nhiên).
       </p>
       <table class="pair-table">
         <thead><tr><th>Cặp Số</th><th>Số Lần Cùng Xuất Hiện</th><th>Lift Lịch Sử (vs kỳ vọng)</th><th>Đánh Giá</th></tr></thead>
         <tbody>${repeatPairs.map(p => {
           const liftWidth = Math.min(100, (p.lift / maxLift) * 100).toFixed(0);
           const cntClass  = p.cnt >= 3 ? "pair-cnt-3" : "pair-cnt-2";
           const liftText  = p.lift > 1.5 ? "📌 Hay đi cùng" : p.lift > 1 ? "Trên mức ngẫu nhiên" : "Bình thường";
           return `<tr>
             <td><div class="pair-balls">
               <span class="mini-ball mb-hot">${p.a}</span>
               <span style="color:var(--text-muted);font-size:0.8rem">+</span>
               <span class="mini-ball mb-hot">${p.b}</span>
             </div></td>
             <td><span class="pair-cnt-badge ${cntClass}">${p.cnt}×</span></td>
             <td>
               <span class="lift-val">${p.lift}×</span>
               <div class="lift-bar-wrap"><div class="lift-bar-fill" style="width:${liftWidth}%"></div></div>
             </td>
             <td style="font-size:0.74rem;color:var(--text-dim)">${liftText}</td>
           </tr>`;
         }).join("")}</tbody>
       </table>`;
  }

  $("draw-analysis-section").style.display = "block";
}

// ── Draw Analysis Tabs & Controls ─────────────────────────────────────────────
function setupDrawAnalysisTabs() {
  // Tab switching
  document.querySelectorAll(".analysis-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".analysis-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".analysis-tab-panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const panel = $(tab.dataset.tab);
      if (panel) panel.classList.add("active");
    });
  });

  // Draw count buttons
  document.querySelectorAll(".draw-count-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".draw-count-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.analysisN = parseInt(btn.dataset.n, 10);
      renderDrawAnalysis();
    });
  });

  // Refresh button
  $("analysis-refresh-btn")?.addEventListener("click", renderDrawAnalysis);
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
  $("generate-btn").textContent = "Đang xử lý…";
  $("results-section").style.display = "none";

  try {
    let lstmModel = state.lstmModel;

    if (mode !== "frequency" && (!lstmModel || state.trainedMode !== mode)) {
      const { buildLSTMModel: buildM, buildSequences: buildS, trainLSTM: trainM } = await import("./lstm-model.js");
      const cfg = GAME_CONFIG[game];
      showStatus("loading", "Đang xây dựng chuỗi LSTM…");
      const { xs, ys } = buildS(state.matrix, seqLen, cfg.max);
      lstmModel = buildM(seqLen);
      $("training-section").style.display = "block";
      initLossChart();
      showStatus("loading", "Đang huấn luyện LSTM… (~60 giây)");
      await trainM(lstmModel, xs, ys, { epochs, batchSize: 32 },
        (ep, tot, logs, lH, vH) => updateLossChart(ep, tot, logs, lH, vH));
      xs.dispose(); ys.dispose();
      state.lstmModel = lstmModel;
      state.trainedMode = mode;
      showStatus("success", "✓ Huấn luyện hoàn tất");
    }

    showStatus("loading", "Đang tạo bộ số…");
    const { generatePredictions: genP } = await import("./predictor.js");
    const { tickets, freqModel } = await genP({ matrix: state.matrix, game, nTickets, mode, lstmModel: state.lstmModel, seqLen, strategy });
    state.freqModel = freqModel;

    const ranked = scoreAndRank(tickets, freqModel, state.matrix);
    renderTickets(ranked, game);
    showStatus("success", `✓ Đã tạo ${tickets.length} bộ số`);

  } catch (err) {
    showStatus("error", `✗ ${err.message}`);
    console.error(err);
  } finally {
    $("generate-btn").disabled = false;
    $("generate-btn").textContent = "✦ Tạo bộ số của tôi";
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
      return `<span class="${cls}" title="Số ${n}">${n}</span>`;
    }).join("");
    const badgeCls = `ticket-badge badge-${tier}`;
    const scoreCls = tier;

    return `<div class="ticket-card tier-${tier}" style="animation-delay:${delay}ms">
      <div class="ticket-header">
        <span class="ticket-num">Vé số ${i + 1}</span>
        <span class="${badgeCls}">${score.label}</span>
      </div>
      <div class="ticket-balls">${balls}</div>
      <div class="score-row">
        <span class="score-num ${scoreCls}">${score.total}</span>
        <div class="score-bar"><div class="score-fill ${scoreCls}" style="width:${score.total}%"></div></div>
      </div>
      <div class="score-breakdown">
        <span title="Xếp hạng Bayesian">📊 Bayes ${score.breakdown.bayes}</span>
        <span title="Z-score khoảng cách">⏳ Khoảng cách ${score.breakdown.gap}</span>
        <span title="Hệ số cặp số">🔗 Cặp ${score.breakdown.pair}</span>
        <span title="Độ trải rộng số">↔ Trải ${score.breakdown.spread}</span>
      </div>
      <div class="ticket-actions">
        <button class="copy-btn" onclick="copyTicket('${ticket.join(", ")}', this)">Sao chép</button>
        <button class="save-btn" id="save-${i}" onclick="saveTicketByIdx(${i}, 'save-${i}')">Lưu</button>
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
        <button class="copy-btn" onclick="copyTicket('${p.ticket.join(", ")}', this)">Sao chép</button>
        <button class="copy-btn delete-btn" onclick="doDeletePick(${p.id})">Xoá</button>
      </div>
      <p class="saved-at">Đã lưu: ${date}</p>
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
    btn.textContent = "Đã sao chép!";
    setTimeout(() => btn.textContent = "Sao chép", 1500);
  });
};

window.saveTicketByIdx = (idx, btnId) => {
  const cached = _ticketCache.get(idx);
  if (!cached) return;
  savePick(cached.ticket, cached.score, cached.game);
  const btn = $(btnId);
  if (btn) { btn.textContent = "✓ Đã lưu"; btn.classList.add("saved"); btn.onclick = null; }
  refreshSavedCount();
};

window.doDeletePick = id => {
  deletePick(id);
  renderSavedPanel();
};

// ── Bao System ────────────────────────────────────────────────────────────────
function setupBao() {
  document.querySelectorAll(".bao-type-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".bao-type-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.baoType = btn.dataset.bao;
      $("bao-generate-btn").disabled = !state.freqModel;
    });
  });
  // default selection
  $("bao-btn-bao7")?.classList.add("active");
  $("bao-generate-btn").addEventListener("click", onGenerateBao);
}

function unlockBaoCard() {
  const card = $("bao-card");
  if (card) { card.style.opacity = "1"; card.style.pointerEvents = "auto"; }
  $("bao-generate-btn").disabled = false;
}

async function onGenerateBao() {
  if (!state.freqModel || !state.matrix.length) return;
  const type   = state.baoType;
  const cfg    = BAO_TYPES[type];
  const maxVal = GAME_CONFIG[state.game].max;

  $("bao-generate-btn").disabled = true;
  $("bao-generate-btn").textContent = "Đang tạo…";

  try {
    // Auto-select top-N Bayesian-ranked numbers as pool
    const n    = cfg.poolSize || 9;
    const pool = Array.from({ length: maxVal }, (_, i) => i + 1)
      .sort((a, b) => state.freqModel.bayesianP(b) - state.freqModel.bayesianP(a))
      .slice(0, n);

    const result = generateBao(pool, type, maxVal);
    renderBaoResults(result);
  } catch (err) {
    showStatus("error", `✗ Bao error: ${err.message}`);
    console.error(err);
  } finally {
    $("bao-generate-btn").disabled = false;
    $("bao-generate-btn").textContent = "🎡 Tạo vé bao số";
  }
}

function renderBaoResults(result) {
  const { type, cfg, pool, tickets, winRates, costFormatted, ticketCount } = result;

  // Meta badge
  $("bao-meta-badge").textContent =
    `${cfg.label} · ${ticketCount} vé · ${costFormatted}`;

  // Info chips
  $("bao-info-row").innerHTML = [
    { label: "Pool", val: pool.join(", ") },
    { label: "Số vé", val: ticketCount.toLocaleString() },
    { label: "Tổng chi phí", val: costFormatted },
    { label: "Đảm bảo", val: cfg.guarantee },
  ].map(({ label, val }) =>
    `<div class="bao-info-chip"><span class="chip-label">${label}</span> ${val}</div>`
  ).join("");

  // Win rate table
  const n = pool.length;
  $("bao-table-body").innerHTML = winRates.map(row => {
    const gtCount = row.guaranteedTickets;
    const gtText  = gtCount > 0
      ? `<strong>${gtCount} vé trúng</strong> nếu có ${row.match} số trúng trong pool`
      : "—";
    return `<tr>
      <td><span class="bao-match-badge bao-match-${row.match}">${row.match}/6</span></td>
      <td>${row.prize}</td>
      <td class="bao-prob">${row.pct}%</td>
      <td class="bao-onein">1 in ${row.oneIn}</td>
      <td class="bao-guarantee">${gtText}</td>
    </tr>`;
  }).join("");

  // Pool ball display (once at top of ticket area)
  const poolBalls = pool.map(b => `<span class="pool-ball">${b}</span>`).join("");

  // Ticket grid
  $("bao-tickets-container").innerHTML =
    `<div class="ticket-card" style="grid-column:1/-1;background:rgba(6,182,212,0.06);border-color:rgba(6,182,212,0.2)">
      <div class="bao-pool-row">
        <p class="bao-pool-label">🎡 Selected Pool — ${n} numbers (top Bayesian rank)</p>
        <div>${poolBalls}</div>
      </div>
    </div>` +
    tickets.map((tkt, i) => {
      const balls = tkt.map(n => `<span class="tball tball-hot">${n}</span>`).join("");
      const delay = i * 40;
      return `<div class="ticket-card" style="animation-delay:${delay}ms">
        <p class="bao-ticket-num">Ticket ${i + 1} of ${tickets.length}</p>
        <div class="ticket-balls">${balls}</div>
        <div class="ticket-actions">
          <button class="copy-btn" onclick="copyTicket('${tkt.join(", ")}',this)">Copy</button>
        </div>
      </div>`;
    }).join("");

  $("bao-section").style.display = "block";
  $("bao-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

// Expose unlockBaoCard for post-data-load calls
export { unlockBaoCard };

