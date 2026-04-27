/**
 * scorer.js
 * Ticket Statistical Confidence Score engine.
 * 
 * Score (0–100) = Hot Ratio×40 + Overdue Ratio×30 + Pair Strength×20 + Range Spread×10
 */

/**
 * Score a single ticket against the fitted FrequencyModel.
 * @param {number[]}       ticket      – sorted array of 6 ball numbers
 * @param {FrequencyModel} freqModel   – fitted model
 * @param {number[][]}     matrix      – full draw matrix (for gap analysis)
 * @returns {{ total: number, breakdown: object }}
 */
export function scoreTicket(ticket, freqModel, matrix) {
  const maxVal   = freqModel.maxVal;
  const nDraws   = matrix.length;
  const counts   = freqModel.counts;

  // ── 1. Hot ratio (40%) ──────────────────────────────────────────────────
  // A ball is "hot" if it's in the top 33% by appearance count
  const threshold = Math.ceil(maxVal * 0.33);
  const hotSet    = new Set(freqModel.hotNumbers.slice(0, threshold));
  const hotCount  = ticket.filter(b => hotSet.has(b)).length;
  const hotScore  = (hotCount / ticket.length) * 100;

  // ── 2. Overdue ratio (30%) ──────────────────────────────────────────────
  // A ball is "overdue" if it hasn't appeared in the last 10+ draws
  const gaps         = freqModel.gapAnalysis();
  const overdueCount = ticket.filter(b => gaps[b] >= 10).length;
  const overdueScore = (overdueCount / ticket.length) * 100;

  // ── 3. Pair co-occurrence strength (20%) ────────────────────────────────
  // Average pairwise co-occurrence score, normalised against expected random rate
  const expectedPairRate = (nDraws * 6 * 5) / (2 * maxVal * (maxVal - 1)); // rough E[pair count]
  let pairTotal = 0, pairCount = 0;
  for (let i = 0; i < ticket.length; i++) {
    for (let j = i + 1; j < ticket.length; j++) {
      const a   = Math.min(ticket[i], ticket[j]);
      const b   = Math.max(ticket[i], ticket[j]);
      const key = `${a},${b}`;
      pairTotal += freqModel.pairMap.get(key) || 0;
      pairCount++;
    }
  }
  const avgPair   = pairCount > 0 ? pairTotal / pairCount : 0;
  const pairScore = Math.min(100, (avgPair / Math.max(1, expectedPairRate)) * 50);

  // ── 4. Range spread (10%) ───────────────────────────────────────────────
  // Reward tickets that cover a wide numerical spread (not all low or all high)
  const min     = Math.min(...ticket);
  const max     = Math.max(...ticket);
  const spread  = max - min;
  const spreadScore = Math.min(100, (spread / (maxVal - 1)) * 100);

  // ── Weighted composite ──────────────────────────────────────────────────
  const total = Math.round(
    hotScore    * 0.40 +
    overdueScore * 0.30 +
    pairScore   * 0.20 +
    spreadScore * 0.10
  );

  return {
    total: Math.max(0, Math.min(100, total)),
    breakdown: {
      hot:     Math.round(hotScore),
      overdue: Math.round(overdueScore),
      pair:    Math.round(pairScore),
      spread:  Math.round(spreadScore),
    },
    tier: total >= 70 ? "gold" : total >= 40 ? "neutral" : "low",
    label: total >= 70 ? "Strong Pick" : total >= 40 ? "Moderate" : "Low Signal",
  };
}

/**
 * Score all tickets and return them sorted best-first.
 */
export function scoreAndRank(tickets, freqModel, matrix) {
  return tickets
    .map(tkt => ({ ticket: tkt, score: scoreTicket(tkt, freqModel, matrix) }))
    .sort((a, b) => b.score.total - a.score.total);
}

// ── localStorage persistence ───────────────────────────────────────────────
const STORAGE_KEY = "vietlott_saved_picks";
const MAX_SAVED   = 20;

export function loadSavedPicks() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch { return []; }
}

export function savePick(ticket, scoreResult, game) {
  const picks = loadSavedPicks();
  const entry = {
    id:        Date.now(),
    ticket,
    game,
    score:     scoreResult.total,
    tier:      scoreResult.tier,
    label:     scoreResult.label,
    breakdown: scoreResult.breakdown,
    savedAt:   new Date().toISOString(),
  };
  picks.unshift(entry);
  if (picks.length > MAX_SAVED) picks.length = MAX_SAVED;
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(picks)); } catch {}
  return picks;
}

export function deletePick(id) {
  const picks = loadSavedPicks().filter(p => p.id !== id);
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(picks)); } catch {}
  return picks;
}

export function clearAllPicks() {
  try { localStorage.removeItem(STORAGE_KEY); } catch {}
}
