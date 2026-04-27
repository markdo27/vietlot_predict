/**
 * scorer.js — Bayesian-Enhanced Ticket Scoring Engine
 *
 * Score S(ω) ∈ [0, 100] is a composite of four statistically grounded components:
 *
 *   S = 0.40 · S_bayes  + 0.30 · S_gap  + 0.20 · S_pair  + 0.10 · S_spread
 *
 *   S_bayes  — Mean Bayesian posterior rank percentile of ticket balls
 *               Uses Beta(α₀+k, β₀+n−k) posterior, NOT raw counts.
 *               Range: balls ranked by posterior mean, top ball = 100.
 *
 *   S_gap    — Geometric Z-score of gap, normalised to [0,100].
 *               Z_gap(b) = (gap_b − μ_G) / σ_G  where G ~ Geometric(6/maxVal).
 *               Positive Z = ball is overdue beyond expected geometric mean.
 *               Score = Φ(Z_gap) × 100  (CDF of standard normal).
 *
 *   S_pair   — Mean pair lift ratio, normalised to [0, 100].
 *               Lift = observed_pairs / expected_pairs under hypergeometric null.
 *               Lift > 1 means the pair co-occurs more than chance predicts.
 *
 *   S_spread — Range entropy of ticket as fraction of maximum possible spread.
 *               Penalises clustering; rewards uniform coverage of the number line.
 *
 * IMPORTANT: A high score means the ticket reflects historically anomalous
 * statistical patterns. It does NOT imply a higher probability of winning.
 * The ground truth probability remains 1/C(maxVal,6) for all tickets.
 */

// ── localStorage constants ────────────────────────────────────────────────────
const STORAGE_KEY = "vietlott_saved_picks";
const MAX_SAVED   = 20;

// ── Standard normal CDF approximation ────────────────────────────────────────
// Hart (1968) rational approximation, |ε| < 7.5 × 10⁻⁸
function normalCDF(z) {
  if (z < -8) return 0;
  if (z >  8) return 1;
  const sign = z >= 0 ? 1 : -1;
  const x    = Math.abs(z) / Math.SQRT2;
  // erfc approximation via Horner's method
  const t    = 1 / (1 + 0.3275911 * x);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
               t * (-1.453152027 + t * 1.061405429))));
  const erfc = poly * Math.exp(-x * x);
  return 0.5 * (1 + sign * (1 - erfc));
}

/**
 * Score a single ticket against the fitted FrequencyModel.
 * @param {number[]}       ticket     — sorted array of 6 ball numbers
 * @param {FrequencyModel} freqModel  — fitted Bayesian model
 * @param {number[][]}     matrix     — full draw matrix
 * @returns {{ total, breakdown, tier, label }}
 */
export function scoreTicket(ticket, freqModel, matrix) {
  const maxVal = freqModel.maxVal;
  const n      = matrix.length;

  // ── Component 1: Bayesian Posterior Rank Score (40%) ─────────────────────
  // Rank all balls by posterior mean (1 = hottest).
  // Score = mean percentile rank of ticket balls (100 = all balls top-third).
  const allBalls = Array.from({ length: maxVal }, (_, i) => i + 1);
  const sorted   = [...allBalls].sort((a, b) =>
    freqModel.bayesianP(b) - freqModel.bayesianP(a)
  );
  const rankMap = new Map(sorted.map((b, i) => [b, i + 1]));

  const rankPercentiles = ticket.map(b => {
    const rank = rankMap.get(b) || maxVal;
    return (1 - (rank - 1) / (maxVal - 1)) * 100; // 100 = rank 1, 0 = rank maxVal
  });
  const bayesScore = rankPercentiles.reduce((s, v) => s + v, 0) / ticket.length;

  // ── Component 2: Geometric Gap Score (30%) ────────────────────────────────
  // Z_gap(b) = (gap_b − μ_G) / σ_G,  where μ_G = maxVal/pick, σ_G = sqrt((1−p)/p²)
  // Map each Z_gap through Φ (normal CDF) to get a probability ∈ (0,1).
  // Score = mean Φ(Z_gap) × 100 across ticket balls.
  // Φ(0) = 0.5 (ball at exactly expected gap → neutral score of 50).
  // Φ(+∞) → 1.0 (extremely overdue → score approaches 100).
  const gapScores = ticket.map(b => normalCDF(freqModel.gapZ(b)) * 100);
  const gapScore  = gapScores.reduce((s, v) => s + v, 0) / ticket.length;

  // ── Component 3: Pair Lift Score (20%) ───────────────────────────────────
  // For all C(6,2) = 15 pairs in the ticket, compute mean lift.
  // Expected lift under H₀ = 1.0. Normalise: lift 2 → 100, lift 0 → 0.
  let liftSum = 0, liftCount = 0;
  for (let i = 0; i < ticket.length; i++) {
    for (let j = i + 1; j < ticket.length; j++) {
      liftSum += freqModel.pairLift(ticket[i], ticket[j]);
      liftCount++;
    }
  }
  const avgLift   = liftCount > 0 ? liftSum / liftCount : 1;
  // Normalise: lift of 2× expected → 100, lift of 0 → 0, expected (1×) → 50
  const pairScore = Math.min(100, Math.max(0, avgLift * 50));

  // ── Component 4: Range Spread Score (10%) ────────────────────────────────
  // Penalise clustering. Score = (max − min) / (maxVal − 1) × 100.
  // A ticket spanning [1, 55] scores 100; [20, 25] scores ~9.
  const min     = Math.min(...ticket);
  const max     = Math.max(...ticket);
  const spread  = Math.min(100, ((max - min) / (maxVal - 1)) * 100);

  // ── Composite score ───────────────────────────────────────────────────────
  const total = Math.round(
    bayesScore * 0.40 +
    gapScore   * 0.30 +
    pairScore  * 0.20 +
    spread     * 0.10
  );
  const clamped = Math.max(0, Math.min(100, total));

  const tier  = clamped >= 70 ? "gold" : clamped >= 40 ? "neutral" : "low";
  const label = clamped >= 70 ? "Strong Pick" : clamped >= 40 ? "Moderate" : "Low Signal";

  return {
    total: clamped,
    breakdown: {
      bayes:   Math.round(bayesScore),
      gap:     Math.round(gapScore),
      pair:    Math.round(pairScore),
      spread:  Math.round(spread),
    },
    tier,
    label,
    // Expose raw Z-scores and lifts for transparency
    meta: {
      avgGapZ:     +(gapScores.map((_, i) => freqModel.gapZ(ticket[i]))
                      .reduce((s, v) => s + v, 0) / ticket.length).toFixed(3),
      avgBayesP:   +(ticket.map(b => freqModel.bayesianP(b))
                      .reduce((s, v) => s + v, 0) / ticket.length).toFixed(5),
      avgPairLift: +avgLift.toFixed(3),
    }
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

// ── localStorage persistence ──────────────────────────────────────────────────
export function loadSavedPicks() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
  catch { return []; }
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
    meta:      scoreResult.meta || {},
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
