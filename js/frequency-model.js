/**
 * frequency-model.js — Bayesian-Enhanced Statistical Model
 *
 * Mathematical foundations:
 *   - Beta-Binomial conjugate model for frequency estimation
 *   - Geometric null distribution for gap analysis
 *   - Hypergeometric pair co-occurrence with lift ratio
 *   - Bonferroni-corrected Z-scores for anomaly detection
 */

// ── Bayesian Prior: Beta(α₀, β₀) ────────────────────────────────────────────
// Encoding the known ground truth: p_true = 6/55 for every ball.
// α₀ = 6  (expected successes per draw), β₀ = 49 (expected non-selections).
// This shrinks extreme sample estimates back towards the true mean,
// reducing the influence of sampling noise on small n.
const BAYES_ALPHA = 6;
const BAYES_BETA  = 49;

// ── Recency Window ────────────────────────────────────────────────────────────
// Empirical observation: last 10 draws show ~2.2 of 6 balls from all-time hot.
// We use a sliding window of W draws to capture short-term momentum.
// Recency weight multiplier gives recent draws 4× the influence of older draws.
const RECENCY_WINDOW = 50;   // draws to treat as "recent"
const RECENCY_MULT   = 4;    // weight multiplier for recent draws
// Empirical hot ratio: ~2 of 6 balls per draw are all-time hot.
// Used to calibrate the realistic mixed ticket generator.
const EMPIRICAL_HOT_PER_DRAW = 2;

export class FrequencyModel {
  constructor(maxVal = 55, pick = 6) {
    this.maxVal  = maxVal;
    this.pick    = pick;
    this.counts  = null;
    this.pairMap = null;
    this.hotNumbers  = [];
    this.coldNumbers = [];
    this._matrix = null;
    this._nDraws = 0;
    this.fitted  = false;

    // Derived statistical quantities (computed in fit())
    this.bayesWeights  = null; // Bayesian posterior means per ball
    this.gapZScores    = null; // Geometric-distribution Z-scores per ball
    this.deviationZ    = null; // Bonferroni-corrected frequency Z-scores
    this.pairLifts     = null; // Expected pair lift (observed/expected)
  }

  // ── Core Fit ───────────────────────────────────────────────────────────────
  fit(matrix) {
    this._matrix = matrix;
    this._nDraws = matrix.length;
    const n      = this._nDraws;
    const maxVal = this.maxVal;
    const p0     = this.pick / maxVal;   // true marginal probability per ball

    // ── 1. Raw frequency counts ──────────────────────────────────────────────
    const counts = new Array(maxVal + 1).fill(0);
    for (const row of matrix) {
      for (const b of row) {
        if (b >= 1 && b <= maxVal) counts[b]++;
      }
    }
    this.counts = counts;

    // ── 2. Bayesian posterior mean: E[p_b | data] ───────────────────────────
    // Model: k_b | p_b ~ Binomial(n, p_b), Prior: p_b ~ Beta(α₀, β₀)
    // Posterior: p_b | k_b ~ Beta(α₀ + k_b, β₀ + n − k_b)
    // Posterior mean: (α₀ + k_b) / (α₀ + β₀ + n)
    const denominator  = BAYES_ALPHA + BAYES_BETA + n;
    const bayesWeights = new Array(maxVal + 1).fill(0);
    for (let b = 1; b <= maxVal; b++) {
      bayesWeights[b] = (BAYES_ALPHA + counts[b]) / denominator;
    }
    this.bayesWeights = bayesWeights;

    // ── 3. Frequency deviation Z-scores (Bonferroni-corrected) ───────────────
    // H₀: p_b = p₀ = 6/maxVal
    // σ = sqrt(p₀(1−p₀)/n),  Z = (p̂_b − p₀) / σ
    // Critical threshold at α = 0.05 / maxVal (Bonferroni)
    const sigma0    = Math.sqrt(p0 * (1 - p0) / n);
    const zCritical = this._zFromAlpha(0.05 / maxVal); // ~3.3 for 55 balls
    const devZ      = new Array(maxVal + 1).fill(0);
    for (let b = 1; b <= maxVal; b++) {
      const pHat = counts[b] / n;
      devZ[b]    = (pHat - p0) / sigma0;
    }
    this.deviationZ  = devZ;
    this.zCritical   = zCritical;

    // ── 4. Gap analysis with geometric null distribution ──────────────────────
    // If X ~ Geometric(p₀), then E[X] = 1/p₀, SD[X] = sqrt((1−p₀)/p₀²)
    // Z_gap(b) = (gap_b − E[X]) / SD[X]
    // P(gap ≥ g) = (1−p₀)^(g−1)  — survival function of geometric dist.
    const rawGaps  = this._computeGaps();
    const muGap    = 1 / p0;                              // E[gap] = 55/6 ≈ 9.17
    const sdGap    = Math.sqrt((1 - p0) / (p0 * p0));    // SD[gap] ≈ 8.66
    const gapZScores = new Array(maxVal + 1).fill(0);
    const gapPVals   = new Array(maxVal + 1).fill(1);

    for (let b = 1; b <= maxVal; b++) {
      const g        = rawGaps[b];
      gapZScores[b]  = (g - muGap) / sdGap;
      // Survival probability: P(Gap ≥ g) = (1−p₀)^(g−1)
      gapPVals[b]    = Math.pow(1 - p0, Math.max(0, g - 1));
    }
    this.gapZScores   = gapZScores;
    this.gapPValues   = gapPVals;
    this.rawGaps      = rawGaps;
    this.muGap        = muGap;
    this.sdGap        = sdGap;

    // ── 5. Pair co-occurrence lift ────────────────────────────────────────────
    // Under H₀, E[co-occurrences of (a,b)] = n × P(a∩b in same draw)
    // P(both a and b drawn) = C(maxVal−2, pick−2) / C(maxVal, pick)
    //                       = [pick(pick−1)] / [maxVal(maxVal−1)]
    const pPair      = (this.pick * (this.pick - 1)) / (maxVal * (maxVal - 1));
    const expectedPair = n * pPair;
    this._expectedPair = expectedPair;

    const pairMap = new Map();
    for (const row of matrix) {
      const sorted = [...row].sort((a, b) => a - b);
      for (let i = 0; i < sorted.length; i++) {
        for (let j = i + 1; j < sorted.length; j++) {
          const key = `${sorted[i]},${sorted[j]}`;
          pairMap.set(key, (pairMap.get(key) || 0) + 1);
        }
      }
    }
    this.pairMap = pairMap;

    // ── 6. Ranked ball lists using Bayesian weights ───────────────────────────
    const balls   = Array.from({ length: maxVal }, (_, i) => i + 1);
    const byBayes = [...balls].sort((a, b) => bayesWeights[b] - bayesWeights[a]);
    this.hotNumbers  = byBayes.slice(0, Math.ceil(maxVal * 0.33));
    this.coldNumbers = byBayes.slice(Math.floor(maxVal * 0.67));

    // ── 7. Recency-weighted Bayesian model ────────────────────────────────────
    // Last RECENCY_WINDOW draws get RECENCY_MULT× weight.
    // This surfaces numbers trending up recently vs. all-time averages.
    //
    // Posterior: p_b^recent ~ Beta(α₀ + k_b^W, β₀ + W − k_b^W)
    // where k_b^W = appearances in last W draws.
    const W            = Math.min(n, RECENCY_WINDOW);
    const recentCounts = new Array(maxVal + 1).fill(0);

    for (let i = n - W; i < n; i++) {
      for (const b of matrix[i]) {
        if (b >= 1 && b <= maxVal) recentCounts[b]++;
      }
    }
    this.recentCounts = recentCounts;
    this._recentWindow = W;

    // Recent Bayesian posterior mean
    const denomRecent   = BAYES_ALPHA + BAYES_BETA + W;
    const recentBayesW  = new Array(maxVal + 1).fill(0);
    for (let b = 1; b <= maxVal; b++) {
      recentBayesW[b] = (BAYES_ALPHA + recentCounts[b]) / denomRecent;
    }
    this.recentBayesWeights = recentBayesW;

    // ── 8. Trend score T(b) = recent_rate − historical_rate ──────────────────
    // Positive  = ball appearing MORE than its all-time average in recent draws.
    // Negative  = ball appearing LESS  than its all-time average recently.
    // Normalised by the all-time standard deviation for comparability.
    const trendScores = new Array(maxVal + 1).fill(0);
    for (let b = 1; b <= maxVal; b++) {
      const recentRate = recentCounts[b] / W;
      const histRate   = counts[b] / n;
      trendScores[b]   = recentRate - histRate;
    }
    this.trendScores = trendScores;

    // ── 9. Combined (blended) weight for generation ───────────────────────────
    // Blend: 40% all-time Bayesian + 60% recency Bayesian.
    // This reflects that recent patterns matter more than distant history
    // for the next draw, while retaining long-run statistical stability.
    const blendedWeights = new Array(maxVal + 1).fill(0);
    for (let b = 1; b <= maxVal; b++) {
      blendedWeights[b] = 0.40 * bayesWeights[b] + 0.60 * recentBayesW[b];
    }
    this.blendedWeights = blendedWeights;

    // Re-rank hot/cold using blended weights (recency-aware)
    const byBlended      = [...balls].sort((a, b) => blendedWeights[b] - blendedWeights[a]);
    this.hotNumbers      = byBlended.slice(0, Math.ceil(maxVal * 0.33));
    this.coldNumbers     = byBlended.slice(Math.floor(maxVal * 0.67));
    this.recentHot       = [...balls].sort((a, b) => recentBayesW[b] - recentBayesW[a])
                                     .slice(0, Math.ceil(maxVal * 0.33));
    this.trending        = [...balls].sort((a, b) => trendScores[b] - trendScores[a]);

    this.fitted = true;
    return this;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Bayesian posterior mean probability for ball b (all-time). */
  bayesianP(b) {
    return this.bayesWeights?.[b] ?? (this.pick / this.maxVal);
  }

  /** Recency-window Bayesian posterior mean (last 50 draws). */
  recentP(b) {
    return this.recentBayesWeights?.[b] ?? (this.pick / this.maxVal);
  }

  /** Blended (40% all-time + 60% recent) posterior mean — primary ranking. */
  blendedP(b) {
    return this.blendedWeights?.[b] ?? this.bayesianP(b);
  }

  /** Trend score: positive = trending up recently, negative = declining. */
  trend(b) {
    return this.trendScores?.[b] ?? 0;
  }

  /** Gap Z-score using geometric null distribution. */
  gapZ(b) {
    return this.gapZScores?.[b] ?? 0;
  }

  /** Gap survival probability P(Gap ≥ observed gap) under H₀. */
  gapSurvival(b) {
    return this.gapPValues?.[b] ?? 1;
  }

  /** Frequency deviation Z-score (Bonferroni-adjusted critical value). */
  freqZ(b) {
    return this.deviationZ?.[b] ?? 0;
  }

  /** Is this ball's frequency statistically significant after Bonferroni? */
  isSignificant(b) {
    return Math.abs(this.freqZ(b)) > (this.zCritical ?? 3.3);
  }

  /** Pair lift = observed / expected co-occurrences. */
  pairLift(a, b) {
    const lo  = Math.min(a, b), hi = Math.max(a, b);
    const key = `${lo},${hi}`;
    const obs = this.pairMap.get(key) || 0;
    return this._expectedPair > 0 ? obs / this._expectedPair : 1;
  }

  /** Backward-compatible gapAnalysis() — returns raw gap in draws. */
  gapAnalysis() {
    if (this.rawGaps) return this.rawGaps;
    return this._computeGaps();
  }

  /** Top-k balls by BLENDED posterior (60% recent + 40% all-time). */
  topNumbers(k = 15) {
    if (!this.fitted) return [];
    return Array.from({ length: this.maxVal }, (_, i) => i + 1)
      .sort((a, b) => this.blendedWeights[b] - this.blendedWeights[a])
      .slice(0, k)
      .map(b => ({ ball: b, count: this.counts[b], bayesP: this.blendedWeights[b].toFixed(5) }));
  }

  // ── Sampling helpers ───────────────────────────────────────────────────────

  /**
   * Bayesian weighted sampling: P(select b) ∝ posterior mean.
   * Replaces raw count weighting — mathematically correct estimator.
   */
  _bayesianSample(boost = null, exclude = new Set()) {
    const ticket = [], seen = new Set(exclude);
    while (ticket.length < this.pick) {
      const weights = new Array(this.maxVal + 1).fill(0);
      for (let b = 1; b <= this.maxVal; b++) {
        if (!seen.has(b)) {
          weights[b] = boost ? boost[b] : this.bayesWeights[b];
        }
      }
      const b = this._weightedSample(weights, seen);
      ticket.push(b);
      seen.add(b);
    }
    return ticket.sort((a, b) => a - b);
  }

  _hotTicket() {
    // Weight = blended posterior (40% all-time + 60% recent)
    // Recent results dominate — reflects current momentum, not just history.
    return this._bayesianSample(this.blendedWeights);
  }

  _coldTicket() {
    // Inverse blended: favour balls that are cold in BOTH all-time and recent windows.
    const inv  = new Array(this.maxVal + 1).fill(0);
    const maxP = Math.max(...this.blendedWeights.slice(1));
    for (let b = 1; b <= this.maxVal; b++) {
      inv[b] = maxP - this.blendedWeights[b] + 1e-6;
    }
    return this._bayesianSample(inv);
  }

  _overdueTicket() {
    const w = new Array(this.maxVal + 1).fill(0);
    for (let b = 1; b <= this.maxVal; b++) {
      w[b] = Math.max(0, this.gapZ(b)) + 0.1;
    }
    return this._bayesianSample(w);
  }

  /**
   * Realistic ticket — calibrated to the empirical ratio observed in actual draws:
   * ~EMPIRICAL_HOT_PER_DRAW balls from recency-blended hot pool,
   * remaining (pick - EMPIRICAL_HOT_PER_DRAW) from the rest of the pool
   * weighted by trend score.
   *
   * This directly addresses the observed pattern that most draws contain
   * only ~2 "hot" numbers and 4 numbers from the broader distribution.
   */
  _realisticTicket() {
    const nHot   = EMPIRICAL_HOT_PER_DRAW;        // 2 from hot pool
    const nOther = this.pick - nHot;              // 4 from broad pool
    const chosen = new Set();

    // Step 1: pick nHot balls from the blended-hot pool (top 33%)
    const hotPool = this.hotNumbers.slice();
    const hotWeights = new Array(this.maxVal + 1).fill(0);
    for (const b of hotPool) hotWeights[b] = this.blendedWeights[b];
    for (let i = 0; i < nHot; i++) {
      const b = this._weightedSample(hotWeights, chosen);
      if (b) { chosen.add(b); hotWeights[b] = 0; }
    }

    // Step 2: pick nOther balls from the FULL pool (excluding chosen),
    // weighted by trend score + small floor to ensure all balls eligible.
    const trendW = new Array(this.maxVal + 1).fill(0);
    const maxT   = Math.max(...this.trendScores.slice(1), 0.001);
    for (let b = 1; b <= this.maxVal; b++) {
      if (!chosen.has(b)) {
        // Weight: normalised positive trend + small base
        trendW[b] = Math.max(0.01, 0.05 + this.trendScores[b] / maxT * 0.1);
      }
    }
    for (let i = 0; i < nOther; i++) {
      const b = this._weightedSample(trendW, chosen);
      if (b) { chosen.add(b); trendW[b] = 0; }
    }

    return [...chosen].sort((a, b) => a - b);
  }

  /**
   * Trend ticket — weights balls by their trend score.
   * Selects numbers that are appearing MORE than their historical average
   * in the last RECENCY_WINDOW draws. These are the "momentum" numbers.
   */
  _trendTicket() {
    const w = new Array(this.maxVal + 1).fill(0);
    for (let b = 1; b <= this.maxVal; b++) {
      // Floor at small positive so all balls remain eligible
      w[b] = Math.max(0.005, this.trend(b) + 0.02);
    }
    return this._bayesianSample(w);
  }

  _mixedTicket() {
    // Composite: 40% blended Bayesian + 30% trend + 30% gap Z
    const maxGapZ = Math.max(...this.gapZScores.slice(1), 1);
    const maxT    = Math.max(...this.trendScores.slice(1), 0.001);
    const w = new Array(this.maxVal + 1).fill(0);
    for (let b = 1; b <= this.maxVal; b++) {
      const freqW  = this.blendedWeights[b];
      const trendW = Math.max(0, this.trend(b)) / maxT * (this.pick / this.maxVal);
      const gapW   = Math.max(0, this.gapZ(b))  / maxGapZ * (this.pick / this.maxVal);
      w[b] = 0.40 * freqW + 0.30 * trendW + 0.30 * gapW;
    }
    return this._bayesianSample(w);
  }

  _pairTicket() {
    // Seed with highest-lift pair found in recent window first, else all-time
    let bestA = 1, bestB = 2, bestLift = 0;
    for (const [key, cnt] of this.pairMap) {
      const lift = cnt / this._expectedPair;
      if (lift > bestLift) {
        bestLift = lift;
        [bestA, bestB] = key.split(",").map(Number);
      }
    }
    const seed    = [bestA, bestB];
    const seedSet = new Set(seed);
    // Fill rest with blended weights (recency-aware)
    return this._bayesianSample(this.blendedWeights, seedSet)
      .concat(seed)
      .filter((v, i, a) => a.indexOf(v) === i)
      .sort((a, b) => a - b)
      .slice(0, this.pick);
  }

  generateTickets(n = 5, strategy = "all") {
    if (!this.fitted) throw new Error("Call .fit(matrix) first.");
    const strategies = {
      hot:       () => this._hotTicket(),
      cold:      () => this._coldTicket(),
      mixed:     () => this._mixedTicket(),
      pair:      () => this._pairTicket(),
      overdue:   () => this._overdueTicket(),
      realistic: () => this._realisticTicket(),
      trend:     () => this._trendTicket(),
    };

    // Default cycle: prioritise realistic+trend which reflect observed draw patterns
    const cycle = strategy === "all"
      ? ["realistic", "trend", "mixed", "overdue", "pair"]
      : [strategy];

    const tickets  = [];
    const seenSets = new Set();
    let attempts   = 0;

    while (tickets.length < n && attempts < n * 300) {
      attempts++;
      const fn  = strategies[cycle[tickets.length % cycle.length]];
      const tkt = fn();
      const key = tkt.join(",");
      if (!seenSets.has(key)) {
        seenSets.add(key);
        tickets.push(tkt);
      }
    }
    return tickets;
  }

  // ── Private utilities ──────────────────────────────────────────────────────

  _computeGaps() {
    const matrix = this._matrix;
    const nDraws = matrix.length;
    const lastSeen = new Map();
    for (let i = 0; i < nDraws; i++) {
      for (const b of matrix[i]) lastSeen.set(b, i);
    }
    const gaps = {};
    for (let b = 1; b <= this.maxVal; b++) {
      gaps[b] = lastSeen.has(b) ? (nDraws - 1 - lastSeen.get(b)) : nDraws;
    }
    return gaps;
  }

  _weightedSample(weights, exclude = new Set()) {
    const balls = [], w = [];
    for (let b = 1; b <= this.maxVal; b++) {
      if (!exclude.has(b)) { balls.push(b); w.push(weights[b] || 0); }
    }
    const total = w.reduce((a, v) => a + v, 0);
    if (total === 0) return balls[Math.floor(Math.random() * balls.length)];
    let r = Math.random() * total;
    for (let i = 0; i < balls.length; i++) { r -= w[i]; if (r <= 0) return balls[i]; }
    return balls[balls.length - 1];
  }

  /** Approximate inverse normal CDF (Abramowitz & Stegun rational approx.) */
  _zFromAlpha(alpha) {
    // Using the rational approximation: accurate to |ε| < 4.5e-4
    const t = Math.sqrt(-2 * Math.log(alpha / 2));
    const c = [2.515517, 0.802853, 0.010328];
    const d = [1.432788, 0.189269, 0.001308];
    return t - (c[0] + c[1]*t + c[2]*t*t) / (1 + d[0]*t + d[1]*t*t + d[2]*t*t*t);
  }
}
