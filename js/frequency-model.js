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
    const balls  = Array.from({ length: maxVal }, (_, i) => i + 1);
    const byBayes = [...balls].sort((a, b) => bayesWeights[b] - bayesWeights[a]);
    this.hotNumbers  = byBayes.slice(0, Math.ceil(maxVal * 0.33));
    this.coldNumbers = byBayes.slice(Math.floor(maxVal * 0.67));

    this.fitted = true;
    return this;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Bayesian posterior mean probability for ball b. */
  bayesianP(b) {
    return this.bayesWeights?.[b] ?? (this.pick / this.maxVal);
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

  /** Top-k balls by Bayesian posterior mean. */
  topNumbers(k = 15) {
    if (!this.fitted) return [];
    return Array.from({ length: this.maxVal }, (_, i) => i + 1)
      .sort((a, b) => this.bayesWeights[b] - this.bayesWeights[a])
      .slice(0, k)
      .map(b => ({ ball: b, count: this.counts[b], bayesP: this.bayesWeights[b].toFixed(5) }));
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
    // Weight = Bayesian posterior — naturally shrinks outliers
    return this._bayesianSample(this.bayesWeights);
  }

  _coldTicket() {
    // Inverse Bayesian: favour balls with low posterior probability
    const inv = new Array(this.maxVal + 1).fill(0);
    const maxP = Math.max(...this.bayesWeights.slice(1));
    for (let b = 1; b <= this.maxVal; b++) {
      inv[b] = maxP - this.bayesWeights[b] + 1e-6;
    }
    return this._bayesianSample(inv);
  }

  _overdueTicket() {
    // Weight = gap Z-score (positive = more overdue than expected)
    // Floor at 0 so we never anti-weight short-gap balls
    const w = new Array(this.maxVal + 1).fill(0);
    for (let b = 1; b <= this.maxVal; b++) {
      w[b] = Math.max(0, this.gapZ(b)) + 0.1; // +0.1 ensures all balls eligible
    }
    return this._bayesianSample(w);
  }

  _mixedTicket() {
    // Composite weight: 50% Bayesian frequency + 50% gap Z (normalised)
    const maxGapZ = Math.max(...this.gapZScores.slice(1), 1);
    const w = new Array(this.maxVal + 1).fill(0);
    for (let b = 1; b <= this.maxVal; b++) {
      const freqW  = this.bayesWeights[b];
      const gapW   = Math.max(0, this.gapZ(b)) / maxGapZ;
      w[b] = 0.5 * freqW + 0.5 * gapW * (this.pick / this.maxVal);
    }
    return this._bayesianSample(w);
  }

  _pairTicket() {
    // Seed with the highest-lift pair, fill with Bayesian weights
    let bestA = 1, bestB = 2, bestLift = 0;
    for (const [key, cnt] of this.pairMap) {
      const lift = cnt / this._expectedPair;
      if (lift > bestLift) {
        bestLift = lift;
        [bestA, bestB] = key.split(",").map(Number);
      }
    }
    const seed   = [bestA, bestB];
    const seedSet = new Set(seed);
    return this._bayesianSample(this.bayesWeights, seedSet)
      .concat(seed)
      .filter((v, i, a) => a.indexOf(v) === i)
      .sort((a, b) => a - b)
      .slice(0, this.pick);
  }

  generateTickets(n = 5, strategy = "all") {
    if (!this.fitted) throw new Error("Call .fit(matrix) first.");
    const strategies = {
      hot:     () => this._hotTicket(),
      cold:    () => this._coldTicket(),
      mixed:   () => this._mixedTicket(),
      pair:    () => this._pairTicket(),
      overdue: () => this._overdueTicket(),
    };
    const cycle = strategy === "all"
      ? ["hot", "mixed", "overdue", "pair", "cold"]
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
