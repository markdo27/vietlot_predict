/**
 * frequency-model.js
 * Statistical frequency, co-occurrence, and gap analysis for lottery draws.
 */

export class FrequencyModel {
  constructor(maxVal = 55, pick = 6) {
    this.maxVal = maxVal;
    this.pick   = pick;
    this.counts      = null; // index 0 unused; counts[b] = appearances of ball b
    this.pairMatrix  = null;
    this.hotNumbers  = [];
    this.coldNumbers = [];
    this._matrix     = null;
    this.fitted      = false;
  }

  fit(matrix) {
    this._matrix = matrix;
    const maxVal = this.maxVal;

    // ── Ball frequency ──────────────────────────────────────────────────────
    const counts = new Array(maxVal + 1).fill(0);
    for (const row of matrix) {
      for (const b of row) {
        if (b >= 1 && b <= maxVal) counts[b]++;
      }
    }
    this.counts = counts;

    // Rank balls
    const balls  = Array.from({length: maxVal}, (_, i) => i + 1);
    const ranked = [...balls].sort((a, b) => counts[b] - counts[a]);
    this.hotNumbers  = ranked.slice(0, Math.ceil(maxVal * 0.33));
    this.coldNumbers = ranked.slice(Math.floor(maxVal * 0.67));

    // ── Pair co-occurrence (sparse approach) ────────────────────────────────
    const pairMap = new Map();
    for (const row of matrix) {
      for (let i = 0; i < row.length; i++) {
        for (let j = i + 1; j < row.length; j++) {
          const key = `${row[i]},${row[j]}`;
          pairMap.set(key, (pairMap.get(key) || 0) + 1);
        }
      }
    }
    this.pairMap = pairMap;
    this.fitted  = true;
    return this;
  }

  // ── Gap analysis ──────────────────────────────────────────────────────────
  gapAnalysis() {
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

  // ── Sampling helpers ──────────────────────────────────────────────────────
  _weightedSample(weights, exclude) {
    const balls = [];
    const w     = [];
    for (let b = 1; b <= this.maxVal; b++) {
      if (!exclude.has(b)) {
        balls.push(b);
        w.push(weights[b] || 0);
      }
    }
    const total = w.reduce((a, v) => a + v, 0);
    if (total === 0) {
      // uniform fallback
      return balls[Math.floor(Math.random() * balls.length)];
    }
    let r = Math.random() * total;
    for (let i = 0; i < balls.length; i++) {
      r -= w[i];
      if (r <= 0) return balls[i];
    }
    return balls[balls.length - 1];
  }

  _hotTicket() {
    const ticket = [], seen = new Set();
    while (ticket.length < this.pick) {
      const b = this._weightedSample(this.counts, seen);
      ticket.push(b);
      seen.add(b);
    }
    return ticket.sort((a,b) => a-b);
  }

  _coldTicket() {
    const maxC = Math.max(...this.counts.slice(1));
    const inv  = this.counts.map(c => maxC - c + 1);
    inv[0] = 0;
    const ticket = [], seen = new Set();
    while (ticket.length < this.pick) {
      const b = this._weightedSample(inv, seen);
      ticket.push(b);
      seen.add(b);
    }
    return ticket.sort((a,b) => a-b);
  }

  _mixedTicket() {
    const hot  = this.hotNumbers.slice(0, 12);
    const cold = this.coldNumbers.slice(0, 12);
    const pool = [...new Set([...hot, ...cold])];
    const shuffled = pool.sort(() => Math.random() - 0.5);
    return shuffled.slice(0, this.pick).sort((a,b) => a-b);
  }

  _pairTicket() {
    let bestA = 1, bestB = 2, bestCount = 0;
    for (const [key, cnt] of this.pairMap) {
      if (cnt > bestCount) {
        bestCount = cnt;
        [bestA, bestB] = key.split(",").map(Number);
      }
    }
    const ticket = [bestA, bestB];
    const seen   = new Set(ticket);
    while (ticket.length < this.pick) {
      const b = this._weightedSample(this.counts, seen);
      ticket.push(b);
      seen.add(b);
    }
    return ticket.sort((a,b) => a-b);
  }

  generateTickets(n = 5, strategy = "all") {
    if (!this.fitted) throw new Error("Call .fit(matrix) first.");
    const strategies = {
      hot:   () => this._hotTicket(),
      cold:  () => this._coldTicket(),
      mixed: () => this._mixedTicket(),
      pair:  () => this._pairTicket(),
    };
    const cycle = strategy === "all"
      ? Object.keys(strategies)
      : [strategy];

    const tickets   = [];
    const seenSets  = new Set();
    let attempts    = 0;

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

  topNumbers(k = 15) {
    if (!this.fitted) return [];
    return Array.from({length: this.maxVal}, (_, i) => i + 1)
      .sort((a, b) => this.counts[b] - this.counts[a])
      .slice(0, k)
      .map(b => ({ ball: b, count: this.counts[b] }));
  }
}
