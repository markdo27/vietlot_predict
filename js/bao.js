/**
 * bao.js — Wheeling / Covering System Engine
 *
 * Bao 7/8/9 : Full wheel — C(n,6) tickets guaranteeing jackpot if all 6
 *              winning numbers fall within your n selected numbers.
 * Bao 5     : Abbreviated covering wheel — minimum tickets such that if
 *              ANY 5 of 6 winning numbers are in your pool, at least one
 *              ticket matches 5 numbers (5-win guarantee).
 */

export const TICKET_PRICE = 10_000; // VND per ticket

// ── Combinatorics utilities ───────────────────────────────────────────────────
/** Generate all k-combinations of arr. */
export function combinations(arr, k) {
  const result = [];
  function bt(start, combo) {
    if (combo.length === k) { result.push([...combo]); return; }
    for (let i = start; i <= arr.length - (k - combo.length); i++) {
      combo.push(arr[i]); bt(i + 1, combo); combo.pop();
    }
  }
  bt(0, []);
  return result;
}

/** C(n, k) — exact binomial coefficient. */
export function C(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  k = Math.min(k, n - k);
  let r = 1;
  for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1);
  return Math.round(r);
}

// ── Full Wheel ────────────────────────────────────────────────────────────────
/**
 * Generate all C(pool.length, 6) tickets (full wheel).
 * Guarantees jackpot if all 6 winning numbers are in pool.
 */
export function fullWheel(pool) {
  return combinations(pool, 6);
}

// ── Abbreviated 5-Guarantee Covering Wheel ────────────────────────────────────
/**
 * Greedy Set Cover for 5-guarantee:
 * Find minimum set of 6-tickets from pool such that every 5-subset of pool
 * is contained in at least one ticket.
 * Guarantee: if 5 of the 6 winning numbers are in your pool → one ticket wins 5-prize.
 */
export function coveringWheel5(pool) {
  const tickets5 = combinations(pool, 6); // candidate tickets
  const targets  = combinations(pool, 5); // all 5-subsets to cover

  // Mark coverage: target i is covered if any chosen ticket contains it
  const covered  = new Array(targets.length).fill(false);
  const chosen   = [];

  // Pre-compute which targets each ticket covers
  const ticketCovers = tickets5.map(tkt =>
    targets.map((tgt, i) => tgt.every(x => tkt.includes(x)) ? i : -1)
           .filter(i => i >= 0)
  );

  let uncovered = targets.length;
  while (uncovered > 0) {
    // Greedy: pick ticket covering most uncovered targets
    let bestIdx = -1, bestCount = -1;
    for (let t = 0; t < tickets5.length; t++) {
      const cnt = ticketCovers[t].filter(i => !covered[i]).length;
      if (cnt > bestCount) { bestCount = cnt; bestIdx = t; }
    }
    if (bestIdx === -1 || bestCount === 0) break;
    chosen.push(tickets5[bestIdx]);
    for (const i of ticketCovers[bestIdx]) {
      if (!covered[i]) { covered[i] = true; uncovered--; }
    }
  }
  return chosen;
}

// ── Win Rate Calculator ───────────────────────────────────────────────────────
/**
 * For a full wheel of n numbers in a maxVal/pick game:
 * Returns prize table — probability that winning draw has exactly j numbers
 * in your pool, and what that guarantees you.
 *
 * P(exactly j winning numbers in pool of n) =
 *   C(n, j) × C(maxVal−n, pick−j) / C(maxVal, pick)
 *
 * When j winning numbers are in pool (full wheel):
 *   - j=6: 1 jackpot ticket guaranteed
 *   - j=5: (n−5) tickets with 5-match guaranteed
 *   - j=4: C(n−4,2) tickets with 4-match guaranteed
 *   - j=3: C(n−3,3) tickets with 3-match guaranteed
 */
export function computeWinRates(n, maxVal = 55, pick = 6) {
  const total = C(maxVal, pick);
  const rows  = [];

  for (let j = pick; j >= 3; j--) {
    const p     = C(n, j) * C(maxVal - n, pick - j) / total;
    const pct   = (p * 100).toFixed(6);
    const oneIn = p > 0 ? Math.round(1 / p).toLocaleString() : "∞";

    // How many of your tickets win a j-match prize given j winning in pool?
    let guaranteedTickets = 0;
    if (j === 6) guaranteedTickets = 1;
    else if (j === 5) guaranteedTickets = n - 5;
    else if (j === 4) guaranteedTickets = C(n - 4, 2);
    else if (j === 3) guaranteedTickets = C(n - 3, 3);

    rows.push({
      match: j,
      prize: PRIZE_NAMES[j],
      probability: p,
      pct,
      oneIn,
      guaranteedTickets,
    });
  }
  return rows;
}

const PRIZE_NAMES = {
  6: "🥇 Jackpot",
  5: "🥈 2nd Prize (5/6)",
  4: "🥉 3rd Prize (4/6)",
  3: "🎖 4th Prize (3/6)",
};

// ── Bao configuration objects ─────────────────────────────────────────────────
export const BAO_TYPES = {
  bao5: {
    key:         "bao5",
    label:       "Bao 5",
    desc:        "5-Guarantee Wheel",
    poolSize:    9,
    type:        "abbreviated",
    guarantee:   "One 5-match ticket if 5 winning numbers are in your pool",
    badgeColor:  "#06b6d4",
  },
  bao7: {
    key:         "bao7",
    label:       "Bao 7",
    desc:        "Full Wheel — 7 numbers",
    poolSize:    7,
    ticketCount: C(7, 6),   // 7
    type:        "full",
    guarantee:   "Jackpot guaranteed if all 6 winning numbers are in your 7",
    badgeColor:  "#a855f7",
  },
  bao8: {
    key:         "bao8",
    label:       "Bao 8",
    desc:        "Full Wheel — 8 numbers",
    poolSize:    8,
    ticketCount: C(8, 6),   // 28
    type:        "full",
    guarantee:   "Jackpot guaranteed if all 6 winning numbers are in your 8",
    badgeColor:  "#8b5cf6",
  },
  bao9: {
    key:         "bao9",
    label:       "Bao 9",
    desc:        "Full Wheel — 9 numbers",
    poolSize:    9,
    ticketCount: C(9, 6),   // 84
    type:        "full",
    guarantee:   "Jackpot guaranteed if all 6 winning numbers are in your 9",
    badgeColor:  "#7c3aed",
  },
};

/**
 * Top-level function: generate Bao tickets and full analysis.
 * pool: array of numbers (already selected/suggested)
 * type: "bao5" | "bao7" | "bao8" | "bao9"
 * maxVal: game max ball (55 or 45)
 */
export function generateBao(pool, type, maxVal = 55) {
  const cfg     = BAO_TYPES[type];
  const n       = pool.length;
  let   tickets;

  if (type === "bao5") {
    tickets = coveringWheel5(pool);
  } else {
    tickets = fullWheel(pool);
  }

  const winRates = computeWinRates(n, maxVal, 6);
  const cost     = tickets.length * TICKET_PRICE;

  return {
    type, cfg, pool, tickets, winRates, cost,
    ticketCount: tickets.length,
    costFormatted: cost.toLocaleString("vi-VN") + " VND",
  };
}
