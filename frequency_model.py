"""
frequency_model.py
------------------
Statistical frequency & co-occurrence model for Vietlott.

Strategies implemented
───────────────────────
1. HOT   – favour the most-drawn numbers
2. COLD  – favour the least-drawn numbers (overdue)
3. MIXED – blend hot and cold (mirrors most human "system" play)
4. PAIR  – seed each ticket with the strongest co-occurring pair,
            then fill remaining slots from hot numbers

All strategies sample *without replacement* and enforce game rules
(unique integers, correct range).
"""

import logging
import random
import numpy as np
from collections import Counter
from data_ingestion import compute_frequency_table, compute_pair_cooccurrence

logger = logging.getLogger(__name__)


class FrequencyModel:
    """
    Fits on a (n_draws × 6) integer matrix and exposes
    a `generate_tickets()` method.
    """

    def __init__(self, max_val: int = 55, pick: int = 6):
        self.max_val = max_val
        self.pick = pick
        self.freq_table: dict | None = None
        self.pair_matrix: np.ndarray | None = None
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────
    def fit(self, matrix: np.ndarray) -> "FrequencyModel":
        """
        Compute frequency and co-occurrence statistics from historical data.

        Parameters
        ----------
        matrix : (n_draws, 6) integer array of historical draws
        """
        if len(matrix) == 0:
            raise ValueError("Cannot fit on an empty draw matrix.")

        self.freq_table  = compute_frequency_table(matrix, self.max_val)
        self.pair_matrix = compute_pair_cooccurrence(matrix, self.max_val)
        self._fitted = True
        logger.info(
            "FrequencyModel fitted on %d draws. "
            "Top-5 hot: %s | Top-5 cold: %s",
            len(matrix),
            self.freq_table["hot_numbers"][:5],
            self.freq_table["cold_numbers"][:5],
        )
        return self

    # ── Internal samplers ─────────────────────────────────────────────────────
    def _weighted_sample(self, weights_array: np.ndarray,
                         exclude: set) -> int:
        """
        Sample one ball proportionally to `weights_array`,
        excluding numbers in `exclude`.
        """
        balls = [b for b in range(1, self.max_val + 1) if b not in exclude]
        w     = np.array([weights_array[b] for b in balls], dtype=float)
        w_sum = w.sum()
        if w_sum == 0:
            return random.choice(balls)
        probs = w / w_sum
        return int(np.random.choice(balls, p=probs))

    def _hot_ticket(self) -> list[int]:
        """Sample a ticket biased toward high-frequency numbers."""
        w = self.freq_table["counts"].astype(float)
        ticket, seen = [], set()
        while len(ticket) < self.pick:
            b = self._weighted_sample(w, seen)
            ticket.append(b)
            seen.add(b)
        return sorted(ticket)

    def _cold_ticket(self) -> list[int]:
        """Sample a ticket biased toward low-frequency (overdue) numbers."""
        counts = self.freq_table["counts"].astype(float)
        max_c  = counts[1:].max()
        # invert: give high weight to rarely seen balls
        w      = (max_c - counts + 1).clip(min=0)
        ticket, seen = [], set()
        while len(ticket) < self.pick:
            b = self._weighted_sample(w, seen)
            ticket.append(b)
            seen.add(b)
        return sorted(ticket)

    def _mixed_ticket(self) -> list[int]:
        """
        Build a ticket by alternating hot and cold picks.
        3 numbers from hot pool, 3 from cold pool.
        """
        hot  = list(self.freq_table["hot_numbers"])
        cold = list(self.freq_table["cold_numbers"])
        pool = list(dict.fromkeys(hot[:12] + cold[:12]))  # deduplicate, order kept
        random.shuffle(pool)
        return sorted(pool[: self.pick])

    def _pair_ticket(self) -> list[int]:
        """
        Seed the ticket with the strongest co-occurring pair,
        then fill remaining slots from hot numbers.
        """
        pm   = self.pair_matrix
        best_a, best_b, best_count = 1, 2, 0
        for a in range(1, self.max_val + 1):
            for b in range(a + 1, self.max_val + 1):
                if pm[a][b] > best_count:
                    best_count = pm[a][b]
                    best_a, best_b = a, b

        ticket = [best_a, best_b]
        seen   = set(ticket)
        w      = self.freq_table["counts"].astype(float)
        while len(ticket) < self.pick:
            b = self._weighted_sample(w, seen)
            ticket.append(b)
            seen.add(b)
        return sorted(ticket)

    # ── Public API ────────────────────────────────────────────────────────────
    def generate_tickets(self, n: int = 5,
                         strategy: str = "mixed") -> list[list[int]]:
        """
        Generate `n` distinct valid lottery tickets.

        Parameters
        ----------
        n        : number of tickets to generate
        strategy : one of "hot", "cold", "mixed", "pair", "all"
                   "all" cycles through all strategies in order.

        Returns
        -------
        list of n sorted lists, each containing `self.pick` integers
        """
        if not self._fitted:
            raise RuntimeError("Call .fit(matrix) before generating tickets.")

        strategies = {
            "hot":   self._hot_ticket,
            "cold":  self._cold_ticket,
            "mixed": self._mixed_ticket,
            "pair":  self._pair_ticket,
        }
        cycle = (list(strategies.keys()) * (n // 4 + 2))[:n] \
                if strategy == "all" else [strategy] * n

        tickets: list[list[int]] = []
        seen_sets: set[frozenset] = set()
        max_attempts = n * 200

        for attempt in range(max_attempts):
            if len(tickets) >= n:
                break
            fn  = strategies.get(cycle[len(tickets)], self._mixed_ticket)
            tkt = fn()
            fs  = frozenset(tkt)
            if fs not in seen_sets:
                seen_sets.add(fs)
                tickets.append(tkt)

        if len(tickets) < n:
            logger.warning("Could only generate %d unique tickets (requested %d).",
                           len(tickets), n)
        return tickets

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def top_numbers(self, k: int = 10) -> list[tuple[int, int]]:
        """Return top-k (ball, count) by frequency."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        counts = self.freq_table["counts"]
        return sorted(
            [(b, counts[b]) for b in range(1, self.max_val + 1)],
            key=lambda x: -x[1]
        )[:k]

    def gap_analysis(self, matrix: np.ndarray) -> dict[int, int]:
        """
        For each ball, compute the number of draws since it last appeared.
        Balls that have never appeared get gap = len(matrix).
        """
        last_seen = {}
        for i, row in enumerate(matrix):
            for b in row:
                last_seen[b] = i
        n_draws = len(matrix)
        gaps = {}
        for b in range(1, self.max_val + 1):
            gaps[b] = n_draws - 1 - last_seen.get(b, -1)
        return gaps
