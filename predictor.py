"""
predictor.py
------------
High-level prediction generator.

Combines outputs from the LSTM and FrequencyModel to produce final
lottery ticket suggestions. Can run in three modes:
  - "frequency"  : pure statistical model (no TensorFlow required)
  - "lstm"       : pure LSTM-based candidates
  - "ensemble"   : merge candidates from both and de-duplicate
"""

import logging
import random
import numpy as np
from data_ingestion import GAME_CONFIG

logger = logging.getLogger(__name__)


# ── Validation helper ─────────────────────────────────────────────────────────
def validate_ticket(ticket: list[int], game: str = "power655") -> bool:
    """
    Return True if `ticket` is a valid ticket for the given game.
    Rules: exactly `pick` unique integers, all within [min, max].
    """
    cfg  = GAME_CONFIG[game]
    mn, mx, pick = cfg["min"], cfg["max"], cfg["pick"]
    return (
        len(ticket) == pick
        and len(set(ticket)) == pick
        and all(mn <= n <= mx for n in ticket)
    )


# ── LSTM-based candidates ─────────────────────────────────────────────────────
def _lstm_candidates(model,
                     last_sequence: np.ndarray,
                     n: int,
                     game: str,
                     n_perturbations: int = 50) -> list[list[int]]:
    """
    Generate `n` distinct tickets from the LSTM model by:
      1. Running one forward pass to get 6 continuous estimates.
      2. Rounding to nearest integers and resolving collisions.
      3. Repeating with small Gaussian noise on the input to get diversity.
    """
    from lstm_model import predict_next_raw

    cfg    = GAME_CONFIG[game]
    mn, mx = cfg["min"], cfg["max"]
    pick   = cfg["pick"]

    tickets:   list[list[int]] = []
    seen_sets: set[frozenset]  = set()

    for _ in range(n_perturbations):
        if len(tickets) >= n:
            break

        # Add small noise for diversity
        noise = np.random.normal(0, 0.02, last_sequence.shape).astype(np.float32)
        seq   = np.clip(last_sequence + noise, 0, 1)

        raw   = predict_next_raw(model, seq, max_val=mx)   # 6 floats
        cands = [int(round(v)) for v in raw]

        # Clamp to valid range
        cands = [max(mn, min(mx, c)) for c in cands]

        # Resolve duplicates by nudging
        resolved: list[int] = []
        used: set[int]      = set()
        for c in cands:
            while c in used:
                c = c + random.choice([-1, 1])
                c = max(mn, min(mx, c))
            resolved.append(c)
            used.add(c)

        tkt = sorted(resolved[:pick])
        if validate_ticket(tkt, game):
            fs = frozenset(tkt)
            if fs not in seen_sets:
                seen_sets.add(fs)
                tickets.append(tkt)

    return tickets


# ── Main prediction function ──────────────────────────────────────────────────
def generate_predictions(
    matrix:        np.ndarray,
    game:          str             = "power655",
    n_tickets:     int             = 5,
    mode:          str             = "ensemble",
    lstm_model                     = None,
    seq_len:       int             = 10,
    freq_strategy: str             = "all",
) -> list[list[int]]:
    """
    Generate lottery ticket predictions.

    Parameters
    ----------
    matrix        : (n_draws, 6) integer array of historical draws
    game          : "power655" or "mega645"
    n_tickets     : how many distinct tickets to produce
    mode          : "frequency" | "lstm" | "ensemble"
    lstm_model    : trained Keras model (required for "lstm" / "ensemble")
    seq_len       : LSTM look-back window
    freq_strategy : strategy passed to FrequencyModel.generate_tickets()

    Returns
    -------
    List of `n_tickets` valid, sorted integer lists
    """
    from frequency_model import FrequencyModel

    cfg    = GAME_CONFIG[game]
    mx     = cfg["max"]

    # ── Fit frequency model ───────────────────────────────────────────────────
    freq_model = FrequencyModel(max_val=mx, pick=cfg["pick"])
    freq_model.fit(matrix)

    freq_tickets: list[list[int]] = []
    lstm_tickets: list[list[int]] = []

    # ── Frequency branch ──────────────────────────────────────────────────────
    if mode in ("frequency", "ensemble"):
        freq_tickets = freq_model.generate_tickets(n=n_tickets,
                                                   strategy=freq_strategy)
        logger.info("Frequency model produced %d tickets.", len(freq_tickets))

    # ── LSTM branch ───────────────────────────────────────────────────────────
    if mode in ("lstm", "ensemble"):
        if lstm_model is None:
            logger.warning("LSTM model not provided; falling back to frequency only.")
        else:
            if len(matrix) < seq_len:
                logger.warning("Not enough history (%d draws) for seq_len=%d. "
                               "Skipping LSTM.", len(matrix), seq_len)
            else:
                normed        = (matrix / mx).astype(np.float32)
                last_seq      = normed[-seq_len:][np.newaxis, ...]  # (1, seq_len, 6)
                lstm_tickets  = _lstm_candidates(lstm_model, last_seq,
                                                 n=n_tickets, game=game)
                logger.info("LSTM model produced %d tickets.", len(lstm_tickets))

    # ── Merge & deduplicate ───────────────────────────────────────────────────
    combined:  list[list[int]] = []
    seen_sets: set[frozenset]  = set()
    for tkt in lstm_tickets + freq_tickets:
        fs = frozenset(tkt)
        if fs not in seen_sets and validate_ticket(tkt, game):
            seen_sets.add(fs)
            combined.append(tkt)
        if len(combined) >= n_tickets:
            break

    # Top-up with fresh frequency tickets if still short
    if len(combined) < n_tickets:
        extras = freq_model.generate_tickets(
            n=(n_tickets - len(combined)) * 3, strategy="hot"
        )
        for tkt in extras:
            fs = frozenset(tkt)
            if fs not in seen_sets and validate_ticket(tkt, game):
                seen_sets.add(fs)
                combined.append(tkt)
            if len(combined) >= n_tickets:
                break

    return combined[:n_tickets]
