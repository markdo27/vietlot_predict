"""
data_ingestion.py
-----------------
Handles loading, validation, and preprocessing of Vietlott historical draw data.

Supported input:
  - CSV  : date, draw_id, n1, n2, n3, n4, n5, n6   (Power 6/55)
           date, draw_id, n1, n2, n3, n4, n5, n6   (Mega 6/45)
  - JSONL: {"id": "...", "date": "...", "result": [n1, n2, n3, n4, n5, n6]}
           (format used by github.com/thanhnhu/vietlott)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Game configuration ────────────────────────────────────────────────────────
GAME_CONFIG = {
    "power655": {"name": "Power 6/55", "min": 1, "max": 55, "pick": 6},
    "mega645":  {"name": "Mega 6/45",  "min": 1, "max": 45, "pick": 6},
}


def detect_game_type(df: pd.DataFrame) -> str:
    """Infer game type from the max number seen in draw columns."""
    number_cols = [c for c in df.columns if c.startswith("n")]
    max_val = df[number_cols].max().max()
    return "power655" if max_val > 45 else "mega645"


# ── CSV loader ────────────────────────────────────────────────────────────────
def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file with columns: date, draw_id, n1, n2, n3, n4, n5, n6.
    Returns a cleaned DataFrame sorted by date (ascending).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_num_cols = ["n1", "n2", "n3", "n4", "n5", "n6"]
    missing = [c for c in required_num_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    for col in required_num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_num_cols)
    logger.info("CSV loaded: %d draws from %s", len(df), filepath)
    return df


# ── JSONL loader (thanhnhu/vietlott format) ───────────────────────────────────
def load_jsonl(filepath: str) -> pd.DataFrame:
    """
    Load a JSONL file where each line has the structure:
        {"id": "00837", "date": "2024-06-18", "result": [3, 12, 22, 35, 41, 45]}

    Returns a DataFrame with columns: date, draw_id, n1..n6
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                nums = sorted(obj.get("result", []))
                if len(nums) < 6:
                    continue
                records.append({
                    "date":    obj.get("date", ""),
                    "draw_id": obj.get("id", ""),
                    "n1": nums[0], "n2": nums[1], "n3": nums[2],
                    "n4": nums[3], "n5": nums[4], "n6": nums[5],
                })
            except json.JSONDecodeError:
                continue

    if not records:
        raise ValueError("No valid records found in JSONL file.")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    logger.info("JSONL loaded: %d draws from %s", len(df), filepath)
    return df


# ── Auto-loader ───────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Detect file format and delegate to the correct loader."""
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".csv":
        return load_csv(filepath)
    elif ext in (".jsonl", ".json"):
        return load_jsonl(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .csv or .jsonl")


# ── Preprocessing helpers ─────────────────────────────────────────────────────
def get_number_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return a (n_draws × 6) integer array of draw numbers."""
    cols = ["n1", "n2", "n3", "n4", "n5", "n6"]
    return df[cols].values.astype(int)


def build_sequences(matrix: np.ndarray,
                    seq_len: int = 10,
                    max_val: int = 55) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) pairs for an LSTM sequence model.

    Each X sample is `seq_len` consecutive draws, normalised to [0, 1].
    Each y label is the *next* draw (also normalised).

    Parameters
    ----------
    matrix  : (n_draws, 6) integer array
    seq_len : look-back window length
    max_val : maximum ball number for the game (55 or 45)

    Returns
    -------
    X : (samples, seq_len, 6)  float32
    y : (samples, 6)           float32
    """
    normed = matrix / max_val  # simple [0,1] normalisation
    X, y = [], []
    for i in range(len(normed) - seq_len):
        X.append(normed[i: i + seq_len])
        y.append(normed[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def compute_frequency_table(matrix: np.ndarray,
                             max_val: int = 55) -> dict:
    """
    Compute per-ball statistics across all historical draws.

    Returns a dict with keys:
        counts      – raw appearance count per ball
        frequency   – appearance rate  per ball
        hot_numbers – top-18 most frequent balls
        cold_numbers– bottom-18 least frequent balls
    """
    flat = matrix.flatten()
    counts = np.zeros(max_val + 1, dtype=int)
    for n in flat:
        if 1 <= n <= max_val:
            counts[n] += 1

    total_draws = len(matrix)
    frequency = counts / (total_draws * 6) if total_draws > 0 else counts

    balls = np.arange(1, max_val + 1)
    ranked = sorted(balls, key=lambda b: -counts[b])

    return {
        "counts":       counts,
        "frequency":    frequency,
        "hot_numbers":  ranked[:18],
        "cold_numbers": ranked[-18:],
    }


def compute_pair_cooccurrence(matrix: np.ndarray, max_val: int = 55) -> np.ndarray:
    """
    Build a (max_val+1) × (max_val+1) co-occurrence matrix.
    pair_matrix[i][j] = number of draws where both i and j appeared.
    """
    pair_matrix = np.zeros((max_val + 1, max_val + 1), dtype=int)
    for row in matrix:
        for a in row:
            for b in row:
                if a != b:
                    pair_matrix[a][b] += 1
    return pair_matrix
