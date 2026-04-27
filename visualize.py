"""
visualize.py
------------
Optional visualization helpers – frequency bar charts and heat maps.

Usage (standalone):
    python visualize.py --file data/power655.jsonl --game power655
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from data_ingestion import load_data, get_number_matrix, GAME_CONFIG
from frequency_model import FrequencyModel


def plot_frequency_bar(freq_model: FrequencyModel,
                       game: str = "power655",
                       top_k: int = 20,
                       save_path: str | None = None) -> None:
    """Bar chart of top-k most and least frequent balls."""
    cfg    = GAME_CONFIG[game]
    counts = freq_model.freq_table["counts"]
    balls  = list(range(1, cfg["max"] + 1))
    vals   = [counts[b] for b in balls]

    # top and bottom k
    ranked = sorted(balls, key=lambda b: -counts[b])
    hot    = ranked[:top_k]
    cold   = ranked[-top_k:]

    colors = ["#ff6b6b" if b in set(hot)
              else "#74b9ff" if b in set(cold)
              else "#dfe6e9"
              for b in balls]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(balls, vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Ball number", fontsize=12)
    ax.set_ylabel("Appearances", fontsize=12)
    ax.set_title(f"{cfg['name']} – Ball Frequency (red = hot, blue = cold)",
                 fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved frequency chart → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pair_heatmap(matrix: np.ndarray,
                      game: str = "power655",
                      top_n: int = 20,
                      save_path: str | None = None) -> None:
    """
    Heatmap of the most-active ball pairs.
    Only shows the top_n balls by total appearances for readability.
    """
    from data_ingestion import compute_pair_cooccurrence, compute_frequency_table

    cfg    = GAME_CONFIG[game]
    pm     = compute_pair_cooccurrence(matrix, cfg["max"])
    ft     = compute_frequency_table(matrix, cfg["max"])
    top    = ft["hot_numbers"][:top_n]
    sub    = pm[np.ix_(top, top)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub, xticklabels=top, yticklabels=top,
                cmap="YlOrRd", linewidths=0.3, ax=ax,
                annot=(top_n <= 20), fmt="d", annot_kws={"size": 7})
    ax.set_title(f"{cfg['name']} – Co-occurrence of Top {top_n} Balls",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved heatmap → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(history, save_path: str | None = None) -> None:
    """Plot LSTM training & validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric in zip(axes, ["loss", "mae"]):
        if metric in history.history:
            ax.plot(history.history[metric],        label="train")
            if f"val_{metric}" in history.history:
                ax.plot(history.history[f"val_{metric}"], label="val", linestyle="--")
            ax.set_title(metric.upper())
            ax.set_xlabel("Epoch")
            ax.legend()

    plt.suptitle("LSTM Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training history → {save_path}")
    else:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietlott visualization tool")
    parser.add_argument("--file", required=True,
                        help="Path to CSV or JSONL data file")
    parser.add_argument("--game", default="power655",
                        choices=["power655", "mega645"])
    parser.add_argument("--chart", default="frequency",
                        choices=["frequency", "heatmap", "both"])
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save charts (omit to display)")
    args = parser.parse_args()

    df  = load_data(args.file)
    mat = get_number_matrix(df)
    fm  = FrequencyModel(max_val=GAME_CONFIG[args.game]["max"],
                         pick=GAME_CONFIG[args.game]["pick"])
    fm.fit(mat)

    import os
    sd = args.save_dir
    if sd:
        os.makedirs(sd, exist_ok=True)

    if args.chart in ("frequency", "both"):
        plot_frequency_bar(fm, game=args.game,
                           save_path=os.path.join(sd, "frequency.png") if sd else None)
    if args.chart in ("heatmap", "both"):
        plot_pair_heatmap(mat, game=args.game,
                          save_path=os.path.join(sd, "heatmap.png") if sd else None)
