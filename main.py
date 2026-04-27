"""
main.py – Vietlott ML Prediction Tool (CLI entry point)
========================================================

Usage examples
--------------
# Frequency analysis only (no TensorFlow needed):
python main.py --file data/power655.jsonl --game power655 --mode frequency

# LSTM training + prediction:
python main.py --file data/power655.jsonl --game power655 --mode lstm --epochs 80

# Ensemble (LSTM + frequency):
python main.py --file data/power655.jsonl --game power655 --mode ensemble

# Load a previously saved LSTM model:
python main.py --file data/power655.jsonl --game power655 --mode ensemble \
               --load-model saved_models/power655.keras

# Mega 6/45:
python main.py --file data/mega645.jsonl --game mega645 --mode frequency

Optional flags:
  --tickets  N       Number of ticket combinations to generate (default 5)
  --seq-len  N       LSTM look-back window in draws           (default 10)
  --epochs   N       LSTM max training epochs                  (default 100)
  --strategy S       Frequency strategy: hot|cold|mixed|pair|all (default all)
  --no-chart         Skip showing frequency bar chart
  --save-model PATH  Where to save the trained LSTM model
  --load-model PATH  Load a pre-trained LSTM model (skips training)
"""

import argparse
import sys
import os
import logging

# ── Colourful terminal output (optional) ─────────────────────────────────────
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    C_TITLE  = Fore.CYAN  + Style.BRIGHT
    C_OK     = Fore.GREEN + Style.BRIGHT
    C_WARN   = Fore.YELLOW
    C_ERR    = Fore.RED   + Style.BRIGHT
    C_TICKET = Fore.MAGENTA + Style.BRIGHT
    C_RST    = Style.RESET_ALL
except ImportError:
    C_TITLE = C_OK = C_WARN = C_ERR = C_TICKET = C_RST = ""

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Disclaimer ────────────────────────────────────────────────────────────────
DISCLAIMER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠️  IMPORTANT DISCLAIMER ⚠️                          ║
║                                                                              ║
║  This tool is for EDUCATIONAL and STATISTICAL EXPERIMENTATION purposes only. ║
║  Vietlott draws are certified random events. No algorithm, model, or         ║
║  statistical method can reliably predict lottery outcomes.                   ║
║  Playing the lottery carries financial risk. Please play responsibly.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_banner() -> None:
    print(C_TITLE + """
 ██╗   ██╗██╗███████╗████████╗██╗      ██████╗ ████████╗
 ██║   ██║██║██╔════╝╚══██╔══╝██║     ██╔═══██╗╚══██╔══╝
 ██║   ██║██║█████╗     ██║   ██║     ██║   ██║   ██║   
 ╚██╗ ██╔╝██║██╔══╝     ██║   ██║     ██║   ██║   ██║   
  ╚████╔╝ ██║███████╗   ██║   ███████╗╚██████╔╝   ██║   
   ╚═══╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝ ╚═════╝    ╚═╝   
          ML Prediction Tool  |  Power 6/55 & Mega 6/45
""" + C_RST)
    print(C_WARN + DISCLAIMER + C_RST)


def print_tickets(tickets: list[list[int]], game_name: str) -> None:
    print(C_OK + f"\n{'─'*50}")
    print(f"  🎰  {game_name}  –  Suggested Tickets")
    print(f"{'─'*50}" + C_RST)
    for i, tkt in enumerate(tickets, 1):
        balls_str = "   ".join(f"{n:>2}" for n in tkt)
        print(C_TICKET + f"  Ticket {i:>2}:  [ {balls_str} ]" + C_RST)
    print(C_OK + f"{'─'*50}\n" + C_RST)


def print_stats(freq_model, matrix, game: str) -> None:
    """Print top-10 hot/cold numbers and overdue gaps."""
    from data_ingestion import GAME_CONFIG
    cfg     = GAME_CONFIG[game]
    top10   = freq_model.top_numbers(k=10)
    gaps    = freq_model.gap_analysis(matrix)
    overdue = sorted(range(1, cfg["max"] + 1), key=lambda b: -gaps[b])[:10]

    print(C_TITLE + "\n📊  Frequency Statistics" + C_RST)
    print("  Top-10 HOT  numbers:", " ".join(f"{b:>2}({c})" for b, c in top10))
    print("  Top-10 COLD numbers:",
          " ".join(f"{b:>2}({freq_model.freq_table['counts'][b]})"
                   for b in freq_model.freq_table["cold_numbers"][:10]))
    print("  Top-10 OVERDUE     :",
          " ".join(f"{b:>2}(gap={gaps[b]})" for b in overdue))
    print()


# ── Argument parsing ──────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vietlott ML Prediction Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file",       required=True,
                   help="Path to historical draw file (.csv or .jsonl)")
    p.add_argument("--game",       default="power655",
                   choices=["power655", "mega645"],
                   help="Game type")
    p.add_argument("--mode",       default="frequency",
                   choices=["frequency", "lstm", "ensemble"],
                   help="Prediction mode")
    p.add_argument("--tickets",    type=int, default=5,
                   help="Number of ticket combinations to generate")
    p.add_argument("--seq-len",    type=int, default=10,
                   help="LSTM look-back window (in draws)")
    p.add_argument("--epochs",     type=int, default=100,
                   help="Max LSTM training epochs")
    p.add_argument("--strategy",   default="all",
                   choices=["hot", "cold", "mixed", "pair", "all"],
                   help="Frequency model sampling strategy")
    p.add_argument("--save-model", default=None,
                   help="Path to save the trained LSTM model (.keras)")
    p.add_argument("--load-model", default=None,
                   help="Path to load a pre-trained LSTM model (skip training)")
    p.add_argument("--no-chart",   action="store_true",
                   help="Suppress the frequency bar chart")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    print_banner()

    # ── Load data ─────────────────────────────────────────────────────────────
    from data_ingestion import load_data, get_number_matrix, GAME_CONFIG
    print(C_OK + f"[1/4] Loading data from: {args.file}" + C_RST)
    try:
        df     = load_data(args.file)
        matrix = get_number_matrix(df)
    except (FileNotFoundError, ValueError) as exc:
        print(C_ERR + f"ERROR loading data: {exc}" + C_RST)
        sys.exit(1)

    cfg = GAME_CONFIG[args.game]
    print(f"      Game   : {cfg['name']}")
    print(f"      Draws  : {len(matrix)}")
    print(f"      Mode   : {args.mode.upper()}")

    # ── Frequency model ───────────────────────────────────────────────────────
    from frequency_model import FrequencyModel
    print(C_OK + "\n[2/4] Fitting frequency model …" + C_RST)
    freq_model = FrequencyModel(max_val=cfg["max"], pick=cfg["pick"])
    freq_model.fit(matrix)
    print_stats(freq_model, matrix, args.game)

    # ── LSTM (optional) ───────────────────────────────────────────────────────
    lstm_model = None
    history    = None

    if args.mode in ("lstm", "ensemble"):
        if args.load_model:
            try:
                from lstm_model import load_lstm_model
                print(C_OK + f"[3/4] Loading saved LSTM from {args.load_model} …" + C_RST)
                lstm_model = load_lstm_model(args.load_model)
            except Exception as exc:
                print(C_WARN + f"WARNING: Could not load model ({exc}). "
                               "Falling back to frequency mode." + C_RST)
        else:
            try:
                from data_ingestion import build_sequences
                from lstm_model import train_lstm

                print(C_OK + "[3/4] Preparing LSTM sequences …" + C_RST)
                X, y = build_sequences(matrix,
                                       seq_len=args.seq_len,
                                       max_val=cfg["max"])
                if len(X) < 20:
                    print(C_WARN + f"WARNING: Only {len(X)} training samples. "
                                   "Consider a longer dataset." + C_RST)

                print(C_OK + "      Training LSTM …" + C_RST)
                lstm_model, history = train_lstm(
                    X, y,
                    seq_len=args.seq_len,
                    epochs=args.epochs,
                    save_path=args.save_model,
                )
            except Exception as exc:
                print(C_WARN + f"WARNING: LSTM training failed ({exc}). "
                               "Falling back to frequency mode." + C_RST)
    else:
        print(C_OK + "[3/4] Skipping LSTM (frequency-only mode)." + C_RST)

    # ── Generate predictions ──────────────────────────────────────────────────
    print(C_OK + "\n[4/4] Generating predictions …" + C_RST)
    from predictor import generate_predictions

    tickets = generate_predictions(
        matrix        = matrix,
        game          = args.game,
        n_tickets     = args.tickets,
        mode          = args.mode if lstm_model is not None else "frequency",
        lstm_model    = lstm_model,
        seq_len       = args.seq_len,
        freq_strategy = args.strategy,
    )

    print_tickets(tickets, cfg["name"])

    # ── Optional chart ────────────────────────────────────────────────────────
    if not args.no_chart:
        try:
            from visualize import plot_frequency_bar
            print(C_OK + "Displaying frequency chart (close window to exit) …" + C_RST)
            plot_frequency_bar(freq_model, game=args.game)
        except Exception as exc:
            print(C_WARN + f"Chart skipped: {exc}" + C_RST)

    # ── Show training loss curve if LSTM was trained ──────────────────────────
    if history is not None and not args.no_chart:
        try:
            from visualize import plot_training_history
            plot_training_history(history)
        except Exception as exc:
            print(C_WARN + f"History chart skipped: {exc}" + C_RST)

    print(C_WARN + "Remember: lottery draws are random. Good luck! 🍀\n" + C_RST)


if __name__ == "__main__":
    main()
