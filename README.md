# 🎰 Vietlott ML Prediction Tool

> **Disclaimer:** This tool is for **educational and statistical experimentation only**. Vietlott lottery draws are certified random events — no machine learning model can reliably predict them. Play responsibly.

---

## Project Structure

```
vietlot/
├── main.py              # CLI entry point
├── data_ingestion.py    # Data loading & preprocessing
├── lstm_model.py        # TensorFlow/Keras LSTM model
├── frequency_model.py   # Statistical frequency & co-occurrence model
├── predictor.py         # Prediction generator (ensemble logic)
├── visualize.py         # Charts: frequency bars & heatmaps
├── requirements.txt     # Python dependencies
└── data/
    ├── power655_sample.jsonl   # Sample Power 6/55 data
    └── mega645_sample.csv      # Sample Mega 6/45 data
```

---

## Supported Data Formats

### CSV (any source)
```
date,draw_id,n1,n2,n3,n4,n5,n6
2024-01-02,00837,3,12,22,35,41,45
```

### JSONL (`github.com/thanhnhu/vietlott` format)
```jsonl
{"id": "00837", "date": "2024-06-18", "result": [3, 12, 22, 35, 41, 45]}
```

To use the real dataset, clone the upstream repo and point `--file` at the JSONL files:
```bash
git clone https://github.com/thanhnhu/vietlott.git upstream_data
# Then use: --file upstream_data/data/power655.jsonl
```

---

## Installation

```bash
# 1. Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **TensorFlow note:** If you only want the statistical model (no GPU/TF required), you can skip TensorFlow and use `--mode frequency`.

---

## Usage

### Frequency-only mode (fastest, no TensorFlow)
```bash
python main.py --file data/power655_sample.jsonl --game power655 --mode frequency
```

### LSTM mode (trains a neural network)
```bash
python main.py --file data/power655_sample.jsonl --game power655 --mode lstm --epochs 80
```

### Ensemble mode (LSTM + frequency, best results)
```bash
python main.py --file data/power655_sample.jsonl --game power655 --mode ensemble --epochs 100
```

### Load a previously saved LSTM model
```bash
python main.py --file data/power655_sample.jsonl --game power655 \
               --mode ensemble --load-model saved_models/power655.keras
```

### Save the trained model after training
```bash
python main.py --file data/power655_sample.jsonl --game power655 \
               --mode lstm --save-model saved_models/power655.keras
```

### Mega 6/45
```bash
python main.py --file data/mega645_sample.csv --game mega645 --mode frequency
```

### Visualizations only
```bash
python visualize.py --file data/power655_sample.jsonl --game power655 --chart both
```

---

## All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | *(required)* | Path to `.csv` or `.jsonl` draw history |
| `--game` | `power655` | `power655` or `mega645` |
| `--mode` | `frequency` | `frequency` / `lstm` / `ensemble` |
| `--tickets` | `5` | Number of ticket combinations to generate |
| `--seq-len` | `10` | LSTM look-back window (draws) |
| `--epochs` | `100` | Max LSTM training epochs |
| `--strategy` | `all` | Frequency strategy: `hot` / `cold` / `mixed` / `pair` / `all` |
| `--save-model` | *(none)* | Where to save the trained LSTM (`.keras`) |
| `--load-model` | *(none)* | Load a pre-trained LSTM (skips training) |
| `--no-chart` | `False` | Suppress matplotlib charts |

---

## How It Works

### 1. Frequency Model (`frequency_model.py`)
- Counts how often each ball has appeared across all historical draws.
- Identifies **hot numbers** (frequently drawn) and **cold/overdue numbers** (rarely drawn).
- Builds a **pair co-occurrence matrix** to find balls that often appear together.
- Sampling strategies: `hot`, `cold`, `mixed`, `pair`, `all`.

### 2. LSTM Model (`lstm_model.py`)
- Treats each draw as a time-step in a sequence.
- Two stacked LSTM layers (128 → 64 units) with Dropout regularization.
- Outputs 6 continuous values → rounded and de-duplicated to valid ball numbers.
- Uses EarlyStopping + ReduceLROnPlateau for efficient training.

### 3. Ensemble (`predictor.py`)
- Generates candidates from both models, merges them, and deduplicates.
- Falls back to frequency-only if LSTM is unavailable or produces too few unique tickets.

---

## Game Rules

| Game | Range | Pick |
|------|-------|------|
| Power 6/55 | 1 – 55 | 6 unique numbers |
| Mega 6/45  | 1 – 45 | 6 unique numbers |

All generated tickets are validated against these rules before output.
