# Chess Blunder Predictor

A real-time chess blunder prediction system that streams any live Lichess game and outputs a blunder risk score after every move — before the next move is made.

Built with a 2-layer LSTM trained on 490,000+ rated games from the Lichess open database, combining clock psychology and Stockfish position evaluations to predict when a player is about to make a critical mistake.

---

## Demo

```
$ python predict.py --live abcd1234 --player black

Connecting to live game: Firouzja vs Carlsen...
Streaming moves...

Move 01 | e2e4   | Risk:  4% ✅ | White: 52% | Clock: 5:00
Move 02 | e7e5   | Risk:  3% ✅ | White: 51% | Clock: 4:58
Move 03 | g1f3   | Risk:  6% ✅ | White: 53% | Clock: 4:55
...
Move 24 | d1d3   | Risk: 61% ⚠️  | White: 67% | Clock: 1:42  ← pressure building
Move 25 | f1d1   | Risk: 88% 🔴 | White: 31% | Clock: 0:45  ← called it
Move 26 | g4f3?? | BLUNDER CONFIRMED 💀
```

---

## How It Works

The model learns that blunders aren't random — they follow patterns:

- **Time pressure** builds over several moves before the mistake
- **Eval drift** indicates a position becoming harder to handle
- **Move complexity** spikes when players face difficult decisions
- **Player ELO** calibrates expectations for a given skill level
- **Game phase** changes how much time pressure matters

Each prediction uses a sliding window of the last 10 moves as context, capturing the psychological arc leading up to a blunder.

---

## Architecture

```
Input: last 10 moves × 11 features
       (time_spent, time_ratio, time_pressure, eval, eval_trend,
        legal_moves, material_balance, move_number, player_elo,
        base_time, game_phase)
       ↓
LSTM (hidden=256, layers=3, dropout=0.4)
       ↓
Linear(256 → 128) + ReLU + Dropout
       ↓
Linear(128 → 1) + Sigmoid
       ↓
Blunder probability: 0.0 → 1.0
```

**Total parameters:** ~800k  
**Test AUC:** 0.77  
**Training data:** 490,000 rated games, 31M moves, 4.7M labeled blunders

---

## Dataset

Games sourced from the [Lichess open database](https://database.lichess.org/) via HuggingFace (`Lichess/standard-chess-games`).

- Only rapid and classical games (base time ≥ 600s)
- Only games with Stockfish computer analysis (clock times + evals)
- Blunder threshold: eval swing of ±1.5+ after a move

---

## Project Structure

```
chess-blunder-predictor/
├── data/
│   ├── bulk_download.py      # Download analyzed games from Lichess API
│   └── hf_download.py        # Stream games from HuggingFace dataset
├── features/
│   └── extract.py            # Parse PGN, extract features, label blunders
├── model/
│   ├── lstm.py               # LSTM model architecture
│   └── train.py              # Training pipeline with weighted sampling
├── predict.py                # Live CLI predictor (streams Lichess games)
└── README.md
```

---

## Setup

```bash
git clone https://github.com/sukhrajsandhar/chess-blunder-predictor
cd chess-blunder-predictor
pip install torch python-chess scikit-learn datasets berserk rich requests
```

---

## Training from Scratch

**1. Download training data**
```bash
python data/hf_download.py        # streams 500k games from HuggingFace
```

**2. Extract features**
```bash
python features/extract.py        # parses PGN, computes features, labels blunders
```

**3. Train**
```bash
cd model && python train.py       # trains LSTM on GPU, saves model.pt
```

Requires CUDA GPU. Trained on RTX 5070 Ti (16GB VRAM) and A100 80GB.

---

## Live Prediction

Stream any ongoing Lichess game by ID:

```bash
python predict.py --live GAME_ID --player white
python predict.py --live GAME_ID --player black
```

Analyze a completed game from PGN file:

```bash
python predict.py --pgn game.pgn --player black --delay 0.3
```

**Flags:**
- `--live` — Lichess game ID to stream in real time
- `--pgn` — path to local PGN file
- `--player` — which player to analyze (`white` or `black`)
- `--threshold` — warning threshold, default `0.65`
- `--delay` — seconds between moves for demo effect, default `0.0`

---

## Features

| Feature | Description |
|---|---|
| `time_spent` | Seconds used on this move |
| `time_ratio` | Time spent vs player's average |
| `time_pressure` | Flag: clock under 5 minutes |
| `eval` | Stockfish centipawn evaluation |
| `eval_trend` | Eval change over last 3 moves |
| `legal_moves` | Position complexity |
| `material_balance` | Piece count difference |
| `move_number` | Where in the game |
| `player_elo` | Skill level of the moving player |
| `base_time` | Time control (10min vs 30min) |
| `game_phase` | Opening / middlegame / endgame |

---

## Results

| Model | AUC | Data |
|---|---|---|
| LSTM (tabular only, 553 games) | 0.71 | 553 games |
| LSTM (tabular only, 5k games) | 0.77 | 5,455 games |
| LSTM (tabular + ELO + game phase, 490k games) | 0.75 | 490,458 games |

The model correctly identifies psychological pressure patterns — time pressure building over multiple moves, eval drift, and position complexity spikes — that precede blunders in rated games.

---

## Tech Stack

- **Python** — data pipeline and CLI
- **PyTorch** — LSTM model and training
- **python-chess** — PGN parsing and board representation
- **Lichess API** — live game streaming via SSE
- **HuggingFace Datasets** — bulk game download
- **rich** — terminal output formatting

---

## Author

**Sukhraj Sandhar**  
BCIT Computer Systems Technology  
[github.com/sukhrajsandhar](https://github.com/sukhrajsandhar)
