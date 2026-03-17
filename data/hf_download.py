from datasets import load_dataset
import os

TARGET_GAMES = 5000
output_path = "data/raw/hf_games.pgn"
os.makedirs("data/raw", exist_ok=True)

print(f"Streaming from HuggingFace Lichess dataset, target: {TARGET_GAMES} games...")

# Load a specific partition using year/month in the path
dataset = load_dataset(
    "Lichess/standard-chess-games",
    split="train",
    streaming=True,
    data_files="data/year=2019/month=01/*.parquet",
)

collected = 0
all_pgn = ""

for item in dataset:
    try:
        pgn_text = item.get("movetext", "")

        if not pgn_text:
            continue

        # Must have evals and clocks
        if "[%eval" not in pgn_text or "[%clk" not in pgn_text:
            continue

        # Filter for rapid or classical (base time >= 600 seconds)
        tc = item.get("TimeControl", "") or ""
        if not tc:
            continue
        base = int(tc.split("+")[0]) if "+" in tc else 0
        if base < 600:
            continue

        # Build a proper PGN string with headers
        headers = [
            f'[Event "{item.get("Event", "?")}"]',
            f'[White "{item.get("White", "?")}"]',
            f'[Black "{item.get("Black", "?")}"]',
            f'[Result "{item.get("Result", "?")}"]',
            f'[WhiteElo "{item.get("WhiteElo", "?")}"]',
            f'[BlackElo "{item.get("BlackElo", "?")}"]',
            f'[TimeControl "{tc}"]',
        ]
        full_pgn = "\n".join(headers) + "\n\n" + pgn_text + "\n\n"
        all_pgn += full_pgn
        collected += 1

        if collected % 500 == 0:
            print(f"  Collected {collected} games...")

        if collected >= TARGET_GAMES:
            break

    except Exception:
        continue

with open(output_path, "w", encoding="utf-8") as f:
    f.write(all_pgn)

print(f"\nDone. {collected} games saved to {output_path}")