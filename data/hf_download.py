from datasets import load_dataset
import os

TARGET_GAMES = 500000
output_path = "data/raw/hf_games.pgn"
os.makedirs("data/raw", exist_ok=True)

print(f"Streaming from HuggingFace Lichess dataset, target: {TARGET_GAMES} games...")

# Pull multiple months to get enough games with evals + rapid/classical
MONTHS = [
    ("2019", "01"), ("2019", "02"), ("2019", "03"), ("2019", "04"),
    ("2019", "05"), ("2019", "06"), ("2019", "07"), ("2019", "08"),
    ("2019", "09"), ("2019", "10"), ("2019", "11"), ("2019", "12"),
    ("2020", "01"), ("2020", "02"), ("2020", "03"), ("2020", "04"),
    ("2020", "05"), ("2020", "06"),
]

collected = 0

with open(output_path, "w", encoding="utf-8") as out_f:
    for year, month in MONTHS:
        if collected >= TARGET_GAMES:
            break

        print(f"Loading {year}-{month}...")
        try:
            dataset = load_dataset(
                "Lichess/standard-chess-games",
                split="train",
                streaming=True,
                data_files=f"data/year={year}/month={month}/*.parquet",
            )
        except Exception as e:
            print(f"  Skipping {year}-{month}: {e}")
            continue

        month_count = 0
        for item in dataset:
            if collected >= TARGET_GAMES:
                break
            try:
                pgn_text = item.get("movetext", "")
                if not pgn_text:
                    continue
                if "[%eval" not in pgn_text or "[%clk" not in pgn_text:
                    continue
                tc = item.get("TimeControl", "") or ""
                if not tc:
                    continue
                base = int(tc.split("+")[0]) if "+" in tc else 0
                if base < 600:
                    continue

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
                out_f.write(full_pgn)
                collected += 1
                month_count += 1

                if collected % 10000 == 0:
                    print(f"  Total collected: {collected} games...")

            except Exception:
                continue

        print(f"  {year}-{month}: {month_count} games | Running total: {collected}")

print(f"\nDone. {collected} games saved to {output_path}")