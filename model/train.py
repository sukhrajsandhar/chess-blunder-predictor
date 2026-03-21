import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, classification_report
from lstm import BlunderPredictor
import random

SEQ_LEN = 10
BATCH_SIZE = 512
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "time_spent",
    "time_ratio",
    "time_pressure",
    "eval",
    "eval_trend",
    "legal_moves",
    "material_balance",
    "move_number",
    "player_elo",
    "base_time",
    "game_phase",
]

class ChessDataset(Dataset):
    def __init__(self, games, mean, std):
        self.mean = mean
        self.std = std
        self.samples = []

        print(f"  Building sequences from {len(games)} games...")
        for games_idx, moves in enumerate(games):
            for i in range(SEQ_LEN, len(moves)):
                window = moves[i - SEQ_LEN:i]
                target = moves[i]

                feats = []
                boards = []
                valid = True

                for m in window:
                    row = []
                    for f in FEATURES:
                        val = m.get(f)
                        if val is None:
                            valid = False
                            break
                        row.append(float(val))
                    if not valid:
                        break
                    feats.append(row)
                    boards.append(m.get("board"))

                if valid and len(feats) == SEQ_LEN and all(b is not None for b in boards):
                    label = target.get("is_blunder", 0)
                    if label is not None:
                        self.samples.append((feats, boards, label))

            if (games_idx + 1) % 5000 == 0:
                print(f"    {games_idx + 1} games processed, {len(self.samples)} sequences so far...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, boards, label = self.samples[idx]
        x_tab = (np.array(feats, dtype=np.float32) - self.mean) / self.std
        x_board = np.array(boards, dtype=np.float32)  # (seq_len, 12, 8, 8)
        return (
            torch.tensor(x_tab, dtype=torch.float32),
            torch.tensor(x_board, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def train():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading data...")
    with open("../data/all_features.json") as f:
        all_moves = json.load(f)

    # Group into games
    games = []
    current_game = []
    prev_move_num = 999
    for move in all_moves:
        if move["move_number"] <= prev_move_num and current_game:
            games.append(current_game)
            current_game = []
        current_game.append(move)
        prev_move_num = move["move_number"]
    if current_game:
        games.append(current_game)

    print(f"Total games: {len(games)}")

    random.seed(42)
    random.shuffle(games)

    # Normalization stats
    print("Computing normalization stats...")
    sample_moves = []
    for g in games[:2000]:
        sample_moves.extend(g)
    sample = np.array([
        [float(m.get(f, 0) or 0) for f in FEATURES]
        for m in sample_moves
    ], dtype=np.float32)
    mean = sample.mean(axis=0)
    std = sample.std(axis=0) + 1e-8
    np.save("../data/mean.npy", mean)
    np.save("../data/std.npy", std)

    # Split
    n = len(games)
    train_games = games[:int(n * 0.7)]
    val_games = games[int(n * 0.7):int(n * 0.85)]
    test_games = games[int(n * 0.85):]

    print(f"Train: {len(train_games)} | Val: {len(val_games)} | Test: {len(test_games)}")

    print("Building datasets...")
    train_ds = ChessDataset(train_games, mean, std)
    val_ds = ChessDataset(val_games, mean, std)
    test_ds = ChessDataset(test_games, mean, std)

    print(f"Train sequences: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Weighted sampler
    labels = np.array([s[2] for s in train_ds.samples])
    blunder_w = 1.0 / (labels.mean() + 1e-8)
    normal_w = 1.0 / (1 - labels.mean() + 1e-8)
    weights = np.where(labels == 1, blunder_w, normal_w)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = BlunderPredictor(
        input_size=len(FEATURES),
        hidden_size=256,
        num_layers=3,
        dropout=0.4,
        board_enc_size=64,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOP = 8

    print("\nTraining...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x_tab, x_board, y_batch in train_loader:
            x_tab = x_tab.to(DEVICE)
            x_board = x_board.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x_tab, x_board)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_tab, x_board, y_batch in val_loader:
                x_tab = x_tab.to(DEVICE)
                x_board = x_board.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                preds = model(x_tab, x_board)
                val_loss += criterion(preds, y_batch).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), "../data/model.pt")
            print(f"  Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Test
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("../data/model.pt"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_tab, x_board, y_batch in test_loader:
            preds = model(x_tab.to(DEVICE), x_board.to(DEVICE))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    binary_preds = (all_preds > 0.5).astype(int)

    print(f"\nTest AUC: {auc:.4f}")
    print(classification_report(all_labels, binary_preds, target_names=["Normal", "Blunder"]))

if __name__ == "__main__":
    train()