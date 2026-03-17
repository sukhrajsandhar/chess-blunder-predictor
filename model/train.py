import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from lstm import BlunderPredictor

SEQ_LEN = 10
BATCH_SIZE = 128
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
]

class ChessDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def build_sequences(all_moves):
    games = {}
    for move in all_moves:
        gid = move.get("game_id", 0)
        if gid not in games:
            games[gid] = []
        games[gid].append(move)

    sequences = []
    labels = []

    for gid, moves in games.items():
        for i in range(SEQ_LEN, len(moves)):
            window = moves[i - SEQ_LEN:i]
            target = moves[i]

            feats = []
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

            if valid and len(feats) == SEQ_LEN:
                label = target.get("is_blunder", 0)
                if label is not None:
                    sequences.append(feats)
                    labels.append(label)

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


def train():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading data...")
    with open("../data/all_features.json") as f:
        all_moves = json.load(f)

    # Assign game IDs
    game_id = 0
    prev_move_num = 999
    for move in all_moves:
        if move["move_number"] <= prev_move_num:
            game_id += 1
        move["game_id"] = game_id
        prev_move_num = move["move_number"]

    print(f"Total moves: {len(all_moves)} across {game_id} games")

    print("Building sequences...")
    X, y = build_sequences(all_moves)
    print(f"Sequences: {X.shape} | Blunder rate: {y.mean():.3f}")

    # Normalize
    X_flat = X.reshape(-1, X.shape[-1])
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0) + 1e-8
    X = (X - mean) / std
    np.save("../data/mean.npy", mean)
    np.save("../data/std.npy", std)

    # Split by game to avoid data leakage
    game_ids = list(range(game_id))
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    # Rebuild sequences with proper splits
    all_moves_by_game = {}
    for move in all_moves:
        gid = move["game_id"]
        if gid not in all_moves_by_game:
            all_moves_by_game[gid] = []
        all_moves_by_game[gid].append(move)

    def get_split(ids):
        seqs, labs = [], []
        for gid in ids:
            moves = all_moves_by_game.get(gid, [])
            for i in range(SEQ_LEN, len(moves)):
                window = moves[i - SEQ_LEN:i]
                target = moves[i]
                feats = []
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
                if valid and len(feats) == SEQ_LEN:
                    label = target.get("is_blunder", 0)
                    if label is not None:
                        seqs.append(feats)
                        labs.append(label)
        X = np.array(seqs, dtype=np.float32)
        y = np.array(labs, dtype=np.float32)
        X = (X - mean) / std
        return X, y

    X_train, y_train = get_split(train_ids)
    X_val, y_val = get_split(val_ids)
    X_test, y_test = get_split(test_ids)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Weighted sampler
    blunder_weight = 1.0 / (y_train.mean() + 1e-8)
    normal_weight = 1.0 / (1 - y_train.mean() + 1e-8)
    weights = np.where(y_train == 1, blunder_weight, normal_weight)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(ChessDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(ChessDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ChessDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Model with higher dropout to reduce overfitting
    model = BlunderPredictor(input_size=len(FEATURES), hidden_size=128, num_layers=2, dropout=0.5).to(DEVICE)
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
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
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
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(DEVICE))
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