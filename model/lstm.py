import torch
import torch.nn as nn


class BoardEncoder(nn.Module):
    """Small CNN to encode board position from 12x8x8 tensor"""
    def __init__(self, out_size=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, out_size),
            nn.ReLU(),
        )

    def forward(self, x):
        # x shape: (batch * seq_len, 12, 8, 8)
        return self.cnn(x)


class BlunderPredictor(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, num_layers=3, dropout=0.4, board_enc_size=64):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.board_enc_size = board_enc_size

        self.board_encoder = BoardEncoder(out_size=board_enc_size)

        # LSTM takes tabular features + board encoding
        self.lstm = nn.LSTM(
            input_size=input_size + board_enc_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_tabular, x_boards):
        # x_tabular: (batch, seq_len, input_size)
        # x_boards: (batch, seq_len, 12, 8, 8)

        batch, seq_len = x_tabular.shape[0], x_tabular.shape[1]

        # Encode all board positions
        boards_flat = x_boards.view(batch * seq_len, 12, 8, 8)
        board_enc = self.board_encoder(boards_flat)
        board_enc = board_enc.view(batch, seq_len, self.board_enc_size)

        # Concatenate tabular + board features
        combined = torch.cat([x_tabular, board_enc], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(combined)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out).squeeze(1)


if __name__ == "__main__":
    model = BlunderPredictor()
    x_tab = torch.randn(32, 10, 11)
    x_board = torch.randn(32, 10, 12, 8, 8)
    out = model(x_tab, x_board)
    print(f"Tabular input: {x_tab.shape}")
    print(f"Board input: {x_board.shape}")
    print(f"Output: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model looks good!")