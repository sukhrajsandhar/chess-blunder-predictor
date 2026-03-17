import torch
import torch.nn as nn

class BlunderPredictor(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.3):
        super(BlunderPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last timestep
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out).squeeze(1)


if __name__ == "__main__":
    # Quick sanity check
    model = BlunderPredictor()
    x = torch.randn(32, 10, 8)  # batch=32, seq=10, features=8
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: {out.min().item():.3f} - {out.max().item():.3f}")
    print("Model looks good!")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")