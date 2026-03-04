import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class CommunicationSystem(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.M = M  # Number of messages (e.g., 256)
        self.n = n  # Channel uses

        # Transmitter (Encoder)
        self.transmitter = nn.Sequential(
            nn.Linear(M, 2 * M),
            nn.ReLU(),
            nn.Linear(2 * M, 2 * n)
        )

        # Receiver (Decoder)
        self.receiver = nn.Sequential(
            nn.Linear(2 * n, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, M)
        )

    def normalize_power(self, x):
        # Normalize complex symbols to unit average power
        power = torch.mean(torch.abs(x) ** 2, dim=1, keepdim=True)
        return x * torch.sqrt(1.0 / (power + 1e-8))

    def forward(self, x, snr):
        # 1. Transmitter
        x = self.transmitter(x)

        # Reshape to complex (batch_size, n)
        real = x[:, :self.n]
        imag = x[:, self.n:]
        x = torch.view_as_complex(torch.stack([real, imag], dim=-1))

        # Power normalization
        x = self.normalize_power(x)

        # ==========================================
        # TODO: 2. Channel Simulation
        # - Generate and apply Rayleigh fading channel (h)
        # - Generate and add AWGN noise based on the given SNR
        # - Perform Channel Equalization (e.g., Zero-Forcing)
        # ==========================================

        # Placeholder for y (Replace this with your channel output after equalization)
        y = x

        # Format for receiver
        input_real = torch.real(y)
        input_imag = torch.imag(y)
        input_features = torch.cat([input_real, input_imag], dim=1)

        # 3. Receiver
        logits = self.receiver(input_features)
        return logits


def train_model(M, n, train_snr, num_epochs=20, batch_size=512):
    model = CommunicationSystem(M, n)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for _ in range(100):  # 100 batches per epoch
            # Generate random messages
            labels = torch.randint(0, M, (batch_size,))
            one_hot = F.one_hot(labels, M).float()

            # ==========================================
            # TODO: Implement the training step
            # - Forward pass
            # - Calculate loss
            # - Backward pass and optimizer step
            # ==========================================

    return model


def evaluate(model, M, test_snrs, num_samples=10000):
    blers = []
    for snr in test_snrs:
        # Generate test data
        labels = torch.randint(0, M, (num_samples,))
        one_hot = F.one_hot(labels, M).float()

        # ==========================================
        # TODO: Implement the evaluation step
        # - Forward pass (without gradients)
        # - Get predictions
        # - Calculate Block Error Rate (BLER) and append to `blers`
        # ==========================================

    return blers

# ==========================================
# TODO: Part (a)
# Fix training SNR = 7 dB.
# Train the network with channel uses n = 32, 64, 128.
# Evaluate over test SNRs [0, 20] dB with steps of 3 dB.
# Plot the BLER vs. Test SNR results.
# ==========================================


# ==========================================
# TODO: Part (b)
# Fix channel uses n = 128.
# Train the network with training SNRs = 1 dB, 7 dB, 20 dB.
# Evaluate over test SNRs [0, 20] dB with steps of 3 dB.
# Plot the BLER vs. Test SNR results.
# ==========================================