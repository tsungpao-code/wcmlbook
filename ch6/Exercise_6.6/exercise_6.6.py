"""
FedAvg Simulation with Packet Loss Impact Analysis
This program demonstrates how packet loss affects federated learning convergence
using the MNIST dataset and a CNN model under non-IID data distribution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

# Experiment Configuration
NUM_CLIENTS = 100  # Total number of participating clients
COMM_ROUNDS = 100  # Global communication rounds
LOCAL_EPOCHS = 5  # Local training epochs per round
BATCH_SIZE = 32  # Batch size for client training
LOSS_RATES = [0.01, 0.05, 0.10]  # Packet loss probabilities to simulate
ALPHA = 0.5  # Dirichlet distribution parameter for non-IID split


# CNN Model Definition for MNIST classification
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5)  # 1 input channel, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, 5)
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)  # 1024 input features, 512 output features
        self.fc2 = nn.Linear(512, 10)  # Output layer for 10 classes

    def forward(self, x):
        # Feature extraction
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Conv -> ReLU -> MaxPool
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Classification
        x = x.view(-1, 1024)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Log probabilities for NLL loss


def dirichlet_split(dataset, num_clients, alpha):
    """
    Partition dataset among clients using Dirichlet distribution
    Args:
        dataset: PyTorch dataset to split
        num_clients: Number of partitions
        alpha: Concentration parameter (smaller alpha = more skewed distribution)
    Returns:
        List of Subset objects for each client
    """
    num_classes = len(dataset.classes)
    client_indices = {i: [] for i in range(num_clients)}

    # Group samples by class
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Split each class's samples according to Dirichlet distribution
    for c in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        splits = (len(class_indices[c]) * np.cumsum(proportions)).astype(int)[:-1]
        client_class_data = np.split(np.random.permutation(class_indices[c]), splits)

        for client_idx in range(num_clients):
            if client_idx < len(client_class_data):
                client_indices[client_idx].extend(client_class_data[client_idx])

    return [Subset(dataset, indices) for indices in client_indices.values()]


def unreliable_channel(updates, loss_rate):
    """
    Simulate packet loss in wireless channel
    Args:
        updates: List of model updates from clients
        loss_rate: Probability of losing a packet
    Returns:
        List where some updates are replaced with None (lost packets)
    """
    return [u if np.random.random() > loss_rate else None for u in updates]


def client_train(client_model, train_loader):
    """
    Local client training routine
    Args:
        client_model: Initial model state
        train_loader: Client's data loader
    Returns:
        Updated model state dict
    """
    client_model.train()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)

    for _ in range(LOCAL_EPOCHS):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)  # Negative log likelihood loss
            loss.backward()
            optimizer.step()

    return client_model.state_dict()


def run_fedavg_experiment(loss_rate):
    """
    Main experiment loop for given packet loss rate
    Args:
        loss_rate: Current packet loss probability
    Returns:
        Tuple of (accuracy_history, loss_history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Create non-IID data partitions
    client_datasets = dirichlet_split(train_dataset, NUM_CLIENTS, ALPHA)
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Initialize global model
    global_model = MNISTCNN().to(device)
    acc_history, loss_history = [], []

    for round in range(COMM_ROUNDS):
        # Random client selection (10% participation)
        selected = np.random.choice(NUM_CLIENTS, size=NUM_CLIENTS //
                                    5, replace=False)

        # Local training phase
        client_updates = []
        for idx in selected:
            client_model = copy.deepcopy(global_model)
            update = client_train(client_model, client_loaders[idx])
            client_updates.append(update)

        # Simulate packet loss
        received_updates = unreliable_channel(client_updates, loss_rate)
        valid_updates = [u for u in received_updates if u is not None]

        # Global aggregation (FedAvg)
        if valid_updates:  # Proceed only if we have updates
            global_weights = {}
            for key in valid_updates[0].keys():
                global_weights[key] = torch.stack([u[key] for u in valid_updates]).mean(0)
            global_model.load_state_dict(global_weights)

        # Evaluation on test set
        global_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Record metrics
        acc = 100. * correct / len(test_loader.dataset)
        loss = test_loss / len(test_loader.dataset)
        acc_history.append(acc)
        loss_history.append(loss)

        print(f'Round {round + 1:02d}, Loss Rate {loss_rate:.0%}: '
              f'Acc {acc:.2f}%, Loss {loss:.4f}, '
              f'Received {len(valid_updates)}/{len(selected)} updates')

    return acc_history, loss_history


# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # Run experiments for different loss rates
    for rate in LOSS_RATES:
        print(f"\n=== Experiment with packet loss rate: {rate:.0%} ===")
        results[rate] = run_fedavg_experiment(rate)

    # Visualization
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    for rate, (acc, _) in results.items():
        plt.plot(acc, label=f'Loss rate {rate:.0%}')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Accuracy under Packet Loss')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    for rate, (_, loss) in results.items():
        plt.plot(loss, label=f'Loss rate {rate:.0%}')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Loss')
    plt.title('Training Loss under Packet Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('packet_loss_impact.png', dpi=300, bbox_inches='tight')
    plt.show()