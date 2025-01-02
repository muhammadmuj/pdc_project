import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import psutil
import time

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(size, 784)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train_and_measure(dataset_size):
    # Model
    model = nn.Linear(784, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Data
    dataset = DummyDataset(dataset_size)
    dataloader = DataLoader(dataset, batch_size=64)

    # Training
    start_time = time.time()
    initial_memory = psutil.virtual_memory().used / (1024 ** 2)  # in MB
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    final_memory = psutil.virtual_memory().used / (1024 ** 2)  # in MB

    training_time = end_time - start_time
    memory_used = final_memory - initial_memory
    return training_time, memory_used

if __name__ == "__main__":
    # Dataset sizes to test
    dataset_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    results = []

    # Measure training times and memory usage
    for dataset_size in dataset_sizes:
        training_time, memory_used = train_and_measure(dataset_size)
        results.append((dataset_size, training_time, memory_used))
        print(f"Dataset Size: {dataset_size}, Training Time: {training_time:.2f}s, Memory Used: {memory_used:.2f} MB")

    # Plot results
    sizes, times, memory_usage = zip(*results)

    plt.figure(figsize=(12, 6))

    # Training Time Plot
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o', label="Training Time")
    plt.xlabel("Dataset Size")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs Dataset Size")
    plt.grid(True)
    plt.legend()

    # Memory Usage Plot
    plt.subplot(1, 2, 2)
    plt.plot(sizes, memory_usage, marker='o', color='orange', label="Memory Usage")
    plt.xlabel("Dataset Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage vs Dataset Size")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import psutil
import time

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(size, 784)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train_and_measure_with_fault_tolerance(dataset_size):
    # Model
    model = nn.Linear(784, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Data
    dataset = DummyDataset(dataset_size)
    dataloader = DataLoader(dataset, batch_size=64)

    training_time = 0
    memory_used = 0
    fault_occurred = False

    # Training
    initial_memory = psutil.virtual_memory().used / (1024 ** 2)  # in MB
    for epoch in range(1):  # Single epoch for simplicity
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(dataloader):
            try:
                if batch_idx == len(dataloader) // 2 and not fault_occurred:
                    # Simulate a fault in the middle of training
                    fault_occurred = True
                    raise RuntimeError("Simulated training failure.")

                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

            except RuntimeError as e:
                print(f"Error encountered: {e}. Recovering from fault...")
                # Recovery logic: Retry the current batch
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

        epoch_training_time = time.time() - start_time
        training_time += epoch_training_time

    final_memory = psutil.virtual_memory().used / (1024 ** 2)  # in MB
    memory_used = final_memory - initial_memory

    return training_time, memory_used, fault_occurred

if __name__ == "__main__":
    # Dataset sizes to test
    dataset_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    results = []

    # Measure training times, memory usage, and fault tolerance
    for dataset_size in dataset_sizes:
        training_time, memory_used, fault_occurred = train_and_measure_with_fault_tolerance(dataset_size)
        results.append((dataset_size, training_time, memory_used, fault_occurred))
        print(f"Dataset Size: {dataset_size}, Training Time: {training_time:.2f}s, Memory Used: {memory_used:.2f} MB, Fault Occurred: {fault_occurred}")

    # Plot results
    sizes, times, memory_usage, faults = zip(*results)

    plt.figure(figsize=(14, 7))

    # Training Time Plot
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o', label="Training Time")
    plt.xlabel("Dataset Size")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs Dataset Size")
    plt.grid(True)
    plt.legend()

    # Memory Usage Plot
    plt.subplot(1, 2, 2)
    plt.plot(sizes, memory_usage, marker='o', color='orange', label="Memory Usage")
    plt.xlabel("Dataset Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage vs Dataset Size")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
