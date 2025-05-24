import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
learning_rate = 0.01
batch_size = 64
epochs = 10

train_images = np.load('benchmarkFolder/MNISTBenchmark/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('benchmarkFolder/MNISTBenchmark/data/train_labels.npy')  # Shape: (60000, 10)

train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
train_labels = torch.argmax(train_labels, axis=1)

dataset = TensorDataset(train_images, train_labels)
data_loader = DataLoader(dataset, batch_size=batch_size) 

# Network model
class QuackNetPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.leaky_relu(self.layer2(x))
        x = self.layer3(x)
        return x

allAc, allLoss = [], []
for i in range(5):
    # Model, loss, optimizer
    model = QuackNetPyTorch()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    accuracies, losses = [], []
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / len(train_labels)
        print(f"Run {i+1}, Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
        accuracies.append(accuracy)
        losses.append(avg_loss)
    allAc.append(accuracies)
    allLoss.append(losses)

meanAccuracy = np.mean(allAc, axis=0)
meanLoss = np.mean(allLoss, axis=0)
print(list(meanAccuracy))
print(list(meanLoss))