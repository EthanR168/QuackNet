import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
learning_rate = 0.01
batch_size = 64
training_epochs = 10
inference_epochs = 1  # Pure forward pass evaluation

# Load data
train_images = np.load('benchmarkFolder/MNISTBenchmark/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('benchmarkFolder/MNISTBenchmark/data/train_labels.npy')  # Shape: (60000, 10)
test_images = np.load('benchmarkFolder/MNISTBenchmark/data/test_images.npy')    # Shape: (10000, 784)
test_labels = np.load('benchmarkFolder/MNISTBenchmark/data/test_labels.npy')    # Shape: (10000, 10)

# Convert to tensors
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
train_labels = torch.argmax(train_labels, axis=1)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)
test_labels = torch.argmax(test_labels, axis=1)

# Create datasets
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

# Training and evaluation
def run_benchmark():
    model = QuackNetPyTorch()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training phase (10 epochs)
    print("=== Training Phase ===")
    for epoch in range(1, training_epochs + 1):
        model.train()
        total_loss, correct = 0, 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / len(train_dataset)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.2f}%")
    
    # Inference-only phase (1 epoch)
    print("\n=== Inference Phase ===")
    model.eval()
    inference_loss, inference_correct = 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            inference_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            inference_correct += pred.eq(target).sum().item()
    
    avg_inference_loss = inference_loss / len(test_loader)
    inference_accuracy = 100. * inference_correct / len(test_dataset)
    print(f"Final Test Metrics - Loss: {avg_inference_loss:.4f}, Accuracy: {inference_accuracy:.2f}%")
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy,
        'test_loss': avg_inference_loss,
        'test_accuracy': inference_accuracy
    }

# Run multiple trials
results = []
for i in range(5):
    print(f"\n=== Trial {i+1} ===")
    results.append(run_benchmark())

# Calculate averages
avg_train_loss = np.mean([r['train_loss'] for r in results])
avg_train_acc = np.mean([r['train_accuracy'] for r in results])
avg_test_loss = np.mean([r['test_loss'] for r in results])
avg_test_acc = np.mean([r['test_accuracy'] for r in results])

print("\n=== Final Benchmark Results ===")
print(f"Average Training Loss (Epoch 10): {avg_train_loss:.4f}")
print(f"Average Training Accuracy: {avg_train_acc:.2f}%")
print(f"Average Test Loss: {avg_test_loss:.4f}")
print(f"Average Test Accuracy: {avg_test_acc:.2f}%")

"""
output:
=== Trial 1 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.6342, Train Accuracy = 59.73%
Epoch 2: Train Loss = 0.5344, Train Accuracy = 85.50%
Epoch 3: Train Loss = 0.3933, Train Accuracy = 88.95%
Epoch 4: Train Loss = 0.3440, Train Accuracy = 90.11%
Epoch 5: Train Loss = 0.3135, Train Accuracy = 90.92%
Epoch 6: Train Loss = 0.2908, Train Accuracy = 91.65%
Epoch 7: Train Loss = 0.2721, Train Accuracy = 92.17%
Epoch 8: Train Loss = 0.2551, Train Accuracy = 92.70%
Epoch 9: Train Loss = 0.2396, Train Accuracy = 93.17%
Epoch 10: Train Loss = 0.2258, Train Accuracy = 93.49%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2157, Accuracy: 93.74%

=== Trial 2 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.7066, Train Accuracy = 55.02%
Epoch 2: Train Loss = 0.5969, Train Accuracy = 83.98%
Epoch 3: Train Loss = 0.4026, Train Accuracy = 88.80%
Epoch 4: Train Loss = 0.3452, Train Accuracy = 90.25%
Epoch 5: Train Loss = 0.3151, Train Accuracy = 91.02%
Epoch 6: Train Loss = 0.2933, Train Accuracy = 91.64%
Epoch 7: Train Loss = 0.2753, Train Accuracy = 92.12%
Epoch 8: Train Loss = 0.2599, Train Accuracy = 92.51%
Epoch 9: Train Loss = 0.2455, Train Accuracy = 92.94%
Epoch 10: Train Loss = 0.2325, Train Accuracy = 93.36%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2273, Accuracy: 93.48%

=== Trial 3 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.7725, Train Accuracy = 53.32%
Epoch 2: Train Loss = 0.5660, Train Accuracy = 85.11%
Epoch 3: Train Loss = 0.3978, Train Accuracy = 88.88%
Epoch 4: Train Loss = 0.3466, Train Accuracy = 90.14%
Epoch 5: Train Loss = 0.3178, Train Accuracy = 90.86%
Epoch 6: Train Loss = 0.2964, Train Accuracy = 91.45%
Epoch 7: Train Loss = 0.2795, Train Accuracy = 92.01%
Epoch 8: Train Loss = 0.2645, Train Accuracy = 92.42%
Epoch 9: Train Loss = 0.2508, Train Accuracy = 92.81%
Epoch 10: Train Loss = 0.2380, Train Accuracy = 93.21%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2267, Accuracy: 93.58%

=== Trial 4 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.7294, Train Accuracy = 56.27%
Epoch 2: Train Loss = 0.5678, Train Accuracy = 85.12%
Epoch 3: Train Loss = 0.4061, Train Accuracy = 88.78%
Epoch 4: Train Loss = 0.3541, Train Accuracy = 89.94%
Epoch 5: Train Loss = 0.3239, Train Accuracy = 90.75%
Epoch 6: Train Loss = 0.3008, Train Accuracy = 91.39%
Epoch 7: Train Loss = 0.2807, Train Accuracy = 91.97%
Epoch 8: Train Loss = 0.2631, Train Accuracy = 92.51%
Epoch 9: Train Loss = 0.2472, Train Accuracy = 92.94%
Epoch 10: Train Loss = 0.2324, Train Accuracy = 93.38%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2213, Accuracy: 93.77%

=== Trial 5 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.6759, Train Accuracy = 54.54%
Epoch 2: Train Loss = 0.5676, Train Accuracy = 84.50%
Epoch 3: Train Loss = 0.4042, Train Accuracy = 88.60%
Epoch 4: Train Loss = 0.3495, Train Accuracy = 90.14%
Epoch 5: Train Loss = 0.3180, Train Accuracy = 90.92%
Epoch 6: Train Loss = 0.2955, Train Accuracy = 91.56%
Epoch 7: Train Loss = 0.2769, Train Accuracy = 92.04%
Epoch 8: Train Loss = 0.2607, Train Accuracy = 92.50%
Epoch 9: Train Loss = 0.2466, Train Accuracy = 92.94%
Epoch 10: Train Loss = 0.2331, Train Accuracy = 93.34%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2240, Accuracy: 93.33%

=== Final Benchmark Results ===
Average Training Loss (Epoch 10): 0.2324
Average Training Accuracy: 93.36%
Average Test Loss: 0.2230
Average Test Accuracy: 93.58%
"""