import numpy as np
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 has 3 channels
])

# Load CIFAR-10 data
trainset = datasets.CIFAR10(root='benchmarkFolder/OptimiserBenchmark/data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='benchmarkFolder/OptimiserBenchmark/data', train=False, download=True, transform=transform)

# Preprocess and save the data as arrays
train_images = trainset.data.astype('float16') / 255.0      # Normalize pixel values to [0,1]
train_images = train_images.transpose(0, 3, 1, 2)

train_labels = np.eye(10, dtype='float16')[np.array(trainset.targets)]  # One-hot encode labels

test_images = testset.data.astype('float16') / 255.0
test_images = test_images.reshape(len(test_images), -1)

test_labels = np.eye(10, dtype='float16')[np.array(testset.targets)]

# Save to disk for reuse
np.save('benchmarkFolder/OptimiserBenchmark/data/train_images.npy', train_images)
np.save('benchmarkFolder/OptimiserBenchmark/data/train_labels.npy', train_labels)
np.save('benchmarkFolder/OptimiserBenchmark/data/test_images.npy', test_images)
np.save('benchmarkFolder/OptimiserBenchmark/data/test_labels.npy', test_labels)