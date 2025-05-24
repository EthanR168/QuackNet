#### Change virtual enviroment to 3.11.9 to use as tensorflow doesnt work on 3.13.1

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.data import Dataset

# Hyperparameters
learning_rate = 0.01
batch_size = 64
epochs = 10
runs = 5

# Load data
train_images = np.load('benchmarkFolder/MNISTBenchmark/data/train_images.npy')  # (60000, 784)
train_labels = np.load('benchmarkFolder/MNISTBenchmark/data/train_labels.npy')  # (60000, 10)
train_labels = np.argmax(train_labels, axis=1)  # convert one-hot to class indices

# Create tf.data.Dataset for batching
dataset = Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.batch(batch_size)

all_accuracies = []
all_losses = []

for run in range(runs):
    # Build model (same architecture)
    model = models.Sequential([
        layers.InputLayer(input_shape=(784,)),
        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dense(64),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    run_accuracies = []
    run_losses = []

    for epoch in range(epochs):
        # Train for one epoch
        history = model.fit(dataset, epochs=1, verbose=0)

        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0] * 100  # convert to percentage

        print(f"Run {run+1}, Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

        run_accuracies.append(accuracy)
        run_losses.append(loss)

    all_accuracies.append(run_accuracies)
    all_losses.append(run_losses)

# Compute mean accuracy and loss across runs per epoch
mean_accuracy = np.mean(all_accuracies, axis=0)
mean_loss = np.mean(all_losses, axis=0)

print("Mean Accuracy per epoch:", list(mean_accuracy))
print("Mean Loss per epoch:", list(mean_loss))
