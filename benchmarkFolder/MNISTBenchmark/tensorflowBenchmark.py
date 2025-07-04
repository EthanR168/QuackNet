#### Change virtual enviroment to 3.11.9 to use as tensorflow doesnt work on 3.13.1

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import time

def load_data():
    """Load and preprocess MNIST datasets"""
    train_images = np.load('benchmarkFolder/MNISTBenchmark/data/train_images.npy')
    train_labels = np.load('benchmarkFolder/MNISTBenchmark/data/train_labels.npy')
    test_images = np.load('benchmarkFolder/MNISTBenchmark/data/test_images.npy')
    test_labels = np.load('benchmarkFolder/MNISTBenchmark/data/test_labels.npy')
    
    # Convert to TensorFlow Dataset format
    train_labels = np.argmax(train_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    
    return (train_images, train_labels), (test_images, test_labels)

def create_model():
    """Create the TensorFlow model"""
    model = models.Sequential([
        layers.InputLayer(input_shape=(784,)),
        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dense(64),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dense(10, activation='softmax')
    ])
    return model

def run_trial(run_num, train_data, test_data, epochs=10, batch_size=64):
    """Run complete training and evaluation trial"""
    print(f"\n=== Trial {run_num} ===")
    print("=== Training Phase ===")
    
    # Prepare datasets
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(1000).batch(batch_size)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.batch(batch_size)
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Training
    train_metrics = {'accuracy': [], 'loss': [], 'time': []}
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        history = model.fit(train_ds, epochs=1, verbose=0)
        epoch_time = time.time() - start_time
        
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0] * 100
        
        train_metrics['accuracy'].append(accuracy)
        train_metrics['loss'].append(loss)
        train_metrics['time'].append(epoch_time)
        
        print(f"Epoch {epoch}: Loss = {loss:.4f}, "
              f"Accuracy = {accuracy:.2f}%, "
              f"Time = {epoch_time:.2f}s")
    
    # Evaluation
    print("\n=== Inference Phase ===")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%, Test Loss: {test_loss:.4f}")
    
    return {
        'train': train_metrics,
        'test': {'accuracy': test_acc*100, 'loss': test_loss}
    }

def main():
    """Main benchmark execution"""
    # Load data
    train_data, test_data = load_data()
    
    # Run benchmark
    all_results = []
    for i in range(5):  # 5 trials
        all_results.append(run_trial(i+1, train_data, test_data))
    
    # Calculate statistics
    final_train_loss = np.mean([trial['train']['loss'][-1] for trial in all_results])
    final_train_acc = np.mean([trial['train']['accuracy'][-1] for trial in all_results])
    avg_test_loss = np.mean([trial['test']['loss'] for trial in all_results])
    avg_test_acc = np.mean([trial['test']['accuracy'] for trial in all_results])
    
    std_test_acc = np.std([trial['test']['accuracy'] for trial in all_results])
    std_test_loss = np.std([trial['test']['loss'] for trial in all_results])
    
    # Final report
    print("\n=== Final Benchmark Results ===")
    print(f"Architecture: 784-128-64-10 (LeakyReLU/Softmax)")
    print(f"Batch Size: 64 | Learning Rate: 0.01 | Optimizer: SGD\n")
    
    print(f"Average Training Loss (Epoch 10): {final_train_loss:.4f}")
    print(f"Average Training Accuracy: {final_train_acc:.2f}%")
    print(f"Average Test Loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
    print(f"Average Test Accuracy: {avg_test_acc:.2f}% ± {std_test_acc:.2f}")
    
    # Save results
    np.save('tensorflow_benchmark_results.npy', all_results)

if __name__ == "__main__":
    main()

"""
output:

=== Trial 1 ===
=== Training Phase ===
Epoch 1: Loss = 0.8573, Accuracy = 77.89%, Time = 2.52s
Epoch 2: Loss = 0.3637, Accuracy = 89.85%, Time = 2.30s
Epoch 3: Loss = 0.3021, Accuracy = 91.39%, Time = 2.38s
Epoch 4: Loss = 0.2686, Accuracy = 92.33%, Time = 2.50s
Epoch 5: Loss = 0.2446, Accuracy = 92.99%, Time = 2.36s
Epoch 6: Loss = 0.2258, Accuracy = 93.56%, Time = 2.49s
Epoch 7: Loss = 0.2097, Accuracy = 94.01%, Time = 2.24s
Epoch 8: Loss = 0.1953, Accuracy = 94.45%, Time = 2.55s
Epoch 9: Loss = 0.1834, Accuracy = 94.78%, Time = 2.40s
Epoch 10: Loss = 0.1727, Accuracy = 95.07%, Time = 2.19s

=== Inference Phase ===
Test Accuracy: 95.15%, Test Loss: 0.1669


=== Trial 2 ===
=== Training Phase ===
Epoch 1: Loss = 0.8411, Accuracy = 78.57%, Time = 2.93s
Epoch 2: Loss = 0.3643, Accuracy = 89.86%, Time = 2.33s
Epoch 3: Loss = 0.3064, Accuracy = 91.18%, Time = 2.38s
Epoch 4: Loss = 0.2746, Accuracy = 92.18%, Time = 2.16s
Epoch 5: Loss = 0.2515, Accuracy = 92.76%, Time = 2.55s
Epoch 6: Loss = 0.2325, Accuracy = 93.37%, Time = 2.50s
Epoch 7: Loss = 0.2169, Accuracy = 93.81%, Time = 2.29s
Epoch 8: Loss = 0.2032, Accuracy = 94.20%, Time = 1.98s
Epoch 9: Loss = 0.1907, Accuracy = 94.54%, Time = 2.33s
Epoch 10: Loss = 0.1795, Accuracy = 94.84%, Time = 2.50s

=== Inference Phase ===
Test Accuracy: 94.63%, Test Loss: 0.1796


=== Trial 3 ===
=== Training Phase ===
Epoch 1: Loss = 0.9151, Accuracy = 75.83%, Time = 2.79s
Epoch 2: Loss = 0.3746, Accuracy = 89.64%, Time = 2.41s
Epoch 3: Loss = 0.3072, Accuracy = 91.24%, Time = 2.41s
Epoch 4: Loss = 0.2722, Accuracy = 92.22%, Time = 2.09s
Epoch 5: Loss = 0.2474, Accuracy = 92.90%, Time = 2.47s
Epoch 6: Loss = 0.2274, Accuracy = 93.52%, Time = 2.58s
Epoch 7: Loss = 0.2108, Accuracy = 93.97%, Time = 2.55s
Epoch 8: Loss = 0.1965, Accuracy = 94.37%, Time = 2.55s
Epoch 9: Loss = 0.1842, Accuracy = 94.67%, Time = 2.17s
Epoch 10: Loss = 0.1730, Accuracy = 94.98%, Time = 2.11s

=== Inference Phase ===
Test Accuracy: 94.96%, Test Loss: 0.1721


=== Trial 4 ===
=== Training Phase ===
Epoch 1: Loss = 0.9143, Accuracy = 75.66%, Time = 2.64s
Epoch 2: Loss = 0.3729, Accuracy = 89.54%, Time = 2.63s
Epoch 3: Loss = 0.3109, Accuracy = 91.09%, Time = 2.51s
Epoch 4: Loss = 0.2768, Accuracy = 92.17%, Time = 2.23s
Epoch 5: Loss = 0.2515, Accuracy = 92.84%, Time = 2.23s
Epoch 6: Loss = 0.2314, Accuracy = 93.40%, Time = 2.54s
Epoch 7: Loss = 0.2145, Accuracy = 93.91%, Time = 2.39s
Epoch 8: Loss = 0.1999, Accuracy = 94.33%, Time = 2.31s
Epoch 9: Loss = 0.1871, Accuracy = 94.69%, Time = 2.23s
Epoch 10: Loss = 0.1759, Accuracy = 94.97%, Time = 2.18s

=== Inference Phase ===
Test Accuracy: 94.89%, Test Loss: 0.1754


=== Trial 5 ===
=== Training Phase ===
Epoch 1: Loss = 0.8748, Accuracy = 77.79%, Time = 2.70s
Epoch 2: Loss = 0.3665, Accuracy = 89.78%, Time = 2.53s
Epoch 3: Loss = 0.3055, Accuracy = 91.31%, Time = 2.23s
Epoch 4: Loss = 0.2726, Accuracy = 92.19%, Time = 2.38s
Epoch 5: Loss = 0.2490, Accuracy = 92.82%, Time = 2.35s
Epoch 6: Loss = 0.2305, Accuracy = 93.43%, Time = 2.15s
Epoch 7: Loss = 0.2153, Accuracy = 93.85%, Time = 1.85s
Epoch 8: Loss = 0.2018, Accuracy = 94.25%, Time = 2.35s
Epoch 9: Loss = 0.1904, Accuracy = 94.58%, Time = 2.30s
Epoch 10: Loss = 0.1794, Accuracy = 94.90%, Time = 2.22s

=== Inference Phase ===
Test Accuracy: 94.77%, Test Loss: 0.1788


Average Training Loss (Epoch 10): 0.1761
Average Training Accuracy: 94.95%
Average Test Loss: 0.1746
Average Test Accuracy: 94.88%
"""