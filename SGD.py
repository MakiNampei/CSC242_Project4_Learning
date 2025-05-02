
import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(z):
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))

def load_data(filename):
    data = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if not parts:
                continue
            features = parts[:-1]
            label = parts[-1]
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

def calculate_accuracy(X, y, theta, bias):
    if X.shape[0] == 0:
        return 0.0
    z = np.dot(X, theta) + bias
    y_hat = sigmoid(z)
    predictions = (y_hat >= 0.5).astype(int)
    accuracy = np.mean(predictions == y.astype(int))
    return accuracy

def train_logistic_regression_minibatch(X_train, y_train, X_dev, y_dev, learning_rate=0.1, epochs=100, batch_size=32):
    N, d = X_train.shape
    theta = np.random.randn(d) * 0.01
    bias = np.random.randn() * 0.01
    dev_accuracy_history = []
    indices = np.arange(N)

    for epoch in range(epochs):
        random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, N, batch_size):
            batch_indices = indices[i:min(i + batch_size, N)]
            if len(batch_indices) == 0:
                continue

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            current_batch_size = len(X_batch)

            z_batch = np.dot(X_batch, theta) + bias
            y_hat_batch = sigmoid(z_batch)
            error_batch = y_hat_batch - y_batch

            gradient_theta = np.dot(X_batch.T, error_batch) / current_batch_size
            gradient_bias = np.sum(error_batch) / current_batch_size

            theta -= learning_rate * gradient_theta
            bias -= learning_rate * gradient_bias

        current_dev_accuracy = calculate_accuracy(X_dev, y_dev, theta, bias)
        dev_accuracy_history.append(current_dev_accuracy)

    return theta, bias, dev_accuracy_history

X_train, y_train = load_data("train.txt")
X_dev, y_dev = load_data("dev.txt")

learning_rate = 1.0
batch_sizes = [1, 8, 16, 32]
epochs = 100
runs = 5

plt.style.use('default')

for batch_size in batch_sizes:
    all_runs_dev_accuracy = []

    for run in range(runs):
        _, _, dev_accuracy_history = train_logistic_regression_minibatch(
            X_train, y_train, X_dev, y_dev,
            learning_rate, epochs, batch_size
        )
        all_runs_dev_accuracy.append(dev_accuracy_history)

    accuracy_array = np.array(all_runs_dev_accuracy)
    min_accuracy = np.min(accuracy_array, axis=0)
    mean_accuracy = np.mean(accuracy_array, axis=0)
    max_accuracy = np.max(accuracy_array, axis=0)

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 7))

    plt.plot(epochs_range, mean_accuracy, label='Mean Accuracy', color='blue', linewidth=2)
    plt.plot(epochs_range, min_accuracy, label='Min Accuracy', linestyle='--', color='orange', linewidth=2)
    plt.plot(epochs_range, max_accuracy, label='Max Accuracy', linestyle='--', color='green', linewidth=2)

    plt.title(f'Dev Accuracy vs. Epoch (Batch Size={batch_size}, LR={learning_rate})', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Dev Accuracy', fontsize=14)
    plt.ylim(0.9, 0.99)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig(f"accuracy_plot_lines_batch_{batch_size}.png")

plt.show()
