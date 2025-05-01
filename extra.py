# Re-run after kernel reset
import numpy as np
import matplotlib.pyplot as plt

# --- Logistic Regression with Mini-batch SGD ---

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_data(filename):
    data, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            *features, label = parts
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

def compute_accuracy(X, y, theta, bias):
    z = np.dot(X, theta) + bias
    y_hat = sigmoid(z)
    predictions = (y_hat >= 0.5).astype(int)
    return np.mean(predictions == y)

def train_logistic_regression_sgd(X_train, y_train, X_dev, y_dev, learning_rate, epochs, batch_size):
    N, d = X_train.shape
    theta = np.random.randn(d)
    bias = np.random.randn()

    train_accuracies = []
    dev_accuracies = []

    for epoch in range(epochs):
        indices = np.arange(N)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            z = np.dot(X_batch, theta) + bias
            y_hat = sigmoid(z)
            error = y_hat - y_batch

            gradient_theta = np.dot(error, X_batch) / len(X_batch)
            gradient_bias = np.sum(error) / len(X_batch)

            theta -= learning_rate * gradient_theta
            bias -= learning_rate * gradient_bias

        # 记录每个 epoch 的准确率
        train_acc = compute_accuracy(X_train, y_train, theta, bias)
        dev_acc = compute_accuracy(X_dev, y_dev, theta, bias)
        train_accuracies.append(train_acc)
        dev_accuracies.append(dev_acc)

    return train_accuracies, dev_accuracies

# Generate and save plots
import os

# Dummy data setup for testing only
os.makedirs("data", exist_ok=True)
with open("data/train.txt", "w") as f:
    for _ in range(100):
        features = np.random.randn(4)
        label = int(np.random.rand() > 0.5)
        f.write(" ".join(f"{x:.4f}" for x in features) + f" {label}\n")

with open("data/dev.txt", "w") as f:
    for _ in range(50):
        features = np.random.randn(4)
        label = int(np.random.rand() > 0.5)
        f.write(" ".join(f"{x:.4f}" for x in features) + f" {label}\n")

# Run experiment
X_train, y_train = load_data("data/train.txt")
X_dev, y_dev = load_data("data/dev.txt")

epochs = 100
learning_rate = 1.0
batch_sizes = [1, 8, 32, 64, len(X_train)]

for batch_size in batch_sizes:
    all_train, all_dev = [], []

    for run in range(5):
        np.random.seed(42 + run)
        train_acc, dev_acc = train_logistic_regression_sgd(
            X_train, y_train, X_dev, y_dev,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        all_train.append(train_acc)
        all_dev.append(dev_acc)

    mean_train = np.mean(all_train, axis=0)
    mean_dev = np.mean(all_dev, axis=0)
    min_dev = np.min(all_dev, axis=0)
    max_dev = np.max(all_dev, axis=0)

    plt.figure(figsize=(8, 5))
    x = range(1, epochs + 1)
    plt.plot(x, mean_dev, label="Mean Dev Accuracy", linewidth=2)
    plt.plot(x, min_dev, linestyle='--', label="Min Dev Accuracy")
    plt.plot(x, max_dev, linestyle='--', label="Max Dev Accuracy")
    plt.title(f"Logistic Regression - Dev Accuracy (Batch size = {batch_size})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"logreg_batch{batch_size}.png")
    plt.close()
