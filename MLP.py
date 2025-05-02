
import numpy as np
import sys
import random
import matplotlib.pyplot as plt

def sigmoid(z):
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def load_data(filename):
    data = []
    labels = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if not parts: continue
                features = parts[:-1]
                label = parts[-1]
                if label not in [0.0, 1.0]:
                    continue
                data.append(features)
                labels.append(label)
        return np.array(data), np.array(labels).reshape(-1, 1)
    except FileNotFoundError:
        print(f"Error: File not found at {filename}", file=sys.stderr)
        sys.exit(1)


def calculate_mlp_accuracy(X, y, W1, b1, W2, b2):
    N = X.shape[0]
    if N == 0: return 0.0
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    predictions = (A2 >= 0.5).astype(int)
    accuracy = np.mean(predictions == y.astype(int))
    return accuracy


def train_mlp_minibatch(X_train, y_train, X_dev, y_dev, hidden_size=10,
                        learning_rate=0.01, epochs=100, batch_size=32):
    N, input_size = X_train.shape
    output_size = 1
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
    b2 = np.zeros((1, output_size))

    dev_accuracy_history = []
    indices = np.arange(N)

    for epoch in range(epochs):
        random.shuffle(indices)
        for i in range(0, N, batch_size):
            batch_indices = indices[i:min(i + batch_size, N)]
            if len(batch_indices) == 0: continue

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            current_batch_size = X_batch.shape[0]

            Z1 = np.dot(X_batch, W1) + b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = sigmoid(Z2)

            dZ2 = A2 - y_batch
            dW2 = np.dot(A1.T, dZ2) / current_batch_size
            db2 = np.sum(dZ2, axis=0, keepdims=True) / current_batch_size

            dA1 = np.dot(dZ2, W2.T)
            dZ1 = dA1 * sigmoid_derivative(Z1)
            dW1 = np.dot(X_batch.T, dZ1) / current_batch_size
            db1 = np.sum(dZ1, axis=0, keepdims=True) / current_batch_size

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        current_dev_accuracy = calculate_mlp_accuracy(X_dev, y_dev, W1, b1, W2, b2)
        dev_accuracy_history.append(current_dev_accuracy)

    return W1, b1, W2, b2, dev_accuracy_history

if __name__ == "__main__":
    DEFAULT_TRAIN_FILE = 'train.txt'
    DEFAULT_DEV_FILE = 'dev.txt'
    LEARNING_RATES_TO_TEST = [1.0, 0.1, 0.01, 0.001, 3.0]
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_HIDDEN_SIZE = 10
    NUM_RUNS = 5

    train_file = DEFAULT_TRAIN_FILE
    dev_file = DEFAULT_DEV_FILE
    num_epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    hidden_size = DEFAULT_HIDDEN_SIZE

    X_train, y_train = load_data(train_file)
    X_dev, y_dev = load_data(dev_file)

    plt.style.use('default')
    results_by_lr = {}

    for learning_rate in LEARNING_RATES_TO_TEST:
        all_runs_dev_accuracy = []

        for run in range(NUM_RUNS):
            _, _, _, _, dev_accuracy_history = train_mlp_minibatch(
                X_train, y_train, X_dev, y_dev,
                hidden_size, learning_rate, num_epochs, batch_size
            )
            all_runs_dev_accuracy.append(dev_accuracy_history)

        accuracy_array = np.array(all_runs_dev_accuracy)
        results_by_lr[learning_rate] = accuracy_array

        min_accuracy = np.min(accuracy_array, axis=0)
        mean_accuracy = np.mean(accuracy_array, axis=0)
        max_accuracy = np.max(accuracy_array, axis=0)

        epochs_range = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 6))

        plt.plot(epochs_range, mean_accuracy, label='Mean Accuracy', color='blue', linewidth=2)
        plt.plot(epochs_range, min_accuracy, label='Min Accuracy', linestyle='--', color='orange', linewidth=2)
        plt.plot(epochs_range, max_accuracy, label='Max Accuracy', linestyle='--', color='green', linewidth=2)

        plt.title(f'MLP Dev Accuracy vs. Epoch (LR = {learning_rate}, Batch Size = {batch_size})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Dev Accuracy', fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=10)
        plt.grid(True)

        plt.savefig(f"mlp_accuracy_plot_lr_{learning_rate}.png")

    plt.show()
