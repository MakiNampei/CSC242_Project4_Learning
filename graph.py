import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_data(filename):
    data = []
    labels = []
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

def train_and_record_dev(X_train, y_train, X_dev, y_dev, learning_rate, epochs):
    N, d = X_train.shape
    theta = np.random.randn(d)
    bias = np.random.randn()

    dev_accuracies = []

    for _ in range(epochs):
        z = np.dot(X_train, theta) + bias
        y_hat = sigmoid(z)

        error = y_hat - y_train
        gradient_theta = np.dot(error, X_train) / N
        gradient_bias = np.sum(error) / N

        theta -= learning_rate * gradient_theta
        bias -= learning_rate * gradient_bias

        dev_acc = compute_accuracy(X_dev, y_dev, theta, bias)
        dev_accuracies.append(dev_acc)

    return dev_accuracies

def plot_mean_min_max(accuracies_list, lr, epochs):
    all_accs = np.array(accuracies_list)  # shape: (runs, epochs)
    mean_acc = np.mean(all_accs, axis=0)
    min_acc = np.min(all_accs, axis=0)
    max_acc = np.max(all_accs, axis=0)

    plt.figure(figsize=(8, 5))
    x = range(1, epochs + 1)
    plt.plot(x, mean_acc, label='Mean Dev Accuracy')
    plt.plot(x, min_acc, label='Min Dev Accuracy', linestyle='--')
    plt.plot(x, max_acc, label='Max Dev Accuracy', linestyle='--')

    plt.title(f"Dev Accuracy per Epoch (Learning Rate = {lr})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    #
    plt.ylim(0.4, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"lr_{lr}.png")
    plt.close()

if __name__ == "__main__":
    learning_rates = [3.0] # [1.0, 0.1, 0.01, 0.001, 3.0]
    epochs = 50
    runs = 5

    X_train, y_train = load_data("train.txt")
    X_dev, y_dev = load_data("dev.txt")

    for lr in learning_rates:
        all_dev_accs = []
        for i in range(runs):
            np.random.seed(1000 + i)
            dev_acc = train_and_record_dev(X_train, y_train, X_dev, y_dev, lr, epochs)
            all_dev_accs.append(dev_acc)
        plot_mean_min_max(all_dev_accs, lr, epochs)
        print(f"Saved plot: lr_{lr}.png")
