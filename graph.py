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

def train_and_record_dev_sgd(X_train, y_train, X_dev, y_dev, learning_rate, epochs, batch_size=1):
    N, d = X_train.shape
    theta = np.random.randn(d)
    bias = np.random.randn()

    dev_accuracies = []

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, N, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            z = np.dot(X_batch, theta) + bias
            y_hat = sigmoid(z)

            error = y_hat - y_batch
            gradient_theta = np.dot(error, X_batch) / batch_size
            gradient_bias = np.sum(error) / batch_size

            theta -= learning_rate * gradient_theta
            bias -= learning_rate * gradient_bias

        dev_acc = compute_accuracy(X_dev, y_dev, theta, bias)
        dev_accuracies.append(dev_acc)

    return dev_accuracies

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def train_and_record_dev_mlp(X_train, y_train, X_dev, y_dev, learning_rate, epochs, batch_size=1):
    N, d = X_train.shape
    W1 = np.random.randn(d, 10)
    b1 = np.zeros(10)
    W2 = np.random.randn(10, 1)
    b2 = np.zeros(1)

    dev_accuracies = []

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, N, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            hidden_input = np.dot(X_batch, W1) + b1
            hidden_output = sigmoid(hidden_input)

            output_input = np.dot(hidden_output, W2) + b2
            output = sigmoid(output_input)

            output_error = output - y_batch
            hidden_error = np.dot(output_error, W2.T) * sigmoid_derivative(hidden_input)

            dW2 = np.dot(hidden_output.T, output_error) / batch_size
            db2 = np.sum(output_error) / batch_size
            dW1 = np.dot(X_batch.T, hidden_error) / batch_size
            db1 = np.sum(hidden_error, axis=0) / batch_size

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        dev_acc = compute_accuracy(X_dev, y_dev, W2, b2)
        dev_accuracies.append(dev_acc)

    return dev_accuracies


if __name__ == "__main__":
    learning_rates = [1.0, 0.1, 0.01, 0.001, 3.0]
    epochs = 50
    runs = 5
    batch_size = 1  # For SGD

    X_train, y_train = load_data("train.txt")
    X_dev, y_dev = load_data("dev.txt")

    for lr in learning_rates:
        all_dev_accs = []
        for i in range(runs):
            np.random.seed(1000 + i)

            # Choose logistic regression or MLP
            print(f"Running SGD for Logistic Regression (LR={lr})...")
            dev_acc_sgd = train_and_record_dev_sgd(X_train, y_train, X_dev, y_dev, lr, epochs, batch_size)
            all_dev_accs.append(dev_acc_sgd)

            print(f"Running MLP for Logistic Regression (LR={lr})...")
            dev_acc_mlp = train_and_record_dev_mlp(X_train, y_train, X_dev, y_dev, lr, epochs, batch_size)
            all_dev_accs.append(dev_acc_mlp)

        plot_mean_min_max(all_dev_accs, lr, epochs)
        print(f"Saved plot: lr_{lr}.png")
