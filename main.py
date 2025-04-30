#GradeScope 版

import numpy as np
import sys

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

def cross_entropy_loss(y, y_hat):
    epsilon = 1e-10
    return -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))

def train_logistic_regression(X, y, learning_rate=0.01, epochs=100):
    N, d = X.shape
    theta = np.random.randn(d)
    bias = np.random.randn()

    for _ in range(epochs):
        z = np.dot(X, theta) + bias
        y_hat = sigmoid(z)

        error = y_hat - y
        gradient_theta = np.dot(error, X) / N
        gradient_bias = np.sum(error) / N

        theta -= learning_rate * gradient_theta
        bias -= learning_rate * gradient_bias

    return theta, bias

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py TRAIN_FILE LEARNING_RATE NUM_EPOCHS")
        sys.exit(1)

    train_file = sys.argv[1]
    learning_rate = float(sys.argv[2])
    num_epochs = int(sys.argv[3])

    X_train, y_train = load_data(train_file)
    theta, bias = train_logistic_regression(X_train, y_train, learning_rate, num_epochs)

    print(" ".join(f"{w:.6f}" for w in np.append(theta, bias)))
