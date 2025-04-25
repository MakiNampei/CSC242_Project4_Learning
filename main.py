import numpy as np

# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 读取训练或测试数据
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

# 逻辑回归训练函数
def train_logistic_regression(X, y, learning_rate=0.001, epochs=100):
    N, d = X.shape
    theta = np.random.randn(d)
    bias = np.random.randn()

    for epoch in range(epochs):
        z = np.dot(X, theta) + bias
        y_hat = sigmoid(z)

        error = y_hat - y
        gradient_theta = np.dot(error, X) / N
        gradient_bias = np.sum(error) / N

        theta -= learning_rate * gradient_theta
        bias -= learning_rate * gradient_bias

        print(f"Epoch {epoch + 1}: weights = {theta}, bias = {bias}")

    return theta, bias

# 使用训练好的模型进行预测
def evaluate_model(X, y, theta, bias):
    z = np.dot(X, theta) + bias
    y_hat = sigmoid(z)
    predictions = (y_hat >= 0.5).astype(int)

    correct = np.sum(predictions == y)
    total = len(y)
    accuracy = correct / total

    print("\n[Dev Set Evaluation]")
    for i, (pred, actual) in enumerate(zip(predictions, y)):
        print(f"Sample {i + 1}: Predicted = {pred}, Actual = {int(actual)}")

    print(f"\nAccuracy on dev set: {accuracy * 100:.2f}%")

# 主函数
if __name__ == "__main__":
    # 训练阶段
    X_train, y_train = load_data("train.txt")
    theta, bias = train_logistic_regression(X_train, y_train, learning_rate=0.1, epochs=2)

    # 测试阶段
    X_dev, y_dev = load_data("dev.txt")
    evaluate_model(X_dev, y_dev, theta, bias)
