import numpy as np
import pandas as pd
import argparse


def _init_data(gate):
    data = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
    }
    if gate == "and":
        data["y"] = [0, 0, 0, 1]
    elif gate == "nand":
        data["y"] = [1, 1, 1, 0]
    else:
        raise ValueError("Gate must be 'and' or 'nand'")
    df = pd.DataFrame(data)
    return df


def _init_params(n_features):
    W = np.random.rand(n_features)
    b = np.random.rand()
    return W, b


def relu(Z):
    return np.maximum(0, Z)


def forward_prop(X, W, b):
    Z = np.dot(X, W) + b
    A = relu(Z)
    return A, Z


def compute_loss(A, Y):
    return np.mean((A - Y) ** 2)


def back_prop(X, Y, A, Z):
    m = X.shape[0]
    dA = 2 * (A - Y) / m
    dZ = dA * (Z > 0)
    dW = np.dot(X.T, dZ)
    db = np.sum(dZ)
    return dW, db


def train(X, Y, W, b, learning_rate, num_iterations):
    for i in range(num_iterations):
        A, Z = forward_prop(X, W, b)
        loss = compute_loss(A, Y)
        dW, db = back_prop(X, Y, A, Z)
        W -= learning_rate * dW
        b -= learning_rate * db
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    return W, b


def predict(X, W, b):
    A, _ = forward_prop(X, W, b)
    return (A > 0.5).astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a perceptron for AND or NAND gate."
    )
    parser.add_argument(
        "--gate",
        type=str,
        choices=["and", "nand"],
        default="and",
        help="Logic gate to train (and or nand)",
    )
    args = parser.parse_args()

    df = _init_data(args.gate)
    X = df[["x1", "x2"]].values
    Y = df["y"].values
    W, b = _init_params(X.shape[1])
    learning_rate = 0.1
    num_iterations = 1000

    W, b = train(X, Y, W, b, learning_rate, num_iterations)
    preds = predict(X, W, b)
    print("Predictions:", preds)
    print("Actual:", Y)
    print("Accuracy:", np.mean(preds == Y))
