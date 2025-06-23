import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def _load_data(gate):
    """Load truth table data for the given gate from a CSV file."""
    file_path = Path(__file__).parent / "data" / f"{gate}.csv"
    if not file_path.exists():
        raise ValueError(f"CSV file for gate '{gate}' not found at {file_path}.")
    df = pd.read_csv(file_path)
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


def train_perceptron(gate, learning_rate=0.1, num_iterations=1000):
    df = _load_data(gate)
    X = df[["x1", "x2"]].values
    Y = df["y"].values
    W, b = _init_params(X.shape[1])
    W, b = train(X, Y, W, b, learning_rate, num_iterations)
    preds = predict(X, W, b)
    accuracy = np.mean(preds == Y)
    loss = compute_loss(forward_prop(X, W, b)[0], Y)
    return {
        "W": W,
        "b": b,
        "accuracy": accuracy,
        "loss": loss,
        "preds": preds,
        "X": X,
        "Y": Y,
        "df": df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a perceptron for a chosen logic gate."
    )
    parser.add_argument(
        "--gate",
        type=str,
        choices=["and", "nand", "or"],
        default="and",
        help="Logic gate to train the perceptron on.",
    )
    args = parser.parse_args()

    df = _load_data(args.gate)
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
    print("Final Weights (W):", W)
    print("Final Bias (b):", b)
