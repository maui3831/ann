import numpy as np
import pandas as pd
from pathlib import Path

def _load_data(gate):
    file_path = Path(__file__).parent / "data" / f"{gate}.csv"
    if not file_path.exists():
        raise ValueError(f"CSV file for gate '{gate}' not found at {file_path}.")
    df = pd.read_csv(file_path)
    return df

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_x, n_h) * 0.01
    b1 = np.zeros((1, n_h))
    W2 = np.random.randn(n_h, n_y) * 0.01
    b2 = np.zeros((1, n_y))
    return W1, b1, W2, b2

def forward_prop(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A1, Z1, A2, Z2

def compute_loss(A2, Y):
    m = Y.shape[0]
    return np.mean((A2 - Y) ** 2)

def back_prop(X, Y, W1, b1, W2, b2, A1, Z1, A2, Z2):
    m = X.shape[0]
    dA2 = 2 * (A2 - Y) / m
    dZ2 = dA2 * A2 * (1 - A2)  
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)  
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

def train_ann(gate, learning_rate=0.1, num_iterations=1000, verbose=False):
    df = _load_data(gate)
    X = df[["x1", "x2"]].values
    Y = df["y"].values.reshape(-1, 1)
    n_x = X.shape[1]
    n_h = 5
    n_y = 1
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        A1, Z1, A2, Z2 = forward_prop(X, W1, b1, W2, b2)
        loss = compute_loss(A2, Y)
        dW1, db1, dW2, db2 = back_prop(X, Y, W1, b1, W2, b2, A1, Z1, A2, Z2)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        if verbose and i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    preds = predict_ann(X, W1, b1, W2, b2)
    accuracy = np.mean(preds == Y)
    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "accuracy": accuracy,
        "loss": loss,
        "preds": preds,
        "X": X,
        "Y": Y,
        "df": df,
    }

def predict_ann(X, W1, b1, W2, b2):
    _, _, A2, _ = forward_prop(X, W1, b1, W2, b2)
    return (A2 > 0.5).astype(int)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a 2-layer ANN for XOR or XNOR gate.")
    parser.add_argument("--gate", type=str, choices=["xor", "xnor"], default="xor", help="Logic gate to train the ANN on.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for training.")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of training iterations.")
    args = parser.parse_args()
    result = train_ann(args.gate, args.learning_rate, args.num_iterations, verbose=True)
    print(f"Trained on {args.gate.upper()} gate")
    print("Predictions:", result["preds"].flatten())
    print("Actual:", result["Y"].flatten())
    print("Accuracy:", result["accuracy"])
    print("Final Loss:", result["loss"])
