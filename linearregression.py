import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# =========================
# Data Loading
# =========================
# CCPP columns: AT (temperature), V (exhaust vacuum),
# AP (ambient pressure), RH (relative humidity), PE (net output, MW)
TARGET = "PE"

def load_data(filename="Folds5x2_pp.csv"):
    # relative path 
    path = Path(__file__).resolve().parent / "CCPP" / filename
    df = pd.read_csv(path)

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    return X, y


# =========================
# Preprocessing
# =========================
def standardize(X_train, X_test):
    # Mean and std from the training set only
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)

    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    return X_train_std, X_test_std, mu, sigma


def add_bias(X):
    # Append a column of 1s as the first column of the design matrix,
    # so w_0 (bias) is part of the parameter vector
    return np.column_stack([np.ones(len(X)), X])


# =========================
# Cost and Prediction
# =========================
def predict(X, w):
    # y_hat = X . w   (N x 1)
    return X @ w


def cost(X, y, w):
    # Cost(w) = 1/(2N) * sum (y_n - w^T x_n)^2   (one half of MSE)
    errors = y - predict(X, w)
    return (errors @ errors) / (2 * len(y))


# =========================
# Ordinary Least Squares
# =========================
def ols(X, y):
    # Normal equation: w_hat = (X^T X)^-1 X^T y
    return np.linalg.inv(X.T @ X) @ X.T @ y


# =========================
# Gradient Descent
# =========================
def gradient_descent(X, y, learning_rate, threshold=1e-8, max_iterations=100_000, seed=42):
    N = len(y)

    # Randomly initialize w in [-1, 1]
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, X.shape[1])

    cost_history = [cost(X, y, w)]

    for k in range(1, max_iterations + 1):
        # gradient -> delta_w = 1/N * X^T (X w - y)   ((D+1) x 1)
        delta_w = X.T @ (predict(X, w) - y) / N

        # w(k) := w(k-1) - alpha * delta_w(k-1); all w_j updated same time
        w = w - learning_rate * delta_w

        cost_history.append(cost(X, y, w))

        # if learning rate too large, cost blew up, will stop early
        if not np.isfinite(cost_history[-1]):
            return w, k, cost_history, False

        # converged ifthe change between two consecutive costs is below the threshold
        if abs(cost_history[-2] - cost_history[-1]) < threshold:
            return w, k, cost_history, True

    return w, max_iterations, cost_history, False


# =========================
# Evaluation Metrics
# =========================
def evaluate(y_actual, y_predicted):
    # error = y - y_hat
    errors = y_actual - y_predicted

    # MSE = 1/N * e^T e
    mse = (errors @ errors) / len(errors)
    mae = np.mean(np.abs(errors))

    # R^2 = 1 - SSE / SST
    sse = errors @ errors
    sst = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - sse / sst

    return mse, mae, r2


def print_metrics(label, y_train, train_pred, y_test, test_pred):
    train = evaluate(y_train, train_pred)
    test = evaluate(y_test, test_pred)

    print(f"\n{label}")
    print(f"{'Metric':<8}{'Training':>12}{'Testing':>12}")
    for name, tr, te in zip(("MSE", "MAE", "R2"), train, test):
        print(f"{name:<8}{tr:>12.4f}{te:>12.4f}")


# =========================
# Main Program
# =========================
def main():
    # X 9568x4, each row is one reading, each col is one ambient feature
    # y 1D, net hourly output for the matching row in X
    X, y = load_data()
    features = list(X.columns)

    # split 80% training / 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize inputs only; y stays in MW so metrics are in real units
    X_train, X_test, mu, sigma = standardize(X_train, X_test)

    print(f"Dataset: {len(X)} examples, {len(features)} features")
    print(f"Training: {len(X_train)} examples, Testing: {len(X_test)} examples\n")

    print(f"{'Feature':<8}{'Mean':>12}{'Std':>12}")
    for feature in features:
        print(f"{feature:<8}{mu[feature]:>12.4f}{sigma[feature]:>12.4f}")

    # design matrix N x (D+1) with a leading column of 1s
    X_train = add_bias(X_train.to_numpy())
    X_test = add_bias(X_test.to_numpy())
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()


    names = ["w0 (bias)"] + [f"w{j} ({f})" for j, f in enumerate(features, 1)]

    # OLS
    ols_w = ols(X_train, y_train)

    print("\n==== OLS ====")
    for name, w_j in zip(names, ols_w):
        print(f"  {name:<10}{w_j:>14.6f}")
    print(f"  Cost(w) = {cost(X_train, y_train, ols_w):.6f}")

    print_metrics("OLS metrics", y_train, predict(X_train, ols_w),
                  y_test, predict(X_test, ols_w))

    # Gradient Descent
    # learning rates within range of 0.001 - 0.1
    learning_rates = [0.001, 0.01, 0.05, 0.1]

    print("\n==== Gradient Descent ====")
    print(f"{'LR':<8}{'Converged':>10}{'Iterations':>12}{'Final cost':>14}{'||w - w_OLS||':>16}")
    runs = {}
    for lr in learning_rates:
        w, iterations, history, converged = gradient_descent(X_train, y_train, lr)
        runs[lr] = w
        print(f"{lr:<8}{str(converged):>10}{iterations:>12}{history[-1]:>14.6f}"
              f"{np.linalg.norm(w - ols_w):>16.8f}")

    # report the fastest learning rate in detail
    best_lr = max(learning_rates)
    gd_w = runs[best_lr]

    print(f"\nGradient descent weights (LR = {best_lr}):")
    for name, w_j in zip(names, gd_w):
        print(f"  {name:<10}{w_j:>14.6f}")

    print_metrics(f"Gradient descent metrics (LR = {best_lr})", y_train,
                  predict(X_train, gd_w), y_test, predict(X_test, gd_w))



if __name__ == "__main__":
    main()
