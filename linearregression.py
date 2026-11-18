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
    # Path relative to linearregression.py
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

# =========================
# Ordinary Least Squares
# =========================
def ols(X_train, y_train):
    # Calculate the OLS weights using the normal equation
    XTX = X_train.T @ X_train
    XTy = X_train.T @ y_train

    # Solve XTX * w = XTy
    weights = np.linalg.solve(XTX, XTy)

    return weights

# =========================
# Prediction
# =========================
def predict(X, weights):
    return X @ weights

# =========================
# Evaluation Metrics
# =========================
def evaluate(y_actual, y_predicted):
    errors = y_actual - y_predicted

    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))

    ss_res = np.sum(errors ** 2)
    ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)

    r2 = 1 - (ss_res / ss_total)

    return mse, mae, r2

# =========================
# Gradient Descent
# =========================
def gradient_descent(X_train, y_train, learning_rate=0.05, max_iterations=30000, tolerance=1e-6):
    n_samples, n_features = X_train.shape

    # Initialize weights to zeros, including the intercept
    weights = np.zeros(n_features)

    # Store training errors to track convergence
    loss_history = []

    for iteration in range(1, max_iterations + 1):
        # Calculate predictions
        predictions = X_train @ weights

        # Calculate prediction errors
        errors = predictions - y_train

        # Compute gradient
        gradient = (X_train.T @ errors) / n_samples

        # Track mean squared error
        mse = np.mean(errors ** 2)
        loss_history.append(mse)

        # Stop if the gradient is sufficiently small
        if np.linalg.norm(gradient) < tolerance:
            return weights, iteration - 1, loss_history, True

        # Update weights
        weights -= learning_rate * gradient

    return weights, max_iterations, loss_history, False

# =========================
# Main Program
# =========================
def main():
    # X 9568x4, each row is one reading, each col is one ambient feature
    # y 1D, net hourly output for the matching row in X
    X, y = load_data()

    # split 80% training / 20% testing 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize inputs only; y stays in MW so metrics are in real units
    X_train, X_test, mu, sigma = standardize(X_train, X_test)

    # Convert DataFrames and Series to NumPy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Add a column of ones for the intercept
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    # Train OLS model
    ols_weights = ols(X_train, y_train)

    print("\n==== OLS Results ====")
    print("OLS Weights:")
    print(ols_weights)

    # Generate predictions
    ols_train_predictions = predict(X_train, ols_weights)
    ols_test_predictions = predict(X_test, ols_weights)

    print(f"Dataset: {len(X)} examples, {X.shape[1]} features")
    print(f"Training: {len(X_train)} examples, Testing: {len(X_test)} examples\n")

    print(f"{'Feature':<8}{'Mean':>12}{'Std':>12}")
    for feature in X.columns:
        print(f"{feature:<8}{mu[feature]:>12.4f}{sigma[feature]:>12.4f}")

    # Self-check: standardized feature columns should have mean 0 and std 1
    assert np.allclose(X_train[:, 1:].mean(axis=0), 0)
    assert np.allclose(X_train[:, 1:].std(axis=0, ddof=0), 1)

    # Evaluate OLS on training data
    train_mse, train_mae, train_r2 = evaluate(y_train, ols_train_predictions)

    # Evaluate OLS on testing data
    test_mse, test_mae, test_r2 = evaluate(y_test, ols_test_predictions)

    print("\nTraining Metrics:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  R²:  {train_r2:.4f}")

    print("\nTesting Metrics:")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²:  {test_r2:.4f}")

    # =========================
    # Gradient Descent Results
    # =========================
    learning_rates = [0.01, 0.05, 0.1]

    for lr in learning_rates:
        gd_weights, iterations, history, converged = gradient_descent(
            X_train, y_train, learning_rate=lr
        )

        # Generate predictions
        gd_train_predictions = predict(X_train, gd_weights)
        gd_test_predictions = predict(X_test, gd_weights)

        # Evaluate training performance
        gd_train_mse, gd_train_mae, gd_train_r2 = evaluate(y_train, gd_train_predictions)

        # Evaluate testing performance
        gd_test_mse, gd_test_mae, gd_test_r2 = evaluate(y_test, gd_test_predictions)

        print(f"\n==== Gradient Descent (LR = {lr}) ====")
        print(f"Converged: {converged}")
        print(f"Iterations: {iterations}")
        print(f"Weights: {gd_weights}")

        print("\nTraining Metrics:")
        print(f"  MSE: {gd_train_mse:.4f}")
        print(f"  MAE: {gd_train_mae:.4f}")
        print(f"  R²:  {gd_train_r2:.4f}")

        print("\nTesting Metrics:")
        print(f"  MSE: {gd_test_mse:.4f}")
        print(f"  MAE: {gd_test_mae:.4f}")
        print(f"  R²:  {gd_test_r2:.4f}")

        # Compare gradient descent weights against OLS
        weight_difference = np.linalg.norm(ols_weights - gd_weights)

        print(f"\nWeight difference from OLS: {weight_difference:.8f}")

if __name__ == "__main__":
    main()
