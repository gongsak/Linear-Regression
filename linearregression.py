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

    print(f"Dataset: {len(X)} examples, {X.shape[1]} features")
    print(f"Training: {len(X_train)} examples, Testing: {len(X_test)} examples\n")

    print(f"{'Feature':<8}{'Mean':>12}{'Std':>12}")
    for feature in X.columns:
        print(f"{feature:<8}{mu[feature]:>12.4f}{sigma[feature]:>12.4f}")

    # Self-check: training set should now be mean 0, std 1
    assert np.allclose(X_train.mean(axis=0), 0)
    assert np.allclose(X_train.std(axis=0, ddof=0), 1)


if __name__ == "__main__":
    main()
