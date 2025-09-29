import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error

from main import load_and_preprocess_data
from utils import first_pred

import matplotlib.pyplot as plt


def ensure_output_dirs():
    os.makedirs("outputs", exist_ok=True)


def get_features_and_target(df: pd.DataFrame):
    # Target to explain: provided probabilities
    target_col = "Predicted probabilities"
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in dataframe. Available: {list(df.columns)}")

    y = df[target_col].astype(float).values
    # Use preprocessed features, keep both prediction columns per instruction not to drop them when loading raw
    # But exclude them from X for surrogate training
    X = df.drop(columns=[c for c in first_pred if c in df.columns], axis=1)

    # Also exclude the target column itself if present after preprocessing
    if target_col in X.columns:
        X = X.drop(columns=[target_col])

    # Defensive: drop any non-numeric columns that may remain
    X_numeric = X.select_dtypes(include=[np.number])
    feature_names = list(X_numeric.columns)
    return X_numeric.values, y, feature_names


def fit_linear_surrogate(X: np.ndarray, y: np.ndarray):
    # We approximate probabilities directly with a simple linear model for interpretability
    lin = LinearRegression()
    lin.fit(X, y)
    y_hat = lin.predict(X)
    return lin, y_hat


def fit_tree_surrogate(X: np.ndarray, y: np.ndarray, max_depth: int = 3, random_state: int = 42):
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    return tree, y_hat


def fidelity_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse


def export_linear_coefficients(model: LinearRegression, feature_names: list, path: str):
    coefs = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
        "model": "linear_surrogate",
    })
    coefs = coefs.sort_values("abs_coefficient", ascending=False)
    coefs.to_csv(path, index=False)
    return coefs


def export_tree_importances(model: DecisionTreeRegressor, feature_names: list, path: str):
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
        "model": "tree_surrogate",
    })
    importances = importances.sort_values("importance", ascending=False)
    importances.to_csv(path, index=False)
    return importances


def save_tree_plot(model: DecisionTreeRegressor, feature_names: list, path: str):
    plt.figure(figsize=(16, 10))
    plot_tree(model, feature_names=feature_names, filled=True, max_depth=3, rounded=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ensure_output_dirs()

    # 1-3) Load raw, preprocess features, keep prediction columns in the raw df
    df = load_and_preprocess_data()

    # 4) Build surrogates explaining provided probabilities
    X, y, feature_names = get_features_and_target(df)

    # Logistic-like interpretable baseline: linear regression on probabilities
    lin_model, lin_pred = fit_linear_surrogate(X, y)
    lin_r2, lin_mse = fidelity_metrics(y, lin_pred)

    # Depth-limited decision tree
    tree_model, tree_pred = fit_tree_surrogate(X, y, max_depth=3)
    tree_r2, tree_mse = fidelity_metrics(y, tree_pred)

    # 6) Interpretability outputs and 7) Save results
    linear_csv = os.path.join("outputs", "step1_linear_coefficients.csv")
    tree_csv = os.path.join("outputs", "step1_tree_importances.csv")
    tree_png = os.path.join("outputs", "step1_tree.png")

    linear_df = export_linear_coefficients(lin_model, feature_names, linear_csv)
    tree_df = export_tree_importances(tree_model, feature_names, tree_csv)
    save_tree_plot(tree_model, feature_names, tree_png)

    # Combined ranking for convenience
    combined = pd.concat([linear_df.rename(columns={"abs_coefficient": "ranking_value"})[["feature", "ranking_value", "model"]],
                          tree_df.rename(columns={"importance": "ranking_value"})[["feature", "ranking_value", "model"]]],
                         ignore_index=True)
    combined_path = os.path.join("outputs", "step1_feature_rankings.csv")
    combined.to_csv(combined_path, index=False)

    # 5) Print fidelity metrics
    print("Surrogate fidelity vs provided 'Predicted probabilities':")
    print(f"Linear surrogate -> R2: {lin_r2:.4f}, MSE: {lin_mse:.6f}")
    print(f"Decision tree    -> R2: {tree_r2:.4f}, MSE: {tree_mse:.6f}")
    print(f"Exports saved to 'outputs/'\n- Coefficients: {linear_csv}\n- Importances: {tree_csv}\n- Combined ranking: {combined_path}\n- Tree PNG: {tree_png}")


if __name__ == "__main__":
    main()


