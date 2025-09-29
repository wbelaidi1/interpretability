import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from main import train_model


def plot_feature_importance(nb_features: int = 10) -> None:
    """
    Plot the top x features based on permutation importance.

    Args:
        nb_features (int, optional): Number of top features to display.
        Defaults to 10.

    Returns:
        None
    """

    pipeline, X_test, y_test = train_model()

    if hasattr(X_test, "columns"):
        feature_names = X_test.columns
    else:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=15, scoring="roc_auc", random_state=42
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    top_features = importance_df.head(nb_features).iloc[::-1]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        color="steelblue",
        alpha=0.8,
    )
    plt.xlabel("Permutation Importance (ROC AUC)", fontsize=12)
    plt.title(
        f"Top {nb_features} Features – Permutation Importance",
        fontsize=14,
        weight="bold",
    )
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
