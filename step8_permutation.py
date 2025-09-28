import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from main import train_model

try:
    from surrogate_models import get_feature_names  # déjà défini chez toi
except ImportError:
    # fallback minimal
    def get_feature_names(pre):
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                fn = list(trans.get_feature_names_out(cols))
            else:
                if isinstance(cols, (list, tuple, np.ndarray)):
                    fn = list(cols)
                else:
                    fn = [cols]
            names.extend(fn)
        return names

# ---------- 1) Compute Permutation Importance ----------
def compute_permutation_importance(metric: str = "roc_auc",
                                   n_repeats: int = 30,
                                   normalize: bool = True) -> pd.DataFrame:
    """
    Calcule la Permutation Importance sur le test set de train_model().
    Renvoie un DataFrame trié décroissant avec colonnes: feature, decrease, std, percent (si normalize=True).
    metric: 'roc_auc' | 'average_precision' | 'accuracy'
    """

    def _score(est, X, y):
        if metric == "accuracy":
            y_hat = est.predict(X)
            return accuracy_score(y, y_hat)
        # métriques "probabilistes"
        if hasattr(est, "predict_proba"):
            y_score = est.predict_proba(X)[:, 1]
        elif hasattr(est, "decision_function"):
            y_score = est.decision_function(X)
        else:
            y_score = est.predict(X)
        if metric == "roc_auc":
            return roc_auc_score(y, y_score)
        elif metric == "average_precision":
            return average_precision_score(y, y_score)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    # 1) pipeline + test
    pipeline, X_test, y_test = train_model()

    # 2) features transformées
    pre = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(pre)

    # 3) permutation importance
    pi = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=_score,     # <- scorer custom
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    means = pi.importances_mean
    stds  = pi.importances_std
    order = np.argsort(means)[::-1]

    df = pd.DataFrame({
        "feature": np.array(feature_names)[order],
        "decrease": means[order],
        "std": stds[order],
    })

    if normalize and df["decrease"].sum() != 0:
        df["percent"] = 100 * df["decrease"] / df["decrease"].sum()

    return df


# ---------- 2) Plot Top-10 ----------
def plot_top10_permutation_importance(pi_df: pd.DataFrame,
                                      metric: str = "roc_auc",
                                      use_percent: bool = True,
                                      show_errorbars: bool = True) -> None:
    """
    Affiche le Top-10 des features par permutation importance.
    use_percent=True -> axe X en 'Contribution (%)' (colonne 'percent' requise).
    """

    df = pi_df.copy()
    k = min(10, len(df))

    if use_percent and "percent" in df.columns:
        vals = df["percent"].to_numpy()[:k]
        xerr = (100 * df["std"] / df["decrease"].sum()).to_numpy()[:k] if show_errorbars else None
        xlabel = "Contribution (%)"
    else:
        vals = df["decrease"].to_numpy()[:k]
        xerr = df["std"].to_numpy()[:k] if show_errorbars else None
        xlabel = f"Mean decrease in {metric}"

    feats = df["feature"].to_numpy()[:k]

    plt.figure(figsize=(9, 0.45 * k + 3))
    plt.barh(range(k), vals, xerr=xerr)
    plt.yticks(range(k), feats)
    plt.xlabel(xlabel)
    plt.title(f"Permutation Importance — Top {k} ({metric})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
