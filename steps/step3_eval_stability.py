import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

from main import create_datasets


def ensure_dirs():
    os.makedirs("reports", exist_ok=True)


def load_artifacts():
    model = joblib.load(os.path.join("models", "best_model.pkl"))
    with open(os.path.join("models", "best_model_meta.json"), "r") as f:
        meta = json.load(f)
    y_test = np.load(os.path.join("preds", "y_test.npy"))
    y_proba_test = np.load(os.path.join("preds", "y_proba_test.npy"))
    return model, meta, y_test, y_proba_test


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
    return {
        "auc": float(auc),
        "brier": float(brier),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_mask = binids == i
        if not np.any(bin_mask):
            continue
        bin_prob = y_prob[bin_mask]
        bin_true = y_true[bin_mask]
        acc = np.mean((bin_prob >= 0.5).astype(int) == bin_true)
        conf = np.mean(bin_prob)
        w = np.mean(bin_mask)
        ece += w * abs(acc - conf)
    return float(ece)


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    # Murphy (1973) decomposition into reliability, resolution, uncertainty
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    p_bar = np.mean(y_true)
    reliability = 0.0
    resolution = 0.0
    uncertainty = p_bar * (1 - p_bar)
    N = len(y_true)
    for i in range(n_bins):
        mask = binids == i
        n_i = np.sum(mask)
        if n_i == 0:
            continue
        o_i = np.mean(y_true[mask])
        p_i = np.mean(y_prob[mask])
        reliability += (n_i / N) * (p_i - o_i) ** 2
        resolution += (n_i / N) * (o_i - p_bar) ** 2
    return float(reliability), float(resolution), float(uncertainty)


def plot_calibration_and_hist(y_true: np.ndarray, y_prob: np.ndarray, path: str):
    import matplotlib.pyplot as plt
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(prob_pred, prob_true, marker="o", label="Model")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Empirical probability")
    axes[0].set_title("Calibration curve")
    axes[0].legend()
    axes[1].hist(y_prob, bins=20, color="#4C72B0", alpha=0.8)
    axes[1].set_title("Predicted probability histogram")
    axes[1].set_xlabel("Probability")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, roc_path: str, pr_path: str):
    import matplotlib.pyplot as plt
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()
    # PR
    from sklearn.metrics import average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()


def plot_score_distribution(y_true: np.ndarray, y_prob: np.ndarray, path: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_true == 0], bins=15, alpha=0.7, label="Class 0")
    plt.hist(y_prob[y_true == 1], bins=15, alpha=0.7, label="Class 1")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Score distribution by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def retrain_with_same_family(best_name: str, best_params: Dict, X_train, y_train):
    if best_name == "XGBClassifier":
        from xgboost import XGBClassifier
        model = XGBClassifier(**best_params)
    elif best_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**best_params)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**best_params)
    model.fit(X_train, y_train)
    return model


def resampling_stability(B: int, best_name: str, best_params: Dict, X_train, y_train, X_test, y_test, feature_names: List[str]):
    rng = np.random.RandomState(42)
    aucs, briers = [], []
    y_probas = []
    fi_matrix = []

    for b in range(B):
        # Bootstrap sample indices
        idx = rng.choice(np.arange(len(y_train)), size=len(y_train), replace=True)
        Xb, yb = X_train[idx], y_train[idx]
        model = retrain_with_same_family(best_name, best_params, Xb, yb)
        prob = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, prob))
        briers.append(brier_score_loss(y_test, prob))
        y_probas.append(prob)

        # Feature importance (native or permutation fallback)
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = getattr(model, "coef_")
            importance = np.abs(coef.ravel())
        else:
            pi = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
            importance = pi.importances_mean
        fi_matrix.append(importance)

    y_probas = np.vstack(y_probas)
    fi_matrix = np.vstack(fi_matrix)

    # Prediction stability vs baseline (first run as baseline)
    base = y_probas[0]
    corrs = []
    rmses = []
    for i in range(1, B):
        corr = np.corrcoef(base, y_probas[i])[0, 1]
        rmse = float(np.sqrt(np.mean((base - y_probas[i]) ** 2)))
        corrs.append(float(corr))
        rmses.append(rmse)

    # FI stability: L2 distance and rank correlation
    from scipy.stats import spearmanr
    l2_dists = []
    rank_corrs = []
    base_fi = fi_matrix[0]
    base_ranks = pd.Series(base_fi, index=feature_names).rank(ascending=False, method="average").values
    for i in range(1, B):
        l2_dists.append(float(np.linalg.norm(base_fi - fi_matrix[i])))
        ranks_i = pd.Series(fi_matrix[i], index=feature_names).rank(ascending=False, method="average").values
        rank_corrs.append(float(spearmanr(base_ranks, ranks_i).correlation))

    # Plots
    import matplotlib.pyplot as plt
    # AUC distribution
    plt.figure(figsize=(6, 4))
    plt.hist(aucs, bins=10, alpha=0.8)
    plt.xlabel("AUC")
    plt.ylabel("Count")
    plt.title("AUC distribution over resamples")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "stability_auc_hist.png"), dpi=200)
    plt.close()

    # FI boxplot across runs
    fi_df = pd.DataFrame(fi_matrix, columns=feature_names)
    plt.figure(figsize=(max(8, 0.25 * len(feature_names)), 5))
    fi_df.boxplot(rot=90)
    plt.ylabel("Importance")
    plt.title("Feature importance dispersion across runs")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "stability_fi_boxplot.png"), dpi=200)
    plt.close()

    # Optional: pairwise correlation heatmap
    try:
        import seaborn as sns
        corr_mat = np.corrcoef(y_probas)
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_mat, cmap="viridis")
        plt.title("Pairwise prediction correlations")
        plt.tight_layout()
        plt.savefig(os.path.join("reports", "stability_pred_corr_heatmap.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    summary = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "brier_mean": float(np.mean(briers)),
        "brier_std": float(np.std(briers, ddof=1)),
        "pred_corr_mean": float(np.mean(corrs)) if len(corrs) else None,
        "pred_corr_std": float(np.std(corrs, ddof=1)) if len(corrs) > 1 else None,
        "pred_rmse_mean": float(np.mean(rmses)) if len(rmses) else None,
        "pred_rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else None,
        "fi_l2_mean": float(np.mean(l2_dists)) if len(l2_dists) else None,
        "fi_l2_std": float(np.std(l2_dists, ddof=1)) if len(l2_dists) > 1 else None,
        "fi_rankcorr_mean": float(np.mean(rank_corrs)) if len(rank_corrs) else None,
        "fi_rankcorr_std": float(np.std(rank_corrs, ddof=1)) if len(rank_corrs) > 1 else None,
    }
    return summary


def subsample_stability(best_name: str, best_params: Dict, X_train, y_train, X_test, y_test, feature_names: List[str]):
    # Two stratified subsamples (70% each, disjoint as much as possible)
    rng = np.random.RandomState(123)
    n = len(y_train)
    idx = np.arange(n)
    rng.shuffle(idx)
    a_idx = idx[: int(0.7 * n)]
    b_idx = idx[int(0.3 * n) :]

    def train_eval(idxs):
        model = retrain_with_same_family(best_name, best_params, X_train[idxs], y_train[idxs])
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        brier = brier_score_loss(y_test, prob)
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "coef_"):
            fi = np.abs(model.coef_.ravel())
        else:
            pi = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
            fi = pi.importances_mean
        return prob, auc, brier, fi

    prob_a, auc_a, brier_a, fi_a = train_eval(a_idx)
    prob_b, auc_b, brier_b, fi_b = train_eval(b_idx)

    from scipy.stats import spearmanr
    l2 = float(np.linalg.norm(fi_a - fi_b))
    rank_corr = float(spearmanr(fi_a, fi_b).correlation)

    # Scatter plot of FI
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(fi_a, fi_b, alpha=0.7)
    lim = [0, max(1e-9 + fi_a.max(), 1e-9 + fi_b.max())]
    plt.plot(lim, lim, "--", color="gray")
    plt.xlabel("FI - subsample A")
    plt.ylabel("FI - subsample B")
    plt.title("FI stability: subsample A vs B")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "stability_subsample_fi_scatter.png"), dpi=200)
    plt.close()

    summary = {
        "auc_a": float(auc_a),
        "brier_a": float(brier_a),
        "auc_b": float(auc_b),
        "brier_b": float(brier_b),
        "fi_l2": l2,
        "fi_rankcorr": rank_corr,
    }
    return summary


def main():
    ensure_dirs()

    model, meta, y_test_loaded, y_proba_test_loaded = load_artifacts()
    X_train, X_test, y_train, y_test = create_datasets()
    # Ensure numpy arrays for safe positional indexing during resampling
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Feature names from meta for consistent ordering
    feature_names = meta.get("feature_names")
    if isinstance(X_test, pd.DataFrame):
        # Reorder to meta order if available
        if feature_names is not None:
            X_test_use = X_test[feature_names].values
            X_train_use = X_train[feature_names].values
        else:
            feature_names = list(X_test.columns)
            X_test_use = X_test.values
            X_train_use = X_train.values
    else:
        X_test_use = X_test
        X_train_use = X_train
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_test_use.shape[1])]

    # Part A — recompute metrics
    metrics_05 = compute_metrics(y_test, y_proba_test_loaded, threshold=0.5)

    # If optimized threshold exists in meta, use it; else replicate 0.5
    thr_opt = meta.get("optimized_threshold", 0.5)
    metrics_opt = compute_metrics(y_test, y_proba_test_loaded, threshold=float(thr_opt))

    # Calibration: ECE and Brier decomposition
    ece = expected_calibration_error(y_test, y_proba_test_loaded, n_bins=10)
    rel, res, unc = brier_decomposition(y_test, y_proba_test_loaded, n_bins=10)
    plot_calibration_and_hist(y_test, y_proba_test_loaded, os.path.join("reports", "calibration_curve_step3.png"))

    # ROC and PR plots
    plot_roc_pr(y_test, y_proba_test_loaded, os.path.join("reports", "roc_step3.png"), os.path.join("reports", "prc_step3.png"))

    # Score distribution
    plot_score_distribution(y_test, y_proba_test_loaded, os.path.join("reports", "score_dist_step3.png"))

    # Save metrics
    with open(os.path.join("reports", "metrics_step3.json"), "w") as f:
        json.dump({"threshold_0_5": metrics_05, "threshold_opt": metrics_opt}, f, indent=2)
    with open(os.path.join("reports", "calibration_step3.json"), "w") as f:
        json.dump({"ece": ece, "brier_reliability": rel, "brier_resolution": res, "brier_uncertainty": unc}, f, indent=2)

    # Part B — Structural Stability
    best_name = meta.get("model_class", "")
    best_params = meta.get("best_params", {})

    resample_summary = resampling_stability(
        B=20,
        best_name=best_name,
        best_params=best_params,
        X_train=X_train_use,
        y_train=y_train,
        X_test=X_test_use,
        y_test=y_test,
        feature_names=feature_names,
    )

    subsample_summary = subsample_stability(
        best_name=best_name,
        best_params=best_params,
        X_train=X_train_use,
        y_train=y_train,
        X_test=X_test_use,
        y_test=y_test,
        feature_names=feature_names,
    )

    with open(os.path.join("reports", "stability_summary.json"), "w") as f:
        json.dump({"resampling": resample_summary, "subsample": subsample_summary}, f, indent=2)

    print(
        f"Test AUC/Brier (0.5): {metrics_05['auc']:.3f}/{metrics_05['brier']:.3f} | "
        f"ECE: {ece:.3f} | AUC over 20 runs: {resample_summary['auc_mean']:.3f}±{resample_summary['auc_std']:.3f} | "
        f"Pred-corr vs baseline: {resample_summary['pred_corr_mean']:.3f}"
    )


if __name__ == "__main__":
    main()


