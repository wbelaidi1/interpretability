import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
import xgboost as xgb

from main import train_model, _load_and_preprocess_data
from utils import tgt, first_pred, numerical_col, ohe, tgt_encoding


def assess_structural_stability_xgb(
    n_runs: int = 8,
    test_size: float = 0.2,
    seeds: list[int] | None = None,
    top_k: int = 15,
):
    """
    Structural stability (same population -> approximately the same model):
    - Use train_model() ONCE to get the *optimized* pipeline and params.
    - Re-sample the SAME dataset multiple times (different seeds).
    - Refit an IDENTICAL pipeline (same preprocessor + same XGB params) for each run.
    - Compare XGB feature_importances_ across runs:
        * Pairwise L2 distances between normalized importance vectors
        * Rank stability (Spearman) on top-k features
    - Track test ROC AUC per run (sanity check).

    Returns:
        dict with feature names, per-run importances (normalized), distance matrix,
        per-feature mean/std, rank_stability summary, test_scores, seeds.
    """
    # 1) Get your optimized pipeline (architecture + tuned params) once
    base_pipeline, _, _ = train_model()
    base_pre = base_pipeline.named_steps["preprocessor"]
    base_xgb = base_pipeline.named_steps["classifier"]
    # extract the EXACT params learned/used by train_model (optuna-tuned)
    xgb_params = {k: v for k, v in base_xgb.get_params(deep=False).items()}

    # 2) Load the same processed table your code uses
    df = _load_and_preprocess_data()
    y = df[tgt].values
    X = df.drop(columns=first_pred + [tgt], axis=1)

    # 3) Rebuild the IDENTICAL preprocessor (same as in train_model)
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_col),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe),
            ("tgt_enc", TargetEncoder(), tgt_encoding),
        ]
    )

    if seeds is None:
        seeds = list(range(123, 123 + n_runs))

    importances = []
    aucs = []
    feature_names = None

    for s in seeds:
        # stratified re-sample (same population, different split)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=s, stratify=y
        )

        model = xgb.XGBClassifier(**xgb_params)

        pipe = Pipeline([("preprocessor", pre), ("classifier", model)])
        pipe.fit(X_tr, y_tr)

        # Feature names after fit 
        if feature_names is None:
            feature_names = _get_feature_names(pipe.named_steps["preprocessor"])

        # XGB importances (structure), L2-normalized
        fi = pipe.named_steps["classifier"].feature_importances_.astype(float)
        fi = fi / (np.linalg.norm(fi) + 1e-12)
        importances.append(fi)

        # Performance sanity check
        y_proba = pipe.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, y_proba))

    importances = np.vstack(importances)  # (n_runs, p)
    aucs = np.asarray(aucs)

    # Pairwise L2 distances between runs
    n = importances.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = np.linalg.norm(importances[i] - importances[j])

    # Rank stability on top-k features (by pooled mean importance)
    pooled_mean = importances.mean(axis=0)
    order = np.argsort(pooled_mean)[::-1]
    top_idx = order[: min(top_k, len(order))]

    rhos = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(importances[i, top_idx], importances[j, top_idx])
            if np.isfinite(rho):
                rhos.append(rho)

    rank_stab = {
        "mean_spearman_topk": float(np.mean(rhos)) if rhos else float("nan"),
        "std_spearman_topk": float(np.std(rhos)) if rhos else float("nan"),
        "top_k": int(min(top_k, len(order))),
    }

    return {
        "feature_names": feature_names,
        "importances_runs": importances,
        "pairwise_distances": D,
        "per_feature_mean": importances.mean(axis=0),
        "per_feature_std": importances.std(axis=0),
        "rank_stability": rank_stab,
        "test_scores": aucs,
        "seeds": seeds,
    }


def _get_feature_names(pre: ColumnTransformer):
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            fn = list(trans.get_feature_names_out(cols))
        else:
            fn = list(cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols])
        names.extend(fn)
    return names


def plot_structural_stability(result: dict, top: int = 20):
    """
    Visual summary:
      A) Heatmap of normalized importances across runs (runs x features)
      B) Bar chart: top-K most variable features (std across runs)
      C) Histogram: pairwise L2 distances (smaller is more stable), with AUC overlay
    """
    feats = np.array(result["feature_names"])
    R = result["importances_runs"]
    stds = result["per_feature_std"]
    D = result["pairwise_distances"]
    aucs = result["test_scores"]

    # A) heatmap
    plt.figure(figsize=(10, max(4, 0.25 * R.shape[1])))
    plt.imshow(R, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Normalized feature importance")
    plt.yticks(range(R.shape[0]), [f"run {i+1}" for i in range(R.shape[0])])
    plt.xticks(range(len(feats)), feats, rotation=90)
    plt.title("Structural stability — importances across runs")
    plt.tight_layout()
    plt.show()

    # B) most unstable features
    k = min(top, len(feats))
    order = np.argsort(stds)[::-1][:k]
    plt.figure(figsize=(9, 0.45 * k + 2))
    plt.barh(range(k), stds[order])
    plt.yticks(range(k), feats[order])
    plt.gca().invert_yaxis()
    plt.xlabel("Std. dev. of normalized importance across runs")
    plt.title(f"Top {k} most unstable features")
    plt.tight_layout()
    plt.show()

    # C) distances + performance
    triu = D[np.triu_indices_from(D, k=1)]
    plt.figure(figsize=(7, 4))
    plt.hist(triu, bins=12, alpha=0.9)
    plt.xlabel("‖ρ(run i) − ρ(run j)‖₂")
    plt.ylabel("Count")
    plt.title(f"Pairwise structural distances | mean AUC = {aucs.mean():.3f} ± {aucs.std():.3f}")
    plt.tight_layout()
    plt.show()
