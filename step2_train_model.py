import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import calibration_curve

from main import create_datasets


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("preds", exist_ok=True)


def compute_class_weights(y: np.ndarray):
    # Return dict for sklearn class_weight and scale_pos_weight for xgb
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return None, 1.0
    scale_pos_weight = neg / max(pos, 1)
    return "balanced", scale_pos_weight


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


def try_models(X_train, y_train, feature_names):
    class_weight, scale_pos_weight = compute_class_weights(y_train)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search_spaces = []

    # RandomForest
    rf = RandomForestClassifier(random_state=42, class_weight=class_weight)
    rf_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 6, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", None],
    }
    search_spaces.append(("RandomForestClassifier", rf, rf_grid))

    # XGBoost (optional if not installed)
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
        xgb_grid = {
            "n_estimators": [300, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "min_child_weight": [1, 5],
            "reg_lambda": [0.0, 1.0, 5.0],
        }
        search_spaces.append(("XGBClassifier", xgb, xgb_grid))
    except Exception:
        pass

    # Optional Logistic baseline
    logit = LogisticRegression(max_iter=2000, class_weight=class_weight, n_jobs=None)
    logit_grid = {
        "penalty": ["l2"],
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs"],
    }
    search_spaces.append(("LogisticRegression", logit, logit_grid))

    best = {
        "model_name": None,
        "estimator": None,
        "best_params": None,
        "cv_auc_mean": -np.inf,
        "cv_auc_std": None,
    }

    for name, estimator, grid in search_spaces:
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="roc_auc",
            cv=skf,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        mean_auc = gs.best_score_
        std_auc = gs.cv_results_["std_test_score"][gs.best_index_]

        if mean_auc > best["cv_auc_mean"]:
            best.update({
                "model_name": name,
                "estimator": gs.best_estimator_,
                "best_params": gs.best_params_,
                "cv_auc_mean": float(mean_auc),
                "cv_auc_std": float(std_auc),
            })

    return best


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, path: str):
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical probability")
    plt.title("Calibration curve (reliability diagram)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ensure_dirs()

    X_train, X_test, y_train, y_test = create_datasets()

    # Guard against leakage: ensure any provided predictions columns are excluded
    # If create_datasets already dropped them, this is a no-op
    if isinstance(X_train, pd.DataFrame):
        drop_cols = [c for c in ["Predictions", "Predicted probabilities"] if c in X_train.columns]
        X_train = X_train.drop(columns=drop_cols, errors="ignore")
        X_test = X_test.drop(columns=drop_cols, errors="ignore")
        feature_names = list(X_train.columns)
        X_train = X_train.values
        X_test = X_test.values
    else:
        feature_names = [f"x{i}" for i in range(X_train.shape[1])]

    # Print class distribution
    def dist(y):
        vals, counts = np.unique(y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}
    print(f"Class distribution - train: {dist(y_train)} | test: {dist(y_test)}")

    # Try models via CV AUC
    best = try_models(X_train, y_train, feature_names)
    print(f"Best model: {best['model_name']}")
    print(f"Best params: {best['best_params']}")
    print(f"CV AUC: {best['cv_auc_mean']:.4f} ± {best['cv_auc_std']:.4f}")

    # Final fit and test evaluation
    best_est = best["estimator"]
    best_est.fit(X_train, y_train)
    y_proba_test = best_est.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_proba_test)
    brier = brier_score_loss(y_test, y_proba_test)
    acc = accuracy_score(y_test, y_pred_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="binary")
    cm = confusion_matrix(y_test, y_pred_test)

    ece = expected_calibration_error(y_test, y_proba_test, n_bins=10)
    calib_path = os.path.join("reports", "calibration_curve_step2.png")
    plot_calibration(y_test, y_proba_test, calib_path)

    # Save predictions
    np.save(os.path.join("preds", "y_test.npy"), y_test)
    np.save(os.path.join("preds", "y_proba_test.npy"), y_proba_test)

    # Save model and metadata
    joblib.dump(best_est, os.path.join("models", "best_model.pkl"))
    meta = {
        "model_class": best["model_name"],
        "best_params": best["best_params"],
        "cv_auc_mean": best["cv_auc_mean"],
        "cv_auc_std": best["cv_auc_std"],
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_names": feature_names,
        "preprocess_version": "main.load_and_preprocess_data@current",
    }
    with open(os.path.join("models", "best_model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Feature importance if available
    fi_path = os.path.join("reports", "fi_test.csv")
    try:
        importances = best_est.feature_importances_
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df.sort_values("importance", ascending=False).to_csv(fi_path, index=False)
    except Exception:
        pass

    # Save metrics
    metrics = {
        "test_auc": float(test_auc),
        "brier": float(brier),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "ece": float(ece),
        "confusion_matrix": cm.tolist(),
        "threshold": 0.5,
    }
    with open(os.path.join("reports", "metrics_step2.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Test AUC: {test_auc:.4f} | Brier: {brier:.4f} | Acc: {acc:.4f} | "
        f"Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ECE: {ece:.4f}"
    )
    print("Artifacts saved: models/, reports/, preds/")


if __name__ == "__main__":
    main()


