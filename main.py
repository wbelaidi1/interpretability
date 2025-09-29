import os
import shap
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
import matplotlib.patches as mpatches


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from category_encoders import TargetEncoder
from lime import lime_tabular

from utils import (
    log_col,
    grd,
    emp_len,
    grd_sub,
    grade_dico,
    emp_lenght_dico,
    sub_grade_dico,
    tgt,
    first_pred,
    numerical_col,
    ohe,
    tgt_encoding,
    best_params,
)


def _load_and_preprocess_data() -> pd.DataFrame:
    """
    Load and preprocess the data.

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Prefer raw dataset in data/, fallback to sample_data if not found
    candidate_paths = [
        "data/data.csv",
        "sample_data/sample.csv",
    ]
    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(
            f"No dataset found. Expected one of: {candidate_paths}."
        )

    df = pd.read_csv(csv_path, index_col=0)

    df = df.rename({"loan duration": "loan_duration"})

    df = df.dropna(subset=[tgt])

    for col in log_col:
        df[f"log_{col}"] = np.log1p(df[col])

    df[f"{grd}_encoded"] = df[grd].map(grade_dico)
    df[f"{emp_len}_encoded"] = df[emp_len].map(emp_lenght_dico)
    df[f"{grd_sub}_encoded"] = df[grd_sub].map(sub_grade_dico)

    df = df.drop(columns=log_col + [grd, grd_sub, emp_len])

    return df


def _create_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create training and testing datasets

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: the training and testing sets
    """
    df = _load_and_preprocess_data()
    y = df[tgt]
    X = df.drop(columns=first_pred + [tgt], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model() -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Train the model and make predictions. This model has been fine-tuned based on AUC thanks to optuna.

    Returns:
        tuple[pd.Series, pd.Series]: the predictions, its probabilities and the true labels associated
    """
    X_train, X_test, y_train, y_test = _create_datasets()

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = np.sqrt(neg / pos)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        **best_params,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_col),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe),
            ("tgt_enc", TargetEncoder(), tgt_encoding),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


def model_performance():
    """Evaluate and display model performance metrics."""
    pipeline, X_test, y_test = train_model()
    # Probabilities and hard predictions
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # --- Metrics ---
    print("AUC:", roc_auc_score(y_test, y_proba))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def plot_roc_pr_curves():
    """Plot ROC and Precision-Recall curves for train and test sets."""
    pipeline, X_test, y_test = train_model()
    X_train, X_test, y_train, y_test = _create_datasets()

    # --- Probabilities ---
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    # --- AUC scores ---
    print("Train AUC:", roc_auc_score(y_train, y_train_proba))
    print("Test AUC :", roc_auc_score(y_test, y_test_proba))

    # --- ROC curves ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(y_train, y_train_proba, name="Train", ax=ax[0])
    RocCurveDisplay.from_predictions(y_test, y_test_proba, name="Test", ax=ax[0])
    ax[0].plot([0, 1], [0, 1], "k--")  # random line
    ax[0].set_title("ROC Curve")

    # --- PR curves ---
    PrecisionRecallDisplay.from_predictions(
        y_train, y_train_proba, name="Train", ax=ax[1]
    )
    PrecisionRecallDisplay.from_predictions(y_test, y_test_proba, name="Test", ax=ax[1])
    ax[1].set_title("Precision-Recall Curve")

    plt.show()


def plot_precision_recall_vs_threshold():
    """Plot precision, recall, and F1-score against varying thresholds."""
    pipeline, X_test, y_test = train_model()
    # Probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Thresholds from 0 to 1
    thresholds = np.linspace(0, 1, 101)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1-score")
    plt.axvline(0.5, color="k", linestyle="--", label="Threshold=0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 vs Threshold")
    plt.legend()
    plt.show()


def performance_base_prediction():
    """Evaluate performance metrics for a baseline prediction."""
    df = _load_and_preprocess_data()
    # Accuracy
    acc = accuracy_score(df["target"], df["Predictions"])

    # Precision, Recall, F1
    prec = precision_score(df["target"], df["Predictions"])
    rec = recall_score(df["target"], df["Predictions"])
    f1 = f1_score(df["target"], df["Predictions"])

    # Confusion Matrix
    cm = confusion_matrix(df["target"], df["Predictions"])

    # Classification report
    report = classification_report(df["target"], df["Predictions"])

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)


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
    plt.yticks(range(R.shape[0]), [f"run {i + 1}" for i in range(R.shape[0])])
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
    plt.title(
        f"Pairwise structural distances | mean AUC = {aucs.mean():.3f} ± {aucs.std():.3f}"
    )
    plt.tight_layout()
    plt.show()


def plot_pdp(feature: str):
    """
    Trace un PDP pour une feature donnée.
    """
    pipeline, X_test, _ = train_model()

    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(pipeline, X_test, [feature], ax=ax)
    plt.title(f"PDP - {feature}")
    plt.show()


def plot_pdp_2d(feature1: str, feature2: str):
    """
    Trace un PDP 2D pour deux features.
    """
    pipeline, X_test, _ = train_model()

    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        pipeline, X_test, [(feature1, feature2)], ax=ax
    )
    plt.title(f"PDP - {feature1} vs {feature2}")
    plt.show()


def create_surrogates_from_xgb(
    max_depth: int = 5,
    min_samples_leaf: int = 20,
    ccp_alpha: float = 0.0,
    l1_C: float = 1.0,
):
    """
    Fit 3 surrogate models (Logit, PLTR, Logit L1) on the predictions of the XGB model.
    Returns:
        models (dict), preprocessor (ColumnTransformer), X_test, y_test
    """
    # split data
    X_train, X_test, y_train, y_test = _create_datasets()

    # train XGB teacher
    pipeline, _, _ = train_model()

    # predicted labels from teacher (this becomes the surrogate target)
    y_hat_train = pipeline.predict(X_train)

    # preprocessor (same as XGB pipeline)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_col),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe),
            ("tgt_enc", TargetEncoder(), tgt_encoding),
        ]
    )

    # logistic regression (L2)
    logit = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    logit.fit(X_train, y_hat_train)

    # PLTR (decision tree)
    pltr = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "to_dense",
                FunctionTransformer(
                    lambda X: X.toarray() if hasattr(X, "toarray") else X,
                    accept_sparse=True,
                ),
            ),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    ccp_alpha=ccp_alpha,
                    random_state=42,
                ),
            ),
        ]
    )
    pltr.fit(X_train, y_hat_train)

    # logistic regression L1
    logit_l1 = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=l1_C,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    logit_l1.fit(X_train, y_hat_train)

    models = {"logit": logit, "pltr": pltr, "logit_l1": logit_l1}
    return models, preprocessor, X_test, y_test


def plot_surrogate_importances(
    models: dict, preprocessor: ColumnTransformer, top: int = 20
):
    """
    Plot feature importances for surrogate models:
    - Logistic / L1 Logistic: absolute coefficients
    - PLTR: Gini importances
    """

    def _get_feature_names(pre):
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                fn = list(trans.get_feature_names_out(cols))
            else:
                fn = list(
                    cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols]
                )
            names.extend(fn)
        return names

    feats = _get_feature_names(preprocessor)

    # logistic (L2)
    if "logit" in models:
        coefs = models["logit"].named_steps["clf"].coef_.ravel()
        abs_coefs = np.abs(coefs)
        idx = np.argsort(abs_coefs)[::-1][:top]
        plt.figure(figsize=(8, 0.4 * len(idx) + 3))
        plt.barh(range(len(idx)), abs_coefs[idx])
        plt.yticks(range(len(idx)), [feats[i] for i in idx])
        plt.xlabel("|Coefficient|")
        plt.title("Top feature importances — Logistic Regression surrogate")
        plt.gca().invert_yaxis()
        plt.show()

    # PLTR
    if "pltr" in models:
        imps = models["pltr"].named_steps["clf"].feature_importances_
        idx = np.argsort(imps)[::-1][:top]
        plt.figure(figsize=(8, 0.4 * len(idx) + 3))
        plt.barh(range(len(idx)), imps[idx])
        plt.yticks(range(len(idx)), [feats[i] for i in idx])
        plt.xlabel("Feature importance (Gini)")
        plt.title("Top feature importances — PLTR surrogate")
        plt.gca().invert_yaxis()
        plt.show()

    # logistic L1
    if "logit_l1" in models:
        coefs = models["logit_l1"].named_steps["clf"].coef_.ravel()
        abs_coefs = np.abs(coefs)
        idx = np.argsort(abs_coefs)[::-1][:top]
        plt.figure(figsize=(8, 0.4 * len(idx) + 3))
        plt.barh(range(len(idx)), abs_coefs[idx])
        plt.yticks(range(len(idx)), [feats[i] for i in idx])
        plt.xlabel("|Coefficient| (L1)")
        plt.title("Top feature importances — Logistic Regression L1 surrogate")
        plt.gca().invert_yaxis()
        plt.show()


def plot_ICE(pipeline, X, feature, n_samples=50):
    values = np.linspace(X[feature].min(), X[feature].max(), 50)

    plt.figure(figsize=(8, 6))

    for i in range(min(n_samples, len(X))):
        row = X.iloc[i : i + 1].copy()
        preds = []
        for val in values:
            row[feature] = val
            pred = pipeline.predict_proba(row)[:, 1][0]
            preds.append(pred)
        plt.plot(values, preds, color="gray", alpha=0.3)

    # PDP
    avg_preds = []
    for val in values:
        X_temp = X.copy()
        X_temp[feature] = val
        avg_preds.append(pipeline.predict_proba(X_temp)[:, 1].mean())
    plt.plot(values, avg_preds, color="red", linewidth=2, label="Average effect (PDP)")

    plt.xlabel(feature)
    plt.ylabel("Predicted probability")
    plt.title(f"ICE + PDP for {feature}")
    plt.legend()
    plt.show()


def _prepare_from_lime_shape() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare datasets for LIME and SHAP explanations.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Transformed training and test datasets
    """
    X_train, X_test, y_train, y_test = _create_datasets()
    pipeline, _, _ = train_model()

    X_train, X_test, y_train, y_test = _create_datasets()
    pipeline, _, _ = train_model()

    # Preprocess X_train and X_test using the pipeline's preprocessor only
    X_train_trans = pipeline.named_steps["preprocessor"].transform(X_train)
    X_test_trans = pipeline.named_steps["preprocessor"].transform(X_test)

    # Get feature names
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Convert transformed arrays back to DataFrames for LIME
    X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names)

    xgb_model = pipeline.named_steps["classifier"]

    # Pick one instance to explain
    i = np.random.randint(0, X_test_trans_df.shape[0])
    instance = X_test_trans_df.iloc[i]

    return i, instance, xgb_model, X_train_trans_df, feature_names


def _explain_instance_with_lime(
    i: int,
    instance: pd.Series,
    xgb_model: object,
    X_train_trans_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """
    Generate and display a LIME explanation for a single instance.

    This function fits a LIME Tabular Explainer on the training data,
    explains the prediction made by the given model for the provided instance,
    and displays the explanation as a bar chart.

    Args:
        i (int):
            Index of the instance being explained. Used for titles/labels.
        instance (pd.Series):
            The feature values of the instance to explain (one observation).
        xgb_model (object):
            Trained XGBoost model or pipeline-compatible estimator.
        X_train_trans_df (pd.DataFrame):
            Training dataset after preprocessing, used to fit LIME.
        feature_names (list[str]):
            Names of the features after preprocessing.

    Returns:
        None
    """
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_trans_df.values,
        feature_names=feature_names,
        class_names=["Not Default", "Default"],
        mode="classification",
    )

    exp = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=xgb_model.predict_proba,
        num_features=5,
        top_labels=1,
    )

    # Show explanation
    exp.show_in_notebook(show_table=True, show_all=False)

    label = exp.available_labels()
    exp.as_pyplot_figure(label=label[0])

    plt.title(f"LIME Explanation for instance {i}")

    green_patch = mpatches.Patch(color="green", label="-> NON-DEFAULT")
    red_patch = mpatches.Patch(color="red", label="<- DEFAULT")
    plt.legend(handles=[green_patch, red_patch], loc="lower right")
    plt.show()


def _explain_instance_with_shap(
    i: int,
    instance: pd.Series,
    xgb_model: object,
    X_train_trans_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """
    Generate and display a SHAP explanation for a single instance.

    This function fits a SHAP Explainer on the training data,
    explains the prediction made by the given model for the provided instance,
    and displays the explanation as a bar chart.

    Args:
        i (int):
            Index of the instance being explained. Used for titles/labels.
        instance (pd.Series):
            The feature values of the instance to explain (one observation).
        xgb_model (object):
            Trained XGBoost model or pipeline-compatible estimator.
        X_train_trans_df (pd.DataFrame):
            Training dataset after preprocessing, used to fit LIME.
        feature_names (list[str]):
            Names of the features after preprocessing.

    Returns:
        None
    """
    x_instance = instance.to_frame().T

    explainer = shap.Explainer(xgb_model, X_train_trans_df, feature_names=feature_names)
    shap_values = explainer(x_instance)

    # --- Waterfall plot ---
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"SHAP Waterfall – Instance {i}")

    # Add custom legend (green = pushes toward Non-Default, red = pushes toward Default)
    green_patch = mpatches.Patch(color="blue", label="Pushes toward Non-Default")
    red_patch = mpatches.Patch(color="red", label="Pushes toward Default")
    plt.legend(handles=[green_patch, red_patch], loc="lower right")

    plt.show()

    # --- Bar plot ---
    shap.plots.bar(shap_values[0], show=False)
    plt.title(f"SHAP Local Feature Importance – Instance {i}")

    green_patch = mpatches.Patch(color="blue", label="Pushes toward Non-Default")
    red_patch = mpatches.Patch(color="red", label="Pushes toward Default")
    plt.legend(handles=[green_patch, red_patch], loc="best")

    plt.show()


def compare_shap_lime() -> None:
    """
    Compare shape and lime explanations for a single instance.

    Returns:
        None
    """

    i, instance, xgb_model, X_train_trans_df, feature_names = _prepare_from_lime_shape()

    y_pred_proba = xgb_model.predict_proba(instance.to_frame().T)[:, 1]

    print(f"Predicted probability of default for instance {i}: {y_pred_proba[0]:.4f}")

    _explain_instance_with_lime(i, instance, xgb_model, X_train_trans_df, feature_names)
    _explain_instance_with_shap(i, instance, xgb_model, X_train_trans_df, feature_names)


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


def bin_ethnicity(s, bin_size=10):
    bins = np.arange(0, 101, bin_size)
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)


def assess_fairness_ethnicity(alpha=0.05, bin_size=10):
    """
    Assess model fairness with respect to binned Pct_afro_american.
    """
    # ---- 1. Load model & test data ----
    pipeline, X_test, y_test = train_model()
    y_true = y_test.values
    yhat = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    # sensitive attribute (Pct_afro_american) aligned with test split
    df = _load_and_preprocess_data()
    _, X_test_full, _, _ = _create_datasets()
    s_test_raw = df.loc[X_test_full.index, "Pct_afro_american"]

    # bin into intervals
    s_test = bin_ethnicity(s_test_raw, bin_size=bin_size)

    # ---- 2. Metrics per group ----
    rows = []
    for group in s_test.cat.categories:
        idx = s_test == group
        if idx.sum() == 0:
            continue
        yt, yp, pr = y_true[idx], yhat[idx], proba[idx]

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)

        rows.append(
            {
                "group": str(group),
                "size": len(yt),
                "positive_rate": yp.mean(),
                "tpr": tpr,
                "fpr": fpr,
                "auc": roc_auc_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan,
            }
        )
    by_group = pd.DataFrame(rows)

    # ---- 3. Gaps vs majority group ----
    ref = by_group.loc[by_group["size"].idxmax(), "group"]
    ref_row = by_group[by_group.group == ref].iloc[0]
    gaps = []
    for _, row in by_group.iterrows():
        gaps.append(
            {
                "group": row.group,
                "ΔDemographicParity": row.positive_rate - ref_row.positive_rate,
                "ΔEqualOpportunity": row.tpr - ref_row.tpr,
                "ΔFPR": row.fpr - ref_row.fpr,
            }
        )
    gaps = pd.DataFrame(gaps)

    # ---- 4. Chi-square test ----
    ct = pd.crosstab(s_test, yhat)
    chi2, p_chi, _, _ = chi2_contingency(ct)

    # ---- 5. Résumé ----
    fair = p_chi >= alpha
    verdict = "FAIR" if fair else "NOT FAIR"

    print("\n=== Metrics by group ===")
    print(by_group)
    print("\n=== Gaps vs group:", ref, "===")
    print(gaps)
    print("\n=== Chi-square test ===")
    print(f"Chi2={chi2:.2f}, p-value={p_chi:.4f}")
    print("\n=== VERDICT ===")
    print(verdict)

    return by_group, gaps, verdict


def plot_positive_rates(by_group):
    plt.figure(figsize=(8, 5))
    sns.barplot(x="group", y="positive_rate", data=by_group, palette="Set2")
    plt.ylabel("Positive prediction rate")
    plt.xlabel("Pct_afro_american (binned %)")
    plt.title("Demographic Parity: Positive prediction rate by group")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_tpr_fpr(by_group):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(x="group", y="tpr", data=by_group, palette="Blues", ax=ax[0])
    ax[0].set_title("True Positive Rate (Recall) by group")
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("Pct_afro_american (binned %)")
    ax[0].set_ylabel("TPR")
    ax[0].tick_params(axis="x", rotation=45)

    sns.barplot(x="group", y="fpr", data=by_group, palette="Reds", ax=ax[1])
    ax[1].set_title("False Positive Rate by group")
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Pct_afro_american (binned %)")
    ax[1].set_ylabel("FPR")
    ax[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_fdpdp(feature: str = "Pct_afro_american"):
    """
    Trace un FDPDP (First Derivative Partial Dependence Plot) pour une feature donnée.
    """
    pipeline, X_test, _ = train_model()

    # 1. Calcul du PDP avec from_estimator
    disp = PartialDependenceDisplay.from_estimator(
        pipeline, X_test, [feature], grid_resolution=50
    )

    # 2. Récupérer les données de la courbe tracée
    # (dans les versions récentes c'est stocké dans disp.lines_)
    line = disp.lines_[0][0]
    x_vals = line.get_xdata()
    pdp_vals = line.get_ydata()

    # 3. Approximation de la dérivée
    fdpdp_vals = np.gradient(pdp_vals, x_vals)

    # 4. Nouveau graphe pour le FDPDP
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, fdpdp_vals, marker="o")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel(feature)
    plt.ylabel("FDPDP (pente du PDP)")
    plt.title(f"FDPDP - {feature}")
    plt.show()


def _statistical_parity_difference(y_true, y_pred, sensitive_feature) -> float:
    """
    Compute the Statistical Parity Difference (SPD) for the model predictions.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        sensitive_feature (pd.Series): Protected attribute values.

    Returns:
        float: The SPD value (max group positive rate - min group positive rate).
    """
    groups = sensitive_feature.unique()
    rates = {}
    for g in groups:
        mask = sensitive_feature == g
        rates[g] = (y_pred[mask] == 1).mean()
    return max(rates.values()) - min(rates.values())


def fairness_pdp():
    """
    Fairness-aware PDP: evaluate how fairness (SPD) changes
    when forcing the sensitive attribute to each possible group.

    Returns:
        dict: Mapping of group -> fairness score
    """
    pipeline, X_test, y_test = train_model()
    sensitive_col = "Pct_afro_american"
    groups = X_test[sensitive_col].unique()
    fairness_scores = []

    for g in groups:
        X_copy = X_test.copy()
        X_copy[sensitive_col] = g  # force everyone to group g
        y_pred = pipeline.predict(X_copy)

        # IMPORTANT: use the modified sensitive feature here
        score = _statistical_parity_difference(y_test, y_pred, X_copy[sensitive_col])
        fairness_scores.append(score)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(groups, fairness_scores, color="skyblue")
    plt.xlabel("Ethnicity group")
    plt.ylabel("Statistical Parity Difference (SPD)")
    plt.title("FPDP – Fairness across Ethnicity")
    plt.show()
