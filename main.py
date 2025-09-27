import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

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


def plot_confusion_matrix(y_test, y_pred, class_names=None):
    """
    Plot confusion matrix with both counts and percentages.
    """
    pipeline, X_test, y_test = train_model()

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_sum = cm.sum()
    cm_perc = cm / cm_sum * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = "0"
            else:
                annot[i, j] = f"{c}\n({p:.1f}%)"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names if class_names is not None else np.arange(ncols),
        yticklabels=class_names if class_names is not None else np.arange(nrows),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def get_accuracy() -> None:
    """
    Compute accuracy, confusion matrix, and classification report.
    Also display them with plots.

    Returns:
        None
    """
    pipeline, X_test, y_test = train_model()

    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Print summary
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print(
        "\nConfusion Matrix:",
        plot_confusion_matrix(y_test, y_pred, class_names=["No Default", "Default"]),
    )


def plot_auc_pr() -> dict:
    """
    Compute and plot ROC AUC and Precision-Recall curve.

    Returns:
        dict: Dictionary containing 'roc_auc' and 'pr_auc'.
    """
    pipeline, X_test, y_test = train_model()

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    # --- ROC AUC ---
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # --- Precision-Recall Curve ---
    pr_auc = average_precision_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}
