from main import train_model, _create_datasets, _load_and_preprocess_data

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# Plotting
import matplotlib.pyplot as plt
import numpy as np

pipeline, X_test, y_test = train_model()
X_train, X_test, y_train, y_test = _create_datasets()
df = _load_and_preprocess_data()

def model_performance(pipeline, X_test, y_test):
    """Evaluate and display model performance metrics."""
    # Probabilities and hard predictions
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = pipeline.predict(X_test)

    # --- Metrics ---
    print("AUC:", roc_auc_score(y_test, y_proba))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def plot_roc_pr_curves(pipeline, X_train, y_train, X_test, y_test):
    """Plot ROC and Precision-Recall curves for train and test sets.""" 
    # --- Probabilities ---
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_test_proba  = pipeline.predict_proba(X_test)[:, 1]

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
    PrecisionRecallDisplay.from_predictions(y_train, y_train_proba, name="Train", ax=ax[1])
    PrecisionRecallDisplay.from_predictions(y_test, y_test_proba, name="Test", ax=ax[1])
    ax[1].set_title("Precision-Recall Curve")

    plt.show()

def plot_precision_recall_vs_threshold(pipeline, X_test, y_test):
    """Plot precision, recall, and F1-score against varying thresholds."""
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
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1-score")
    plt.axvline(0.5, color="k", linestyle="--", label="Threshold=0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 vs Threshold")
    plt.legend()
    plt.show()

def performance_base_prediction(df):
    """Evaluate performance metrics for a baseline prediction."""
    # Accuracy
    acc = accuracy_score(df['target'], df['Predictions'])

    # Precision, Recall, F1
    prec = precision_score(df['target'], df['Predictions'])
    rec = recall_score(df['target'], df['Predictions'])
    f1 = f1_score(df['target'], df['Predictions'])

    # Confusion Matrix
    cm = confusion_matrix(df['target'], df['Predictions'])

    # Classification report
    report = classification_report(df['target'], df['Predictions'])

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)