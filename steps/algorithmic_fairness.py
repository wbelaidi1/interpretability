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

from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

from main import _load_and_preprocess_data, _create_datasets, train_model

def bin_ethnicity(s, bin_size=10):
    bins = np.arange(0, 101, bin_size)
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
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
        idx = (s_test == group)
        if idx.sum() == 0:
            continue
        yt, yp, pr = y_true[idx], yhat[idx], proba[idx]

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)

        rows.append({
            "group": str(group),
            "size": len(yt),
            "positive_rate": yp.mean(),
            "tpr": tpr,
            "fpr": fpr,
            "auc": roc_auc_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan
        })
    by_group = pd.DataFrame(rows)

    # ---- 3. Gaps vs majority group ----
    ref = by_group.loc[by_group["size"].idxmax(), "group"]
    ref_row = by_group[by_group.group == ref].iloc[0]
    gaps = []
    for _, row in by_group.iterrows():
        gaps.append({
            "group": row.group,
            "ΔDemographicParity": row.positive_rate - ref_row.positive_rate,
            "ΔEqualOpportunity": row.tpr - ref_row.tpr,
            "ΔFPR": row.fpr - ref_row.fpr,
        })
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
    plt.figure(figsize=(8,5))
    sns.barplot(x="group", y="positive_rate", data=by_group, palette="Set2")
    plt.ylabel("Positive prediction rate")
    plt.xlabel("Pct_afro_american (binned %)")
    plt.title("Demographic Parity: Positive prediction rate by group")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_tpr_fpr(by_group):
    fig, ax = plt.subplots(1, 2, figsize=(14,5))

    sns.barplot(x="group", y="tpr", data=by_group, palette="Blues", ax=ax[0])
    ax[0].set_title("True Positive Rate (Recall) by group")
    ax[0].set_ylim(0,1)
    ax[0].set_xlabel("Pct_afro_american (binned %)")
    ax[0].set_ylabel("TPR")
    ax[0].tick_params(axis='x', rotation=45)

    sns.barplot(x="group", y="fpr", data=by_group, palette="Reds", ax=ax[1])
    ax[1].set_title("False Positive Rate by group")
    ax[1].set_ylim(0,1)
    ax[1].set_xlabel("Pct_afro_american (binned %)")
    ax[1].set_ylabel("FPR")
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


