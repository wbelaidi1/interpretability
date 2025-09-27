# surrogate_models.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

# ---- on réutilise vos colonnes/encodages définis dans utils.py ----
from category_encoders import TargetEncoder
from utils import numerical_col, ohe, tgt_encoding

# ---- on réutilise vos datasets déjà créés par main.py ----
from main import _create_datasets


def make_preprocessor() -> ColumnTransformer:
    """Construit le même préprocesseur (num + OHE + TargetEncoder) que dans ton pipeline XGB."""
    try:
        ohe_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)  # sklearn >= 1.2
    except TypeError:
        ohe_enc = OneHotEncoder(handle_unknown="ignore", sparse=True)         # sklearn < 1.2

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_col),
            ("ohe", ohe_enc, ohe),
            ("tgt_enc", TargetEncoder(), tgt_encoding),
        ],
        remainder="drop",
    )
    return pre


def get_feature_names(pre: ColumnTransformer):
    """Récupère les noms de features après préprocesseur (num + OHE + TE)."""
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            fn = list(trans.get_feature_names_out(cols))
        else:
            # passthrough ou encodeur sans méthode
            if isinstance(cols, (list, tuple, np.ndarray)):
                fn = list(cols)
            else:
                fn = [cols]
        names.extend(fn)
    return names


def build_models(pre: ColumnTransformer):
    """Construit les deux surrogates: Logistic Regression et PLTR (arbre)."""
    # Régression logistique 'normale' (L2 par défaut), pondération balanced vue le déséquilibre
    logit = Pipeline(steps=[
        ("preprocessor", pre),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Petit arbre interprétable (PLTR)
    def _to_dense(X):
        return X.toarray() if hasattr(X, "toarray") else X

    pltr = Pipeline(steps=[
        ("preprocessor", pre),
        ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
        ("clf", DecisionTreeClassifier(
            max_depth=5,            # un peu plus permissif pour éviter AUC~0.5
            min_samples_leaf=20,
            ccp_alpha=0.0,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # (Optionnel) Logit pénalisée L1 pour parcimonie
    penalised_lr = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    return logit, pltr, penalised_lr


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    """Accuracy, ROC AUC, PR AUC sur le test."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }


def plot_importances_logit(logit_pipe: Pipeline, top: int = 20):
    """Barplot des |coefficients| (importance) pour la logit."""
    feats = get_feature_names(logit_pipe.named_steps["preprocessor"])
    coefs = logit_pipe.named_steps["clf"].coef_.ravel()
    abs_coefs = np.abs(coefs)
    k = min(top, len(abs_coefs))
    idx = np.argsort(abs_coefs)[::-1][:k]

    plt.figure(figsize=(8, 0.4 * k + 3))
    plt.barh(range(k), abs_coefs[idx])
    plt.yticks(range(k), [feats[i] for i in idx])
    plt.xlabel("|Coefficient|")
    plt.title(f"Top {k} Feature Importances — Logistic Regression")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_importances_pltr(pltr_pipe: Pipeline, top: int = 20):
    """Barplot des importances Gini pour l’arbre (PLTR)."""
    feats = get_feature_names(pltr_pipe.named_steps["preprocessor"])
    imps = pltr_pipe.named_steps["clf"].feature_importances_
    if (imps > 0).sum() == 0:
        print("Aucune importance non nulle (arbre trop contraint).")
        return
    k = min(top, (imps > 0).sum())
    idx = np.argsort(imps)[::-1][:k]

    plt.figure(figsize=(8, 0.4 * k + 3))
    plt.barh(range(k), imps[idx])
    plt.yticks(range(k), [feats[i] for i in idx])
    plt.xlabel("Feature importance (Gini)")
    plt.title(f"Top {k} Feature Importances — Decision Tree (PLTR)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def run():
    """
    Point d’entrée : crée les datasets via main._create_datasets,
    construit les surrogates, les entraîne, les évalue et trace les importances.
    Fonctions imbriquées pour garder un flux clair.
    """
    X_train, X_test, y_train, y_test = _create_datasets()
    pre = make_preprocessor()
    logit, pltr, penalised_lr = build_models(pre)

    # --- fit ---
    logit.fit(X_train, y_train)
    pltr.fit(X_train, y_train)
    penalised_lr.fit(X_train, y_train)

    # --- eval imbriquée ---
    def show_metrics(name: str, model: Pipeline):
        m = evaluate(model, X_test, y_test)
        print(f"{name:<14} | Acc: {m['accuracy']:.3f} | ROC AUC: {m['roc_auc']:.3f} | PR AUC: {m['pr_auc']:.3f}")

    print("\n=== Surrogates (balanced) ===")
    show_metrics("Logit", logit)
    show_metrics("PLTR", pltr)
    show_metrics("Logit L1", penalised_lr)

    # --- plots imbriqués ---
    def plots():
        plot_importances_logit(logit, top=20)
        plot_importances_pltr(pltr, top=20)
        # Règles de l'arbre (utile pour l'explicabilité glob.)
        feats = get_feature_names(pltr.named_steps["preprocessor"])
        print("\nRègles PLTR:")
        print(export_text(pltr.named_steps["clf"], feature_names=feats))

    plots()


if __name__ == "__main__":
    run()
