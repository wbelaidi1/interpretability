# surrogates.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, roc_auc_score

from category_encoders import TargetEncoder
from utils import numerical_col, ohe, tgt_encoding
from main import _create_datasets


# 1) Créer/entraîner les trois modèles surrogates (Logit, PLTR, Logit L1)
def create_surrogate_models(
    max_depth: int = 5,
    min_samples_leaf: int = 20,
    ccp_alpha: float = 0.0,
    l1_C: float = 1.0,
    use_class_weight: bool = False,
):
    """
    Construit le préprocesseur donné, split les données via _create_datasets(),
    et entraîne trois surrogates :
      - Logistic Regression (L2 par défaut)
      - Decision Tree (PLTR)
      - Logistic Regression L1 (penalisée)

    Returns:
        models (dict), preprocessor (ColumnTransformer), X_test, y_test
    """
    # splits depuis les fonctions existantes
    X_train, X_test, y_train, y_test = _create_datasets()

    # préprocesseur EXACTEMENT comme demandé
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_col),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe),
            ("tgt_enc", TargetEncoder(), tgt_encoding),
        ]
    )

    class_weight = "balanced" if use_class_weight else None

    # Logistic regression "normale"
    logit_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            random_state=42,
            class_weight=class_weight,
        ))
    ])

    # PLTR (arbre)
    to_dense = FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X,
                                   accept_sparse=True)
    pltr_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("to_dense", to_dense),
        ("clf", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=42,
            class_weight=class_weight,
        ))
    ])

    # Logistic regression L1 (penalisée)
    penalised_lr = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=l1_C,
            max_iter=2000,
            n_jobs=-1,
            class_weight=class_weight,
        ))
    ])

    # fit
    logit_pipe.fit(X_train, y_train)
    pltr_pipe.fit(X_train, y_train)
    penalised_lr.fit(X_train, y_train)

    models = {
        "logit": logit_pipe,
        "pltr": pltr_pipe,
        "logit_l1": penalised_lr,
    }
    return models, preprocessor, X_test, y_test


# 2) Évaluer (Accuracy + ROC AUC) les modèles sur X_test, y_test
def evaluate_models(models: dict, X_test, y_test) -> dict:
    """
    Calcule Accuracy et ROC AUC pour chaque modèle.
    """
    out = {}
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        out[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
    return out


# 3) Tracer les feature importances (|coef| pour logit/L1, Gini pour PLTR)
def plot_feature_importances(models: dict, preprocessor: ColumnTransformer, top: int = 20):
    """
    Trace deux/trois barplots (selon modèles fournis) des importances :
      - Logistic / Logistic L1 : |coefficients|
      - PLTR : feature_importances_ (Gini)
    """
    # helper local pour récupérer les noms de colonnes transformées
    def _get_feature_names(pre):
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

    feats = _get_feature_names(preprocessor)

    # --- Logistic Regression (L2) ---
    if "logit" in models:
        coefs = models["logit"].named_steps["clf"].coef_.ravel()
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

    # --- PLTR ---
    if "pltr" in models:
        imps = models["pltr"].named_steps["clf"].feature_importances_
        nz = imps > 0
        if nz.sum() == 0:
            print("PLTR: aucune importance non nulle (arbre trop contraint).")
        else:
            feats_pltr = np.array(feats)[nz]
            imps_pltr = imps[nz]
            k = min(top, len(imps_pltr))
            idx = np.argsort(imps_pltr)[::-1][:k]

            plt.figure(figsize=(8, 0.4 * k + 3))
            plt.barh(range(k), imps_pltr[idx])
            plt.yticks(range(k), feats_pltr[idx])
            plt.xlabel("Feature importance (Gini)")
            plt.title(f"Top {k} Feature Importances — Decision Tree (PLTR)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

    # --- Logistic Regression L1 ---
    if "logit_l1" in models:
        coefs = models["logit_l1"].named_steps["model"].coef_.ravel()
        abs_coefs = np.abs(coefs)
        k = min(top, len(abs_coefs))
        idx = np.argsort(abs_coefs)[::-1][:k]

        plt.figure(figsize=(8, 0.4 * k + 3))
        plt.barh(range(k), abs_coefs[idx])
        plt.yticks(range(k), [feats[i] for i in idx])
        plt.xlabel("|Coefficient| (L1)")
        plt.title(f"Top {k} Feature Importances — Penalised Logistic Regression (L1)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    # (optionnel) règles de l’arbre
    if "pltr" in models:
        print("\nRègles PLTR:")
        print(export_text(models["pltr"].named_steps["clf"], feature_names=feats))
