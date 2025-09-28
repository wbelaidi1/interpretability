import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from category_encoders import TargetEncoder

from utils import numerical_col, ohe, tgt_encoding
from main import _create_datasets, train_model


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
    logit = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ])
    logit.fit(X_train, y_hat_train)

    # PLTR (decision tree)
    pltr = Pipeline([
        ("preprocessor", preprocessor),
        ("to_dense", FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X,
                                         accept_sparse=True)),
        ("clf", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=42
        ))
    ])
    pltr.fit(X_train, y_hat_train)

    # logistic regression L1
    logit_l1 = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear", C=l1_C,
            max_iter=2000, random_state=42
        ))
    ])
    logit_l1.fit(X_train, y_hat_train)

    models = {"logit": logit, "pltr": pltr, "logit_l1": logit_l1}
    return models, preprocessor, X_test, y_test

def plot_surrogate_importances(models: dict, preprocessor: ColumnTransformer, top: int = 20):
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
                fn = list(cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols])
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
