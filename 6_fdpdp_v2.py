"""
Step 6 - FDPDP (First Derivative Partial Dependence Plots)
----------------------------------------------------------
Implémentation de la méthode FDPDP pour identifier les zones
où une feature a le plus d'impact sur la prédiction du modèle.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from main import create_datasets
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from utils import numerical_col, ohe, tgt_encoding


def get_trained_pipeline():
    """
    Entraîne le pipeline (préprocessing + XGBoost).
    Retourne le pipeline entraîné et le X_test associé.
    """
    X_train, X_test, y_train, y_test = create_datasets()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=2,
        random_state=42,
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

    return pipeline, X_test


def plot_fdpdp(feature: str):
    """
    Trace un FDPDP (First Derivative Partial Dependence Plot) pour une feature donnée.
    """
    pipeline, X_test = get_trained_pipeline()

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




if __name__ == "__main__":
    # Exemple d'utilisation
    pipeline, X_test = get_trained_pipeline()
    print("Features disponibles:", X_test.columns.tolist()[:20])  # aperçu

    # Exemple FDPDP simple
    plot_fdpdp("log_annual_inc")
