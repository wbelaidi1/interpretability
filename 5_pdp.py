"""
Step 5 - PDP (Partial Dependence Plots)
---------------------------------------
Implémentation de  la méthode PDP pour interpréter notre modèle.
Comparaison avec les surrogate models (Step 4).
"""

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
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


def plot_pdp(feature: str):
    """
    Trace un PDP pour une feature donnée.
    """
    pipeline, X_test = get_trained_pipeline()

    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        pipeline, X_test, [feature], ax=ax
    )
    plt.title(f"PDP - {feature}")
    plt.show()


def plot_pdp_2d(feature1: str, feature2: str):
    """
    Trace un PDP 2D pour deux features.
    """
    pipeline, X_test = get_trained_pipeline()

    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        pipeline, X_test, [(feature1, feature2)], ax=ax
    )
    plt.title(f"PDP - {feature1} vs {feature2}")
    plt.show()


if __name__ == "__main__":
    # Exemple d'utilisation : adapter les noms selon X_test.columns
    pipeline, X_test = get_trained_pipeline()
    print("Features disponibles:", X_test.columns.tolist()[:20])  # aperçu

    # Exemple PDP simple
    plot_pdp("int_rate")

    # Exemple PDP 2D
    plot_pdp_2d("int_rate", "log_annual_inc")  # 
