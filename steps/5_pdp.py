"""
Step 5 - PDP (Partial Dependence Plots)
---------------------------------------
Implémente la méthode PDP pour interpréter notre modèle.
Comparaison avec les surrogate models (Step 4) prévue dans le notebook/slides.
"""

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from main import train_model


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


if __name__ == "__main__":
    # Exemple d'utilisation : adapter les noms selon X_test.columns
    pipeline, X_test, _ = train_model()
    print("Features disponibles:", X_test.columns.tolist()[:20])  # aperçu

    # Exemple PDP simple
    plot_pdp("int_rate")

    # Exemple PDP 2D
    plot_pdp_2d("int_rate", "log_annual_inc")  #
