import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from lime import lime_tabular

from main import train_model, create_datasets


def _prepare_from_lime_shape() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare datasets for LIME and SHAP explanations.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Transformed training and test datasets
    """
    X_train, X_test, y_train, y_test = create_datasets()
    pipeline, _, _ = train_model()

    X_train, X_test, y_train, y_test = create_datasets()
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
