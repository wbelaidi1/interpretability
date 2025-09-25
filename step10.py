import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from lime import lime_tabular

from main import train_model, create_datasets


def explain_instance_with_lime() -> None:
    """
    Explain a single instance using LIME, displaying the explanation plot.

    Returns:
        None
    """
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

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_trans_df.values,
        feature_names=feature_names,
        class_names=["Not Default", "Default"],
        mode="classification",
    )
    # Pick one instance to explain
    i = np.random.randint(0, X_test_trans_df.shape[0])
    instance = X_test_trans_df.iloc[i]

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

    plt.title(f"LIME Explanation for instance {i}, class {label[0]}")

    green_patch = mpatches.Patch(color="green", label="-> NON-DEFAULT")
    red_patch = mpatches.Patch(color="red", label="<- DEFAULT")
    plt.legend(handles=[green_patch, red_patch], loc="lower right")
    plt.show()
