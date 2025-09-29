import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from main import train_model, _create_datasets, _load_and_preprocess_data
from utils import (
    log_col, grd, emp_len, grd_sub, grade_dico, emp_lenght_dico, sub_grade_dico,
    tgt, first_pred, numerical_col, ohe, tgt_encoding
)

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def build_preprocessor():
    """Return a ColumnTransformer with scaling, OHE, and Target Encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_col),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe),
            ("tgt_enc", TargetEncoder(), tgt_encoding),
        ]
    )


def fit_and_analyze_model(X, y, preprocessor, model, title):
    """
    Fit a logistic model, plot top 10 coefficients, and return coefficients DataFrame.
    """
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(X, y)

    # Extract coefficients
    coefficients = model.coef_[0]
    ohe_features = list(preprocessor.named_transformers_["ohe"].get_feature_names_out(ohe))
    feature_names = numerical_col + ohe_features + tgt_encoding

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    }).sort_values(by="coefficient", key=abs, ascending=False)

    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    top_coef_df = coef_df.nlargest(10, "abs_coef").sort_values(by="abs_coef", ascending=False)

    # Plot
    norm = plt.Normalize(top_coef_df["abs_coef"].min(), top_coef_df["abs_coef"].max())
    colors = plt.cm.viridis(norm(top_coef_df["abs_coef"]))

    plt.figure(figsize=(8, 6))
    sns.barplot(x="coefficient", y="feature", data=top_coef_df, palette=colors)
    plt.title(f"Top 10 coefficients - {title}")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return pipeline, coef_df


def evaluate_model(pipeline, X, y_true, y_pred_ref, name="Model"):
    """Evaluate accuracy against a reference prediction."""
    y_pred_train = pipeline.predict(X)
    acc = accuracy_score(y_true, y_pred_train)
    print(f"{name} Accuracy: {acc:.4f}")
    return acc


def plot_top10_coefficient_evolution(coef_df):
    """
    Plot evolution of top 10 coefficients for logistic regression vs PLTR.
    """
    coef_df['max_abs_coef'] = coef_df[['coef_logistic_regression_1',
                                       'coef_logistic_regression_2',
                                       'coef_PLTR_1',
                                       'coef_PLTR_2']].abs().max(axis=1)
    top10_df = coef_df.nlargest(10, 'max_abs_coef')

    # Logistic Regression
    coef_log = top10_df[['feature', 'coef_logistic_regression_1', 'coef_logistic_regression_2']]
    coef_log_melt = coef_log.melt(id_vars='feature',
                                  value_vars=['coef_logistic_regression_1', 'coef_logistic_regression_2'],
                                  var_name='Step', value_name='Coefficient')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='feature', hue='Step', data=coef_log_melt, palette='Blues')
    plt.title("Evolution of Top 10 Logistic Regression Coefficients")
    plt.tight_layout()
    plt.show()

    # PLTR
    coef_pltr = top10_df[['feature', 'coef_PLTR_1', 'coef_PLTR_2']]
    coef_pltr_melt = coef_pltr.melt(id_vars='feature',
                                    value_vars=['coef_PLTR_1', 'coef_PLTR_2'],
                                    var_name='Step', value_name='Coefficient')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='feature', hue='Step', data=coef_pltr_melt, palette='Greens')
    plt.title("Evolution of Top 10 PLTR Coefficients")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------

def plot_surrogate_models():
    # Load pipeline and data
    pipeline, X_test, y_test = train_model()
    X_train, X_test, y_train, y_test = _create_datasets()
    df = _load_and_preprocess_data()

    # Base data
    X = df.drop(columns=first_pred + [tgt])
    y_pred_ref = pipeline.predict(X)

    preprocessor = build_preprocessor()

    # Step 1: Logistic Regression (no penalty)
    model_1 = LogisticRegression(max_iter=200)
    pipeline_1, coef_df_1 = fit_and_analyze_model(X, y_pred_ref, preprocessor, model_1, "Logistic Regression")
    evaluate_model(pipeline_1, X, y_pred_ref, y_pred_ref, "Logistic Regression")

    # Step 2: Logistic Regression with L1 penalty (PLTR)
    model_2 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000)
    pipeline_2, coef_df_2 = fit_and_analyze_model(X, y_pred_ref, preprocessor, model_2, "PLTR (L1 Regularization)")
    evaluate_model(pipeline_2, X, y_pred_ref, y_pred_ref, "PLTR")

    # Repeat with y_pred = df["Predictions"]
    y_pred = df["Predictions"]

    model_1_step1 = LogisticRegression(max_iter=200)
    pipeline_1_step1, coef_df_1_step1 = fit_and_analyze_model(X, y_pred, preprocessor, model_1_step1, "Logistic Regression (step1)")
    evaluate_model(pipeline_1_step1, X, y_pred, y_pred, "Logistic Regression Step1")

    model_2_step1 = LogisticRegression(penalty="l2", solver="liblinear", C=0.1, max_iter=1000)
    pipeline_2_step1, coef_df_2_step1 = fit_and_analyze_model(X, y_pred, preprocessor, model_2_step1, "PLTR (L2 Regularization)")
    evaluate_model(pipeline_2_step1, X, y_pred, y_pred, "PLTR Step1")

    # Merge coefficients
    coef_df_combined = pd.DataFrame({
        "feature": coef_df_1_step1["feature"],
        "coef_logistic_regression_1": coef_df_1_step1["coefficient"],
        "coef_PLTR_1": coef_df_2_step1["coefficient"],
        "coef_logistic_regression_2": coef_df_1["coefficient"],
        "coef_PLTR_2": coef_df_2["coefficient"],
    })

    # Plot evolution
    plot_top10_coefficient_evolution(coef_df_combined)