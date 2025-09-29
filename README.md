## Interpretability, Stability and Algorithmic Fairness — Project

### Group Members
- Adrien Senghor
- Grégoire Bidault
- Lea Tinelli
- Valentin Henry-Léo
- Victor Soto
- William Belaidi

### Project Structure
- `main.py`: End‑to‑end pipeline utilities (data prep, training, plots, interpretability, fairness).
- `utils.py`: Column names, encoders, and constants used by the pipeline.
- `steps/`: Earlier stepwise scripts (surrogate models, PDP/ICE, permutation importance, stability, fairness). Functionality is centralized in `main.py`.
- `data/`: Put your main dataset here.
- `sample_data/`: Tiny sample for quick trials.
- `reports/`: Optional place for saving figures if you choose to do so.

### Setup
We recommend a virtual environment.

```bash
pip install -r requirements.txt
```

### Data
  - `data/dataproject2025.csv` (dataset)

The pipeline automatically creates log‑transformed and encoded features, then drops the originals accordingly.

### How to Use (import from main)
Open a Python session or a notebook and import the functions you need from `main.py`.


### 1) Train and get the fitted pipeline + test split
pipeline, X_test, y_test = train_model()

### 2) Basic performance
model_performance()
plot_roc_pr_curves()
plot_precision_recall_vs_threshold()

### 3) Global interpretability
plot_pdp("some_feature_name")
plot_pdp_2d("feature_a", "feature_b")
plot_feature_importance(nb_features=10)

### 4) Local interpretability (LIME vs SHAP for one instance)
compare_shap_lime()

### 5) Structural stability of feature importances
result = assess_structural_stability_xgb(n_runs=8, test_size=0.2)
plot_structural_stability(result, top=20)

### 6) Fairness analysis by binned ethnicity
by_group, gaps, verdict = assess_fairness_ethnicity(alpha=0.05, bin_size=10)
plot_positive_rates(by_group)
plot_tpr_fpr(by_group)
fairness_pdp()  # fairness‑aware PDP (SPD across groups)
```

Notes:
- Figures are shown via `matplotlib`. If you prefer files, call `plt.savefig("reports/your_figure.png")` right before `plt.show()` or adapt the functions.
- The sensitive attribute used in fairness utilities is `Pct_afro_american` (binned for groupwise metrics).