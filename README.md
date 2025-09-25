# Interpretability Project

## Group Members
- Adrien Senghor
- Grégoire Bidault
- Lea Tinelli
- Valentin Henry-Léo
- Victor Soto
- William Belaidi

## Project Structure
- `main.py`: Data loading and preprocessing functions, and train/test split helper
- `utils.py`: Column names and encoding dictionaries used during preprocessing
- `data/`: The dataset
- `sample_data/`: Example CSV for reference

## Setup
We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

## Data
- Expected file: `data/data.csv` (first column is an index).
- The script will create encoded/log-transformed features and drop originals accordingly.

## Usage
Run the preprocessing and obtain train/test splits:
```bash
python main.py
```

## Notes
- The code uses `LabelEncoder` for `emp_title`, `purpose`, and `home_ownership`.
- Grade mappings are defined in `utils.py`. Adjust as needed for your data.

## Reminder

#### Technical Steps

#### 1. Interpret provided DP (Default Probability):
- Use the estimated default probability (DP) from dataset.
- Implement 1–2 surrogate models to interpret the unknown model generating DP.

#### 2. Build your own black-box ML model to forecast default.
- Each group must develop their own (no collaboration across groups).

#### 3. Evaluate forecasting performance & structural stability of your model.

#### 4. Global interpretability (part 1):
- Implement 1–2 surrogate models for your own model.
- Compare with Step 1.

#### 5. Global interpretability (part 2):
- Implement Partial Dependence Plots (PDP) for your own model.
- Compare with Step 4.

#### 6. Local interpretability (part 1):
- Implement LIME and/or ICE on your model.

#### 7. Local interpretability (part 2):
- Implement SHAP on your model.
- Compare results with Step 6.

#### 8. Performance interpretability:
- Implement Permutation Importance.
- Check if drivers of predictive performance (Step 8) align with SHAP drivers (Step 7).

#### 9. Fairness analysis:
- Assess fairness of your model w.r.t. ethnicity of borrower (protected attribute).
- Discuss findings.

#### 10. Fairness interpretability:
- Implement a Fairness Partial Dependence Plot (FPDP) using a fairness measure.
- Discuss findings.