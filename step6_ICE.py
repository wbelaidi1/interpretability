from main import train_model, _create_datasets
import numpy as np
import matplotlib.pyplot as plt

pipeline, X_test, y_test = train_model()
X_train, X_test, y_train, y_test = _create_datasets()

def plot_ICE(pipeline, X, feature, n_samples=50):
    values = np.linspace(X[feature].min(), X[feature].max(), 50)
    
    plt.figure(figsize=(8,6))
    
    for i in range(min(n_samples, len(X))):
        row = X.iloc[i:i+1].copy()  # copy a single row
        preds = []
        for val in values:
            row[feature] = val
            pred = pipeline.predict_proba(row)[:,1][0]
            preds.append(pred)
        plt.plot(values, preds, color='gray', alpha=0.3)
    
    # PDP
    avg_preds = []
    for val in values:
        X_temp = X.copy()
        X_temp[feature] = val
        avg_preds.append(pipeline.predict_proba(X_temp)[:,1].mean())
    plt.plot(values, avg_preds, color='red', linewidth=2, label='Average effect (PDP)')
    
    plt.xlabel(feature)
    plt.ylabel("Predicted probability")
    plt.title(f"ICE + PDP for {feature}")
    plt.legend()
    plt.show()

