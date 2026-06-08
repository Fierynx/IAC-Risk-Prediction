import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_counterfactuals(model, scaler, df, feature_names, top_n=10):
    logging.info("Generating counterfactuals for high-risk instances...")
    
    # We'll pick the most confident defect predictions
    X_raw = df[feature_names].values
    X_scaled = scaler.transform(X_raw)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # Find top instances predicted as defects
    df_pred = df.copy()
    df_pred['risk_prob'] = probs
    high_risk = df_pred.sort_values('risk_prob', ascending=False).head(top_n)
    
    counterfactuals = []
    
    for idx, row in high_risk.iterrows():
        base_prob = row['risk_prob']
        x_base = row[feature_names].values.reshape(1, -1).copy()
        
        # We try to perturb top features to drop risk below 0.5
        for i, feat in enumerate(feature_names):
            x_pert = x_base.copy()
            # If feature > 0, try reducing it by 50%
            if x_pert[0, i] > 0:
                x_pert[0, i] = x_pert[0, i] * 0.5
            else:
                x_pert[0, i] = 1 # try increasing it
                
            prob_pert = model.predict_proba(scaler.transform(x_pert))[0, 1]
            
            if prob_pert < base_prob - 0.1: # Significant drop
                counterfactuals.append({
                    'repo': row['repo'],
                    'filepath': row['filepath'],
                    'base_risk': base_prob,
                    'new_risk': prob_pert,
                    'feature_changed': feat,
                    'original_value': x_base[0, i],
                    'new_value': x_pert[0, i]
                })
                break # Just record one strong CF per file
                
    cf_df = pd.DataFrame(counterfactuals)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    cf_df.to_csv("results/tables/counterfactuals.csv", index=False)
    logging.info("Saved counterfactuals.csv")


def main():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    logging.info("Loading model and data for Explainability...")
    model = joblib.load("models/rf_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    
    df = pd.read_csv("data/processed/features.csv")
    df.fillna(0, inplace=True)
    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    
    # Use TreeExplainer for Random Forest
    logging.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    # SHAP takes a while, so compute on a sample of 1000
    sample_idx = np.random.choice(X_scaled.shape[0], min(1000, X_scaled.shape[0]), replace=False)
    X_sample = X_scaled[sample_idx]
    
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP returns a list of arrays for classification (one per class). We want class 1 (defect)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        # In newer shap versions it might just return an array of shape (N, features, classes)
        if len(shap_values.shape) == 3:
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values

    # 1. Global Bar Plot
    logging.info("Generating SHAP Global Bar plot...")
    plt.figure()
    shap.summary_plot(shap_values_class1, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig("results/figures/shap_global_bar.png", bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Beeswarm Plot
    logging.info("Generating SHAP Beeswarm plot...")
    plt.figure()
    shap.summary_plot(shap_values_class1, X_sample, feature_names=feature_names, show=False)
    plt.savefig("results/figures/shap_beeswarm.png", bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Counterfactuals
    generate_counterfactuals(model, scaler, df, feature_names)

    logging.info("Explainability phase complete. All plots saved to results/figures/")

if __name__ == "__main__":
    main()
