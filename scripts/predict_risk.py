import argparse
import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import extractors from our feature extraction script
import importlib.util

spec = importlib.util.spec_from_file_location("extract_features", str(Path(__file__).parent / "03_extract_features.py"))
extract_features = importlib.util.module_from_spec(spec)
sys.modules["extract_features"] = extract_features
spec.loader.exec_module(extract_features)

from extract_features import extract_product_metrics, extract_structural_metrics, extract_semantic_metrics, extract_iac_specific

def get_tool_from_extension(filepath):
    ext = Path(filepath).suffix
    if ext in ['.yml', '.yaml']:
        return 'ansible'
    elif ext in ['.tf', '.tfvars']:
        return 'terraform'
    elif ext == '.rb':
        return 'chef'
    else:
        return 'unknown'

def extract_features_for_file(filepath):
    tool = get_tool_from_extension(filepath)
    if tool == 'unknown':
        print(f"Error: Unsupported file extension for {filepath}")
        sys.exit(1)
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)
        
    feats = {}
    feats.update(extract_product_metrics(content))
    feats.update(extract_structural_metrics(content, tool))
    # Get real process metrics from Git history
    import git
    try:
        repo = git.Repo(Path.cwd(), search_parent_directories=True)
        # Attempt to get real history using the extractor
        from extract_features import extract_process_and_evo_metrics
        # git log needs the relative path from repo root
        rel_path = Path(filepath).resolve().relative_to(Path(repo.working_dir))
        proc_feats = extract_process_and_evo_metrics(repo, str(rel_path))
        
        if proc_feats:
            feats.update(proc_feats)
        else:
            # File is brand new (uncommitted), genuine risk is low
            feats.update({
                'num_commits': 0, 'num_authors': 1, 'file_age_days': 0,
                'last_modified_days': 0, 'edit_frequency': 0,
                'is_recently_active': 1, 'stability_score': 0
            })
            
    except Exception as e:
        print(f"Warning: Could not extract Git history ({e}). Assuming new file.")
        feats.update({
            'num_commits': 0, 'num_authors': 1, 'file_age_days': 0,
            'last_modified_days': 0, 'edit_frequency': 0,
            'is_recently_active': 1, 'stability_score': 0
        })

    # Add churn approximation for the new file state
    if 'churn_total' not in feats:
        feats['churn_total'] = feats.get('num_commits', 0) * feats['loc'] * 0.1
        feats['churn_avg'] = feats['churn_total'] / max(feats.get('num_commits', 1), 1)
        feats['loc_growth_rate'] = feats['loc'] / max(feats.get('file_age_days', 1), 1)
        feats['churn_volatility'] = feats.get('churn_avg', 0) * 0.5
    
    return feats

def main():
    parser = argparse.ArgumentParser(description="Predict defect risk for an IaC script.")
    parser.add_argument("--file", required=True, help="Path to the IaC file (.tf, .yml, .rb)")
    args = parser.parse_args()
    
    # Load model artifacts
    model_path = Path(__file__).parent.parent / "models" / "rf_model.joblib"
    scaler_path = Path(__file__).parent.parent / "models" / "scaler.joblib"
    features_path = Path(__file__).parent.parent / "models" / "feature_names.joblib"
    
    if not model_path.exists():
        print("Error: Model artifacts not found. Run training phase first.")
        sys.exit(1)
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    
    feats = extract_features_for_file(args.file)
    
    # Align features with model
    df = pd.DataFrame([feats])
    # Fill any missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    
    risk_prob = model.predict_proba(X_scaled)[0, 1]
    
    print("\n" + "="*50)
    print(f"IaC Risk Scan Report: {Path(args.file).name}")
    print("="*50)
    print(f"Risk Probability: {risk_prob:.2%}")
    
    if risk_prob < 0.3:
        print("Risk Level: LOW")
    elif risk_prob < 0.6:
        print("Risk Level: MEDIUM")
    else:
        print("Risk Level: HIGH")
        print("\nWARNING: This script has a high probability of containing defects.")
        
    # Get top contributing factors for this specific file
    import shap
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_scaled)
    # handle different shap versions
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        vals = shap_vals[0, :, 1]
    else:
        vals = shap_vals[0]
        
    top_indices = np.argsort(vals)[-3:][::-1]
    print("\nTop Risk Drivers (Local Explanations):")
    for idx in top_indices:
        if vals[idx] > 0:
            feat_name = feature_names[idx]
            val = df.iloc[0][feat_name]
            print(f"- {feat_name} = {val} (Increases risk by {vals[idx]:.3f})")
            
    print("="*50 + "\n")
    
    # Return non-zero exit code if high risk to break CI
    if risk_prob >= 0.6:
        sys.exit(2)

if __name__ == "__main__":
    main()
