import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, f1_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def train_nn(X_train, y_train, X_test, epochs=20, batch_size=32):
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FeedforwardNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).numpy().flatten()
    return preds, model

def evaluate(y_true, y_prob):
    # Avoid zero-division or errors if all labels are same
    if len(np.unique(y_true)) == 1:
        return {'auc_pr': 0, 'mcc': 0, 'f1': 0, 'brier': 0}
        
    y_pred = (y_prob >= 0.5).astype(int)
    auc_pr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    return {'auc_pr': auc_pr, 'mcc': mcc, 'f1': f1, 'brier': brier}

def main():
    df = pd.read_csv("data/processed/features.csv")
    logging.info(f"Loaded {len(df)} records.")
    
    # Preprocessing
    df.fillna(0, inplace=True)
    
    # Feature categories for ablation
    product_cols = ['loc', 'blank_lines', 'comment_lines', 'avg_line_length', 'max_nesting_depth', 'num_keys', 'file_size_bytes', 'token_count']
    process_cols = ['num_commits', 'num_authors', 'file_age_days', 'last_modified_days', 'edit_frequency', 'is_recently_active', 'stability_score', 'churn_total', 'churn_avg', 'loc_growth_rate', 'churn_volatility']
    struct_cols = ['num_resources', 'num_variables', 'num_dependencies', 'num_modules', 'dependency_ratio']
    semantic_cols = ['has_hardcoded_secret', 'has_version_pinning', 'uses_latest_tag', 'num_anti_patterns']
    iac_cols = ['has_error_handling', 'has_idempotency_check', 'has_privilege_escalation', 'template_complexity', 'num_conditionals', 'num_loops']
    
    all_features = product_cols + process_cols + struct_cols + semantic_cols + iac_cols
    
    X = df[all_features].values
    y = df['is_defect'].values
    
    models = {
        'RF': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGB': XGBClassifier(n_estimators=100, scale_pos_weight=(len(y)-sum(y))/sum(y) if sum(y) > 0 else 1, random_state=42, eval_metric='logloss'),
        'LR': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    }
    
    scaler = StandardScaler()
    
    results = []
    
    # 1. Within-Project (Stratified 5-Fold over all data as baseline)
    logging.info("Running Within-Project Baseline...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, model in models.items():
        metrics_list = []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            # SMOTE for imbalance
            try:
                sm = SMOTE(random_state=42)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except ValueError:
                pass
                
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)[:, 1]
            metrics_list.append(evaluate(y_te, y_prob))
            
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        avg_metrics.update({'model': model_name, 'experiment': 'Within-Project'})
        results.append(avg_metrics)
        
    # Also run NN for Within-Project
    metrics_list = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = scaler.fit_transform(X[train_idx]), scaler.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        y_prob, _ = train_nn(X_tr, y_tr, X_te, epochs=10)
        metrics_list.append(evaluate(y_te, y_prob))
    avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    avg_metrics.update({'model': 'Feedforward NN', 'experiment': 'Within-Project'})
    results.append(avg_metrics)
        
    # 2. Cross-Project (Leave-One-Repo-Out)
    logging.info("Running Cross-Project LOPO...")
    repos = df['repo'].unique()
    for model_name, model in models.items():
        metrics_list = []
        for repo in repos:
            train_mask = df['repo'] != repo
            test_mask = df['repo'] == repo
            
            if sum(test_mask) < 10 or sum(df[test_mask]['is_defect']) < 1:
                continue
                
            X_tr, X_te = X[train_mask], X[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            try:
                sm = SMOTE(random_state=42)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except ValueError:
                pass
                
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)[:, 1]
            metrics_list.append(evaluate(y_te, y_prob))
            
        if metrics_list:
            avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
            avg_metrics.update({'model': model_name, 'experiment': 'Cross-Project'})
            results.append(avg_metrics)

    # 3. Cross-Tool
    logging.info("Running Cross-Tool Transfer...")
    tools = df['tool'].unique()
    for target_tool in tools:
        train_mask = df['tool'] != target_tool
        test_mask = df['tool'] == target_tool
        
        if sum(test_mask) < 10 or sum(df[test_mask]['is_defect']) < 1:
            continue
            
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        
        mets = evaluate(y_te, y_prob)
        mets.update({'model': 'RF', 'experiment': f'Train:Others->Test:{target_tool}'})
        results.append(mets)

    # 4. Ablation Study for RF (Within-Project)
    logging.info("Running Ablation Study...")
    categories = {
        'Product': product_cols,
        'Process': process_cols,
        'Structural': struct_cols,
        'Semantic': semantic_cols,
        'IaC': iac_cols,
        'All Combined': all_features
    }
    
    ablation_results = []
    for cat_name, feats in categories.items():
        X_cat = df[feats].values
        metrics_list = []
        for train_idx, test_idx in skf.split(X_cat, y):
            X_tr, X_te = X_cat[train_idx], X_cat[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)[:, 1]
            metrics_list.append(evaluate(y_te, y_prob))
            
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        avg_metrics.update({'category': cat_name})
        ablation_results.append(avg_metrics)

    # Save outputs
    res_df = pd.DataFrame(results)
    ab_df = pd.DataFrame(ablation_results)
    
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    
    # Save main results
    wp_df = res_df[res_df['experiment'] == 'Within-Project']
    cp_df = res_df[res_df['experiment'] == 'Cross-Project']
    ct_df = res_df[res_df['experiment'].str.startswith('Train')]
    
    wp_df.to_csv("results/tables/within_project_results.csv", index=False)
    cp_df.to_csv("results/tables/cross_project_results.csv", index=False)
    ct_df.to_csv("results/tables/cross_tool_results.csv", index=False)
    ab_df.to_csv("results/tables/ablation_results.csv", index=False)
    
    # Train and save the final RF model for CI/CD
    logging.info("Training final production model...")
    final_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    X_scaled = scaler.fit_transform(X)
    final_rf.fit(X_scaled, y)
    
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(final_rf, "models/rf_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(all_features, "models/feature_names.joblib")
    
    logging.info("Saved final model and scaler. Done.")

if __name__ == "__main__":
    main()
