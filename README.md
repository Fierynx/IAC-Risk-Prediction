# Explainable Cross-Project Risk Prediction for Infrastructure-as-Code (IaC)

This repository contains the end-to-end pipeline for predicting defect risks in Infrastructure-as-Code (IaC) scripts (Ansible, Terraform, Chef).

## Features
- **Data Mining**: Extracts and labels defects from 15 real open-source GitHub repositories.
- **Feature Engineering**: Calculates ~35 IaC-specific metrics spanning static code, graph structure, semantics, and process history.
- **Explainable ML**: Trains Random Forest, XGBoost, and Deep Learning models, and evaluates them on cross-project scenarios. Outputs SHAP feature importance and counterfactuals.
- **CI/CD Ready**: Includes a Dockerized CLI tool to scan new pull requests for risky IaC changes.

## Directory Structure
- `data/`: Raw cloned repos and processed CSVs (`labels.csv`, `features.csv`).
- `scripts/`: Python pipeline scripts.
- `models/`: Pickled trained models.
- `results/`: Output CSV tables and SHAP figures for publication.

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
pip install imbalanced-learn
```

Run the pipeline sequentially:
```bash
# 1. Clone Repos
python scripts/01_collect_repos.py

# 2. Mine Commits and Label Defects
python scripts/02_label_defects.py

# 3. Extract Features
python scripts/03_extract_features.py

# 4. Train Models
python scripts/04_train_models.py

# 5. Generate Explanations
python scripts/05_explain.py
```

## CI/CD Usage
To predict the risk of a single script locally:
```bash
python scripts/predict_risk.py --file my_script.tf
```

Or using Docker:
```bash
docker build -t iac-risk .
docker run -v $(pwd):/app/workspace -w /app/workspace iac-risk --file my_script.tf
```
