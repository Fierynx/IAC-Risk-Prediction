import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import git
import re
from datetime import datetime
from tqdm import tqdm
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_product_metrics(content):
    lines = content.splitlines()
    loc = len([l for l in lines if l.strip()])
    blank_lines = len(lines) - loc
    comment_lines = len([l for l in lines if l.strip().startswith('#') or l.strip().startswith('//')])
    avg_line_length = np.mean([len(l) for l in lines]) if lines else 0
    
    # Simple max indentation heuristic
    max_indent = 0
    for l in lines:
        if l.strip():
            indent = len(l) - len(l.lstrip())
            max_indent = max(max_indent, indent)
    max_nesting_depth = max_indent // 2  # Assuming 2 spaces per indent
    
    num_keys = len(re.findall(r'^\s*[\w-]+:', content, re.MULTILINE))
    token_count = len(content.split())
    
    return {
        'loc': loc,
        'blank_lines': blank_lines,
        'comment_lines': comment_lines,
        'avg_line_length': avg_line_length,
        'max_nesting_depth': max_nesting_depth,
        'num_keys': num_keys,
        'file_size_bytes': len(content.encode('utf-8')),
        'token_count': token_count
    }

def extract_process_and_evo_metrics(repo, filepath):
    try:
        # Use git log to get stats
        commits = list(repo.iter_commits(paths=filepath))
        if not commits:
            return None
            
        num_commits = len(commits)
        authors = set([c.author.email for c in commits])
        
        # Calculate file age and edit frequency
        first_commit_date = datetime.fromtimestamp(commits[-1].committed_date)
        last_commit_date = datetime.fromtimestamp(commits[0].committed_date)
        now = datetime.now()
        
        file_age_days = (now - first_commit_date).days
        last_modified_days = (now - last_commit_date).days
        edit_frequency = num_commits / max(file_age_days / 30.0, 1.0)
        
        # Process commit stats to get churn
        total_churn = 0
        loc_history = []
        churn_history = []
        commit_dates = [datetime.fromtimestamp(c.committed_date) for c in commits]
        
        max_gap = 0
        for i in range(len(commit_dates)-1):
            gap = (commit_dates[i] - commit_dates[i+1]).days
            if gap > max_gap:
                max_gap = gap
                
        # To get exact lines added/deleted per commit is very expensive for 5000 files x history.
        # We'll approximate using touch_count which we already have, or lightweight git log --numstat
        # But for speed, let's just use the touch_count from labels and basic metrics.
        
        return {
            'num_commits': num_commits,
            'num_authors': len(authors),
            'file_age_days': file_age_days,
            'last_modified_days': last_modified_days,
            'edit_frequency': edit_frequency,
            'is_recently_active': 1 if last_modified_days < 30 else 0,
            'stability_score': max_gap
        }
    except Exception as e:
        return None

def extract_structural_metrics(content, tool):
    metrics = {
        'num_resources': 0,
        'num_variables': 0,
        'num_dependencies': 0,
        'num_modules': 0
    }
    
    if tool == 'terraform':
        metrics['num_resources'] = len(re.findall(r'^\s*resource\s+"', content, re.MULTILINE))
        metrics['num_variables'] = len(re.findall(r'var\.', content))
        metrics['num_dependencies'] = len(re.findall(r'depends_on\s*=', content))
        metrics['num_modules'] = len(re.findall(r'^\s*module\s+"', content, re.MULTILINE))
    elif tool == 'ansible':
        metrics['num_resources'] = len(re.findall(r'^\s*-\s+name:', content, re.MULTILINE))
        metrics['num_variables'] = len(re.findall(r'\{\{\s*[\w\.]+\s*\}\}', content))
        metrics['num_dependencies'] = len(re.findall(r'notify:', content))
        metrics['num_modules'] = len(re.findall(r'include_role:|import_role:', content))
    elif tool == 'chef':
        metrics['num_resources'] = len(re.findall(r'^\s*\w+\s+[\'"].*?[\'"]\s+do', content, re.MULTILINE))
        metrics['num_variables'] = len(re.findall(r'node\[[\'"]', content))
        metrics['num_dependencies'] = len(re.findall(r'notifies\s+:', content))
        metrics['num_modules'] = len(re.findall(r'include_recipe', content))
        
    metrics['dependency_ratio'] = metrics['num_dependencies'] / max(metrics['num_resources'], 1)
    return metrics

def extract_semantic_metrics(content):
    has_secret = 1 if re.search(r'(?i)password\s*:\s*.+|secret\s*:\s*.+|api_key\s*:\s*.+', content) else 0
    has_pinning = 1 if re.search(r'version\s*=|version\s*:\s*[\d\.]+|>=|==', content) else 0
    uses_latest = 1 if re.search(r':latest\b|latest\s*=', content) else 0
    num_anti_patterns = len(re.findall(r'777|0.0.0.0/0|privileged:\s*true', content))
    
    return {
        'has_hardcoded_secret': has_secret,
        'has_version_pinning': has_pinning,
        'uses_latest_tag': uses_latest,
        'num_anti_patterns': num_anti_patterns,
        # tfidf will be added later in model training
    }

def extract_iac_specific(content, tool):
    metrics = {
        'has_error_handling': 0,
        'has_idempotency_check': 0,
        'has_privilege_escalation': 0,
        'template_complexity': 0,
        'num_conditionals': 0,
        'num_loops': 0
    }
    
    if tool == 'ansible':
        metrics['has_error_handling'] = 1 if 'ignore_errors:' in content else 0
        metrics['has_idempotency_check'] = 1 if 'creates:' in content or 'removes:' in content else 0
        metrics['has_privilege_escalation'] = 1 if 'become: true' in content or 'become: yes' in content else 0
        metrics['template_complexity'] = len(re.findall(r'\{\{.*?\}\}|\{%.*?%\}', content))
        metrics['num_conditionals'] = len(re.findall(r'\s+when:', content))
        metrics['num_loops'] = len(re.findall(r'\s+loop:|\s+with_items:', content))
    elif tool == 'terraform':
        metrics['has_error_handling'] = 1 if 'lifecycle {' in content else 0
        metrics['template_complexity'] = len(re.findall(r'\$\{.*?\}', content))
        metrics['num_conditionals'] = len(re.findall(r'count\s*=\s*.*?==', content))
        metrics['num_loops'] = len(re.findall(r'for_each\s*=', content))
    elif tool == 'chef':
        metrics['has_error_handling'] = 1 if 'rescue ' in content or 'ignore_failure ' in content else 0
        metrics['has_idempotency_check'] = 1 if 'not_if ' in content or 'only_if ' in content else 0
        metrics['template_complexity'] = len(re.findall(r'<%=.*?%>', content))
        metrics['num_conditionals'] = len(re.findall(r'\bif\b|\bunless\b', content))
        metrics['num_loops'] = len(re.findall(r'\.each\s+do', content))
        
    return metrics

def main():
    labels_path = Path("data/processed/labels.csv")
    repos_path = Path("data/repos.json")
    
    if not labels_path.exists() or not repos_path.exists():
        logging.error("Missing labels.csv or repos.json. Run previous phases.")
        return
        
    df = pd.read_csv(labels_path)
    with open(repos_path, "r") as f:
        repos_info = json.load(f)
        
    repo_map = {r['repo']: git.Repo(r['local_path']) for r in repos_info if r['success'] and Path(r['local_path']).exists()}
    
    features_list = []
    
    logging.info(f"Extracting features for {len(df)} files...")
    
    # For speed, pre-load file contents at HEAD. If deleted, skip.
    for _, row in tqdm(df.iterrows(), total=len(df)):
        repo_name = row['repo']
        tool = row['tool']
        filepath = row['filepath']
        
        if repo_name not in repo_map:
            continue
            
        repo = repo_map[repo_name]
        full_path = Path(repo.working_dir) / filepath
        
        if not full_path.exists():
            continue
            
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            continue
            
        # Product
        feats = extract_product_metrics(content)
        
        # Process & Evo
        proc = extract_process_and_evo_metrics(repo, filepath)
        if proc:
            feats.update(proc)
        else:
            # Fallback
            feats.update({
                'num_commits': row.get('touch_count', 1),
                'num_authors': 1, 'file_age_days': 0, 'last_modified_days': 0,
                'edit_frequency': 0, 'is_recently_active': 0, 'stability_score': 0
            })
            
        # Add basic churn approximation
        feats['churn_total'] = feats['num_commits'] * feats['loc'] * 0.1 # proxy
        feats['churn_avg'] = feats['churn_total'] / max(feats['num_commits'], 1)
        feats['loc_growth_rate'] = feats['loc'] / max(feats['file_age_days'], 1)
        feats['churn_volatility'] = feats['churn_avg'] * 0.5
            
        # Structural
        feats.update(extract_structural_metrics(content, tool))
        
        # Semantic
        feats.update(extract_semantic_metrics(content))
        
        # IaC Specific
        feats.update(extract_iac_specific(content, tool))
        
        # Base info
        feats['repo'] = repo_name
        feats['tool'] = tool
        feats['filepath'] = filepath
        feats['is_defect'] = row['is_defect']
        
        features_list.append(feats)
        
    features_df = pd.DataFrame(features_list)
    output_path = Path("data/processed/features.csv")
    features_df.to_csv(output_path, index=False)
    logging.info(f"Successfully extracted features for {len(features_df)} files.")
    logging.info(f"Features saved to {output_path}")

if __name__ == "__main__":
    main()
