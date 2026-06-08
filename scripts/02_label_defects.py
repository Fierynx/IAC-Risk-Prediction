import os
import json
import re
import pandas as pd
from pathlib import Path
import logging
import git
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFECT_KEYWORDS = re.compile(
    r"\b(fix|bug|error|fault|defect|patch|hotfix|issue|crash|fail|broken|resolve|correct)\b",
    re.IGNORECASE
)

def is_iac_file(tool, filepath):
    """Check if the given file matches the IaC language for the tool."""
    path = Path(filepath)
    if tool == "ansible":
        return path.suffix in [".yml", ".yaml"]
    elif tool == "terraform":
        return path.suffix in [".tf", ".tfvars"]
    elif tool == "chef":
        return path.suffix == ".rb"
    return False

def extract_labels_from_repo(repo_info):
    local_path = repo_info["local_path"]
    tool = repo_info["tool"]
    repo_name = repo_info["repo"]
    
    if not repo_info["success"] or not Path(local_path).exists():
        logging.warning(f"Skipping {repo_name} because it was not successfully cloned.")
        return []

    try:
        git_repo = git.Repo(local_path)
    except git.exc.InvalidGitRepositoryError:
        logging.error(f"Invalid Git repository at {local_path}")
        return []

    labels = []
    
    # We will iterate through commits on the default branch
    try:
        commits = list(git_repo.iter_commits())
    except git.exc.GitCommandError as e:
        logging.error(f"Failed to get commits for {repo_name}: {e}")
        return []
        
    logging.info(f"Mining {len(commits)} commits in {repo_name}...")
    
    for commit in commits:
        msg = commit.message
        
        if not msg:
            continue
            
        # Check if it's a defect-fixing commit
        is_defect = bool(DEFECT_KEYWORDS.search(msg))
        
        # Skip trivial or irrelevant commits
        if "merge pull request" in msg.lower() or "merge branch" in msg.lower():
            continue
            
        # Inspect files modified in this commit
        try:
            # Get modified files compared to parents
            if not commit.parents:
                continue # Skip initial commit
                
            diffs = commit.parents[0].diff(commit)
            for diff in diffs:
                # We only care about modifications
                if diff.change_type in ['M', 'A']:
                    filepath = diff.b_path
                    if filepath and is_iac_file(tool, filepath):
                        labels.append({
                            "repo": repo_name,
                            "tool": tool,
                            "filepath": filepath,
                            "commit_sha": commit.hexsha,
                            "commit_date": datetime.fromtimestamp(commit.committed_date).isoformat(),
                            "is_defect": 1 if is_defect else 0,
                            "commit_message": msg.strip().replace("\n", " ")
                        })
        except Exception as e:
            logging.warning(f"Error processing commit {commit.hexsha} in {repo_name}: {e}")
            continue

    return labels

def main():
    registry_path = Path("data/repos.json")
    if not registry_path.exists():
        logging.error("repos.json not found. Run Phase 2 first.")
        return

    with open(registry_path, "r") as f:
        repos = json.load(f)
        
    all_labels = []
    
    for repo_info in repos:
        repo_labels = extract_labels_from_repo(repo_info)
        all_labels.extend(repo_labels)
        
    if not all_labels:
        logging.error("No labels extracted!")
        return
        
    df = pd.DataFrame(all_labels)
    
    # Deduplicate: if a file was EVER touched by a defect commit, it's marked as defect-prone
    file_level_df = df.groupby(["repo", "tool", "filepath"]).agg({
        "is_defect": "max",
        "commit_sha": "count" # number of times file was touched
    }).reset_index()
    file_level_df.rename(columns={"commit_sha": "touch_count"}, inplace=True)
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the detailed commit-level log
    commit_level_path = output_dir / "commit_labels.csv"
    df.to_csv(commit_level_path, index=False)
    
    # Save the file-level labels
    file_level_path = output_dir / "labels.csv"
    file_level_df.to_csv(file_level_path, index=False)
    
    logging.info(f"Mined {len(df)} total file-commit events.")
    logging.info(f"Generated labels for {len(file_level_df)} unique IaC files.")
    defect_rate = file_level_df['is_defect'].mean() * 100
    logging.info(f"Overall Defect Rate: {defect_rate:.2f}%")
    logging.info(f"Labels saved to {file_level_path}")

if __name__ == "__main__":
    main()
