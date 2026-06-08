import os
import json
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REPOS = [
    {"tool": "ansible", "repo": "geerlingguy/ansible-role-docker"},
    {"tool": "ansible", "repo": "geerlingguy/ansible-role-mysql"},
    {"tool": "ansible", "repo": "ansible/ansible-examples"},
    {"tool": "ansible", "repo": "debops/debops"},
    {"tool": "ansible", "repo": "ansible/awx"},
    {"tool": "terraform", "repo": "terraform-aws-modules/terraform-aws-s3-bucket"},
    {"tool": "terraform", "repo": "terraform-aws-modules/terraform-aws-vpc"},
    {"tool": "terraform", "repo": "terraform-aws-modules/terraform-aws-eks"},
    {"tool": "terraform", "repo": "gruntwork-io/terragrunt-infrastructure-live-example"},
    {"tool": "terraform", "repo": "antonbabenko/pre-commit-terraform"},
    {"tool": "chef", "repo": "sous-chefs/apache2"},
    {"tool": "chef", "repo": "chef-cookbooks/mysql"},
    {"tool": "chef", "repo": "sous-chefs/postgresql"},
    {"tool": "chef", "repo": "sous-chefs/nginx"},
    {"tool": "chef", "repo": "facebook/chef-cookbooks"}
]

def clone_repo(tool, repo_slug, base_dir="data/raw"):
    repo_name = repo_slug.split("/")[-1]
    target_dir = Path(base_dir) / tool / repo_name
    
    # Ensure tool directory exists
    (Path(base_dir) / tool).mkdir(parents=True, exist_ok=True)
    
    if target_dir.exists() and (target_dir / ".git").exists():
        logging.info(f"Repo {repo_slug} already exists at {target_dir}. Skipping clone.")
        return True, str(target_dir)

    url = f"https://github.com/{repo_slug}.git"
    logging.info(f"Cloning {url} into {target_dir}...")
    
    try:
        # Clone with depth 500
        subprocess.run(
            ["git", "clone", "--depth", "500", url, str(target_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Successfully cloned {repo_slug}.")
        return True, str(target_dir)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone {repo_slug}. Error: {e.stderr}")
        return False, str(target_dir)

def main():
    base_dir = "data/raw"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    success_count = 0
    
    for repo_info in REPOS:
        tool = repo_info["tool"]
        repo_slug = repo_info["repo"]
        
        success, path = clone_repo(tool, repo_slug, base_dir)
        
        repo_info["success"] = success
        repo_info["local_path"] = path
        results.append(repo_info)
        
        if success:
            success_count += 1
            
    # Save repo registry
    registry_path = Path("data/repos.json")
    with open(registry_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"Finished cloning. Successfully cloned {success_count}/{len(REPOS)} repos.")
    logging.info(f"Registry saved to {registry_path}")

if __name__ == "__main__":
    main()
