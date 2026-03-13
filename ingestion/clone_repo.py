"""
clone_repo.py — Clone a GitHub repository locally for analysis.

Uses GitPython to clone repos into /repos/{repo_name}.
"""

import os
from git import Repo


REPOS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "repos")


def clone_repository(repo_url: str, target_dir: str | None = None) -> str:
    """Clone a GitHub repository to a local directory.
    
    Args:
        repo_url: Full GitHub URL (e.g., https://github.com/karpathy/nanoGPT)
        target_dir: Optional custom target directory. Defaults to repos/{repo_name}.
    
    Returns:
        Path to the cloned repository.
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    
    if target_dir is None:
        target_dir = os.path.join(REPOS_DIR, repo_name)
    
    if os.path.exists(target_dir):
        print(f"[INFO] Repository already exists at {target_dir}, pulling latest...")
        repo = Repo(target_dir)
        repo.remotes.origin.pull()
    else:
        print(f"[INFO] Cloning {repo_url} into {target_dir}...")
        os.makedirs(target_dir, exist_ok=True)
        Repo.clone_from(repo_url, target_dir)
    
    print(f"[INFO] Repository ready at {target_dir}")
    return target_dir


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clone_repo.py <github_url>")
        sys.exit(1)
    clone_repository(sys.argv[1])
