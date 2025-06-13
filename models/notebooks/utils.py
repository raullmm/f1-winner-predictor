# utils.py
from sklearn.model_selection import train_test_split
import yaml, os, hashlib

def seeded_split(X, y, params_file="params.yaml"):
    """
    Train/val split where the random_state comes from params.yaml.
    That makes the split deterministic *and* visible in version control.
    """
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"{params_file} missing; add a 'seed:' key.")
    seed = yaml.safe_load(open(params_file))["seed"]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
