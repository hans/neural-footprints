import os
import yaml
from pathlib import Path


def load_config(path=None):
    if path is None:
        env_path = os.environ.get("CONFIG_FILE")
        if env_path:
            path = Path(__file__).resolve().parent.parent / env_path
        else:
            path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
