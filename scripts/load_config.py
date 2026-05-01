import yaml
from pathlib import Path


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
