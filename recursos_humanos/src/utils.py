"""utils.py - helper utilities for the Attrition project"""
import os
import logging
from pathlib import Path
from datetime import datetime
import yaml

def setup_logging(log_file: str = "logs/pipeline.log"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def timestamp():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def version_name(prefix="model"):
    return f"{prefix}_{timestamp()}"

def read_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
