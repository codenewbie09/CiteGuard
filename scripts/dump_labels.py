"""Dump ~20 real answers with sentence-level CiteGuard verdicts to results/manual_labels.csv."""
import sys

from src.config import load_config
from src.eval import dump_labels

dump_labels(load_config(), full="--sample" not in sys.argv, force="--force" in sys.argv)
