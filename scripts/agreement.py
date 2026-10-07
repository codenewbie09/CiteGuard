"""Agreement (accuracy, Cohen's kappa, unsupported-P/R/F1) between human labels and CiteGuard."""
from src.config import load_config
from src.eval import agreement

agreement(load_config())
