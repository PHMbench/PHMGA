"""Model-side feature, baseline, and analysis exports."""

from .features import build_feature_matrix
from .inquirer import build_similarity_artifacts
from .shallow_ml import run_shallow_ml_baseline

__all__ = ["build_feature_matrix", "build_similarity_artifacts", "run_shallow_ml_baseline"]
