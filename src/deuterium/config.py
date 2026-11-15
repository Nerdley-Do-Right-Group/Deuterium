"""Centralized configuration and default file paths for Deuterium.

Keeping paths in one place means when someone inevitably renames a folder,
we only have to fix it here instead of playing grep-whack-a-mole.
"""

from pathlib import Path

# Resolve the repository root relative to this file (src/deuterium/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_PATH = REPO_ROOT / "SMSSpamCollection"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "spam_model.joblib"

VALID_LABELS = frozenset({"ham", "spam"})
SUPPORTED_ALGOS = ("logreg", "nb")


def metadata_path_for(model_path: Path) -> Path:
    """Return the sidecar JSON metadata path for a given model artifact."""
    return model_path.with_suffix(".meta.json")
