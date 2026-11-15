"""ML pipeline construction, training, evaluation, persistence, and inference.

No GPUs, no 40 GB transformer weights, no waiting 3 hours for an epoch to
finish. Just TF-IDF + classic linear/probabilistic classifiers that train in
under a second and hit ~98% accuracy on SMS spam.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from deuterium.config import SUPPORTED_ALGOS, metadata_path_for


@dataclass(frozen=True)
class EvaluationMetrics:
    """Classification metrics computed on a held-out test set."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    test_samples: int


@dataclass(frozen=True)
class PredictionResult:
    """Single-message prediction output."""

    message: str
    label: str
    spam_probability: float
    ham_probability: float

    @property
    def confidence(self) -> float:
        """Probability of the predicted class (in [0.0, 1.0])."""
        return self.spam_probability if self.label == "spam" else self.ham_probability


def build_pipeline(algo: str = "logreg") -> Pipeline:
    """Build an untrained scikit-learn text classification pipeline.

    Args:
        algo: Classifier key — ``"logreg"`` (Logistic Regression, default) or
            ``"nb"`` (Complement Naive Bayes).
    """
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algorithm {algo!r}. Choose from {SUPPORTED_ALGOS}.")

    # Unigrams + bigrams catch both single suspicious words ("free", "claim")
    # and classic two-word spam combos ("won prize", "call now", "std txt").
    # sublinear_tf=True dampens messages that scream "FREE FREE FREE FREE".
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    if algo == "logreg":
        # class_weight="balanced" stops the model from lazily guessing "ham"
        # for everything just because 86.6% of the dataset is ham.
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    else:
        # ComplementNB is specifically designed for imbalanced text datasets.
        classifier = ComplementNB()

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def evaluate_pipeline(
    pipeline: Pipeline,
    x_test: list[str],
    y_test: list[str],
) -> EvaluationMetrics:
    """Evaluate a fitted pipeline against test messages and labels."""
    predictions = pipeline.predict(x_test)

    # Explicit label order ["ham", "spam"] guarantees [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=["ham", "spam"]).ravel()

    return EvaluationMetrics(
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        recall=float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        f1=float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        test_samples=len(y_test),
    )


def save_model(
    pipeline: Pipeline,
    model_path: Path,
    algo: str,
    metrics: EvaluationMetrics,
    train_samples: int,
) -> Path:
    """Persist the trained pipeline to disk along with a JSON metadata sidecar."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    meta = {
        "algo": algo,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_samples": train_samples,
        "metrics": asdict(metrics),
    }
    meta_path = metadata_path_for(model_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path


def load_model(model_path: Path) -> Pipeline:
    """Load a previously trained pipeline from disk.

    Raises:
        FileNotFoundError: With a friendly nudge to run `deuterium train` first.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run `deuterium train` first — the AI can't guess spam by telepathy (yet)."
        )
    return joblib.load(model_path)


def predict_message(pipeline: Pipeline, message: str) -> PredictionResult:
    """Classify a single message as 'spam' or 'ham' with class probabilities."""
    if not message.strip():
        raise ValueError("Cannot classify an empty message. Try typing actual words.")

    probabilities = pipeline.predict_proba([message])[0]
    classes = list(pipeline.classes_)
    prob_by_class = dict(zip(classes, probabilities))

    spam_prob = float(prob_by_class.get("spam", 0.0))
    ham_prob = float(prob_by_class.get("ham", 0.0))
    label = "spam" if spam_prob >= ham_prob else "ham"

    return PredictionResult(
        message=message,
        label=label,
        spam_probability=spam_prob,
        ham_probability=ham_prob,
    )
