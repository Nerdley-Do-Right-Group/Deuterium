"""Tests for ML pipeline building, training, persistence, and inference."""

from pathlib import Path

import pytest

from deuterium.config import metadata_path_for
from deuterium.model import (
    build_pipeline,
    evaluate_pipeline,
    load_model,
    predict_message,
    save_model,
)

# A tiny synthetic dataset — just enough for TF-IDF (min_df=2) to learn patterns fast
TINY_HAM = [
    "Hey are we still meeting for lunch today?",
    "I will be home in ten minutes, see you soon",
    "Can you send me the meeting notes for today?",
    "See you at the office tomorrow morning",
    "Thanks for helping me with the project notes",
    "Lunch is at noon, see you there",
]
TINY_SPAM = [
    "WINNER! Claim your FREE cash prize now!",
    "FREE entry to win cash prize, text WIN now!",
    "URGENT! You won a FREE prize reward, claim today!",
    "Win cash now! Call to claim your FREE prize!",
    "Congratulations! Claim your FREE cash reward now!",
    "Text WIN now for a FREE cash prize!",
]


@pytest.mark.parametrize("algo", ["logreg", "nb"])
def test_train_save_load_and_predict(tmp_path: Path, algo: str) -> None:
    messages = TINY_HAM + TINY_SPAM
    labels = ["ham"] * len(TINY_HAM) + ["spam"] * len(TINY_SPAM)

    pipeline = build_pipeline(algo=algo)
    pipeline.fit(messages, labels)

    metrics = evaluate_pipeline(pipeline, messages, labels)
    assert metrics.accuracy >= 0.8

    model_path = tmp_path / "models" / f"{algo}.joblib"
    save_model(pipeline, model_path, algo=algo, metrics=metrics, train_samples=len(messages))
    assert model_path.exists()
    assert metadata_path_for(model_path).exists()

    reloaded = load_model(model_path)
    spam_res = predict_message(reloaded, "FREE cash prize! Claim your WIN reward now!")
    ham_res = predict_message(reloaded, "See you tomorrow morning for lunch")

    assert spam_res.label == "spam"
    assert ham_res.label == "ham"
    assert 0.0 <= spam_res.confidence <= 1.0


def test_load_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="deuterium train"):
        load_model(tmp_path / "missing.joblib")


def test_predict_empty_message_raises() -> None:
    pipeline = build_pipeline("logreg")
    pipeline.fit(TINY_HAM + TINY_SPAM, ["ham"] * len(TINY_HAM) + ["spam"] * len(TINY_SPAM))
    with pytest.raises(ValueError, match="empty message"):
        predict_message(pipeline, "   ")
