"""End-to-end tests for the Deuterium CLI subcommands."""

import json
from pathlib import Path

from deuterium.cli import main


def _write_mini_dataset(path: Path) -> None:
    rows = [
        "ham\tHey are we still meeting for lunch today?",
        "ham\tI will be home in ten minutes, see you soon",
        "ham\tCan you send me the meeting notes for today?",
        "ham\tSee you at the office tomorrow morning",
        "ham\tThanks for helping me with the project notes",
        "ham\tLunch is at noon, see you there",
        "ham\tAre you free for coffee tomorrow morning?",
        "ham\tJust checking in about the meeting notes",
        "spam\tWINNER! Claim your FREE cash prize now!",
        "spam\tFREE entry to win cash prize, text WIN now!",
        "spam\tURGENT! You won a FREE prize reward, claim today!",
        "spam\tWin cash now! Call to claim your FREE prize!",
        "spam\tCongratulations! Claim your FREE cash reward now!",
        "spam\tText WIN now for a FREE cash prize!",
        "spam\tFREE cash reward waiting, claim your prize now!",
        "spam\tWIN a FREE prize today, call now to claim!",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_cli_train_predict_and_evaluate(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "mini.tsv"
    model = tmp_path / "model.joblib"
    _write_mini_dataset(dataset)

    # 1. Train
    exit_code = main(["train", "--data", str(dataset), "--model", str(model), "--test-size", "0.25"])
    assert exit_code == 0
    assert model.exists()
    train_out = capsys.readouterr().out
    assert "Accuracy" in train_out

    # 2. Predict (human-readable)
    exit_code = main(["predict", "FREE cash prize! Claim now!", "--model", str(model)])
    assert exit_code == 0
    predict_out = capsys.readouterr().out
    assert "Prediction: SPAM" in predict_out

    # 3. Predict (JSON)
    exit_code = main(["predict", "See you at lunch tomorrow", "--model", str(model), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "ham"
    assert "confidence" in payload

    # 4. Evaluate
    exit_code = main(["evaluate", "--data", str(dataset), "--model", str(model)])
    assert exit_code == 0
    eval_out = capsys.readouterr().out
    assert "Confusion Matrix" in eval_out


def test_cli_predict_without_model_returns_error(tmp_path: Path, capsys) -> None:
    exit_code = main(["predict", "hello", "--model", str(tmp_path / "nope.joblib")])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "deuterium train" in err
