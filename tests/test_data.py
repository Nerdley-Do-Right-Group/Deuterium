"""Tests for dataset loading, validation, and stratified splitting."""

from pathlib import Path

import pytest

from deuterium.data import load_dataset, split_dataset


def test_load_valid_dataset(tmp_path: Path) -> None:
    sample = tmp_path / "sample.tsv"
    sample.write_text(
        "ham\tHey, want to grab coffee?\n"
        "spam\tFREE prize! Call 0800-NOT-A-SCAM now!\n"
        "\n",  # trailing blank line should be ignored gracefully
        encoding="utf-8",
    )

    messages, labels = load_dataset(sample)
    assert messages == [
        "Hey, want to grab coffee?",
        "FREE prize! Call 0800-NOT-A-SCAM now!",
    ]
    assert labels == ["ham", "spam"]


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "ghost_dataset.tsv")


def test_load_malformed_line_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text("this line forgot its tab separator\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected '<label>"):
        load_dataset(bad)


def test_load_unknown_label_raises(tmp_path: Path) -> None:
    bad = tmp_path / "unknown_label.tsv"
    bad.write_text("bacon\tNot ham or spam\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown label"):
        load_dataset(bad)


def test_split_dataset_preserves_stratification() -> None:
    messages = [f"ham msg {i}" for i in range(80)] + [f"spam msg {i}" for i in range(20)]
    labels = ["ham"] * 80 + ["spam"] * 20

    split = split_dataset(messages, labels, test_size=0.2, random_state=42)
    assert len(split.x_train) == 80
    assert len(split.x_test) == 20
    assert split.y_test.count("spam") == 4
    assert split.y_test.count("ham") == 16
