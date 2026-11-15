"""Dataset loading, validation, and train/test splitting for Deuterium.

The SMSSpamCollection file is tab-separated with no header row:
    ham\tHey, are we still on for lunch?
    spam\tWINNER!! Claim your free £900 prize right now!
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from deuterium.config import VALID_LABELS


@dataclass(frozen=True)
class DatasetSplit:
    """Container for stratified train and test splits."""

    x_train: list[str]
    x_test: list[str]
    y_train: list[str]
    y_test: list[str]


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Load and validate a tab-separated spam/ham dataset.

    Args:
        path: Path to the dataset file (e.g. ``SMSSpamCollection``).

    Returns:
        A ``(messages, labels)`` tuple of equal-length string lists.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a non-empty line is missing a tab separator, has an
            unknown label, has an empty message, or the file is empty.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Did you delete SMSSpamCollection or pass the wrong --data path?"
        )

    messages: list[str] = []
    labels: list[str] = []

    # UTF-8 handles £, curly quotes, and whatever else 2005-era SMS gateways threw in
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                # Skip trailing blank lines gracefully instead of crashing on line 5575
                continue

            if "\t" not in line:
                raise ValueError(
                    f"Malformed row at {path}:{line_no} — expected '<label>\\t<message>', got: {line!r}"
                )

            # Split on the FIRST tab only; messages themselves might contain tabs
            label, message = line.split("\t", maxsplit=1)
            label = label.strip().lower()

            if label not in VALID_LABELS:
                raise ValueError(
                    f"Unknown label {label!r} at {path}:{line_no}. Expected one of {sorted(VALID_LABELS)}."
                )

            if not message.strip():
                raise ValueError(f"Empty message at {path}:{line_no}. Silence is golden, but not trainable.")

            messages.append(message)
            labels.append(label)

    if not messages:
        raise ValueError(f"Dataset at '{path}' contains no valid rows. Training on air rarely works.")

    return messages, labels


def split_dataset(
    messages: list[str],
    labels: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split messages and labels into stratified train and test sets.

    Why stratified? Because ~86.6% of SMS messages are 'ham' and only ~13.4%
    are 'spam'. Without stratification, a unlucky random split could skew the
    class ratio and make evaluation metrics lie to us.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {test_size}.")

    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return DatasetSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
