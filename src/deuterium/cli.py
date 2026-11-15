"""Command-line interface for Deuterium.

Provides three subcommands:
    deuterium train     Train a new spam detection model on a dataset
    deuterium predict   Classify a single message as SPAM or HAM
    deuterium evaluate  Evaluate a saved model on a dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from deuterium import __version__
from deuterium.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    SUPPORTED_ALGOS,
)
from deuterium.data import load_dataset, split_dataset
from deuterium.model import (
    EvaluationMetrics,
    build_pipeline,
    evaluate_pipeline,
    load_model,
    predict_message,
    save_model,
)


def _format_metrics(metrics: EvaluationMetrics) -> str:
    """Format evaluation metrics into a human-readable summary block."""
    return "\n".join([
        f"Test samples : {metrics.test_samples}",
        f"Accuracy     : {metrics.accuracy:.2%}",
        f"Precision    : {metrics.precision:.2%}  (how often 'spam' predictions are actually spam)",
        f"Recall       : {metrics.recall:.2%}  (how much actual spam we caught)",
        f"F1 Score     : {metrics.f1:.2%}",
        "Confusion Matrix (rows=actual, cols=predicted):",
        f"  ham  -> ham: {metrics.true_negatives:>4} | spam (false alarm): {metrics.false_positives:>4}",
        f"  spam -> ham (missed): {metrics.false_negatives:>4} | spam: {metrics.true_positives:>4}",
    ])


def _handle_train(args: argparse.Namespace) -> int:
    """Run the `train` subcommand."""
    messages, labels = load_dataset(args.data)
    split = split_dataset(messages, labels, test_size=args.test_size)

    pipeline = build_pipeline(algo=args.algo)
    pipeline.fit(split.x_train, split.y_train)

    metrics = evaluate_pipeline(pipeline, split.x_test, split.y_test)
    saved_path = save_model(
        pipeline=pipeline,
        model_path=args.model,
        algo=args.algo,
        metrics=metrics,
        train_samples=len(split.x_train),
    )

    print(f"Trained '{args.algo}' model on {len(split.x_train)} messages.")
    print(f"Saved model to: {saved_path}")
    print()
    print(_format_metrics(metrics))
    return 0


def _handle_predict(args: argparse.Namespace) -> int:
    """Run the `predict` subcommand."""
    pipeline = load_model(args.model)
    result = predict_message(pipeline, args.message)

    if args.json:
        payload = {
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "spam_probability": round(result.spam_probability, 4),
            "ham_probability": round(result.ham_probability, 4),
        }
        print(json.dumps(payload, indent=2))
    else:
        verdict = result.label.upper()
        print(
            f"Prediction: {verdict} ({result.confidence:.1%} confidence | "
            f"spam={result.spam_probability:.1%}, ham={result.ham_probability:.1%})"
        )
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    """Run the `evaluate` subcommand."""
    pipeline = load_model(args.model)
    messages, labels = load_dataset(args.data)
    metrics = evaluate_pipeline(pipeline, messages, labels)

    print(f"Evaluated model '{args.model}' on '{args.data}':")
    print()
    print(_format_metrics(metrics))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse CLI parser."""
    parser = argparse.ArgumentParser(
        prog="deuterium",
        description="Deuterium: Train and invoke an SMS spam detector from your terminal.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser(
        "train",
        help="Train the spam detection model on a dataset and save it to disk.",
    )
    train_parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to tab-separated dataset (default: {DEFAULT_DATASET_PATH.name}).",
    )
    train_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output path for the trained model (default: {DEFAULT_MODEL_PATH}).",
    )
    train_parser.add_argument(
        "--algo",
        choices=SUPPORTED_ALGOS,
        default="logreg",
        help="Classifier algorithm: 'logreg' (Logistic Regression) or 'nb' (Complement Naive Bayes).",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing (default: 0.2).",
    )
    train_parser.set_defaults(handler=_handle_train)

    # --- predict ---
    predict_parser = subparsers.add_parser(
        "predict",
        help="Classify a single message as SPAM or HAM.",
    )
    predict_parser.add_argument(
        "message",
        type=str,
        help="The message text to classify (wrap in quotes).",
    )
    predict_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model file (default: {DEFAULT_MODEL_PATH}).",
    )
    predict_parser.add_argument(
        "--json",
        action="store_true",
        help="Output prediction as JSON instead of human-readable text.",
    )
    predict_parser.set_defaults(handler=_handle_predict)

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved model against a labeled dataset.",
    )
    eval_parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to tab-separated dataset (default: {DEFAULT_DATASET_PATH.name}).",
    )
    eval_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model file (default: {DEFAULT_MODEL_PATH}).",
    )
    eval_parser.set_defaults(handler=_handle_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns process exit code (0 on success, 1 on error)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
