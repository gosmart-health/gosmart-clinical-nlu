"""Command line interface: train, parse, and inspect intent models.

Used by developers and DevOps to produce the single-file model bundle from the
``training/`` directory and to smoke-test it. See ``docs/training.md`` for the
operational procedure.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path

from gosmart_clinical_nlu.parsing import parse_utterance
from gosmart_clinical_nlu.pipeline import (
    load_intent_models,
    save_bundle,
    train_intent_models,
)
from gosmart_clinical_nlu.training import TrainingDataError, load_training_dir

PACKAGE_DISTRIBUTION = "gosmart-clinical-nlu"
DEFAULT_TRAINING_DIR = "training"
DEFAULT_BUNDLE = "models/intent_models.zip"


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except (TrainingDataError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gosmart-clinical-nlu",
        description="Clinical NLU intent models: train, parse, inspect.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_package_version()}"
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train", help="train all intents and write the bundle"
    )
    train_parser.add_argument(
        "--training-dir", type=Path, default=Path(DEFAULT_TRAINING_DIR)
    )
    train_parser.add_argument("--out", type=Path, default=Path(DEFAULT_BUNDLE))
    train_parser.add_argument(
        "--epochs", type=int, default=None, help="override training config"
    )
    train_parser.add_argument(
        "--seed", type=int, default=None, help="override training config"
    )
    train_parser.set_defaults(func=_run_train)

    parse_parser = subparsers.add_parser(
        "parse", help="parse one utterance with a bundle"
    )
    parse_parser.add_argument("text", help="the user utterance")
    parse_parser.add_argument("--model", type=Path, default=Path(DEFAULT_BUNDLE))
    parse_parser.add_argument(
        "--ref-date",
        type=_ref_date,
        default=None,
        metavar="YYYY-MM-DD",
        help="reference date for relative date expressions",
    )
    parse_parser.set_defaults(func=_run_parse)

    inspect_parser = subparsers.add_parser(
        "inspect", help="show what is inside a bundle"
    )
    inspect_parser.add_argument("--model", type=Path, default=Path(DEFAULT_BUNDLE))
    inspect_parser.set_defaults(func=_run_inspect)
    return parser


def _ref_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid ref date {value!r}: expected YYYY-MM-DD"
        ) from error


def _package_version() -> str:
    try:
        return metadata.version(PACKAGE_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


def _run_train(args: argparse.Namespace) -> int:
    corpus = load_training_dir(args.training_dir)
    models = train_intent_models(corpus, epochs=args.epochs, seed=args.seed)
    out = save_bundle(models, args.out)
    print(f"trained {len(models.names)} intent(s): {', '.join(models.names)}")
    print(f"spaCy: {models.spacy_version}")
    print(f"content hash: {models.content_hash}")
    print(f"bundle: {out}")
    return 0


def _run_parse(args: argparse.Namespace) -> int:
    models = load_intent_models(args.model)
    result = parse_utterance(args.text, models, ref_date=args.ref_date)
    print(result.model_dump_json(indent=2))
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    models = load_intent_models(args.model)
    print(f"spaCy: {models.spacy_version}")
    print(f"content hash: {models.content_hash}")
    for name in models.names:
        trained = models[name]
        entity_labels = ", ".join(sorted(trained.spec.entity_map)) or "-"
        required = ", ".join(trained.spec.required_fields) or "-"
        print(
            f"{name}: label={trained.spec.label} "
            f"min_confidence={trained.spec.min_confidence} "
            f"required={required} entity_labels={entity_labels}"
        )
    return 0
