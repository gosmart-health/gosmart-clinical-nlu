"""Shared fixtures: a small hermetic training corpus and models trained from it."""

import json
from pathlib import Path
from typing import Any

import pytest

from gosmart_clinical_nlu.pipeline import IntentModels, train_intent_models
from gosmart_clinical_nlu.training import TrainingCorpus, load_training_dir

PATTERNS = [
    {"label": "MODALITY", "pattern": [{"LOWER": {"IN": ["ct", "mri", "us"]}}]},
    {
        "label": "PATIENT_ID",
        "pattern": [
            {"LOWER": {"IN": ["patient", "mrn", "id"]}, "OP": "?"},
            {"TEXT": {"REGEX": r"^\d{6,10}$"}},
        ],
    },
    {"label": "REL_DATE", "pattern": [{"LOWER": {"IN": ["today", "yesterday"]}}]},
    {
        "label": "REL_DATE",
        "pattern": [
            {"LOWER": "last"},
            {
                "LOWER": {
                    "IN": [
                        "week",
                        "month",
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                    ]
                }
            },
        ],
    },
]

FIND_EXAMPLES = [
    "Find a CT patient 12345678 from yesterday",
    "Pull up yesterday's CT for 12345678",
    "Show me MRI scans for mrn 98765432 last week",
]

REPORT_EXAMPLES = [
    "Get the radiology report for patient 12345678",
    "Read impression on CT for 12345678",
    "Show me the report for mrn 98765432",
]


def write_training_dir(
    root: Path,
    *,
    config: dict[str, Any] | None = None,
    find_examples: list[str] | None = None,
    report_examples: list[str] | None = None,
    min_confidence: float | None = None,
    required_fields: list[str] | None = None,
) -> Path:
    """Create a per-intent training directory under ``root``."""
    meta_overrides: dict[str, Any] = {}
    if min_confidence is not None:
        meta_overrides["min_confidence"] = min_confidence
    if required_fields is not None:
        meta_overrides["required_fields"] = required_fields
    intents = [
        (
            "find_study",
            "FIND_STUDY",
            find_examples if find_examples is not None else FIND_EXAMPLES,
        ),
        (
            "get_report",
            "GET_REPORT",
            report_examples if report_examples is not None else REPORT_EXAMPLES,
        ),
    ]
    for name, label, examples in intents:
        intent_dir = root / name
        intent_dir.mkdir(parents=True)
        meta = {
            "name": name,
            "label": label,
            "description": f"{name} test intent",
            **meta_overrides,
        }
        (intent_dir / "intent.json").write_text(json.dumps(meta), encoding="utf-8")
        (intent_dir / "patterns.json").write_text(
            json.dumps(PATTERNS), encoding="utf-8"
        )
        (intent_dir / "examples.json").write_text(
            json.dumps({"positive": examples, "negative": []}), encoding="utf-8"
        )
    if config is not None:
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def training_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A training directory shared by the whole test session."""
    return write_training_dir(tmp_path_factory.mktemp("training"))


@pytest.fixture(scope="session")
def corpus(training_dir: Path) -> TrainingCorpus:
    return load_training_dir(training_dir)


@pytest.fixture(scope="session")
def models(corpus: TrainingCorpus) -> IntentModels:
    """Intent models trained once for the session (in-memory only)."""
    return train_intent_models(corpus, epochs=50)


@pytest.fixture
def training_dir_factory(tmp_path: Path):
    """Return a callable writing a fresh training directory under ``tmp_path``."""

    def _factory(**kwargs: Any) -> Path:
        return write_training_dir(tmp_path / "training", **kwargs)

    return _factory
