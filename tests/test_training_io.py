"""Tests for the training-data schemas and loaders."""

import json
from pathlib import Path

import pytest
from conftest import FIND_EXAMPLES, PATTERNS, REPORT_EXAMPLES, write_training_dir

from gosmart_clinical_nlu.training import TrainingDataError, load_training_dir


def test_per_intent_layout_loads(training_dir: Path, corpus) -> None:
    names = [item.name for item in corpus.intents]
    assert names == ["find_study", "get_report"]  # sorted by name
    assert (corpus.config.epochs, corpus.config.seed) == (20, 0)  # repo seed config
    find_study = next(item for item in corpus.intents if item.name == "find_study")
    assert find_study.label == "FIND_STUDY"
    assert find_study.patterns[0].label == "MODALITY"
    assert find_study.examples.positive


def test_aggregated_layout_loads(tmp_path: Path) -> None:
    intents = [
        {
            "name": "find_study",
            "label": "FIND_STUDY",
            "description": "study search",
            "patterns": PATTERNS,
            "examples": {"positive": FIND_EXAMPLES},
        },
        {
            "name": "get_report",
            "label": "GET_REPORT",
            "description": "report fetch",
            "patterns": PATTERNS,
            "examples": {"positive": REPORT_EXAMPLES},
        },
    ]
    payload = {"config": {"epochs": 5, "seed": 1}, "intents": intents}
    (tmp_path / "intents.json").write_text(json.dumps(payload), encoding="utf-8")
    corpus = load_training_dir(tmp_path)
    assert corpus.config.epochs == 5
    assert corpus.config.seed == 1
    assert [item.name for item in corpus.intents] == ["find_study", "get_report"]
    assert corpus.intents[0].patterns[0].label == "MODALITY"


def test_both_layouts_normalize_identically(tmp_path: Path, training_dir: Path) -> None:
    per_intent = load_training_dir(training_dir)
    payload = {
        "config": {"epochs": 20, "seed": 0, "default_min_confidence": 0.6},
        "intents": [item.model_dump(mode="json") for item in per_intent.intents],
    }
    (tmp_path / "intents.json").write_text(json.dumps(payload), encoding="utf-8")
    aggregated = load_training_dir(tmp_path)
    assert aggregated.content_hash() == per_intent.content_hash()


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(TrainingDataError, match="not found"):
        load_training_dir(tmp_path / "nope")


def test_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(TrainingDataError, match="no intents found"):
        load_training_dir(tmp_path)


def test_missing_examples_file(tmp_path: Path) -> None:
    root = tmp_path / "training"
    intent_dir = root / "find_study"
    intent_dir.mkdir(parents=True)
    (intent_dir / "intent.json").write_text(
        json.dumps({"name": "find_study", "label": "FIND_STUDY"}), encoding="utf-8"
    )
    with pytest.raises(TrainingDataError, match="missing examples file"):
        load_training_dir(root)


def test_directory_name_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "training"
    intent_dir = root / "wrong_dir"
    intent_dir.mkdir(parents=True)
    (intent_dir / "intent.json").write_text(
        json.dumps({"name": "find_study", "label": "FIND_STUDY"}), encoding="utf-8"
    )
    (intent_dir / "examples.json").write_text(
        json.dumps({"positive": ["hello"]}), encoding="utf-8"
    )
    with pytest.raises(TrainingDataError, match="does not match directory name"):
        load_training_dir(root)


def test_duplicate_labels(tmp_path: Path) -> None:
    """Duplicate labels are rejected (checked on the aggregated layout, where
    directory names do not already enforce unique intent names)."""
    intents = [
        {
            "name": "first",
            "label": "SAME_LABEL",
            "examples": {"positive": ["hello"]},
        },
        {
            "name": "second",
            "label": "SAME_LABEL",
            "examples": {"positive": ["world"]},
        },
    ]
    (tmp_path / "intents.json").write_text(
        json.dumps({"intents": intents}), encoding="utf-8"
    )
    with pytest.raises(TrainingDataError, match="duplicate label"):
        load_training_dir(tmp_path)


def test_no_positive_examples(tmp_path: Path) -> None:
    root = write_training_dir(tmp_path / "training", find_examples=[])
    with pytest.raises(TrainingDataError, match="no positive examples"):
        load_training_dir(root)


def test_invalid_json(tmp_path: Path) -> None:
    root = write_training_dir(tmp_path / "training")
    (root / "find_study" / "examples.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(TrainingDataError, match="invalid JSON"):
        load_training_dir(root)


def test_min_confidence_out_of_range(tmp_path: Path) -> None:
    root = write_training_dir(tmp_path / "training", min_confidence=1.5)
    with pytest.raises(TrainingDataError, match=r"within \[0, 1\]"):
        load_training_dir(root)


def test_content_hash_changes_with_data(tmp_path: Path) -> None:
    baseline = load_training_dir(write_training_dir(tmp_path / "a"))
    changed = load_training_dir(
        write_training_dir(
            tmp_path / "b", find_examples=[*FIND_EXAMPLES, "one more CT please"]
        )
    )
    assert baseline.content_hash() != changed.content_hash()
