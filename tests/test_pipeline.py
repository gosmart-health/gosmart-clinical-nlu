"""Tests for pipeline building and training."""

import json
from pathlib import Path

import pytest
from conftest import write_training_dir

from gosmart_clinical_nlu.pipeline import NEGATIVE_LABEL, train_intent_models
from gosmart_clinical_nlu.training import TrainingDataError, load_training_dir

TEXT = "Find a CT patient 12345678 from yesterday"


def test_patterns_survive_training(models) -> None:
    """Regression: patterns added after training must still match (spaCy W036)."""
    doc = models["find_study"].nlp(TEXT)
    labels = {ent.label_ for ent in doc.ents}
    assert {"MODALITY", "PATIENT_ID", "REL_DATE"} <= labels


def test_textcat_has_one_vs_rest_labels(models) -> None:
    for name in models.names:
        textcat = models[name].nlp.get_pipe("textcat")
        expected = {models[name].spec.label, NEGATIVE_LABEL}
        assert set(textcat.labels) == expected


def test_training_is_deterministic(corpus) -> None:
    first = train_intent_models(corpus, epochs=10)
    second = train_intent_models(corpus, epochs=10)
    for name in first.names:
        assert first[name].nlp(TEXT).cats == second[name].nlp(TEXT).cats


def test_unknown_intent_in_training_data(tmp_path: Path) -> None:
    root = write_training_dir(tmp_path)
    bogus = root / "bogus"
    bogus.mkdir()
    (bogus / "intent.json").write_text(
        json.dumps({"name": "bogus", "label": "BOGUS"}), encoding="utf-8"
    )
    (bogus / "examples.json").write_text(
        json.dumps({"positive": ["hello world"]}), encoding="utf-8"
    )
    corpus = load_training_dir(root)
    with pytest.raises(TrainingDataError, match="unknown intent"):
        train_intent_models(corpus, epochs=2)


def test_epochs_override_validated(corpus) -> None:
    with pytest.raises(TrainingDataError, match="epochs must be >= 1"):
        train_intent_models(corpus, epochs=0)
