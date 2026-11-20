"""Tests for parse_utterance: dispatch, slot extraction, and the failover signal."""

from datetime import datetime
from pathlib import Path

import pytest

from gosmart_clinical_nlu.intents import ReportQueryIntent, StudyQueryIntent
from gosmart_clinical_nlu.parsing import parse_utterance
from gosmart_clinical_nlu.pipeline import train_intent_models
from gosmart_clinical_nlu.training import load_training_dir

REF_DATE = datetime(2026, 11, 19)


def test_dispatches_find_study_and_extracts_slots(models) -> None:
    result = parse_utterance(
        "Find a CT patient 12345678 from yesterday", models, ref_date=REF_DATE
    )
    assert result.intent == "find_study"
    assert result.status == "matched"
    assert isinstance(result.data, StudyQueryIntent)
    assert result.data.modality == "CT"
    assert result.data.patient_id == "12345678"
    assert result.data.study_date == "20261118"


def test_normalizes_modality_and_relative_dates(models) -> None:
    result = parse_utterance(
        "Show me MRI scans for mrn 98765432 last week", models, ref_date=REF_DATE
    )
    assert result.intent == "find_study"
    assert result.data.modality == "MR"
    assert result.data.patient_id == "98765432"
    assert result.data.study_date == "20261112"


def test_dispatches_get_report(models) -> None:
    result = parse_utterance(
        "Get the radiology report for patient 12345678", models, ref_date=REF_DATE
    )
    assert result.intent == "get_report"
    assert result.status == "matched"
    assert isinstance(result.data, ReportQueryIntent)
    assert result.data.patient_id == "12345678"


def test_candidates_cover_all_intents(models) -> None:
    result = parse_utterance("hello there", models, ref_date=REF_DATE)
    assert set(result.candidates) == {"find_study", "get_report"}
    assert result.intent in result.candidates
    assert result.threshold == pytest.approx(0.6)


def test_low_confidence_signal(tmp_path: Path, training_dir_factory) -> None:
    """A threshold of 1.0 can never be reached, so the failover signal fires
    while best-effort slots are still returned."""
    corpus = load_training_dir(training_dir_factory(min_confidence=1.0))
    trained = train_intent_models(corpus, epochs=50)
    result = parse_utterance(
        "Find a CT patient 12345678 from yesterday", trained, ref_date=REF_DATE
    )
    assert result.status == "low_confidence"
    assert result.threshold == pytest.approx(1.0)
    assert result.data is not None  # best-effort slots are still returned
    assert result.missing_required == []


def test_missing_required_field(tmp_path: Path, training_dir_factory) -> None:
    corpus = load_training_dir(training_dir_factory(required_fields=["patient_id"]))
    trained = train_intent_models(corpus, epochs=50)
    result = parse_utterance("hello there", trained, ref_date=REF_DATE)
    assert result.status == "low_confidence"
    assert "patient_id" in result.missing_required


def test_empty_model_set_rejected() -> None:
    from gosmart_clinical_nlu.pipeline import IntentModels

    empty = IntentModels(models={}, content_hash="x", spacy_version="0.0.0")
    with pytest.raises(ValueError, match="no intents"):
        parse_utterance("hello", empty)
