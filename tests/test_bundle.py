"""Tests for the single-file bundle save/load round trip."""

import json
import zipfile
from pathlib import Path

import pytest
import spacy
from conftest import write_training_dir
from pydantic import BaseModel

from gosmart_clinical_nlu.intents import (
    IntentSpec,
    register_intent_spec,
    unregister_intent_spec,
)
from gosmart_clinical_nlu.pipeline import (
    load_intent_models,
    save_bundle,
    train_intent_models,
)
from gosmart_clinical_nlu.training import TrainingDataError, load_training_dir

TEXT = "Find a CT patient 12345678 from yesterday"


def test_round_trip_predictions_equal(tmp_path: Path, models) -> None:
    out = save_bundle(models, tmp_path / "bundle.zip")
    loaded = load_intent_models(out)
    assert set(loaded.names) == set(models.names)
    for name in models.names:
        original = models[name].nlp(TEXT)
        restored = loaded[name].nlp(TEXT)
        assert original.cats == pytest.approx(restored.cats)
        assert [(e.text, e.label_) for e in original.ents] == [
            (e.text, e.label_) for e in restored.ents
        ]


def test_manifest_contents(tmp_path: Path, models) -> None:
    out = save_bundle(models, tmp_path / "bundle.zip")
    with zipfile.ZipFile(out) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        top_level = {arcname.split("/")[0] for arcname in bundle.namelist()}
    assert manifest["spacy_version"] == spacy.__version__
    assert manifest["content_hash"] == models.content_hash
    assert {entry["name"] for entry in manifest["intents"]} == set(models.names)
    assert {"manifest.json", "find_study", "get_report"} <= top_level
    find_study_entry = next(e for e in manifest["intents"] if e["name"] == "find_study")
    assert find_study_entry["label"] == "FIND_STUDY"
    assert find_study_entry["min_confidence"] == pytest.approx(0.6)


def test_bundle_is_single_file(tmp_path: Path, models) -> None:
    out = save_bundle(models, tmp_path / "bundle.zip")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_load_missing_bundle(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_intent_models(tmp_path / "nope.zip")


def test_load_non_bundle_zip(tmp_path: Path) -> None:
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as bundle:
        bundle.writestr("readme.txt", "not a bundle")
    with pytest.raises(Exception, match="not an intent bundle"):
        load_intent_models(empty)


def test_spacy_version_mismatch_warns(tmp_path: Path, models, monkeypatch) -> None:
    out = save_bundle(models, tmp_path / "bundle.zip")
    monkeypatch.setattr(spacy, "__version__", "9.9.9")
    with pytest.warns(UserWarning, match="trained with spaCy"):
        loaded = load_intent_models(out)
    assert loaded.spacy_version == models.spacy_version


def test_load_unregistered_bundle_intent_fails(tmp_path: Path) -> None:
    class EphemeralIntent(BaseModel):
        note: str | None = None

    spec = IntentSpec(
        name="ephemeral_intent",
        label="EPHEMERAL_INTENT",
        model_cls=EphemeralIntent,
        entity_map={},
    )
    register_intent_spec(spec)
    try:
        root = write_training_dir(tmp_path / "training")
        extra = root / "ephemeral_intent"
        extra.mkdir()
        (extra / "intent.json").write_text(
            json.dumps({"name": "ephemeral_intent", "label": "EPHEMERAL_INTENT"}),
            encoding="utf-8",
        )
        (extra / "examples.json").write_text(
            json.dumps({"positive": ["trigger ephemeral intent"]}),
            encoding="utf-8",
        )
        trained = train_intent_models(load_training_dir(root), epochs=5)
        bundle = save_bundle(trained, tmp_path / "bundle.zip")
    finally:
        unregister_intent_spec("ephemeral_intent")

    with pytest.raises(TrainingDataError, match="not registered.*register_intent_spec"):
        load_intent_models(bundle)
