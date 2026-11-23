"""Tests for the extensible intent registry, plugin discovery, and custom models."""

import json
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from conftest import write_training_dir
from pydantic import BaseModel, ValidationError

import gosmart_clinical_nlu.intents as intents_mod
from gosmart_clinical_nlu import (
    INTENT_ENTRY_POINT_GROUP,
    EntityBinding,
    IntentSpec,
    ParsedUtterance,
    discover_intent_specs,
    get_intent_spec,
    list_intent_specs,
    load_intent_models,
    load_training_dir,
    main,
    parse_utterance,
    register_intent_spec,
    save_bundle,
    train_intent_models,
    unregister_intent_spec,
)
from gosmart_clinical_nlu.intents.normalizers import normalize_patient_id


class OrderExamIntent(BaseModel):
    """External intent slot model simulating a peer-project definition."""

    exam_type: str | None = None
    patient_id: str | None = None
    priority: str | None = None


class CancelExamIntent(BaseModel):
    """Second external intent slot model for multi-intent plugin tests."""

    order_id: str | None = None


def _normalize_priority(raw: str, ref_date: datetime | None = None) -> str:
    return raw.strip().upper()


ORDER_EXAM_SPEC = IntentSpec(
    name="order_exam",
    label="ORDER_EXAM",
    model_cls=OrderExamIntent,
    entity_map={
        "EXAM_TYPE": EntityBinding(field="exam_type"),
        "PATIENT_ID": EntityBinding(field="patient_id", normalize=normalize_patient_id),
        "PRIORITY": EntityBinding(field="priority", normalize=_normalize_priority),
    },
    description="Place an imaging exam order.",
)

CANCEL_EXAM_SPEC = IntentSpec(
    name="cancel_exam",
    label="CANCEL_EXAM",
    model_cls=CancelExamIntent,
    entity_map={"ORDER_ID": EntityBinding(field="order_id")},
    description="Cancel an imaging exam order.",
)


@dataclass
class _FakeEntryPoint:
    name: str
    value: str
    target: Any
    group: str = INTENT_ENTRY_POINT_GROUP

    def load(self) -> Any:
        if isinstance(self.target, Exception):
            raise self.target
        return self.target


@pytest.fixture(autouse=True)
def isolated_registry():
    """Snapshot and restore global intent registry state around every test."""
    saved_by_name = dict(intents_mod._STATE.by_name)
    saved_loaded = intents_mod._STATE.entry_points_loaded
    saved_discovered = intents_mod._STATE.discovered_specs
    try:
        intents_mod._STATE.by_name.clear()
        intents_mod._STATE.by_name.update(
            {spec.name: spec for spec in intents_mod.INTENT_SPECS}
        )
        intents_mod._STATE.entry_points_loaded = False
        intents_mod._STATE.discovered_specs = ()
        yield
    finally:
        intents_mod._STATE.by_name.clear()
        intents_mod._STATE.by_name.update(saved_by_name)
        intents_mod._STATE.entry_points_loaded = saved_loaded
        intents_mod._STATE.discovered_specs = saved_discovered


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEntryPoint]
) -> None:
    def _fake_entry_points(*, group: str | None = None) -> list[_FakeEntryPoint]:
        if group is None:
            return list(eps)
        return [ep for ep in eps if ep.group == group]

    monkeypatch.setattr(metadata, "entry_points", _fake_entry_points)


def _write_training_with_order_exam(root: Path) -> Path:
    write_training_dir(root)
    order_dir = root / "order_exam"
    order_dir.mkdir()
    (order_dir / "intent.json").write_text(
        json.dumps(
            {
                "name": "order_exam",
                "label": "ORDER_EXAM",
                "description": "Order an exam",
            }
        ),
        encoding="utf-8",
    )
    (order_dir / "patterns.json").write_text(
        json.dumps(
            [
                {
                    "label": "EXAM_TYPE",
                    "pattern": [{"LOWER": {"IN": ["chest", "brain"]}}, {"LOWER": "ct"}],
                },
                {
                    "label": "PATIENT_ID",
                    "pattern": [
                        {"LOWER": {"IN": ["patient", "mrn"]}, "OP": "?"},
                        {"TEXT": {"REGEX": r"^\d{6,10}$"}},
                    ],
                },
                {
                    "label": "PRIORITY",
                    "pattern": [{"LOWER": {"IN": ["stat", "urgent", "routine"]}}],
                },
            ]
        ),
        encoding="utf-8",
    )
    (order_dir / "examples.json").write_text(
        json.dumps(
            {
                "positive": [
                    "Order a stat chest CT for patient 12345678",
                    "Place an urgent brain CT order for mrn 98765432",
                    "Schedule a routine chest CT order for patient 55551234",
                ],
                "negative": [],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_builtin_intents_listed_in_sorted_order() -> None:
    specs = list_intent_specs()
    assert [spec.name for spec in specs] == ["find_study", "get_report"]


def test_intent_spec_rejects_non_basemodel_class() -> None:
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="Pydantic BaseModel subclass"):
        IntentSpec(
            name="bad_intent",
            label="BAD_INTENT",
            model_cls=NotAModel,  # type: ignore[arg-type]
            entity_map={},
        )


def test_register_and_lookup_custom_intent() -> None:
    returned = register_intent_spec(ORDER_EXAM_SPEC)
    assert returned == ORDER_EXAM_SPEC
    assert get_intent_spec("order_exam") == ORDER_EXAM_SPEC
    assert "order_exam" in [spec.name for spec in list_intent_specs()]


def test_register_identical_spec_is_idempotent() -> None:
    register_intent_spec(ORDER_EXAM_SPEC)
    same_copy = IntentSpec(
        name=ORDER_EXAM_SPEC.name,
        label=ORDER_EXAM_SPEC.label,
        model_cls=ORDER_EXAM_SPEC.model_cls,
        entity_map=dict(ORDER_EXAM_SPEC.entity_map),
        description=ORDER_EXAM_SPEC.description,
    )
    assert register_intent_spec(same_copy) == ORDER_EXAM_SPEC


def test_register_conflicting_spec_rejected_without_replace() -> None:
    register_intent_spec(ORDER_EXAM_SPEC)
    conflicting = IntentSpec(
        name="order_exam",
        label="ORDER_EXAM",
        model_cls=CancelExamIntent,
        entity_map={},
    )
    with pytest.raises(ValueError, match="already registered"):
        register_intent_spec(conflicting)


def test_register_conflicting_spec_allowed_with_replace() -> None:
    register_intent_spec(ORDER_EXAM_SPEC)
    replacement = IntentSpec(
        name="order_exam",
        label="ORDER_EXAM",
        model_cls=CancelExamIntent,
        entity_map={},
    )
    assert register_intent_spec(replacement, replace=True) == replacement
    assert get_intent_spec("order_exam").model_cls is CancelExamIntent


def test_register_rejects_non_intent_spec() -> None:
    with pytest.raises(TypeError, match="expected an IntentSpec"):
        register_intent_spec("not-a-spec")  # type: ignore[arg-type]


def test_unregister_intent_spec() -> None:
    register_intent_spec(ORDER_EXAM_SPEC)
    removed = unregister_intent_spec("order_exam")
    assert removed == ORDER_EXAM_SPEC
    with pytest.raises(KeyError, match="unknown intent 'order_exam'"):
        get_intent_spec("order_exam")
    with pytest.raises(KeyError, match="unknown intent 'order_exam'"):
        unregister_intent_spec("order_exam")


def test_discover_entry_point_single_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("order_exam", "graphite.intents:ORDER", ORDER_EXAM_SPEC)],
    )
    spec = get_intent_spec("order_exam")
    assert spec == ORDER_EXAM_SPEC


def test_discover_entry_point_callable_returning_single_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "order_exam", "graphite.intents:make_order", lambda: ORDER_EXAM_SPEC
            )
        ],
    )
    assert get_intent_spec("order_exam") == ORDER_EXAM_SPEC


def test_discover_entry_point_callable_returning_multiple_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "graphite",
                "graphite.intents:get_specs",
                lambda: [ORDER_EXAM_SPEC, CANCEL_EXAM_SPEC],
            )
        ],
    )
    discovered = discover_intent_specs()
    assert discovered == (ORDER_EXAM_SPEC, CANCEL_EXAM_SPEC)
    assert get_intent_spec("order_exam") == ORDER_EXAM_SPEC
    assert get_intent_spec("cancel_exam") == CANCEL_EXAM_SPEC


def test_discover_entry_point_iterable_of_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "graphite",
                "graphite.intents:INTENT_SPECS",
                (ORDER_EXAM_SPEC, CANCEL_EXAM_SPEC),
            )
        ],
    )
    assert [spec.name for spec in list_intent_specs()] == [
        "cancel_exam",
        "find_study",
        "get_report",
        "order_exam",
    ]


def test_discover_entry_point_reentrant_import_time_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module that calls register_intent_spec on import AND exposes the spec
    as an entry point must neither deadlock nor raise a duplicate error."""

    def _simulate_import_and_return() -> IntentSpec:
        register_intent_spec(ORDER_EXAM_SPEC)
        return ORDER_EXAM_SPEC

    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "order_exam",
                "graphite.intents:ORDER",
                _simulate_import_and_return,
            )
        ],
    )
    assert get_intent_spec("order_exam") == ORDER_EXAM_SPEC


def test_discover_entry_point_lazy_once_and_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _counting_entry_points(*, group: str | None = None) -> list[_FakeEntryPoint]:
        nonlocal calls
        calls += 1
        return [_FakeEntryPoint("order_exam", "graphite:ORDER", ORDER_EXAM_SPEC)]

    monkeypatch.setattr(metadata, "entry_points", _counting_entry_points)

    assert discover_intent_specs() == (ORDER_EXAM_SPEC,)
    assert discover_intent_specs() == (ORDER_EXAM_SPEC,)
    assert calls == 1

    assert discover_intent_specs(reload=True) == (ORDER_EXAM_SPEC,)
    assert calls == 2


def test_discover_entry_point_invalid_target_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pointing at the BaseModel class itself rather than an IntentSpec
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("bad_cls", "graphite:OrderExamIntent", OrderExamIntent)],
    )
    with pytest.raises(TypeError, match="entry point 'bad_cls'.*OrderExamIntent"):
        discover_intent_specs()

    # Iterable containing a non-IntentSpec item
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("bad_iter", "graphite:SPECS", [ORDER_EXAM_SPEC, 42])],
    )
    with pytest.raises(TypeError, match="entry point 'bad_iter'.*index 1: int"):
        discover_intent_specs()


def test_discover_entry_point_load_exception_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "broken",
                "graphite.missing:SPEC",
                ImportError("No module named 'graphite'"),
            )
        ],
    )
    with pytest.raises(RuntimeError, match="failed to load entry point 'broken'"):
        discover_intent_specs()


def test_discover_entry_point_conflict_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conflicting_builtin = IntentSpec(
        name="find_study",
        label="FIND_STUDY",
        model_cls=CancelExamIntent,
        entity_map={},
    )
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("hijack", "graphite:CONFLICT", conflicting_builtin)],
    )
    with pytest.raises(ValueError, match="entry point 'hijack'.*already registered"):
        discover_intent_specs()


def test_custom_intent_end_to_end_train_bundle_parse_and_serialize(
    tmp_path: Path,
) -> None:
    register_intent_spec(ORDER_EXAM_SPEC)
    corpus = load_training_dir(_write_training_with_order_exam(tmp_path / "training"))
    trained = train_intent_models(corpus, epochs=50)
    bundle = save_bundle(trained, tmp_path / "models.zip")
    loaded = load_intent_models(bundle)

    result = parse_utterance("Order a stat chest CT for patient 12345678", loaded)
    assert result.status == "matched"
    assert result.intent == "order_exam"
    assert isinstance(result.data, OrderExamIntent)
    assert result.data.exam_type == "chest CT"
    assert result.data.patient_id == "12345678"
    assert result.data.priority == "STAT"

    # Regression check: Pydantic v2 must serialize concrete subclass fields (not {})
    dumped = result.model_dump()
    assert dumped["data"] == {
        "exam_type": "chest CT",
        "patient_id": "12345678",
        "priority": "STAT",
    }
    json_payload = json.loads(result.model_dump_json())
    assert json_payload["data"] == dumped["data"]


def test_parsed_utterance_rejects_non_basemodel_data() -> None:
    with pytest.raises(ValidationError):
        ParsedUtterance(
            status="matched",
            intent="order_exam",
            confidence=0.95,
            threshold=0.6,
            candidates={"order_exam": 0.95},
            data="not-a-model",  # type: ignore[arg-type]
        )


def test_cli_end_to_end_with_entry_point_discovered_intent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("order_exam", "graphite.intents:ORDER", ORDER_EXAM_SPEC)],
    )
    assert main(["list-intents"]) == 0
    listed = str(capsys.readouterr().out)
    assert "find_study (built-in):" in listed
    assert "order_exam (external):" in listed
    assert "OrderExamIntent" in listed

    training_dir = _write_training_with_order_exam(tmp_path / "training")
    bundle = tmp_path / "bundle.zip"
    assert (
        main(
            [
                "train",
                "--training-dir",
                str(training_dir),
                "--out",
                str(bundle),
                "--epochs",
                "50",
            ]
        )
        == 0
    )
    assert "order_exam" in str(capsys.readouterr().out)

    assert main(["inspect", "--model", str(bundle)]) == 0
    inspected = str(capsys.readouterr().out)
    assert "order_exam: label=ORDER_EXAM" in inspected

    assert (
        main(
            [
                "parse",
                "--model",
                str(bundle),
                "Order a stat chest CT for patient 12345678",
            ]
        )
        == 0
    )
    parsed = json.loads(str(capsys.readouterr().out))
    assert parsed["status"] == "matched"
    assert parsed["intent"] == "order_exam"
    assert parsed["data"] == {
        "exam_type": "chest CT",
        "patient_id": "12345678",
        "priority": "STAT",
    }


def test_cli_broken_entry_point_exits_with_code_1(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "broken_plugin",
                "graphite.bad:SPEC",
                ImportError("missing dependency"),
            )
        ],
    )
    assert main(["list-intents"]) == 1
    err = str(capsys.readouterr().err)
    assert "error:" in err
    assert "broken_plugin" in err
