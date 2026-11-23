"""Utterance parsing: dispatch to the best intent pipeline.

:func:`parse_utterance` scores the utterance against every intent pipeline,
dispatches to the highest-scoring one, extracts its slots, and wraps the
result in a :class:`ParsedUtterance` envelope whose ``status`` and
``candidates`` fields are the failover signal for an LLM playbook.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, SerializeAsAny
from spacy.tokens import Doc

from gosmart_clinical_nlu.intents import IntentSpec
from gosmart_clinical_nlu.pipeline import IntentModels

IntentData = SerializeAsAny[BaseModel]


class ParsedUtterance(BaseModel):
    """Result envelope returned by :func:`parse_utterance`.

    ``status == "matched"`` means the top intent cleared its
    ``min_confidence`` and every required slot was extracted; anything else is
    ``low_confidence`` and the caller should consider failing over to the LLM
    playbook. ``data`` always carries the best-effort parsed slots of the top
    intent (any Pydantic :class:`~pydantic.BaseModel` subclass), and
    ``candidates`` carries every intent's score.
    """

    status: Literal["matched", "low_confidence"]
    intent: str
    confidence: float
    threshold: float
    candidates: dict[str, float]
    missing_required: list[str] = []
    data: IntentData | None = None


def parse_utterance(
    text: str, models: IntentModels, *, ref_date: datetime | None = None
) -> ParsedUtterance:
    """Score ``text`` against every intent pipeline and dispatch to the best.

    Ties are broken deterministically by intent name (lexicographically first
    wins). ``ref_date`` is the reference "now" for relative date expressions;
    it defaults to the current local time.
    """
    if len(models) == 0:
        raise ValueError("models contains no intents")
    scores: dict[str, float] = {}
    docs: dict[str, Doc] = {}
    for name in models.names:
        trained = models[name]
        doc = trained.nlp(text)
        scores[name] = float(doc.cats.get(trained.spec.label, 0.0))
        docs[name] = doc
    best = max(sorted(scores), key=lambda name: scores[name])
    trained = models[best]
    data = _extract_slots(trained.spec, docs[best], ref_date)
    missing = [
        field for field in trained.spec.required_fields if getattr(data, field) is None
    ]
    confidence = scores[best]
    matched = confidence >= trained.spec.min_confidence and not missing
    return ParsedUtterance(
        status="matched" if matched else "low_confidence",
        intent=best,
        confidence=round(confidence, 4),
        threshold=trained.spec.min_confidence,
        candidates={name: round(score, 4) for name, score in sorted(scores.items())},
        missing_required=missing,
        data=data,
    )


def _extract_slots(spec: IntentSpec, doc: Doc, ref_date: datetime | None) -> BaseModel:
    """Build the intent's pydantic model from the matched entities."""
    slots: dict[str, Any] = {}
    for ent in doc.ents:
        binding = spec.entity_map.get(ent.label_)
        if binding is None:
            continue
        if slots.get(binding.field) is not None:
            continue  # leftmost match wins
        value = binding.normalize(ent.text, ref_date) if binding.normalize else ent.text
        if value is None:
            continue
        slots[binding.field] = value
    return spec.model_cls(**slots)
