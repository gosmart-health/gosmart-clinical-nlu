"""Intent registry: one self-contained spaCy pipeline per intent.

Each :class:`IntentSpec` ties together the training label, the Pydantic slot
model, and the mapping from EntityRuler labels to model fields. Add a new
intent by creating a module in this package and appending an entry to
``INTENT_SPECS``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from pydantic import BaseModel

from gosmart_clinical_nlu.intents.base import EntityBinding
from gosmart_clinical_nlu.intents.find_study import ENTITY_MAP as FIND_STUDY_ENTITY_MAP
from gosmart_clinical_nlu.intents.find_study import StudyQueryIntent
from gosmart_clinical_nlu.intents.get_report import ENTITY_MAP as GET_REPORT_ENTITY_MAP
from gosmart_clinical_nlu.intents.get_report import ReportQueryIntent

DEFAULT_MIN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class IntentSpec:
    """Static definition of one intent (the code side of the training data)."""

    name: str
    label: str
    model_cls: type[BaseModel]
    entity_map: Mapping[str, EntityBinding]
    description: str = ""
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    required_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        model_fields = self.model_cls.model_fields
        unknown_required = [
            name for name in self.required_fields if name not in model_fields
        ]
        if unknown_required:
            raise ValueError(
                f"intent {self.name!r}: required fields missing on "
                f"{self.model_cls.__name__}: {unknown_required}"
            )
        unknown_bindings = [
            binding.field
            for binding in self.entity_map.values()
            if binding.field not in model_fields
        ]
        if unknown_bindings:
            raise ValueError(
                f"intent {self.name!r}: entity bindings reference missing fields on "
                f"{self.model_cls.__name__}: {unknown_bindings}"
            )

    def with_overrides(
        self,
        *,
        description: str | None = None,
        min_confidence: float | None = None,
        required_fields: tuple[str, ...] | None = None,
    ) -> IntentSpec:
        """Return a copy with training-data/manifest overrides applied."""
        return replace(
            self,
            description=description if description else self.description,
            min_confidence=min_confidence
            if min_confidence is not None
            else self.min_confidence,
            required_fields=required_fields
            if required_fields is not None
            else self.required_fields,
        )


INTENT_SPECS: tuple[IntentSpec, ...] = (
    IntentSpec(
        name="find_study",
        label="FIND_STUDY",
        model_cls=StudyQueryIntent,
        entity_map=FIND_STUDY_ENTITY_MAP,
        description="Search for imaging studies for a patient.",
    ),
    IntentSpec(
        name="get_report",
        label="GET_REPORT",
        model_cls=ReportQueryIntent,
        entity_map=GET_REPORT_ENTITY_MAP,
        description="Fetch the radiology report for a study.",
    ),
)

_INTENT_BY_NAME = {spec.name: spec for spec in INTENT_SPECS}


def get_intent_spec(name: str) -> IntentSpec:
    """Return the registered :class:`IntentSpec` for ``name``."""
    try:
        return _INTENT_BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(_INTENT_BY_NAME))
        raise KeyError(f"unknown intent {name!r} (registered: {known})") from None
