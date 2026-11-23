"""Intent registry: one self-contained spaCy pipeline per intent.

Each :class:`IntentSpec` ties together the training label, the Pydantic slot
model, and the mapping from EntityRuler labels to model fields. Built-in intents
live in :data:`INTENT_SPECS`; peer projects can register additional intents
programmatically via :func:`register_intent_spec` or declaratively via the
``gosmart_clinical_nlu.intents`` Python entry-point group.
"""

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from importlib import metadata

from pydantic import BaseModel

from gosmart_clinical_nlu.intents.base import EntityBinding
from gosmart_clinical_nlu.intents.find_study import ENTITY_MAP as FIND_STUDY_ENTITY_MAP
from gosmart_clinical_nlu.intents.find_study import StudyQueryIntent
from gosmart_clinical_nlu.intents.get_report import ENTITY_MAP as GET_REPORT_ENTITY_MAP
from gosmart_clinical_nlu.intents.get_report import ReportQueryIntent

DEFAULT_MIN_CONFIDENCE = 0.6
INTENT_ENTRY_POINT_GROUP = "gosmart_clinical_nlu.intents"


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
        if not (
            isinstance(self.model_cls, type) and issubclass(self.model_cls, BaseModel)
        ):
            raise TypeError(
                f"intent {self.name!r}: model_cls must be a Pydantic BaseModel "
                f"subclass (got {self.model_cls!r})"
            )
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


@dataclass
class _RegistryState:
    """Mutable process-wide intent registry state."""

    by_name: dict[str, IntentSpec] = field(
        default_factory=lambda: {spec.name: spec for spec in INTENT_SPECS}
    )
    entry_points_loaded: bool = False
    discovered_specs: tuple[IntentSpec, ...] = ()


_REGISTRY_LOCK = threading.RLock()
_STATE = _RegistryState()
_INTENT_BY_NAME: dict[str, IntentSpec] = _STATE.by_name


def register_intent_spec(spec: IntentSpec, *, replace: bool = False) -> IntentSpec:
    """Register an :class:`IntentSpec` so training, loading, and parsing recognize it.

    Re-registering an equal :class:`IntentSpec` under the same name is an
    idempotent no-op. Registering a different spec under an already-registered
    name raises :class:`ValueError` unless ``replace=True``.
    """
    if not isinstance(spec, IntentSpec):
        raise TypeError(f"expected an IntentSpec instance, got {type(spec).__name__}")
    with _REGISTRY_LOCK:
        if spec.name in _STATE.by_name:
            existing = _STATE.by_name[spec.name]
            if existing == spec:
                return existing
            if not replace:
                raise ValueError(
                    f"intent {spec.name!r} is already registered; "
                    f"pass replace=True to override"
                )
        _STATE.by_name[spec.name] = spec
        return spec


def unregister_intent_spec(name: str) -> IntentSpec:
    """Remove and return the :class:`IntentSpec` registered under ``name``."""
    with _REGISTRY_LOCK:
        try:
            return _STATE.by_name.pop(name)
        except KeyError:
            known = ", ".join(sorted(_STATE.by_name))
            raise KeyError(f"unknown intent {name!r} (registered: {known})") from None


def _iter_entry_point_specs(entry_point: metadata.EntryPoint) -> list[IntentSpec]:
    """Load one entry point and normalize its target to a list of IntentSpecs."""
    source = f"entry point {entry_point.name!r} ({entry_point.value})"
    try:
        loaded = entry_point.load()
    except Exception as error:
        raise RuntimeError(f"failed to load {source}: {error}") from error

    if callable(loaded) and not isinstance(loaded, type):
        try:
            loaded = loaded()
        except Exception as error:
            raise RuntimeError(f"failed to invoke {source}: {error}") from error

    if isinstance(loaded, IntentSpec):
        return [loaded]

    if isinstance(loaded, Iterable) and not isinstance(loaded, (str, bytes)):
        specs = list(loaded)
        for index, item in enumerate(specs):
            if not isinstance(item, IntentSpec):
                got = item.__name__ if isinstance(item, type) else type(item).__name__
                raise TypeError(
                    f"{source} yielded non-IntentSpec item at index {index}: {got}"
                )
        return specs

    got = loaded.__name__ if isinstance(loaded, type) else type(loaded).__name__
    raise TypeError(
        f"{source} must resolve to an IntentSpec, an iterable of IntentSpec, "
        f"or a zero-arg callable returning one (got {got})"
    )


def discover_intent_specs(*, reload: bool = False) -> tuple[IntentSpec, ...]:
    """Discover and register intents declared under :data:`INTENT_ENTRY_POINT_GROUP`.

    Discovery runs lazily at most once per process unless ``reload=True``.
    Returns the discovered :class:`IntentSpec` objects in deterministic order.
    """
    with _REGISTRY_LOCK:
        if _STATE.entry_points_loaded and not reload:
            return _STATE.discovered_specs
        discovered: list[IntentSpec] = []
        entry_points = metadata.entry_points(group=INTENT_ENTRY_POINT_GROUP)
        for ep in sorted(entry_points, key=lambda item: (item.name, item.value)):
            for spec in _iter_entry_point_specs(ep):
                try:
                    registered = register_intent_spec(spec)
                except ValueError as error:
                    raise ValueError(
                        f"entry point {ep.name!r} ({ep.value}) conflict: {error}"
                    ) from None
                discovered.append(registered)
        _STATE.discovered_specs = tuple(discovered)
        _STATE.entry_points_loaded = True
        return _STATE.discovered_specs


def get_intent_spec(name: str) -> IntentSpec:
    """Return the registered :class:`IntentSpec` for ``name``."""
    with _REGISTRY_LOCK:
        discover_intent_specs()
        try:
            return _STATE.by_name[name]
        except KeyError:
            known = ", ".join(sorted(_STATE.by_name))
            raise KeyError(f"unknown intent {name!r} (registered: {known})") from None


def list_intent_specs() -> tuple[IntentSpec, ...]:
    """Return all registered :class:`IntentSpec` objects, sorted by name."""
    with _REGISTRY_LOCK:
        discover_intent_specs()
        return tuple(_STATE.by_name[name] for name in sorted(_STATE.by_name))


__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "INTENT_ENTRY_POINT_GROUP",
    "INTENT_SPECS",
    "EntityBinding",
    "IntentSpec",
    "ReportQueryIntent",
    "StudyQueryIntent",
    "discover_intent_specs",
    "get_intent_spec",
    "list_intent_specs",
    "register_intent_spec",
    "unregister_intent_spec",
]
