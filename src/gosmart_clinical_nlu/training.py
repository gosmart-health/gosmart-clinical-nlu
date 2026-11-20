"""Training-data schemas and loaders for the intent models.

Two layouts are supported under a training directory:

* Per intent — ``<dir>/<intent_name>/intent.json`` plus ``patterns.json`` and
  ``examples.json``, and an optional top-level ``config.json``.
* Aggregated — a single ``<dir>/intents.json`` holding the config and every
  intent.

Both layouts normalize to :class:`TrainingCorpus`. All JSON is validated with
pydantic so malformed training data fails fast with a precise message.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from gosmart_clinical_nlu.intents import DEFAULT_MIN_CONFIDENCE

AGGREGATED_FILENAME = "intents.json"
META_FILENAME = "intent.json"
PATTERNS_FILENAME = "patterns.json"
EXAMPLES_FILENAME = "examples.json"
CONFIG_FILENAME = "config.json"

_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
_LABEL_PATTERN = r"^[A-Z][A-Z0-9_]*$"


class TrainingDataError(ValueError):
    """The training directory is missing, malformed, or inconsistent."""


class TrainingConfig(BaseModel):
    """Training knobs shared by all intents."""

    epochs: int = 20
    seed: int = 0
    default_min_confidence: float = DEFAULT_MIN_CONFIDENCE

    @field_validator("epochs")
    @classmethod
    def _validate_epochs(cls, value: int) -> int:
        if value < 1:
            raise ValueError("epochs must be >= 1")
        return value

    @field_validator("default_min_confidence")
    @classmethod
    def _validate_default_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("default_min_confidence must be within [0, 1]")
        return value


class TokenPatternSpec(BaseModel):
    """An EntityRuler token pattern (``{"label": ..., "pattern": [...]}``)."""

    label: str = Field(pattern=_LABEL_PATTERN)
    pattern: list[dict[str, Any]]


class PhrasePatternSpec(BaseModel):
    """An EntityRuler phrase pattern (``{"label": ..., "phrase_pattern": [...]}``)."""

    label: str = Field(pattern=_LABEL_PATTERN)
    phrase_pattern: list[str]


PatternSpec = TokenPatternSpec | PhrasePatternSpec

_PATTERN_ADAPTER = TypeAdapter(PatternSpec)


def to_ruler_pattern(spec: PatternSpec) -> dict[str, Any]:
    """Convert a pattern spec into an ``EntityRuler.add_patterns`` entry."""
    if isinstance(spec, TokenPatternSpec):
        return {"label": spec.label, "pattern": spec.pattern}
    return {"label": spec.label, "phrase_pattern": spec.phrase_pattern}


class ExamplesSpec(BaseModel):
    """Utterances for one intent; negatives are optional extras."""

    positive: list[str] = []
    negative: list[str] = []


class IntentTrainingSpec(BaseModel):
    """Everything needed to train one intent (matches ``intent.json`` fields).

    In the aggregated layout the same flat shape is used per entry of the
    top-level ``intents`` list, plus ``patterns`` and ``examples``.
    """

    name: str = Field(pattern=_NAME_PATTERN)
    label: str = Field(pattern=_LABEL_PATTERN)
    description: str = ""
    min_confidence: float | None = None
    required_fields: list[str] = []
    patterns: list[PatternSpec] = []
    examples: ExamplesSpec

    @field_validator("min_confidence")
    @classmethod
    def _validate_min_confidence(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        return value


class TrainingCorpus(BaseModel):
    """The validated training content of a training directory."""

    config: TrainingConfig = Field(default_factory=TrainingConfig)
    intents: list[IntentTrainingSpec]

    def content_hash(self) -> str:
        """Deterministic sha256 over the canonical training JSON."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_training_dir(path: Path | str) -> TrainingCorpus:
    """Load a training directory in either layout and validate it."""
    root = Path(path)
    if not root.is_dir():
        raise TrainingDataError(f"training directory not found: {root}")
    aggregated = root / AGGREGATED_FILENAME
    corpus = (
        _load_aggregated(aggregated) if aggregated.is_file() else _load_per_intent(root)
    )
    _validate_corpus(corpus, source=str(root))
    return corpus


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TrainingDataError(f"{path}: invalid JSON ({error})") from error


def _validated(model: Any, data: Any, source: str) -> Any:
    try:
        return TypeAdapter(model).validate_python(data)
    except ValidationError as error:
        raise TrainingDataError(f"{source}: {error}") from error


def _load_aggregated(path: Path) -> TrainingCorpus:
    data = _read_json(path)
    if not isinstance(data, dict) or "intents" not in data:
        raise TrainingDataError(f"{path}: expected an object with an 'intents' list")
    config = _validated(TrainingConfig, data.get("config", {}), str(path))
    intents = [
        _validated(IntentTrainingSpec, item, f"{path}#intents[{index}]")
        for index, item in enumerate(data["intents"])
    ]
    return TrainingCorpus(config=config, intents=intents)


def _load_per_intent(root: Path) -> TrainingCorpus:
    config = TrainingConfig()
    config_path = root / CONFIG_FILENAME
    if config_path.is_file():
        config = _validated(TrainingConfig, _read_json(config_path), str(config_path))
    intent_dirs = sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / META_FILENAME).is_file()
    )
    if not intent_dirs:
        raise TrainingDataError(
            f"{root}: no intents found - expected {AGGREGATED_FILENAME} or per-intent "
            f"directories containing {META_FILENAME}"
        )
    intents = []
    for intent_dir in intent_dirs:
        meta_path = intent_dir / META_FILENAME
        data = _read_json(meta_path)
        if not isinstance(data, dict):
            raise TrainingDataError(f"{meta_path}: expected a JSON object")
        data["patterns"] = _load_patterns(intent_dir / PATTERNS_FILENAME)
        examples_path = intent_dir / EXAMPLES_FILENAME
        if not examples_path.is_file():
            raise TrainingDataError(f"{examples_path}: missing examples file")
        data["examples"] = _read_json(examples_path)
        intents.append(_validated(IntentTrainingSpec, data, str(meta_path)))
        if intents[-1].name != intent_dir.name:
            raise TrainingDataError(
                f"{meta_path}: name {intents[-1].name!r} does not match directory name "
                f"{intent_dir.name!r}"
            )
    return TrainingCorpus(config=config, intents=intents)


def _load_patterns(path: Path) -> list[PatternSpec]:
    if not path.is_file():
        return []
    data = _read_json(path)
    if not isinstance(data, list):
        raise TrainingDataError(f"{path}: expected a list of patterns")
    return [
        _validated(PatternSpec, item, f"{path}[{index}]")
        for index, item in enumerate(data)
    ]


def _validate_corpus(corpus: TrainingCorpus, source: str) -> None:
    names = [item.name for item in corpus.intents]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise TrainingDataError(
            f"{source}: duplicate intent name(s): {duplicate_names}"
        )
    labels = [item.label for item in corpus.intents]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        raise TrainingDataError(f"{source}: duplicate label(s): {duplicate_labels}")
    for item in corpus.intents:
        if not item.examples.positive:
            raise TrainingDataError(
                f"{source}: intent {item.name!r} has no positive examples"
            )
