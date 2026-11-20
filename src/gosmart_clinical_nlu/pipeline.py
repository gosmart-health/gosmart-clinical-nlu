"""Build, train, and persist the per-intent spaCy pipelines.

Each intent gets a self-contained one-vs-rest pipeline: blank ``en`` plus
``entity_ruler`` (that intent's slot patterns) plus ``textcat`` with two
labels (the intent's label and :data:`NEGATIVE_LABEL`). Trained pipelines are
packed into a single ZIP bundle (one ``to_disk()`` tree per intent plus a
manifest) that :func:`load_intent_models` restores with ``spacy.load()``.
"""

import tempfile
import warnings
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from random import Random

import spacy
from pydantic import BaseModel
from spacy.language import Language
from spacy.training import Example
from spacy.util import fix_random_seed

from gosmart_clinical_nlu.intents import IntentSpec, get_intent_spec
from gosmart_clinical_nlu.training import (
    IntentTrainingSpec,
    TrainingCorpus,
    TrainingDataError,
    to_ruler_pattern,
)

NEGATIVE_LABEL = "__NEGATIVE__"
MANIFEST_FILENAME = "manifest.json"
PACKAGE_DISTRIBUTION = "gosmart-clinical-nlu"


@dataclass(frozen=True)
class TrainedIntent:
    """One trained intent pipeline plus its effective spec."""

    spec: IntentSpec
    nlp: Language


@dataclass(frozen=True)
class IntentModels(Mapping[str, TrainedIntent]):
    """Immutable name-to-pipeline mapping for a trained model set."""

    models: Mapping[str, TrainedIntent]
    content_hash: str
    spacy_version: str

    def __getitem__(self, key: str) -> TrainedIntent:
        return self.models[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    @property
    def names(self) -> tuple[str, ...]:
        """All intent names, in registration order."""
        return tuple(self.models)


class BundleIntentEntry(BaseModel):
    """Per-intent entry of the bundle manifest."""

    name: str
    label: str
    description: str = ""
    min_confidence: float
    required_fields: list[str] = []


class BundleManifest(BaseModel):
    """Manifest embedded in every bundle file."""

    schema_version: int = 1
    package_version: str
    spacy_version: str
    created_utc: str
    content_hash: str
    intents: list[BundleIntentEntry]


def _package_version() -> str:
    try:
        return metadata.version(PACKAGE_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return "unknown"


def _one_vs_rest_examples(
    spec: IntentSpec,
    training: IntentTrainingSpec,
    all_training: Sequence[IntentTrainingSpec],
) -> list[tuple[str, dict[str, float]]]:
    """Build the (text, cats) pairs for one intent's one-vs-rest training."""
    pairs = [
        (text, {spec.label: 1.0, NEGATIVE_LABEL: 0.0})
        for text in training.examples.positive
    ]
    negatives = list(training.examples.negative)
    for other in all_training:
        if other.name != training.name:
            negatives.extend(other.examples.positive)
    pairs.extend((text, {spec.label: 0.0, NEGATIVE_LABEL: 1.0}) for text in negatives)
    return pairs


def build_intent_pipeline(
    spec: IntentSpec,
    training: IntentTrainingSpec,
    *,
    all_training: Sequence[IntentTrainingSpec],
    epochs: int,
    seed: int,
) -> TrainedIntent:
    """Train one intent's one-vs-rest pipeline deterministically.

    The EntityRuler patterns are added AFTER training: ``EntityRuler.initialize()``
    (invoked through ``nlp.initialize``) clears any patterns added beforehand,
    which silently produced empty ``doc.ents`` (spaCy warning W036).
    """
    fix_random_seed(seed)
    nlp = spacy.blank("en")
    nlp.add_pipe("entity_ruler")
    textcat = nlp.add_pipe("textcat")
    textcat.add_label(spec.label)
    textcat.add_label(NEGATIVE_LABEL)

    pairs = _one_vs_rest_examples(spec, training, all_training)
    examples = [
        Example.from_dict(nlp.make_doc(text), {"cats": cats}) for text, cats in pairs
    ]

    rng = Random(seed)
    optimizer = nlp.initialize(lambda: list(examples))
    for _ in range(epochs):
        ordered = list(examples)
        rng.shuffle(ordered)
        for example in ordered:
            nlp.update([example], sgd=optimizer)

    ruler = nlp.get_pipe("entity_ruler")
    ruler.add_patterns([to_ruler_pattern(pattern) for pattern in training.patterns])
    return TrainedIntent(spec=spec, nlp=nlp)


def train_intent_models(
    corpus: TrainingCorpus, *, epochs: int | None = None, seed: int | None = None
) -> IntentModels:
    """Train every intent in ``corpus``; ``epochs``/``seed`` override the config."""
    epochs = corpus.config.epochs if epochs is None else epochs
    seed = corpus.config.seed if seed is None else seed
    if epochs < 1:
        raise TrainingDataError(f"epochs must be >= 1 (got {epochs})")
    trained: dict[str, TrainedIntent] = {}
    for training in sorted(corpus.intents, key=lambda item: item.name):
        try:
            base_spec = get_intent_spec(training.name)
        except KeyError as error:
            raise TrainingDataError(str(error)) from None
        spec = base_spec.with_overrides(
            description=training.description,
            min_confidence=training.min_confidence,
            required_fields=tuple(training.required_fields),
        )
        trained[spec.name] = build_intent_pipeline(
            spec, training, all_training=corpus.intents, epochs=epochs, seed=seed
        )
    return IntentModels(
        models=trained,
        content_hash=corpus.content_hash(),
        spacy_version=spacy.__version__,
    )


def save_bundle(models: IntentModels, out: Path | str) -> Path:
    """Write ``models`` into a single-file ZIP bundle at ``out``."""
    out = Path(out)
    manifest = _build_manifest(models)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        for name in sorted(models.names):
            models[name].nlp.to_disk(stage / name)
        (stage / MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
            for file_path in sorted(stage.rglob("*")):
                if file_path.is_file():
                    bundle.write(
                        file_path, arcname=file_path.relative_to(stage).as_posix()
                    )
    return out


def load_intent_models(path: Path | str) -> IntentModels:
    """Load a single-file bundle written by :func:`save_bundle`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model bundle not found: {path}")
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(path) as bundle:
            bundle.extractall(temp)
        stage = Path(temp)
        manifest_path = stage / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise TrainingDataError(
                f"{path}: not an intent bundle ({MANIFEST_FILENAME} missing)"
            )
        manifest = BundleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        _warn_on_spacy_mismatch(manifest)
        models: dict[str, TrainedIntent] = {}
        for entry in manifest.intents:
            try:
                base_spec = get_intent_spec(entry.name)
            except KeyError as error:
                raise TrainingDataError(
                    f"{path}: bundle intent {entry.name!r} is not registered in this "
                    f"gosmart-clinical-nlu version ({error})"
                ) from None
            spec = base_spec.with_overrides(
                description=entry.description,
                min_confidence=entry.min_confidence,
                required_fields=tuple(entry.required_fields),
            )
            models[entry.name] = TrainedIntent(
                spec=spec, nlp=spacy.load(stage / entry.name)
            )
    if not models:
        raise TrainingDataError(f"{path}: bundle contains no intents")
    return IntentModels(
        models=models,
        content_hash=manifest.content_hash,
        spacy_version=manifest.spacy_version,
    )


def _build_manifest(models: IntentModels) -> BundleManifest:
    entries = [
        BundleIntentEntry(
            name=models[name].spec.name,
            label=models[name].spec.label,
            description=models[name].spec.description,
            min_confidence=models[name].spec.min_confidence,
            required_fields=list(models[name].spec.required_fields),
        )
        for name in sorted(models.names)
    ]
    return BundleManifest(
        package_version=_package_version(),
        spacy_version=spacy.__version__,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        content_hash=models.content_hash,
        intents=entries,
    )


def _warn_on_spacy_mismatch(manifest: BundleManifest) -> None:
    saved = manifest.spacy_version.split(".")[:2]
    current = spacy.__version__.split(".")[:2]
    if saved != current:
        warnings.warn(
            f"intent bundle was trained with spaCy {manifest.spacy_version} but spaCy "
            f"{spacy.__version__} is running; serialized models are only guaranteed "
            f"compatible within minor versions",
            stacklevel=2,
        )
