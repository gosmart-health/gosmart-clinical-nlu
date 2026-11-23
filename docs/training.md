# Training and updating intents — operations guide

Audience: developers and DevOps who train, update, or deploy the clinical NLU
intent models. For the runtime API (parsing utterances, the failover contract)
see the project `README.md`; for architecture background see
`docs/design_notes/intent_nlu_design.md`.

## 1. Overview

- Each **intent** is one self-contained spaCy pipeline: a blank English
  pipeline + an `entity_ruler` (the intent's slot patterns) + a `textcat`
  classifier trained *one-vs-rest* (this intent vs everything else).
- All trained pipelines are packed into **one artifact file**
  (`models/intent_models.zip` by default): a ZIP containing one spaCy
  `to_disk()` directory per intent plus a `manifest.json`.
- Training input lives in `training/` as plain JSON. Changing training data for
  an existing intent never requires code changes; adding a *new* intent can be
  done **either from a peer project** (without modifying `gosmart-clinical-nlu`,
  via Python entry points or `register_intent_spec()`) **or as a built-in
  intent** in this repository, plus a training directory.

## 2. Prerequisites

```bash
uv sync        # creates .venv and installs the package + deps (editable)
```

Requires Python 3.14 (see `.python-version`). No model downloads, no network
access, no GPU: training a typical intent set takes seconds on a CPU.

## 3. Training data reference

### 3.1 Per-intent layout (recommended)

```
training/
├── config.json                  # optional global knobs
├── find_study/
│   ├── intent.json              # identity + thresholds
│   ├── patterns.json            # EntityRuler slot patterns
│   └── examples.json            # training utterances
└── get_report/
    ├── intent.json
    ├── patterns.json
    └── examples.json
```

`intent.json`:

| Field             | Type         | Default | Notes                                             |
|-------------------|--------------|---------|---------------------------------------------------|
| `name`            | string       | —       | `[a-z][a-z0-9_]*`; must match the directory name  |
| `label`           | string       | —       | `[A-Z][A-Z0-9_]*`; the textcat label; unique      |
| `description`     | string       | `""`    | Human-readable summary                            |
| `min_confidence`  | float | null | null    | Threshold for "matched"; null uses config default |
| `required_fields` | string[]     | `[]`    | Slot fields that must be extracted to be "matched"|

`patterns.json` — a JSON array of spaCy `EntityRuler` patterns:

```json
[
  {"label": "MODALITY", "pattern": [{"LOWER": {"IN": ["ct", "mri", "us"]}}]},
  {"label": "PATIENT_ID", "pattern": [
      {"LOWER": {"IN": ["patient", "mrn", "id"]}, "OP": "?"},
      {"TEXT": {"REGEX": "^\\d{6,10}$"}}
  ]}
]
```

Notes:

- Each pattern `label` must have a matching entry in the intent's entity map
  (code side) for its value to reach a slot field; unmatched labels are
  extracted but ignored at parse time.
- `"OP"` is a **token-level** key (sibling of `"LOWER"`/`"TEXT"`). Nesting it
  inside an attribute dict is rejected by spaCy (warning W035).
- Phrase patterns (`{"label": ..., "phrase_pattern": [...]}`) are also
  accepted.

`examples.json`:

```json
{
  "positive": ["Find a CT patient 12345678 from yesterday"],
  "negative": []
}
```

- `positive` (required, at least one): utterances of THIS intent.
- `negative` (optional): extra counter-examples for THIS intent. You normally
  leave it empty: training automatically uses every *other* intent's positive
  utterances as negatives (one-vs-rest). If a training corpus has only a single
  intent, provide explicit `negative` examples so the one-vs-rest classifier
  learns a boundary.

`config.json` (optional):

```json
{"epochs": 20, "seed": 0, "default_min_confidence": 0.6}
```

### 3.2 Aggregated layout (single file)

Alternatively, keep everything in `training/intents.json`:

```json
{
  "config": {"epochs": 20, "seed": 0, "default_min_confidence": 0.6},
  "intents": [
    {
      "name": "find_study",
      "label": "FIND_STUDY",
      "description": "Search for imaging studies.",
      "min_confidence": 0.6,
      "required_fields": [],
      "patterns": [ ... same shape as patterns.json ... ],
      "examples": {"positive": ["..."], "negative": []}
    }
  ]
}
```

If both `intents.json` and per-intent directories exist, `intents.json` wins.
Both layouts produce identical training results (verified by test).

### 3.3 Validation rules

The loader rejects (exit code 1) among others: missing/duplicate intent names
or labels, intent directories whose `name` does not match the directory name,
intents without positive examples, out-of-range `min_confidence`, malformed
JSON, broken plugin entry points, and unknown intent names (see §7).

## 4. Training

```bash
# Optional: check which built-in and peer-project intents are registered
uv run gosmart-clinical-nlu list-intents

# Train all intents under --training-dir and write the single-file bundle
uv run gosmart-clinical-nlu train [--training-dir ./training] [--out models/intent_models.zip] [--epochs N] [--seed N]
```

- `--training-dir` defaults to `./training`; `--out` defaults to
  `models/intent_models.zip` (the `models/` directory is git-ignored).
- `--epochs` / `--seed` override `config.json`.
- Before training, the CLI automatically discovers any external intents declared
  by installed peer packages under the `gosmart_clinical_nlu.intents` entry-point
  group (§7.1).
- Output: one line per step, ending with the artifact path and the
  **content hash** — a sha256 over the canonical training JSON. Two bundles
  with the same hash were trained from identical data; use it to correlate a
  deployed artifact with its source data.
- Exit codes: `0` success, `1` training-data, plugin, or I/O errors (message on
  stderr), `2` command-line usage errors.

Training is deterministic for a given `(training data, epochs, seed)`: the
same inputs produce the same model behavior.

## 5. Verifying a bundle

```bash
# Which intents are registered in this environment (built-in + plugins)?
uv run gosmart-clinical-nlu list-intents

# What is inside a trained bundle?
uv run gosmart-clinical-nlu inspect --model models/intent_models.zip

# Smoke-test one utterance (prints the full JSON result):
uv run gosmart-clinical-nlu parse --model models/intent_models.zip \
    "Show me MRI scans for mrn 98765432 last week" --ref-date 2026-11-19
```

`parse` prints the `ParsedUtterance` envelope. Check that:

- `intent` is the expected one and `status` is `"matched"`.
- `candidates` shows a healthy margin between the top intent and the rest.
- `data` carries the expected slots (e.g. `modality: "MR"` for "MRI",
  `study_date` as a DICOM `YYYYMMDD` string, or all fields of your custom
  Pydantic intent model).

## 6. Updating an existing intent (no code changes)

1. Edit the intent's `examples.json` (add/remove utterances), `patterns.json`
   (slot rules), or `intent.json` (`min_confidence`, `required_fields`).
2. Re-run training (§4). Every intent is re-trained into a fresh bundle;
   there is no incremental state to preserve.
3. Verify with §5, then distribute the new bundle file.

## 7. Adding a new intent (code + data)

There are two ways to add a new intent:

- **§7.1 From a peer project** (e.g., `gosmart-graphite`) without modifying
  `gosmart-clinical-nlu` — recommended for application-specific intents.
- **§7.2 Built-in inside `gosmart-clinical-nlu`** — for core clinical intents
  shared by every consumer.

### 7.1 Adding a new intent from a peer project (recommended)

#### Step 1: Define the Pydantic slot model, entity map, and `IntentSpec`

Create a module in your peer project (for example,
`src/gosmart_graphite/intents/order_exam.py`). Any Pydantic `BaseModel`
subclass is supported in `ParsedUtterance.data` (and serializes all of its
fields cleanly via `model_dump()` / `model_dump_json()`):

```python
from datetime import datetime
from pydantic import BaseModel

from gosmart_clinical_nlu import EntityBinding, IntentSpec
from gosmart_clinical_nlu.intents.normalizers import (
    normalize_modality,
    normalize_patient_id,
)


class OrderExamIntent(BaseModel):
    """Slots for placing an imaging exam order."""

    modality: str | None = None
    patient_id: str | None = None
    priority: str | None = None


def normalize_priority(raw: str, ref_date: datetime | None = None) -> str:
    return raw.strip().upper()


ORDER_EXAM_SPEC = IntentSpec(
    name="order_exam",
    label="ORDER_EXAM",
    model_cls=OrderExamIntent,
    entity_map={
        "MODALITY": EntityBinding(field="modality", normalize=normalize_modality),
        "PATIENT_ID": EntityBinding(field="patient_id", normalize=normalize_patient_id),
        "PRIORITY": EntityBinding(field="priority", normalize=normalize_priority),
    },
    description="Place an imaging exam order for a patient.",
)
```

Normalizer signature: `(raw: str, ref_date: datetime | None = None) -> Any`.
Omit `normalize` on `EntityBinding(field="...")` to pass matched text verbatim.
Returning `None` skips the entity match.

#### Step 2: Register the intent

Choose **Option A** (declarative entry points, recommended so the CLI works
automatically) or **Option B** (programmatic registration):

**Option A — `pyproject.toml` entry point (recommended for CLI + runtime):**

Declare the intent under `[project.entry-points."gosmart_clinical_nlu.intents"]`
in the peer project's `pyproject.toml`:

```toml
[project.entry-points."gosmart_clinical_nlu.intents"]
order_exam = "gosmart_graphite.intents.order_exam:ORDER_EXAM_SPEC"
```

Entry-point targets may resolve to any of three forms:

1. A single `IntentSpec` instance (`pkg.mod:ORDER_EXAM_SPEC`).
2. A zero-argument callable returning an `IntentSpec` or an iterable of
   `IntentSpec`s (`pkg.mod:get_intent_specs`).
3. An iterable (tuple/list) of `IntentSpec` instances (`pkg.mod:INTENT_SPECS`).

After editing `pyproject.toml`, reinstall the peer package metadata and verify
discovery:

```bash
uv sync
uv run gosmart-clinical-nlu list-intents
```

You should see your intent listed as `order_exam (external): ...` alongside the
built-in intents.

**Option B — Programmatic registration (scripts, notebooks, tests):**

```python
from gosmart_clinical_nlu import register_intent_spec
from gosmart_graphite.intents.order_exam import ORDER_EXAM_SPEC

register_intent_spec(ORDER_EXAM_SPEC)
```

Notes:
- Re-registering an equal `IntentSpec` is an idempotent no-op (safe if a module
  both calls `register_intent_spec()` on import and exposes an entry point).
- Registering a different spec under an existing name raises `ValueError`
  unless `replace=True` is passed.
- `unregister_intent_spec(name)` removes a spec (useful in test teardown).

#### Step 3: Create the training data

Create a training directory in your peer project (for example, `training/`).
Include a subdirectory for your new intent (or add an entry to `intents.json`
per §3.2):

```
training/
├── config.json              # optional
├── find_study/              # include if you want built-in intents in the same bundle
├── get_report/
└── order_exam/
    ├── intent.json
    ├── patterns.json
    └── examples.json
```

`training/order_exam/intent.json`:

```json
{
  "name": "order_exam",
  "label": "ORDER_EXAM",
  "description": "Place an imaging exam order for a patient.",
  "min_confidence": 0.6,
  "required_fields": []
}
```

`training/order_exam/patterns.json`:

```json
[
  {"label": "MODALITY", "pattern": [{"LOWER": {"IN": ["ct", "mri", "us", "ultrasound"]}}]},
  {"label": "PATIENT_ID", "pattern": [
      {"LOWER": {"IN": ["patient", "mrn", "id"]}, "OP": "?"},
      {"TEXT": {"REGEX": "^\\d{6,10}$"}}
  ]},
  {"label": "PRIORITY", "pattern": [{"LOWER": {"IN": ["stat", "urgent", "routine"]}}]}
]
```

`training/order_exam/examples.json`:

```json
{
  "positive": [
    "Order a stat CT for patient 12345678",
    "Place an urgent MRI order for mrn 98765432",
    "Schedule a routine ultrasound for patient 55551234"
  ],
  "negative": []
}
```

> **One-vs-rest tip:** Every intent in `--training-dir` contributes its
> `positive` examples as `negative` examples for all *other* intents in that
> directory. If your peer project wants one unified bundle covering both
> built-in intents and custom intents, include all of them in the same training
> directory. If you train a bundle with only a single custom intent, add
> explicit `negative` examples in `examples.json`.

#### Step 4: Train, verify, and parse

With entry points (Option A), run the standard CLI inside the peer project's
environment:

```bash
uv run gosmart-clinical-nlu train \
    --training-dir ./training \
    --out models/intent_models.zip

uv run gosmart-clinical-nlu inspect --model models/intent_models.zip

uv run gosmart-clinical-nlu parse --model models/intent_models.zip \
    "Order a stat CT for patient 12345678"
```

Or train and parse from Python (works with both Option A and Option B):

```python
from gosmart_clinical_nlu import (
    load_intent_models,
    load_training_dir,
    parse_utterance,
    save_bundle,
    train_intent_models,
)
from gosmart_graphite.intents.order_exam import OrderExamIntent

corpus = load_training_dir("training")
models = train_intent_models(corpus)
save_bundle(models, "models/intent_models.zip")

loaded = load_intent_models("models/intent_models.zip")
result = parse_utterance("Order a stat CT for patient 12345678", loaded)
if result.status == "matched" and isinstance(result.data, OrderExamIntent):
    ...  # result.data.modality == "CT", result.data.priority == "STAT"
```

### 7.2 Adding a built-in intent inside `gosmart-clinical-nlu`

For shared clinical intents maintained directly in this repository:

1. Create `src/gosmart_clinical_nlu/intents/<name>.py` with the intent's
   Pydantic slot model and an `ENTITY_MAP` mapping EntityRuler labels to model
   fields (optionally with normalizers; see `find_study.py`).
2. Append an `IntentSpec` entry to `INTENT_SPECS` in
   `src/gosmart_clinical_nlu/intents/__init__.py`.
3. Create `training/<name>/` with `intent.json`, `patterns.json`, and
   `examples.json` (§3.1).
4. Train and verify (§4–§5). Existing intents keep working; only the new
   intent is added to the bundle.

Forgetting step 2 (or failing to register a peer-project intent) fails training
with `unknown intent '<name>'`.

## 8. Deploying the bundle

The bundle is a single file — copy it and point the peer project at it:

```python
from gosmart_clinical_nlu import load_intent_models, parse_utterance

models = load_intent_models("models/intent_models.zip")  # once per process
result = parse_utterance(user_text, models, ref_date=today)
if result.status == "low_confidence":
    ...  # hand off to the LLM playbook, passing result.candidates for context
```

- Load **once per process** at startup; `IntentModels` is an immutable,
  read-only mapping.
- If the bundle contains **external (peer-project) intents**, the serving
  environment must have those intents registered before calling
  `load_intent_models()` — either automatically via installed package entry
  points (`gosmart_clinical_nlu.intents`) or via `register_intent_spec()`.
- `ParsedUtterance.data` holds the concrete `BaseModel` subclass for the matched
  intent and serializes all of its fields in `model_dump()` and
  `model_dump_json()`.
- The manifest records the spaCy version used for training. Loading a bundle
  with a different spaCy minor version emits a `UserWarning` — serialized
  models are only guaranteed compatible within minor versions, so keep the
  `spacy` requirement pinned when deploying.
- `required_fields`/`min_confidence` stored in the manifest override the code
  defaults at load time — the bundle is self-contained; the `training/` JSON
  is not needed at inference.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `training directory not found: ...` | Wrong `--training-dir`; run from the repo root or pass an absolute path. |
| `no intents found - expected intents.json or ...` | The directory has no `intents.json` and no per-intent subdirectories containing `intent.json`. |
| `name 'x' does not match directory name 'y'` | Rename the directory to match `intent.json`'s `name` (or vice versa). |
| `duplicate intent name(s)/label(s)` | Two intents share a name or textcat label; labels must be unique. |
| `unknown intent 'x'` | Training data references an intent that is not registered. Check `uv run gosmart-clinical-nlu list-intents`; if declared in `pyproject.toml` entry points, re-run `uv sync` so package metadata is refreshed (§7.1). |
| `bundle intent 'x' is not registered` | The bundle was trained with an external intent (or newer built-in intent) not registered in the current environment. Install the peer package providing the entry point or call `register_intent_spec()` before `load_intent_models()` (§8). |
| `failed to load entry point ...` / `entry point ... conflict` | A package declares a broken or colliding entry point under `gosmart_clinical_nlu.intents`. Fix the target import path, return type (`IntentSpec` / factory / iterable), or duplicate intent name. |
| Parse returns `null` slots for inputs that used to work | The bundle predates a pattern fix — retrain (§4). Never add `entity_ruler` patterns before training in custom code; spaCy clears them during initialization. |
| `UserWarning: intent bundle was trained with spaCy ...` | spaCy version drift between training and serving (§8). |

## 10. Test suite

```bash
uv run pytest        # loaders, registry & plugins, training, bundle round-trip, parsing, CLI
uv run ruff check .
uv run ruff format --check .
```
