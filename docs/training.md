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
- Training input lives in `training/` as plain JSON. Changing intent data never
  requires code changes; adding a *new* intent requires a small code change
  (a Pydantic model + a registry entry) plus a training directory.

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
  utterances as negatives (one-vs-rest).

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
JSON, and unknown intent names (see §5).

## 4. Training

```bash
uv run gosmart-clinical-nlu train [--training-dir ./training] [--out models/intent_models.zip] [--epochs N] [--seed N]
```

- `--training-dir` defaults to `./training`; `--out` defaults to
  `models/intent_models.zip` (the `models/` directory is git-ignored).
- `--epochs` / `--seed` override `config.json`.
- Output: one line per step, ending with the artifact path and the
  **content hash** — a sha256 over the canonical training JSON. Two bundles
  with the same hash were trained from identical data; use it to correlate a
  deployed artifact with its source data.
- Exit codes: `0` success, `1` training-data or I/O errors (message on
  stderr), `2` command-line usage errors.

Training is deterministic for a given `(training data, epochs, seed)`: the
same inputs produce the same model behavior.

## 5. Verifying a bundle

```bash
# What is inside?
uv run gosmart-clinical-nlu inspect --model models/intent_models.zip

# Smoke-test one utterance (prints the full JSON result):
uv run gosmart-clinical-nlu parse --model models/intent_models.zip \
    "Show me MRI scans for mrn 98765432 last week" --ref-date 2026-11-19
```

`parse` prints the `ParsedUtterance` envelope. Check that:

- `intent` is the expected one and `status` is `"matched"`.
- `candidates` shows a healthy margin between the top intent and the rest.
- `data` carries the expected slots (e.g. `modality: "MR"` for "MRI",
  `study_date` as a DICOM `YYYYMMDD` string).

## 6. Updating an existing intent (no code changes)

1. Edit the intent's `examples.json` (add/remove utterances), `patterns.json`
   (slot rules), or `intent.json` (`min_confidence`, `required_fields`).
2. Re-run training (§4). Every intent is re-trained into a fresh bundle;
   there is no incremental state to preserve.
3. Verify with §5, then distribute the new bundle file.

## 7. Adding a new intent (code + data)

1. Create `src/gosmart_clinical_nlu/intents/<name>.py` with the intent's
   Pydantic slot model and an `ENTITY_MAP` mapping EntityRuler labels to model
   fields (optionally with normalizers; see `find_study.py`).
2. Register it in `INTENT_SPECS` in `src/gosmart_clinical_nlu/intents/__init__.py`.
3. Create `training/<name>/` with `intent.json`, `patterns.json`,
   `examples.json` (§3.1).
4. Train and verify (§4–§5). Existing intents keep working; only the new
   intent is added to the bundle.

Forgetting step 2 fails training with `unknown intent '<name>'`.

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
| `unknown intent 'x'` | Training data references an intent that is not registered in code (§7 step 2), or a bundle is loaded by an older package version (§8). |
| Parse returns `null` slots for inputs that used to work | The bundle predates a pattern fix — retrain (§4). Never add `entity_ruler` patterns before training in custom code; spaCy clears them during initialization. |
| `UserWarning: intent bundle was trained with spaCy ...` | spaCy version drift between training and serving (§8). |

## 10. Test suite

```bash
uv run pytest        # 41 tests: loaders, training, bundle round-trip, parsing, CLI
uv run ruff check .
uv run ruff format --check .
```

