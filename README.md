<!--suppress ALL -->
# gosmart-clinical-nlu

Clinical NLU library shared across peer projects (Python, managed with [uv](https://docs.astral.sh/uv/)).

## Layout

Uses the standard Python `src` layout, which keeps the importable package
separated from project tooling and guarantees tests run against the installed
package rather than the working directory:

```
.
├── src/
│   └── gosmart_clinical_nlu/   # the importable package (underscores — required by Python)
│       ├── __init__.py         # public API re-exports
│       ├── core.py
│       ├── cli.py              # train / parse / inspect / list-intents subcommands
│       ├── training.py         # training-data schemas + loaders (per-intent & aggregated)
│       ├── pipeline.py         # per-intent pipeline training + single-file bundle
│       ├── parsing.py          # parse_utterance dispatch + result envelope
│       ├── intents/            # per-intent pydantic slot models + extensible registry
│       └── __main__.py
├── training/                   # training data (JSON) — see docs/training.md
├── tests/                      # test suite (pytest)
├── docs/                       # design notes + ops guides
├── pyproject.toml
└── uv.lock
```

The distribution name is `gosmart-clinical-nlu` (hyphens are allowed in
distribution names), but Python module names cannot contain hyphens, so the
import name is `gosmart_clinical_nlu`.

## Development

```bash
uv sync                                # create/update .venv, install the package (editable) + dev deps
uv run pytest                          # run the test suite
uv run gosmart-clinical-nlu            # run the console entry point
uv run python -m gosmart_clinical_nlu  # same thing, via module execution
uv build                               # build sdist + wheel for distribution
```

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
`main`, on every **pull request targeting `dev` or `main`** (feature branches
into `dev`, and `dev` → `main`), and on manual `workflow_dispatch`. It is a
**Linux-only** workflow (single `ubuntu-latest` job, no macos/Windows targets)
that:

1. installs `uv` and Python 3.14 (`astral-sh/setup-uv`, version kept in sync
   with `.python-version`),
2. installs the project with locked dependencies (`uv sync --locked` — fails
   when `uv.lock` is out of date),
3. runs the ruff lint/format gate,
4. runs the unit tests (`uv run pytest`),
5. builds the sdist and wheel (`uv build`) and uploads them as a workflow
   artifact named `dist-<commit sha>`.

## Intent NLU

The package turns free-text clinical utterances into validated, per-intent
Pydantic structures using one self-contained spaCy pipeline per intent
(blank `en` + `entity_ruler` slot patterns + a one-vs-rest `textcat`
classifier). Training data lives in `training/` as JSON; the trained result is
packed into a single artifact file that peer projects load with
`spacy.load()`.

### Command line (repo checkout)

```bash
uv run gosmart-clinical-nlu list-intents   # show built-in & plugin-discovered intents
uv run gosmart-clinical-nlu train          # ./training -> models/intent_models.zip
uv run gosmart-clinical-nlu inspect --model models/intent_models.zip
uv run gosmart-clinical-nlu parse --model models/intent_models.zip \
    "Find a CT patient 12345678 from yesterday" --ref-date 2026-11-19
```

### Python API (peer projects)

```python
from gosmart_clinical_nlu import load_intent_models, parse_utterance

models = load_intent_models("models/intent_models.zip")  # once per process
result = parse_utterance("Show me MRI for mrn 98765432 last week", models)

result.status  # "matched" | "low_confidence"  <- LLM failover signal
result.intent  # "find_study"
result.confidence  # 0.9998 (threshold in result.threshold)
result.candidates  # {"find_study": 0.9998, "get_report": 0.0003}
result.data  # StudyQueryIntent(modality="MR", patient_id="98765432",
#                study_date="20261112")  (DICOM DA)
```

`status == "matched"` requires the top intent to clear its `min_confidence`
AND all `required_fields` to be extracted. On `low_confidence` the best-effort
`data` is still returned and `candidates` carries every intent's score, so a
caller can fail over to an LLM playbook with full context.

### Defining custom intents in a peer project

Peer projects can define and train new Pydantic intent models **without
modifying `gosmart-clinical-nlu`**. `ParsedUtterance.data` accepts any Pydantic
`BaseModel` subclass (and preserves all subclass fields during JSON/dict
serialization).

Register custom intents either declaratively via `pyproject.toml` entry points
(recommended — automatically discovered by `gosmart-clinical-nlu train`,
`parse`, `inspect`, `list-intents`, and the Python API):

```toml
# peer-project/pyproject.toml
[project.entry-points."gosmart_clinical_nlu.intents"]
order_exam = "my_peer_pkg.intents.order_exam:ORDER_EXAM_SPEC"
```

or programmatically at runtime:

```python
from pydantic import BaseModel
from gosmart_clinical_nlu import EntityBinding, IntentSpec, register_intent_spec
from gosmart_clinical_nlu.intents.normalizers import normalize_patient_id


class OrderExamIntent(BaseModel):
    patient_id: str | None = None


ORDER_EXAM_SPEC = IntentSpec(
    name="order_exam",
    label="ORDER_EXAM",
    model_cls=OrderExamIntent,
    entity_map={
        "PATIENT_ID": EntityBinding(field="patient_id", normalize=normalize_patient_id),
    },
    description="Place an imaging exam order.",
)

register_intent_spec(ORDER_EXAM_SPEC)
```

See [`docs/training.md`](docs/training.md) §7 for the full step-by-step guide
on authoring custom intent models, structuring training data, and training
bundles in a peer project.

### Training data and operations

- Training data formats, external intent authoring, the train/verify/deploy
  procedure, and troubleshooting live in [`docs/training.md`](docs/training.md).
- Architecture notes: [`docs/design_notes/intent_nlu_design.md`](docs/design_notes/intent_nlu_design.md).
- The artifact is a ZIP of one spaCy `to_disk()` tree per intent plus a
  `manifest.json` (spaCy version, content hash of the training data,
  per-intent thresholds). `models/` is git-ignored — bundles are build
  artifacts, not source.

## Using from a peer project

Add the package straight from GitHub (no PyPI release required):

```bash
# from the default branch (main)
uv add "gosmart-clinical-nlu @ git+https://github.com/gosmart-health/gosmart-clinical-nlu.git"

# from the dev branch
uv add "gosmart-clinical-nlu @ git+https://github.com/gosmart-health/gosmart-clinical-nlu.git@dev"

# pinned to a tag/release
uv add "gosmart-clinical-nlu @ git+https://github.com/gosmart-health/gosmart-clinical-nlu.git@v0.2.0"

# or, while iterating locally on a sibling clone
uv add --editable ../gosmart-clinical-nlu
```

### Importing the package

The distribution name is `gosmart-clinical-nlu`, but Python module names
cannot contain hyphens, so the import name is `gosmart_clinical_nlu`:

```python
import gosmart_clinical_nlu

gosmart_clinical_nlu.__version__  # '0.2.0'

# The public API is re-exported at the package root:
from gosmart_clinical_nlu import print_hi

print_hi("peer project")  # Hi, peer project
```

### Running it

After `uv sync` inside a clone of this repo:

```bash
uv run gosmart-clinical-nlu               # console entry point
uv run python -m gosmart_clinical_nlu     # equivalent, via module execution
```
