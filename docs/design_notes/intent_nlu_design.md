# Intent NLU design notes

## Architecture

**Per-intent, self-contained, one-vs-rest.** Each intent is a complete spaCy
pipeline built on `spacy.blank("en")`:

1. `entity_ruler` — deterministic token/phrase patterns extracting that
   intent's slot entities (e.g. `MODALITY`, `PATIENT_ID`, `REL_DATE`).
2. `textcat` — a two-label classifier: the intent's own label vs
   `__NEGATIVE__`. Training pairs are built one-vs-rest: an intent's positives
   are labeled `1.0`, and every other intent's positives (plus explicit
   negatives) are labeled `0.0`.

`parse_utterance` runs the utterance through every pipeline, dispatches to the
highest-scoring intent (ties broken lexicographically by name for
determinism), extracts slots from that pipeline's `doc.ents` via the intent's
entity map, normalizes values (modality synonyms to DICOM codes, digit
extraction, `dateparser` with `RELATIVE_BASE` to DICOM DA), and validates them
into the intent's Pydantic model.

Consequences: adding an intent never retrains existing ones; entity patterns
cannot collide across intents; per-intent thresholds; running N small
pipelines per utterance is cheap.

## Result envelope (the failover contract)

`ParsedUtterance` separates *routing* facts (status, intent, confidence,
threshold, candidates, missing_required) from *slot data* (`data`).
`status == "matched"` iff `confidence >= min_confidence` and no required slot
is missing; otherwise `low_confidence`. Best-effort `data` is always returned,
and `candidates` exposes the full score distribution so an LLM playbook can
re-rank or disambiguate.

## Serialization

The trained set is stored as a single ZIP: one spaCy `to_disk()` tree per
intent plus `manifest.json` (schema version, package/spaCy versions, UTC
timestamp, sha256 content hash of the canonical training JSON, and per-intent
runtime settings). Loading unzips to a temp dir, `spacy.load()`s each tree
(fully in-memory), applies the manifest overrides to the code-side registry
specs, then deletes the temp dir. spaCy's `Language.from_bytes` on a fresh
blank pipeline silently drops components (deserializers are registered from
components already on the instance), which is why the bundle wraps
`to_disk`/`spacy.load` instead of raw bytes.

## spaCy pitfalls found and fixed (3.8.18)

- `EntityRuler.initialize()` unconditionally calls `self.clear()`, so patterns
  added *before* `nlp.initialize()`/training are silently wiped (warning
  W036). The pipeline adds patterns **after** training.
- In token patterns, `"OP"` is a token-level key; nesting it inside an
  attribute dict such as `{"LOWER": {...}, "OP": "?"}` discards the subpattern
  (warning W035).
- `Config` lives in `confection` since spaCy 3.4 (`from confection import
  Config`), not `spacy.config`.

## Files

- `training.py` — pydantic schemas + loaders for both training layouts.
- `pipeline.py` — `build_intent_pipeline`, `train_intent_models`,
  `save_bundle`, `load_intent_models`, manifest models.
- `parsing.py` — `ParsedUtterance`, `parse_utterance`, slot extraction.
- `intents/` — `IntentSpec` registry, per-intent Pydantic models, normalizers.
- `cli.py` — `train` / `parse` / `inspect` subcommands.
- `docs/training.md` — operator procedure.
