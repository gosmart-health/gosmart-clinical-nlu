"""Design-note sample: single-pipeline intent parsing with spaCy.

This is the original exploration sample. It stays intentionally simple: ONE
pipeline with an EntityRuler and a TextCategorizer, trained inline.

The production architecture differs (see docs/design_notes/intent_nlu_design.md):
one self-contained pipeline per intent, trained from ./training JSON, and
bundled into a single artifact file. Fixes applied to this sample vs. the
first draft:

1. EntityRuler patterns are added AFTER training. ``EntityRuler.initialize()``
   (invoked via ``nlp.initialize``) clears any patterns added beforehand, which
   silently produced empty ``doc.ents`` (warning W036).
2. The PATIENT_ID pattern's ``"OP": "?"`` is a token-level key, not a key
   inside the ``LOWER`` attribute dict (which spaCy rejected with W035).
3. ``nlp.initialize(...)`` replaces the deprecated ``nlp.begin_training()``.
"""

from datetime import datetime

import dateparser
import spacy
from pydantic import BaseModel
from spacy.training import Example


# 1. Define your strict output schema (maps cleanly to FHIR / DICOM C-FIND)
class StudyQueryIntent(BaseModel):
    intent: str
    confidence: float
    modality: str | None = None
    patient_id: str | None = None
    study_date: str | None = None


def build_clinical_nlu() -> spacy.Language:
    # Blank English pipeline (zero external cloud calls, ~10MB footprint)
    nlp = spacy.blank("en")

    # --- STAGE B: Trainable Intent Classifier (TextCategorizer) ---
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("FIND_STUDY")
    textcat.add_label("GET_REPORT")

    # Training utterances (easily maintained in a JSON/YAML file)
    train_data = [
        (
            "Find a CT patient 12345678 from yesterday",
            {"cats": {"FIND_STUDY": 1.0, "GET_REPORT": 0.0}},
        ),
        (
            "Pull up yesterday's CT for 12345678",
            {"cats": {"FIND_STUDY": 1.0, "GET_REPORT": 0.0}},
        ),
        (
            "Show me MRI scans for mrn 98765432 last week",
            {"cats": {"FIND_STUDY": 1.0, "GET_REPORT": 0.0}},
        ),
        (
            "Get the radiology report for patient 12345678",
            {"cats": {"FIND_STUDY": 0.0, "GET_REPORT": 1.0}},
        ),
        (
            "Read impression on CT for 12345678",
            {"cats": {"FIND_STUDY": 0.0, "GET_REPORT": 1.0}},
        ),
    ]

    # Quick local training loop (~1 second on CPU)
    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, annotations))
    optimizer = nlp.initialize(lambda: list(examples))
    for _ in range(15):
        for example in examples:
            nlp.update([example], sgd=optimizer)

    # --- STAGE A: Deterministic Rules (EntityRuler) ---
    # NOTE: patterns are added AFTER training on purpose — EntityRuler.initialize()
    # (called by nlp.initialize above) clears patterns added beforehand.
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [
        # DICOM Modalities (case-insensitive)
        {
            "label": "MODALITY",
            "pattern": [
                {
                    "LOWER": {
                        "IN": [
                            "ct",
                            "mri",
                            "mr",
                            "us",
                            "ultrasound",
                            "xray",
                            "x-ray",
                            "pet",
                        ]
                    }
                }
            ],
        },
        # Deterministic Patient ID / MRN (e.g., 6-10 digits, optionally preceded by 'patient' or 'mrn')
        # "OP" is a token-level key (W035 otherwise).
        {
            "label": "PATIENT_ID",
            "pattern": [
                {"LOWER": {"IN": ["patient", "mrn", "id"]}, "OP": "?"},
                {"TEXT": {"REGEX": r"^\d{6,10}$"}},
            ],
        },
        # Relative date triggers
        {"label": "REL_DATE", "pattern": [{"LOWER": {"IN": ["today", "yesterday"]}}]},
        {
            "label": "REL_DATE",
            "pattern": [
                {"LOWER": "last"},
                {
                    "LOWER": {
                        "IN": [
                            "week",
                            "month",
                            "monday",
                            "tuesday",
                            "wednesday",
                            "thursday",
                            "friday",
                        ]
                    }
                },
            ],
        },
    ]
    ruler.add_patterns(patterns)

    return nlp


def parse_utterance(
    nlp: spacy.Language, text: str, ref_date: datetime
) -> StudyQueryIntent:
    doc = nlp(text)
    top_intent = max(doc.cats, key=doc.cats.get)

    modality = None
    patient_id = None
    study_date = None

    for ent in doc.ents:
        if ent.label_ == "MODALITY":
            # Normalize synonyms to DICOM Modality codes
            norm = ent.text.upper()
            modality = (
                "MR" if norm == "MRI" else ("US" if norm == "ULTRASOUND" else norm)
            )
        elif ent.label_ == "PATIENT_ID":
            # Strip optional prefix words like "patient" or "mrn"
            patient_id = "".join(filter(str.isdigit, ent.text))
        elif ent.label_ == "REL_DATE":
            parsed_dt = dateparser.parse(ent.text, settings={"RELATIVE_BASE": ref_date})
            if parsed_dt:
                study_date = parsed_dt.strftime("%Y%m%d")  # DICOM DA format

    return StudyQueryIntent(
        intent=top_intent,
        confidence=round(doc.cats[top_intent], 3),
        modality=modality,
        patient_id=patient_id,
        study_date=study_date,
    )


# --- Execution ---
if __name__ == "__main__":
    nlp_engine = build_clinical_nlu()
    result = parse_utterance(
        nlp_engine,
        "Can you pull up CT for patient 12345678 from yesterday?",
        ref_date=datetime(2026, 11, 19),
    )
    print(result.model_dump_json(indent=2))
