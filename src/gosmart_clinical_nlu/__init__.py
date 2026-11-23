"""gosmart-clinical-nlu: clinical NLU library shared across peer projects.

The distribution name is ``gosmart-clinical-nlu``; the importable package name
is ``gosmart_clinical_nlu`` (Python module names cannot contain hyphens).

Intent NLU quickstart::

    from gosmart_clinical_nlu import load_intent_models, parse_utterance

    models = load_intent_models("models/intent_models.zip")
    result = parse_utterance("Find a CT for patient 12345678", models)
    if result.status == "low_confidence":
        ...  # fail over to the LLM playbook
"""

from gosmart_clinical_nlu.cli import main
from gosmart_clinical_nlu.core import print_hi
from gosmart_clinical_nlu.intents import (
    DEFAULT_MIN_CONFIDENCE,
    INTENT_ENTRY_POINT_GROUP,
    EntityBinding,
    IntentSpec,
    discover_intent_specs,
    get_intent_spec,
    list_intent_specs,
    register_intent_spec,
    unregister_intent_spec,
)
from gosmart_clinical_nlu.parsing import ParsedUtterance, parse_utterance
from gosmart_clinical_nlu.pipeline import (
    IntentModels,
    load_intent_models,
    save_bundle,
    train_intent_models,
)
from gosmart_clinical_nlu.training import load_training_dir

__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "INTENT_ENTRY_POINT_GROUP",
    "EntityBinding",
    "IntentModels",
    "IntentSpec",
    "ParsedUtterance",
    "discover_intent_specs",
    "get_intent_spec",
    "list_intent_specs",
    "load_intent_models",
    "load_training_dir",
    "main",
    "parse_utterance",
    "print_hi",
    "register_intent_spec",
    "save_bundle",
    "train_intent_models",
    "unregister_intent_spec",
]
__version__ = "0.2.0"
