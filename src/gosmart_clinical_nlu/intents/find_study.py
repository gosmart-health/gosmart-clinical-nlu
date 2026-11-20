"""``find_study`` intent: search for imaging studies."""

from pydantic import BaseModel

from gosmart_clinical_nlu.intents.base import EntityBinding
from gosmart_clinical_nlu.intents.normalizers import (
    normalize_modality,
    normalize_patient_id,
    normalize_study_date,
)


class StudyQueryIntent(BaseModel):
    """Slots for a DICOM study search request (maps to a C-FIND query)."""

    modality: str | None = None
    patient_id: str | None = None
    study_date: str | None = None  # DICOM DA value (YYYYMMDD)


ENTITY_MAP = {
    "MODALITY": EntityBinding(field="modality", normalize=normalize_modality),
    "PATIENT_ID": EntityBinding(field="patient_id", normalize=normalize_patient_id),
    "REL_DATE": EntityBinding(field="study_date", normalize=normalize_study_date),
}
