"""``get_report`` intent: fetch the radiology report for a study."""

from pydantic import BaseModel

from gosmart_clinical_nlu.intents.base import EntityBinding
from gosmart_clinical_nlu.intents.normalizers import (
    normalize_modality,
    normalize_patient_id,
    normalize_study_date,
)


class ReportQueryIntent(BaseModel):
    """Slots for a radiology report retrieval request."""

    modality: str | None = None
    patient_id: str | None = None
    study_date: str | None = None  # DICOM DA value (YYYYMMDD)


ENTITY_MAP = {
    "MODALITY": EntityBinding(field="modality", normalize=normalize_modality),
    "PATIENT_ID": EntityBinding(field="patient_id", normalize=normalize_patient_id),
    "REL_DATE": EntityBinding(field="study_date", normalize=normalize_study_date),
}
