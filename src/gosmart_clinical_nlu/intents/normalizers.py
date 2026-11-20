"""Entity normalizers: raw entity text to normalized slot values."""

from datetime import datetime

import dateparser

# Upper-cased trigger word (punctuation stripped) to DICOM Modality code.
_MODALITY_SYNONYMS = {
    "MRI": "MR",
    "ULTRASOUND": "US",
    "PET": "PT",
    "XRAY": "CR",
}


def normalize_modality(raw: str, ref_date: datetime | None = None) -> str:
    """Map trigger words to DICOM Modality codes (CT, MR, US, PT, CR, ...)."""
    code = raw.strip().upper().replace("-", "").replace(" ", "")
    return _MODALITY_SYNONYMS.get(code, code)


def normalize_patient_id(raw: str, ref_date: datetime | None = None) -> str:
    """Strip optional prefix words ('patient', 'mrn') from a matched ID."""
    digits = "".join(char for char in raw if char.isdigit())
    return digits if digits else raw.strip()


def normalize_study_date(raw: str, ref_date: datetime | None = None) -> str | None:
    """Parse a matched relative date into a DICOM DA value (``YYYYMMDD``)."""
    base = ref_date or datetime.now()
    parsed = dateparser.parse(raw, settings={"RELATIVE_BASE": base})
    if parsed is None:
        return None
    return parsed.strftime("%Y%m%d")
