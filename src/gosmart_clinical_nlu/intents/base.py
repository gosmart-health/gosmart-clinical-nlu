"""Shared building blocks for intent definitions."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class EntityBinding:
    """Maps an EntityRuler label to a pydantic field of the intent model.

    ``normalize`` receives the matched entity text and the caller-supplied
    reference date (used for relative dates) and returns the slot value. When
    no normalizer is set, the raw entity text is used verbatim.
    """

    field: str
    normalize: Callable[[str, datetime | None], Any] | None = None
