"""gosmart-clinical-nlu: clinical NLU library shared across peer projects.

The distribution name is ``gosmart-clinical-nlu``; the importable package name
is ``gosmart_clinical_nlu`` (Python module names cannot contain hyphens).

Example::

    from gosmart_clinical_nlu import print_hi, __version__
"""

from gosmart_clinical_nlu.core import print_hi

__all__ = ["print_hi", "main"]
__version__ = "0.1.0"


def main() -> None:
    """Console entry point for ``gosmart-clinical-nlu``."""
    print_hi("gosmart-clinical-nlu")
