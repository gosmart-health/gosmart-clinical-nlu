"""Tests for the gosmart_clinical_nlu package.

These tests import the installed (editable) package, which is exactly how a
peer project will consume it.
"""

import gosmart_clinical_nlu


def test_version_is_exposed() -> None:
    assert gosmart_clinical_nlu.__version__ == "0.1.0"


def test_public_api_reexports_core() -> None:
    assert callable(gosmart_clinical_nlu.print_hi)


def test_print_hi_output(capsys) -> None:
    gosmart_clinical_nlu.print_hi("World")
    assert capsys.readouterr().out == "Hi, World\n"


def test_main_entry_point(capsys) -> None:
    gosmart_clinical_nlu.main()
    assert capsys.readouterr().out == "Hi, gosmart-clinical-nlu\n"
