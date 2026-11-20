"""Tests for the command line interface."""

import json
from pathlib import Path

import pytest

from gosmart_clinical_nlu.cli import main


def test_train_then_parse(tmp_path: Path, capsys, training_dir_factory) -> None:
    training = training_dir_factory()
    bundle = tmp_path / "models" / "bundle.zip"
    assert main(["train", "--training-dir", str(training), "--out", str(bundle)]) == 0
    assert "find_study" in capsys.readouterr().out
    assert bundle.is_file()

    exit_code = main(
        [
            "parse",
            "--model",
            str(bundle),
            "Find a CT patient 12345678 from yesterday",
            "--ref-date",
            "2026-11-19",
        ]
    )
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "matched"
    assert result["intent"] == "find_study"
    assert result["data"]["patient_id"] == "12345678"


def test_inspect(tmp_path: Path, capsys, training_dir_factory) -> None:
    bundle = tmp_path / "bundle.zip"
    assert (
        main(
            [
                "train",
                "--training-dir",
                str(training_dir_factory()),
                "--out",
                str(bundle),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["inspect", "--model", str(bundle)]) == 0
    out = capsys.readouterr().out
    assert "find_study" in out
    assert "GET_REPORT" in out


def test_parse_missing_bundle(tmp_path: Path, capsys) -> None:
    exit_code = main(["parse", "--model", str(tmp_path / "nope.zip"), "hello"])
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_train_with_missing_training_dir(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "train",
            "--training-dir",
            str(tmp_path / "nope"),
            "--out",
            str(tmp_path / "bundle.zip"),
        ]
    )
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_help(capsys) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out.lower()


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "gosmart-clinical-nlu" in capsys.readouterr().out
