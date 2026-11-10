from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from autotomeqc.algorithms.thickness import ThicknessQC
from autotomeqc.core.models import QCCriteria


@pytest.fixture
def qc():
    cfg = SimpleNamespace(
        weights_path="missing.joblib", min_confidence=0.1, tolerance_nm=20.0
    )
    q = ThicknessQC(cfg)   # the weights file does not exist; predict() is patched below
    return q


def _predicted(nm, conf=0.9):
    return QCCriteria(pass_status=True, label=str(nm), conf=conf, message=f"Predicted {nm} nm")


IMG = np.zeros((10, 10, 3), dtype=np.uint8)


def test_no_target_keeps_old_behaviour(qc):
    with patch.object(qc, "predict", return_value=_predicted(160)):
        assert qc.check(IMG).pass_status is True


def test_within_tolerance_passes(qc):
    with patch.object(qc, "predict", return_value=_predicted(160)):
        res = qc.check(IMG, expected_nm=180)   # off by 20, tolerance 20
    assert res.pass_status is True
    assert "expected 180 nm" in res.message


def test_outside_tolerance_fails_with_reason(qc):
    with patch.object(qc, "predict", return_value=_predicted(160)):
        res = qc.check(IMG, expected_nm=200)   # off by 40
    assert res.pass_status is False
    assert "200 nm target" in res.reason


def test_tolerance_comes_from_config(qc):
    qc.tolerance_nm = 50.0
    with patch.object(qc, "predict", return_value=_predicted(160)):
        assert qc.check(IMG, expected_nm=200).pass_status is True
