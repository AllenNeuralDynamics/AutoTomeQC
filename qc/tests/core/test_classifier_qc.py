import logging
from unittest.mock import patch

import pytest

from autotomeqc.algorithms.base import ClassifierQC, YOLOClassifier
from autotomeqc.core.models import QCCriteria


def test_classifier_qc_is_abstract():
    with pytest.raises(TypeError):
        ClassifierQC()


def _yolo(pass_labels, min_conf=0.5):
    with patch("autotomeqc.algorithms.base.YOLO", side_effect=RuntimeError("no weights")):
        return YOLOClassifier("missing.pt", 224, pass_labels, min_conf)


def _pred(label, conf):
    return QCCriteria(pass_status=True, label=label, conf=conf)


def test_yolo_passes_accepted_confident_label():
    qc = _yolo(["full_section"])
    with patch.object(qc, "predict", return_value=_pred("full_section", 0.9)):
        res = qc.check(None)
    assert res.pass_status is True and res.reason is None


def test_yolo_fails_bad_label_with_defect_reason():
    qc = _yolo(["full_section"])
    with patch.object(qc, "predict", return_value=_pred("partial", 0.99)):
        res = qc.check(None)
    assert res.pass_status is False and res.reason == "Defect Detected: partial"


def test_yolo_fails_low_confidence():
    qc = _yolo(["ANY"], min_conf=0.5)
    with patch.object(qc, "predict", return_value=_pred("x", 0.2)):
        res = qc.check(None)
    assert res.pass_status is False and res.reason.startswith("Low Confidence")


def test_failed_prediction_is_returned_as_is():
    qc = _yolo(["ANY"])
    failed = QCCriteria(pass_status=False, label="Error", conf=0.0, message="boom")
    with patch.object(qc, "predict", return_value=failed):
        assert qc.check(None) is failed
