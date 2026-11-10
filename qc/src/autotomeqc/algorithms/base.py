import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional
from ultralytics import YOLO
from cv2 import resize, INTER_NEAREST
from autotomeqc.core.models import QCCriteria


class ClassifierQC(ABC):
    """
    Contract for model-based QC modules (YOLO classifiers, the thickness random forest).

    Subclasses implement predict(); check() is shared: it runs predict() and turns the result into
    pass/fail. Both return a QCCriteria, so the pipeline treats every classifier the same way.
    A subclass changes what counts as a pass by overriding _failure_reason().
    """
    min_confidence: float
    log: logging.Logger

    @abstractmethod
    def predict(self, image: np.ndarray) -> QCCriteria:
        """Runs the model. pass_status=True here only means 'the prediction ran'; check() decides QC."""

    def _failure_reason(self, result: QCCriteria) -> Optional[str]:
        """Why the prediction fails QC, or None if it passes. Default: confidence only; subclasses add criteria."""
        if (result.conf or 0.0) < self.min_confidence:
            return f"Low Confidence ({result.conf} < {self.min_confidence})"
        return None

    def check(self, image: np.ndarray) -> QCCriteria:
        """Standard QC check: predict, then validate against the config criteria."""
        result = self.predict(image)
        if not result.pass_status:  # prediction failed: return the error result as is
            return result
        reason = self._failure_reason(result)
        if reason:
            result.pass_status = False
            result.reason = reason
            self.log.debug(reason)
        return result


class YOLOClassifier(ClassifierQC):
    """
    Generic wrapper that handles Loading, Prediction, and QC Evaluation.
    """
    def __init__(self, model_path: str, img_size: int, pass_labels: List[str], min_conf: float):
        self.model = None
        self.model_name = model_path.split("/")[-1]
        self.img_size = img_size
        self.pass_labels = pass_labels
        self.min_confidence = min_conf
        self.log = logging.getLogger(self.__class__.__name__)

        try:
            self.model = YOLO(model_path)
            # Warmup inference
            self.model(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8), verbose=False)
            self.log.debug(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            self.log.warning(f"Could not load {model_path}. Error: {e}")

    def predict(self, image: np.ndarray) -> QCCriteria:
        if self.model is None or image is None:
            return QCCriteria(pass_status=False, label="Unknown", conf=0.0, message="Model not loaded")

        try:
            if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
                image = resize(image, (self.img_size, self.img_size), interpolation=INTER_NEAREST)
            results = self.model(image, verbose=False)
            top_idx = results[0].probs.top1  # Index of highest confidence
            conf = float(results[0].probs.top1conf)
            label = results[0].names[top_idx]
            return QCCriteria(pass_status=True, label=label, conf=round(conf, 4))
        except Exception as e:
            self.log.error(f"Prediction failed for {self.model_name}: {e}")
            return QCCriteria(pass_status=False, label="Error", conf=0.0, message=str(e))

    def _failure_reason(self, result: QCCriteria) -> Optional[str]:
        """The label must be an accepted one, and the confidence high enough."""
        if "ANY" not in self.pass_labels and result.label not in self.pass_labels:
            return f"Defect Detected: {result.label}"
        return super()._failure_reason(result)