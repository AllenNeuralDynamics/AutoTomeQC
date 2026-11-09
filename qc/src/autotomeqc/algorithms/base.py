import logging
import numpy as np
from typing import List
from ultralytics import YOLO
from cv2 import resize, INTER_NEAREST
from autotomeqc.core.models import QCCriteria


class YOLOClassifier:
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
        """Runs the model. Here pass_status=True only means 'the prediction ran'; check() decides QC."""
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

    def check(self, image: np.ndarray) -> QCCriteria:
        """
        Standard QC Check: Predicts -> Validates against Config Criteria
        """
        result = self.predict(image)
        if not result.pass_status:  # prediction failed: return the error result as is
            return result

        # Check Label
        is_valid_label = "ANY" in self.pass_labels or result.label in self.pass_labels
        # Check Confidence
        is_confident = (result.conf or 0.0) >= self.min_confidence

        # Final Decision using pass_status
        result.pass_status = is_valid_label and is_confident
        if not is_valid_label:
            result.reason = f"Defect Detected: {result.label}"
            self.log.debug(result.reason)
        elif not is_confident:
            result.reason = f"Low Confidence ({result.conf} < {self.min_confidence})"
            self.log.debug(result.reason)

        return result