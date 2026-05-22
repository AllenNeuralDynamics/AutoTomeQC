import logging
import numpy as np
from typing import List
from ultralytics import YOLO
from cv2 import resize, INTER_NEAREST


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

    def predict(self, image: np.ndarray) -> dict:
        if self.model is None or image is None:
            return {"error": "Model not loaded", "label": "Unknown", "conf": 0.0}

        try:
            if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
                image = resize(image, (self.img_size, self.img_size), interpolation=INTER_NEAREST)
            results = self.model(image, verbose=False)
            top_idx = results[0].probs.top1  # Index of highest confidence
            conf = float(results[0].probs.top1conf)
            label = results[0].names[top_idx]
            return {"label": label, "conf": round(conf, 4), "error": None}
        except Exception as e:
            self.log.error(f"Prediction failed for {self.model_name}: {e}")
            return {"error": str(e), "label": "Error", "conf": 0.0}

    def check(self, image: np.ndarray) -> dict:
        """
        Standard QC Check: Predicts -> Validates against Config Criteria
        """
        result = self.predict(image)
        if result.get("error"):
            result["pass_status"] = False  # Changed key
            return result

        # Check Label
        if "ANY" in self.pass_labels:
            is_valid_label = True
        else:
            is_valid_label = result.get("label") in self.pass_labels
            
        # Check Confidence
        is_confident = result.get("conf", 0) >= self.min_confidence

        # Final Decision using pass_status
        if is_valid_label and is_confident:
            result["pass_status"] = True
        else:
            result["pass_status"] = False
            if not is_valid_label:
                result["reason"] = f"Defect Detected: {result.get('label')}"
                self.log.debug(result["reason"])
            elif not is_confident:
                result["reason"] = f"Low Confidence ({result.get('conf')} < {self.min_confidence})"
                self.log.debug(result["reason"])

        return result