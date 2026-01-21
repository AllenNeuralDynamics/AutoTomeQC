import logging
import numpy as np
from ultralytics import YOLO
from cv2 import resize, INTER_NEAREST

logger = logging.getLogger(__name__)

class YOLOClassifier:
    """
    Generic wrapper that handles Loading, Prediction, and QC Evaluation.
    """
    def __init__(self, model_path: str, img_size: int, pass_labels: list, min_conf: float):
        self.model = None
        self.model_name = model_path.split("/")[-1]
        self.img_size = img_size
        self.pass_labels = pass_labels
        self.min_confidence = min_conf

        try:
            logger.info(f"Loading model: {self.model_name}...")
            self.model = YOLO(model_path)
            # Warmup inference
            self.model(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8), verbose=False)
            logger.info(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load {model_path}. Error: {e}")

    def predict(self, image: np.ndarray) -> dict:
        if self.model is None or image is None:
            return {"error": "Model not loaded", "label": "Unknown", "conf": 0.0}

        try:
            if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
                image = resize(image, (self.img_size, self.img_size), interpolation=INTER_NEAREST)
            results = self.model(image, verbose=False)
            top_idx = results[0].probs.top1
            conf = float(results[0].probs.top1conf)
            label = results[0].names[top_idx]
            return {"label": label, "conf": round(conf, 4), "error": None}
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_name}: {e}")
            return {"error": str(e), "label": "Error", "conf": 0.0}

    def check(self, image: np.ndarray) -> dict:
        """
        Standard QC Check: Predicts -> Validates against Config Criteria
        """
        result = self.predict(image)
        if result.get("error"):
            result["pass"] = False
            return result

        # Check Label
        # Handle "ANY" wildcard (for Thickness check)
        if self.pass_labels == "ANY":
            is_valid_label = True
        else:
            is_valid_label = result["label"] in self.pass_labels
            
        # Check Confidence
        is_confident = result["conf"] >= self.min_confidence

        # 3. Final Decision
        if is_valid_label and is_confident:
            result["pass"] = True
        else:
            result["pass"] = False
            if not is_valid_label:
                result["reason"] = f"Defect Detected: {result['label']}"
            elif not is_confident:
                result["reason"] = f"Low Confidence ({result['conf']} < {self.min_confidence})"

        return result