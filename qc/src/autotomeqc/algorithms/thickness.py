# autotomeqc/algorithms/thickness.py
"""Section thickness QC from interference colour, using a random forest.

The model (weights/thickness_rf.joblib) was trained on 17 colour features
(HSV, Lab, colour spread) of segmented sections on a black background.
`masks()` and `extract()` below must stay identical to the training codes, 
otherwise the features no longer match what the forest was trained on.
"""
import logging
from typing import List

import cv2
import joblib
import numpy as np


def masks(bgr: np.ndarray):
    """Return (section, kept) boolean masks. Same as the training notebook."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = (gray > 12).astype(np.uint8)                                  # 1. not background
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.erode(m, np.ones((15, 15), np.uint8)) > 0                 # 2. drop blurred edge
    if not m.any():
        raise ValueError("No section pixels found (image is empty or the section is too small)")
    v = bgr.max(axis=2).astype(np.float32) / 255
    kept = m & (v > np.percentile(v[m], 15))                          # 3. drop darkest 15%
    return m, kept


def extract(bgr: np.ndarray) -> dict:
    """The colour features the forest was trained on (HSV, Lab, colour spread)."""
    _, kept = masks(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)   # H scaled to 0-255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # HSV: saturation-weighted circular mean hue (grey pixels have meaningless hue)
    h = hsv[..., 0][kept] / 255 * 2 * np.pi
    s = hsv[..., 1][kept] / 255
    hx, hy = np.sum(s * np.cos(h)), np.sum(s * np.sin(h))
    hue = np.arctan2(hy, hx)

    # Lab arrays rescaled to standard ranges
    L = lab[..., 0][kept] * 100 / 255
    a = lab[..., 1][kept] - 128
    b = lab[..., 2][kept] - 128

    return {
        "hue_sin": float(np.sin(hue)), "hue_cos": float(np.cos(hue)),
        "sat": float(np.median(s) * 100), "val": float(np.median(hsv[..., 2][kept]) / 255 * 100),
        "L": float(np.median(L)), "a": float(np.median(a)), "b": float(np.median(b)),
        "a_p10": float(np.percentile(a, 10)), "a_p90": float(np.percentile(a, 90)),
        "b_p10": float(np.percentile(b, 10)), "b_p90": float(np.percentile(b, 90)),
        "L_p10": float(np.percentile(L, 10)), "L_p90": float(np.percentile(L, 90)),
        "a_std": float(a.std()), "b_std": float(b.std()), "L_std": float(L.std()),
        "hue_spread": float(1 - np.hypot(hx, hy) / max(np.sum(s), 1e-9)),   # 0 = one hue, ->1 = scattered
    }


class ThicknessQC:
    """Predicts section thickness (nm) with the colour random forest.

    the pipeline treats it like the other QC modules. The input is the segmented section crop (BGR, black background), used at
    its native size: the feature code (15 px erode) depends on the resolution the model was
    trained at (640x640), so the image is NOT resized.
    """

    RANGE_NM = 20   # report the probability that the true thickness is within +-RANGE_NM

    def __init__(self, config):
        """
        Args:
            config: An instance of AlgorithmSettings (CONFIG.qc.thickness);
                    weights_path must point to thickness_rf.joblib
        """
        self.model = None
        self.features: List[str] = []
        self.classes: np.ndarray = np.array([])
        self.model_name = config.weights_path.replace("\\", "/").split("/")[-1]
        self.pass_labels = config.pass_labels
        self.min_confidence = config.min_confidence
        self.log = logging.getLogger(self.__class__.__name__)

        try:
            bundle = joblib.load(config.weights_path)
            self.model = bundle["model"]
            self.features = list(bundle["features"])
            self.classes = np.asarray(bundle["classes"])
            # warm-up so the first real frame is not slow
            self.model.predict_proba(np.zeros((1, len(self.features))))
            self.log.debug(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            self.model = None
            self.log.warning(f"Could not load {config.weights_path}. Error: {e}")

    def predict(self, image: np.ndarray) -> dict:
        if self.model is None:
            return {"error": "Model not loaded", "label": "Unknown", "conf": 0.0}
        if image is None:
            return {"error": "Input image is None", "label": "Error", "conf": 0.0}

        try:
            feats = extract(image)
            x = np.array([[feats[k] for k in self.features]])
            proba = self.model.predict_proba(x)[0]
            classes = self.model.classes_
            top = int(np.argmax(proba))
            nm = int(classes[top])
            near = np.abs(classes - nm) <= self.RANGE_NM
            return {
                "label": str(nm),
                "conf": round(float(proba[top]), 4),
                "metric": {
                    "thickness_nm": nm,
                    f"conf_within_{self.RANGE_NM}nm": round(float(proba[near].sum()), 4),
                },
                "error": None,
            }
        except Exception as e:
            self.log.error(f"Prediction failed for {self.model_name}: {e}")
            return {"error": str(e), "label": "Error", "conf": 0.0}

    def check(self, image: np.ndarray) -> dict:
        """Standard QC check: predict, then validate against the config criteria."""
        result = self.predict(image)
        if result.get("error"):
            result["pass_status"] = False
            return result

        is_valid_label = "ANY" in self.pass_labels or result.get("label") in self.pass_labels
        is_confident = result.get("conf", 0) >= self.min_confidence

        result["pass_status"] = bool(is_valid_label and is_confident)
        if not is_valid_label:
            result["reason"] = f"Thickness not accepted: {result.get('label')} nm"
        elif not is_confident:
            result["reason"] = f"Low Confidence ({result.get('conf')} < {self.min_confidence})"
        return result
