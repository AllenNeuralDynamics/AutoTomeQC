# autotomeqc/algorithms/thickness.py
"""Section thickness QC from interference colour, using a random forest.

The model (weights/thickness_rf.joblib) was trained on 17 colour features
(HSV, Lab, colour spread) of segmented sections on a black background.
"""
import logging
from typing import List, Optional

import cv2
import joblib
import numpy as np

from autotomeqc.core.models import QCCriteria


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


def normalise_brightness(bgr: np.ndarray, kept: np.ndarray, target: float) -> np.ndarray:
    """Uniform gain so the median V of the used pixels equals `target` (removes session brightness)."""
    gain = target / max(float(np.median(bgr.max(axis=2)[kept])) / 255, 1e-3)
    return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def extract(bgr: np.ndarray, norm_v: Optional[float] = None) -> dict:
    """The colour features the forest was trained on (HSV, Lab, colour spread).

    norm_v: brightness the section is normalised to before measuring (the model's saved `norm_v`).
    None = no normalisation, for models trained before normalisation was added.
    """
    _, kept = masks(bgr)
    if norm_v is not None:
        bgr = normalise_brightness(bgr, kept, norm_v)
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

    Same interface as YOLOClassifier (`predict` and `check` both return a QCCriteria),
    so the pipeline treats it like the other QC modules.
    The input is the segmented section crop (BGR, black background), used at
    its native size: the feature code (15 px erode) depends on the resolution the model was
    trained at (640x640), so the image is NOT resized.
    """

    def __init__(self, config):
        """
        Args:
            config: An instance of ThicknessSettings (CONFIG.qc.thickness);
                    weights_path must point to thickness_rf.joblib
        """
        self.model = None
        self.features: List[str] = []
        self.classes: np.ndarray = np.array([])
        self.norm_v: Optional[float] = None
        self.model_name = config.weights_path.replace("\\", "/").split("/")[-1]
        self.min_confidence = config.min_confidence
        self.tolerance_nm = config.tolerance_nm   # one tolerance: used for the target check and the P(within) report
        self.log = logging.getLogger(self.__class__.__name__)

        try:
            bundle = joblib.load(config.weights_path)
            self.model = bundle["model"]
            self.features = list(bundle["features"])
            self.classes = np.asarray(bundle["classes"])
            self.norm_v = bundle.get("norm_v")   # None for models trained without brightness normalisation
            # warm-up so the first real frame is not slow
            self.model.predict_proba(np.zeros((1, len(self.features))))
            self.log.debug(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            self.model = None
            self.log.warning(f"Could not load {config.weights_path}. Error: {e}")

    def predict(self, image: np.ndarray) -> QCCriteria:
        """Runs the forest. Here pass_status=True only means 'the prediction ran'; check() decides QC."""
        if self.model is None:
            return QCCriteria(pass_status=False, label="Unknown", conf=0.0, message="Model not loaded")
        if image is None:
            return QCCriteria(pass_status=False, label="Error", conf=0.0, message="Input image is None")

        try:
            feats = extract(image, self.norm_v)
            x = np.array([[feats[k] for k in self.features]])
            proba = self.model.predict_proba(x)[0]
            classes = self.model.classes_
            nm = int(classes[int(np.argmax(proba))])
            within = float(proba[np.abs(classes - nm) <= self.tolerance_nm].sum())
            return QCCriteria(
                pass_status=True,
                label=str(nm),
                conf=round(float(proba.max()), 4),
                message=f"Predicted {nm} nm; P(within ±{self.tolerance_nm:g} nm) = {within:.2f}",
            )
        except Exception as e:
            self.log.error(f"Prediction failed for {self.model_name}: {e}")
            return QCCriteria(pass_status=False, label="Error", conf=0.0, message=str(e))

    def check(self, image: np.ndarray, expected_nm: Optional[float] = None) -> QCCriteria:
        """Standard QC check: predict, then validate against the config criteria.

        Passes when the confidence is >= min_confidence and, if expected_nm is given (e.g. the
        instrument setting), the prediction is within the config's tolerance_nm of it.
        Without expected_nm it only reports the thickness.
        """
        result = self.predict(image)
        if not result.pass_status:  # prediction failed: return the error result as is
            return result

        is_confident = (result.conf or 0.0) >= self.min_confidence

        within_tolerance = True   # no expected_nm given = nothing to compare, so this check does not apply
        if expected_nm is not None:
            diff = float(result.label) - expected_nm   # label is the predicted thickness in nm, as text
            within_tolerance = abs(diff) <= self.tolerance_nm
            result.message = (f"{result.message}; expected {expected_nm:g} nm "
                              f"(off by {diff:+g} nm, tolerance ±{self.tolerance_nm:g} nm)")

        result.pass_status = is_confident and within_tolerance
        if not is_confident:
            result.reason = f"Low Confidence ({result.conf} < {self.min_confidence})"
        elif not within_tolerance:
            result.reason = (f"Thickness {result.label} nm is more than ±{self.tolerance_nm:g} nm "
                             f"from the {expected_nm:g} nm target")
        return result
