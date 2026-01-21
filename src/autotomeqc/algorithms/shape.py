import cv2
import numpy as np

def check_shape(image: np.ndarray) -> dict:
    """
    Determines if tissue is Hexagon or Diamond using contours.
    """
    if image is None:
        return {"pass": False, "label": "No Image", "vertices": 0}

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"pass": False, "label": "Empty", "vertices": 0}

    # Analyze largest contour
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
    vertices = len(approx)

    # Classify
    if vertices == 4:
        label = "Diamond"
    elif vertices == 6:
        label = "Hexagon"
    else:
        label = f"Irregular ({vertices})"

    return {
        "pass": True, # Shape usually just logs the type, unless you want to enforce one
        "label": label,
        "vertices": vertices
    }