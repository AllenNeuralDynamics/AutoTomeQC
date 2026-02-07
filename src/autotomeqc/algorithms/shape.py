# autotomeqc/algorithms/shape.py
from typing import Optional
import cv2
import logging
from pathlib import Path
from autotomeqc.utils.io import save_debug_image

class ShapeQC:
    def __init__(self, config, output_dir: Path):
        """
        Args:
            config: An instance of AlgorithmSettings (CONFIG.qc.shape).
            output_dir (Path): The global output directory Path object.
        """
        self.shape_config = config
        self.output_dir = output_dir
        self.save_debug_img = getattr(config, "save_debug_img", True)
        
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info("ShapeQC initialized.")

    def check(self, image, filename: Optional[str] = None):
        """
        Analyzes the section shape (Diamond vs Hexagon).
        
        Args:
            image (np.ndarray): The segmented image section.
            filename (str, optional): The base filename for saving debug images.
        """
        try:
            # 1. Validation
            if image is None:
                return {"pass": False, "error": "Input image is None", "label": "Error"}

            # 2. Preprocessing (Grayscale + Threshold)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

            # 3. Find Contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                self.log.warning("ShapeQC: No contours found.")
                return {"pass": False, "error": "No contours found", "label": "Empty"}

            # Assume largest contour is the section
            cnt = max(contours, key=cv2.contourArea)

            # 4. Convex Hull & Approx Polygon
            hull = cv2.convexHull(cnt)
            
            # Epsilon: 3% of arc length is a standard "sweet spot" for shape approximation
            epsilon = 0.03 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            num_vertices = len(approx)

            # 5. Classification Logic
            # <= 5 vertices -> Diamond (allows for blunt corners), >= 6 -> Hexagon
            if num_vertices <= 5:
                shape_label = "Diamond"
            else:
                shape_label = "Hexagon"

            # 6. Pass/Fail Logic (Noise filter)
            if cv2.contourArea(cnt) < 100:
                return {"pass": False, "error": "Contour too small", "label": "Noise"}

            self.log.debug(f"ShapeQC: Detected {shape_label} (v={num_vertices})")

            # 7. Visualization & Saving Logic
            # Check if saving is enabled in config
            if self.save_debug_img:
                vis_img = image.copy()
                
                # Draw the approximated polygon (Green)
                cv2.drawContours(vis_img, [approx], -1, (0, 255, 0), 2)
                
                # Draw the vertices (Red Dots)
                for point in approx:
                    x, y = point[0]
                    cv2.circle(vis_img, (x, y), 3, (0, 0, 255), -1)
                
                # Add Text Label
                text = f"{shape_label} (v={num_vertices})"
                cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 255), 2)

                # Save the image if filename is provided
                if filename:
                    save_path = self.output_dir / f"{filename}_shape.jpg"
                    save_debug_image(vis_img, save_path)
                    self.log.debug(f"Saved shape debug image to {save_path}")

            return {
                "label": shape_label,
                "metric": num_vertices,
                "message": f"Detected {shape_label} (vertices={num_vertices})",
                "pass": True,
            }

        except Exception as e:
            self.log.error(f"ShapeQC crashed: {e}")
            return {"pass": False, "error": str(e), "label": "Error"}