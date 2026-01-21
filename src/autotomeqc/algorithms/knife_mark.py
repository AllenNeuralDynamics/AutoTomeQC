# autotome/algorithms/knife_mark.py
from .base import YOLOClassifier

class KnifeMarksQC(YOLOClassifier):
    def __init__(self, config):
        """
        Args:
            config: The entire 'qc' dictionary from config.yaml
        """
        qc_config = config["knife_mark"]
        super().__init__(
            model_path=qc_config["weights_path"], 
            img_size=qc_config["img_size"], 
            pass_labels=qc_config["pass_labels"], 
            min_conf=qc_config.get("min_confidence", 0.5)
        )
