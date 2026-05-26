# autotome/algorithms/knife_mark.py
from .base import YOLOClassifier

class KnifeMarksQC(YOLOClassifier):
    def __init__(self, config):
        """
        Args:
            config: An instance of AlgorithmSettings (CONFIG.qc.knife_marks)
        """
        super().__init__(
            model_path=config.weights_path,
            img_size=config.img_size,
            pass_labels=config.pass_labels,
            min_conf=config.min_confidence
        )
