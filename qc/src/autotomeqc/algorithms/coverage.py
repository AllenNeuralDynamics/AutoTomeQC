from .base import YOLOClassifier

class SectionCoverageQC(YOLOClassifier):
    def __init__(self, config):
        """
        Args:
            config: An instance of AlgorithmSettings (CONFIG.qc.thickness_consistency)
        """
        super().__init__(
            model_path=config.weights_path,
            img_size=config.img_size,
            pass_labels=config.pass_labels,
            min_conf=config.min_confidence
        )