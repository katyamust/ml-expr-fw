from sentiment_analysis.evaluation import EvaluationMetrics


class NLPSampleEvaluationMetrics(EvaluationMetrics):
    """
    Class to hold the actual values the evaluation created, e.g. precision, recall, MSE.
    """

    def __init__(self, validation_score):
        self.validation_score = validation_score
        super().__init__()

    def get_metrics(self):
        return {"validation_score": self.validation_score}

    def __repr__(self):
        return f"validation_score: {self.validation_score}"
