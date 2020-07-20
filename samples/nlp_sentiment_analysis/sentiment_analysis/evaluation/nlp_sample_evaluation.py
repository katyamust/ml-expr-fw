from sentiment_analysis.evaluation import Evaluator
from sentiment_analysis.evaluation import NLPSampleEvaluationMetrics
from sklearn.metrics import accuracy_score


class NLPSampleEvaluator(Evaluator):
    """
    Class to hold the logic for how the model is evaluated.
    """
    def __init__(self):
        super().__init__()

    def evaluate(self, predicted, actual) -> NLPSampleEvaluationMetrics:
        # This is where actual evaluation takes place.
        val_score = accuracy_score(actual, predicted)
        return NLPSampleEvaluationMetrics(
            validation_score=val_score
        )