from abc import ABC, abstractmethod

from src.evaluation import EvaluationResult


class Evaluator(ABC):
    """
    Holds the logic for evaluating model results
    """

    @abstractmethod
    def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Method for running evaluations on model predictions
        :param kwargs: for example: predicted, actual (or other inputs required for evaluation)
        :return: EvaluationResult
        """
        pass
