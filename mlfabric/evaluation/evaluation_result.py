import json
from abc import ABC, abstractmethod
from typing import Dict


class EvaluationResult(ABC):
    """
    Class which holds the evaluation output for one model run.
    For example, precision or recall, MSE, accuracy etc.
    """
    pass

    @abstractmethod
    def get_metrics(self) -> Dict:
        """
        Return the evaluation result's metrics you wish to be stored in the experiment logging system
        :return: A dictionary with names of values of metrics to store
        """
        pass
