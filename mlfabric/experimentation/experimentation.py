from abc import ABC, abstractmethod

from mlfabric.evaluation import EvaluationResult


class Experimentation(ABC):
    """
    Abstract class for model experimentation using various loggers (azure ml, mlflow)
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def set_experiment(self, name, artifact_location=None):
        """
        Set experiment name and parameters
        :param name: experiment name
        :param artifact_location: mlflow artifact_location
        :return: None
        """
        pass

    @abstractmethod
    def start_run(self):
        """
        Start logging
        :return: None
        """
        pass

    @abstractmethod
    def end_run(self):
        """
        End logging
        :return: None
        """
        pass

    @abstractmethod
    def log_param(self, key, value):
        """
        Log one parameter
        :param key: Param name
        :param value: Param value
        :return: None
        """
        pass

    @abstractmethod
    def log_params(self, params):
        """
        Log multiple parameters
        :param params: parameters to log
        :return: None
        """
        pass

    @abstractmethod
    def log_metric(self, key, value, step=None):
        """
        Log metric value.
        :param key: Metric name
        :param value: Metric value
        :param step: index of metric value within run (optional)
        :return: None
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics, step=None):
        """
        Log multiple metrics at once.
        :return:
        """
        pass

    @abstractmethod
    def log_image(self, title, fig):
        """
        Logs a matplotlib image
        :param title: Title of figure
        :param fig: Figure object
        :return:
        """
        pass

    def log_evaluation_result(self, evaluation_result: EvaluationResult):
        try:
            self.log_metrics(evaluation_result.get_metrics())
        except Exception as e:
            raise Exception(f"Failed to read the contents of the EvaluationResult "
                            f"based class as metrics. "
                            f"Consider implementing this method yourself or "
                            f"make sure your implementation contains "
                            f"only metrics as fields. "
                            f"Exception: {e}")
