import logging

from src.evaluation import EvaluationResult, Evaluator
from src.experimentation import Experimentation
from src.models import BaseModel


class ExperimentRunner:
    def __init__(
        self,
        model: BaseModel,
        X_train,
        X_test,
        dataset_name: str,
        dataset_version: str,
        evaluator: Evaluator,
        y_test=None,
        y_train=None,
        log_experiment: bool = True,
        experiment_logger: Experimentation = None,
        experiment_name: str = None,
        **experiment_params_to_log,
    ):
        """
        Runs one model, evaluates results, and stores all parameters
        and metrics to the experimentation service
        :param model: model instance (of type BaseModel)
        :param X_train: The set the model should be evaluated on
        :param y_train: Training set tagged vald be evaluated on
        :param X_test: Test set tagged vald be evaluated on
        :param y_test: Test set tagged values (labels)
        :param dataset_name: Name of raw dataset used
        :param dataset_version: Version of raw dataset used
        :param evaluator: Logic for model and results evaluation
        :param log_experiment: Whether to log this experiment
        into the experimentation service or not
        :param experiment_logging: Experimentation service instance
        (e.g. MlflowExperimentation)
        :param experiment_name: Name of experiment,
        to be used by the experimentation service
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.evaluator = evaluator
        self.experiment_logger = experiment_logger
        self.log_experiment = log_experiment
        self.experiment_name = experiment_name
        self._metrics = []  # Metrics gathered during experiment

        logging.info(f"Starting experiment: {self.experiment_name}...")

        if self.log_experiment:
            if not self.experiment_logger:
                raise ValueError(
                    "Experimentation system not passed, cannot log experiment"
                )

            if not self.experiment_name:
                raise ValueError(
                    "Experiment name must be specified for the experiment logging system"
                )

            logging.info(f"Connecting to {self.experiment_logger.name}")
            self.experiment_logger.set_experiment(name=experiment_name)
            self.experiment_logger.start_run()

            if model.hyper_params:
                self.experiment_logger.log_params(model.hyper_params)
            if experiment_params_to_log:
                self.experiment_logger.log_params(experiment_params_to_log)

            self.experiment_logger.log_param("model_name", self.model.model_name)
            self.experiment_logger.log_param("dataset_name", self.dataset_name)
            self.experiment_logger.log_param("dataset_ver", self.dataset_version)

    def run(self) -> EvaluationResult:
        """
        Performs model fitting and evaluation
        :return: evaluation results
        """
        self.fit_model()

        evaluation_result = self.evaluate()

        return evaluation_result

    def fit_model(self) -> None:
        logging.info(
            f"Fitting model {self.model.model_name} on {len(self.X_train)} samples"
        )

        self.model.fit(X=self.X_train, y=self.y_train)

    def evaluate(self) -> EvaluationResult:
        """
        Runs evaluation on the given model and test set
        :return: EvaluationResult
        """

        predictions = self.model.predict(X=self.X_test)

        evaluation_result = self.evaluator.evaluate(predictions, self.y_test)
        self.experiment_logger.log_evaluation_result(evaluation_result)

        return evaluation_result
