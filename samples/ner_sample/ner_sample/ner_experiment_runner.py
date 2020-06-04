import logging

from flair.data import Corpus

from ner_sample import LoggableObject
from ner_sample.data import ConllDataLoader
from ner_sample.evaluation import EvaluationMetrics, StepEvaluationMetrics, NEREvaluator
from ner_sample.experimentation import Experimentation
from ner_sample.models import BaseModel

logger = logging.getLogger(__name__)


class NERExperimentRunner:
    def __init__(
        self,
        model: BaseModel,
        corpus: Corpus,
        data_loader: ConllDataLoader,
        evaluator: NEREvaluator,
        log_experiment: bool = True,
        experiment_logger: Experimentation = None,
        experiment_name: str = None,
        **experiment_params_to_log,
    ):
        """
        Runs one model, evaluates results, and stores all parameters
        and metrics in the experimentation service

        :param model: model instance (of type BaseModel)
        :param corpus: The object holding the various datasets
        :param y_train: Training set tagged vald be evaluated on
        :param X_test: Test set tagged vald be evaluated on
        :param y_test: Test set tagged values (labels)
        :param data_loader: DataLoader instance used to load data
        :param dataset_version: Version of raw dataset used
        :param evaluator: Logic for model and results evaluation
        :param log_experiment: Whether to log this experiment
        into the experimentation service or not
        :param experiment_logging: Experimentation service instance
        (e.g. MlflowExperimentation)
        :param experiment_name: Name of experiment,
        to be used by the experimentation service

        :example:

        # Call experiment runner and log all objects' parameters:
        experiment_runner = ExperimentRunner(
            model=mock_model,
            corpus=corpus,
            data_loader=data_loader,
            evaluator=evaluator,
            experiment_logger=experiment_logger,
            experiment_name="Text",
            )

        # Option 1: Fit, predict and evaluate :
        results = experiment_runner.run()
        print(results)

        # Option 2: Predict and evaluate ((model is already fitted):
        results = experiment_runner.evaluate()
        print(results)

        # Option 3: Run each part separately:
        experiment_runner.fit_model()
        experiment_runner.predict()
        results = experiment_runner.evaluate()

        """
        self.model = model
        self.corpus = corpus
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.experiment_logger = experiment_logger
        self.log_experiment = log_experiment
        self.experiment_name = experiment_name
        self._evaluation_metrics = []  # Metrics gathered during experiment
        self._predictions = []  # Predictions gathered during experiment

        logger.info(f"Starting experiment: {self.experiment_name}...")

        if self.log_experiment:
            if not self.experiment_logger:
                raise ValueError(
                    "Experimentation system not passed, cannot log experiment"
                )

            if not self.experiment_name:
                raise ValueError(
                    "Experiment name must be specified for the experiment logging system"
                )

            logger.info(f"Connecting to {self.experiment_logger.name}")
            self.experiment_logger.set_experiment(name=experiment_name)
            self.experiment_logger.start_run()

            self._log_loggable_object(model, "Model")
            self._log_loggable_object(evaluator, "Evaluator")
            self._log_loggable_object(data_loader, "DataLoader")

            if model.preprocessor:
                self._log_loggable_object(model.preprocessor, "Preprocessor")
            if model.postprocessor:
                self._log_loggable_object(model.postprocessor, "Postprocessor")

            # Log additional inputs to this class
            if experiment_params_to_log:
                logger.info(
                    f"Logging these experiments as well: {experiment_params_to_log}"
                )
                self.experiment_logger.log_params(experiment_params_to_log)

    def _log_loggable_object(self, loggable_object: LoggableObject, object_name):
        if loggable_object:
            params = loggable_object.get_params()
            metrics = loggable_object.get_metrics()

            self.experiment_logger.log_params(params if params else {})
            self.experiment_logger.log_metrics(metrics if metrics else {})
            self.experiment_logger.log_param(object_name, loggable_object.name)

    def run(self) -> EvaluationMetrics:
        """
        Performs model fitting and evaluation
        :return: evaluation results
        """
        self.fit_model()

        self.predict()

        evaluation_result = self.evaluate()

        return evaluation_result

    def fit_model(self) -> None:
        logger.info(f"Fitting model {self.model.name}")

        self.model.fit(X=self.corpus, y=None)

    def predict(self):
        """
        Calls the model predict function with the input X_test
        :return: None
        """
        logger.info(f"Running model.predict() using model {self.model.name}")
        # Copy gold NER to new label and assign O to all ner labels (to be populated during inference)
        for sentence in self.corpus.test:
            # Move gold tags to a new tag (gold_ner)
            [
                token.add_tag_label("gold_ner", token.get_tag("ner"))
                for token in sentence.tokens
            ]
            # Erase tags prior to prediction to verify that target tags aren't leaking
            [token.set_label("ner", value="O") for token in sentence.tokens]

        self._predictions = self.model.predict(self.corpus.test)

    def evaluate(self) -> EvaluationMetrics:
        """
        Runs evaluation on the given model and test set
        :return: EvaluationResult
        """

        if self._predictions is None:
            logger.info("Predictions not found, running model.predict")
            self.predict()

        evaluation_result = self.evaluator.evaluate(self._predictions)
        if self.log_experiment:
            if isinstance(evaluation_result, StepEvaluationMetrics):
                for step in evaluation_result.get_steps():
                    step_metrics = evaluation_result.get_metrics(step=step)
                    self.experiment_logger.log_metrics(step_metrics)
            else:
                self.experiment_logger.log_evaluation_result(evaluation_result)

        self._evaluation_metrics = evaluation_result
        return self._evaluation_metrics

    def get_predictions(self):
        """
        Get already calculated predictions.
        :return: Model predictions on X_test
        """
        if self._predictions is None:
            logger.info(
                "Model was not predicted. Make sure you called `predict_model()` to calculate predictions"
            )
            return None
        else:
            return self._predictions

    def get_evaluation_metrics(self):
        if self._evaluation_metrics is None:
            logger.info(
                "Evaluation metrics are empty. "
                "Make sure you ran a full experiment (train, predict and evaluate) "
                "prior to calling this method."
            )
            return None
        else:
            return self._evaluation_metrics
