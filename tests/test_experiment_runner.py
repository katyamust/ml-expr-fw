from mlfabric import ExperimentRunner, Evaluator, EvaluationResult
from mlfabric.experimentation import Experimentation, MlflowExperimentation
from mlfabric.models import BaseModel


class MockModel(BaseModel):
    def __init__(self,
                 model_name=None,
                 **hyper_params):
        self.x = None
        super().__init__(model_name=model_name, hyper_params=hyper_params)

    def fit(self, X, y=None, experimentation: Experimentation = None, **fit_params) -> None:
        self.x = X

    def predict(self, X):
        return self.x == X


class MockEvaluationResult(EvaluationResult):
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall

    def get_metrics(self):
        return {"precision": self.precision, "recall": self.recall}


class MockEvaluator(Evaluator):

    def evaluate(self, predicted, actual) -> EvaluationResult:
        return MockEvaluationResult(0.5, 0.7)


def test_experiment_runner():
    X_train = [1, 2, 3, 4, 5]
    y_train = [1, 1, 1, 0, 0]
    X_test = [1, 2, 3, 4, 4]
    y_test = [1, 1, 1, 1, 1]

    model = MockModel(model_name="Mock", param1="hello", param2="world")
    evaluator = MockEvaluator()
    experiment_logger = MlflowExperimentation()
    experiment_runner = ExperimentRunner(model=model,
                                         X_train=X_train,
                                         X_test=X_test,
                                         y_train=y_train,
                                         y_test=y_test,
                                         dataset_name="mock dataset",
                                         dataset_version="V1",
                                         evaluator=evaluator,
                                         experiment_logger=experiment_logger,
                                         experiment_name="Text")
    results = experiment_runner.run()

    assert results.precision == 0.5
    assert results.recall == 0.7
