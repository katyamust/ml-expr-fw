# Experiment template
*This is an experiment template, used to auto-generate notebooks for new experiments*

## Experiment description



##### Jupyter helpers:

```python
%reload_ext autoreload
%autoreload
```

Define imports

```python
import pandas as pd
import numpy as np

from src import ExperimentRunner, Evaluator, EvaluationResult
from src.models import BaseModel
from src.experimentation import MlflowExperimentation

```

## Load data
*replace MyDataLoader with your DataLoader implementation*

```python
data_loader = MyDataLoader()
data_loader.download_dataset()
dataset_for_modeling = data_loader.get_dataset()
pickle_data = pickle.load(dataset_for_modeling)
X_train, y_train = pickle_data['X_train'], pickle_data['y_train']
X_test, y_test = pickle_data['X_test'], pickle_data['y_test']
```

Define experimentation object, which will be used for logging the experiments parameters, metrics and artifacts
*Replace MlflowExperimentation if you use a different experimentation system*
```python
experimentation = MlflowExperimentation()
``` 

Create preprocessor
```python
class MyPreprocessor(Prepreprocessor):
    def preprocess(self, X):
        pass


preprocessor = MyPreprocessor()

```

Create model/logic:
```python
class MyModel(BaseModel):
    def fit(self, X, y=None, **fit_params) -> None:
        pass

    def predict(self, X):
        pass


my_model = MyModel(preprocessor = preprocessor)
```

Define evaluation
```python
class MyEvaluator(Evaluator):
    def evaluate(self, **kwargs) -> EvaluationResult:
        pass

evaluator = MyEvaluator()
```


Run experiment

```python
experiment_runner = ExperimentRunner(model = my_model,
                                     X_train=X_train,
                                     y_train=y_train,
                                     X_test=X_test,
                                     y_test=y_test,
                                     log_experiment=True,
                                     experimentation=experimentation,
                                     evaluator=evaluator,
                                     experiment_name="My experiment")

```