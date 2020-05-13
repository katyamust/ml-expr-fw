# Sentiment Analysis template

## Experiment description
Describe the flow in this notebook here


##### Jupyter helpers:

```python
%reload_ext autoreload
%autoreload
```

Define imports

```python
import pickle

import pandas as pd
import numpy as np
import spacy

from src import ExperimentRunner, Evaluator, EvaluationResult
from src.data_loader import BlobDataLoader
from src.preprocessing import SpacyPreprocessor
from src.evaluation import SentimentEvaluator
from src.models import BaseModel
from src.experimentation import MlflowExperimentation

```

## Load data
```python
DATASET_NAME = "ABC"
DATASET_VERSION = "4.0"

data_loader = BlobDataLoader(
    dataset_name=DATASET_NAME,
    dataset_version=DATASET_VERSION)

data_loader.download_dataset()
dataset_for_modeling = data_loader.get_dataset()
pickle_data = pickle.load(dataset_for_modeling)
X_train, y_train = pickle_data['X_train'], pickle_data['y_train']
X_test, y_test = pickle_data['X_test'], pickle_data['y_test']
```

Define experimentation logger (which logs params, metrics and code)
```python
experiment_logger = MlflowExperimentation()
``` 

Create preprocessor / use existing
```python

preprocessor = SpacyPreprocessor(model='en_core_web_sm')

```

Define new model/logic:
```python
class MyModel(BaseModel):

    def __init__(self, preprocessor)
        # Define your model fields here
        pass

    def fit(self, X, y=None, **fit_params) -> None:
        # Put your model fit logic here        
        pass

    def predict(self, X):
        pub your model predict logic here
        pass



my_model = MyModel(preprocessor = preprocessor, experiment_logger = experiment_logger)

my_model.fit(X=X_train, y=y_train)
```

Define evaluation
```python

evaluator = SentimentEvaluator()
```


Run experiment

```python
experiment_runner = ExperimentRunner(model = my_model,
                                     X_train=X_train,
                                     y_train=y_train,
                                     X_test=X_test,
                                     y_test=y_test,
                                     dataset_name=DATASET_NAME,
                                     dataset_version=DATASET_VERSION,
                                     log_experiment=True,
                                     experiment_logger=experiment_logger,
                                     evaluator=evaluator,
                                     experiment_name="My experiment")

results = experiment_runner.evaluate()

```