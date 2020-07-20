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

from sentiment_analysis.data import NLPSampleDataLoader
from sentiment_analysis.models import SentimentClassifier
from sentiment_analysis.data_processing.text import SpacyTextProcessor
from sentiment_analysis.experimentation import MlflowExperimentation
from sentiment_analysis.evaluation import EvaluationMetrics, Evaluator
from sentiment_analysis import ExperimentRunner

```

## Load data
*replace MyDataLoader with your DataLoader implementation*

```python
data_loader = NLPSampleDataLoader("imdb", 1.0)
data_loader.download_dataset()
imdb_df_train, imdb_df_test = data_loader.get_dataset()

X_train, y_train = imdb_df_train['text'], imdb_df_train['label']
X_test, y_test = imdb_df_test['text'], imdb_df_test['label']
```

Define experimentation object, which will be used for logging the experiments parameters, metrics and artifacts
*Replace MlflowExperimentation if you use a different experimentation system*
```python
experimentation = MlflowExperimentation()
``` 

Create preprocessor for handling data preprocessing, feature engineering etc.
```python
preprocessor = SpacyTextProcessor()

```

Create model/logic:
```python
evaluator = NLPSampleEvaluator()
```


Run experiment

```python

from sentiment_analysis import ExperimentRunner
 
experiment_runner = ExperimentRunner(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    data_loader=data_loader,
    log_experiment=True,
    experiment_logger=experimentation,
    evaluator=evaluator,
    experiment_name="Experiment",
)

results = experiment_runner.run()
print(results)

```