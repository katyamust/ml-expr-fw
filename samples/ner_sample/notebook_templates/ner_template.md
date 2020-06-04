# Experiment template

*This is an experiment template, used to auto-generate notebooks for new experiments*

## Experiment description

##### Jupyter helpers

```python
%reload_ext autoreload
%autoreload
```

Define imports

```python
from flair.data import Corpus

from ner_sample.data import ConllDataLoader
from ner_sample.experimentation import MlflowExperimentation
from ner_sample.evaluation import NEREvaluator
from ner_sample import NERExperimentRunner
```

## Load data
Download (if missing) the Conll-2003 dataset from Github and load it into memory using a flair Corpus object

```python
data_loader = ConllDataLoader()
data_loader.download_dataset()
corpus: Corpus = data_loader.get_dataset()
corpus
```

Define experimentation object, which will be used for logging the experiments parameters, metrics and artifacts

```python
experimentation = MlflowExperimentation()
```

Model:

```python
class MyNERModel(BaseModel):
    def fit(self, corpus, y=None) -> None:
        pass

    def predict(self, X):
        pass

```

```python
model = MyNERModel()

model.fit(corpus)
```

Define evaluation

```python
evaluator = NEREvaluator()
```

Run experiment

```python
experiment_runner = NERExperimentRunner(
    model=model,
    corpus=corpus,
    data_loader=data_loader,
    log_experiment=True,
    experiment_logger=experimentation,
    evaluator=evaluator,
    experiment_name="Experiment"
)
```

```python
# results = experiment_runner.run()
results = experiment_runner.evaluate()
print(results)
```
