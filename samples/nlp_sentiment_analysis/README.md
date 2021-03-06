nlp_sentiment_analysis
==============================

Example repo with a Sentiment Analysis model


How to run
------------

It's advised to use a virtual env (like conda) for this project.
For conda, creating a new environment:
```sh
conda create -n nlp_sentiment_analysis python=3.7
conda activate nlp_sentiment_analysis
```

Install requirements
```sh
pip install -r requirements.txt
```

Install the package locally in interactive model
```sh
pip install -e .
```


Experiment flow
------------

Several objects are used throughout the experiment flow. Specifically:
- [DataLoader](sentiment_analysis/data/data_loader.py): For loading data
- [DataProcessor](sentiment_analysis/data_processing/data_processor.py): For pre and post processing (e.g. feature engineering)
- [Evaluator](sentiment_analysis/evaluation/evaluator.py): For defining the logic for evaluation
- [Experimentation](sentiment_analysis/experimentation/experimentation.py): For defining how the code, params and metrics are logged for future reference
- [BaseModel](sentiment_analysis/models/base_model.py): For defining the actual model logic (fit, predict)
- [ExperimentRunner](sentiment_analysis/experiment_runner.py): For orchestrating an experiment.

Here is an example flow:
See [](notebook_templates/example_template.md) For an example of an experiment structure

And this is the template for generating a new notebook for a new experiment:
See [](notebook_templates/notebook_template.md) For starter code for creating a new experiment flow


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── data_processing<- Classes for pre and post procesing logic
    │   │
    │   ├── evaluation     <- Classes for evaluating models and storing metrics
    │   │
    │   ├── experimentation<- Classes for managing experiment logging
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
