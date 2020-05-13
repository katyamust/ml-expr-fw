Fabric ML
==============================

Experimentation framework for data scientists and data science teams


## What does this do?
First, it builds on top of the awesome 
[CookieCutter-DataScience](https://drivendata.github.io/cookiecutter-data-science/) template which takes care of your project structure and python package setup.
Then, it adds more functionality for conducting structured, 
reproducible and robust experimentation.

Here's How:
##### 1. By defining a clear structure for the main building blocks

- Getting data
- Preprocessing, feature engineering and postprocessing
- Modeling
- Evaluation
- Running experiments
    
This structure brings easier testing, experimentation and model productization, 
as all models, preprocesors and postprocessors have the same API.

##### 2. By providing a template for all notebooks
All notebooks looks the same: Get the data, preprocess it, fit a model, evaluate. 
Why not make this standardized so you won't have any errors?

This template helps you set up new experiments easier. The template is written in a text file so it's easier to source-control it.
See this example: [template_example](notebook_templates/example_template.md).

Generate a new notebook using:
```sh
python generate_notebook --name my_new_notebook.ipynb
```

See notebook generation options:
```sh
python generate_notebook --help
```

##### 3. By getting experiment logging for free
The experiments are structured in a way, that you will always be able to reproduce any experiment, 
as code, hyperparams, data versions and metrics are stored in one of the provided experimentation engines 

##### 4. By leveraging the CookieCutter Data science project structure

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
