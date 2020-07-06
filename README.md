# Fabric ML

Generic experimentation framework for data scientists and data science teams.
It provides a quick starter code for reproducible machine learning.

![Experiment flow](assets/experiment-flow.png)

## TL;DR: What this framework does

- First, it builds on top of the awesome
[CookieCutter-DataScience](https://drivendata.github.io/cookiecutter-data-science/) template which takes care of your project structure and python package setup.
- Then, it provides functionality for conducting rigorous experimentation: structured, reproducible and robust.

It does so by providing these main capabilities:

1. **Collaborate** easily by defining a clear structure to the main building blocks used during model experimentation
2. **Experiment** easily by defining a project template for all Jupyter notebooks
3. **Reproduce** easily by automagically adding experiment logging capabilities to every experiment
4. **Productize** easily by defining code in classes (with defined APIs)

## How to install
To create a new package for a new experiment:

a. Install `cookiecutter` (preferably in a virtual environment):
```
pip install cookiecutter
```

or (through conda):

```
conda config --add channels conda-forge
conda install cookiecutter
```

b. In an empty folder, generate the folder structure and prebuilt classes:

```sh
cookiecutter git+https://csefy19.visualstudio.com/CSECodeShare/_git/MLExperimentationFramework
```

c. From the root of the folder, install requirements:

```sh
conda create -n ENV_NAME python=3.7
conda activate ENV_NAME
pip install -e .
```

## Example usage
- [This notebook](samples/ner_sample/notebooks/flair_ner.html) shows the full implementation of a Named Entity Recognition problem
- [This notebook](samples/ner_sample/notebooks/flair_ner_clean.html) shows the same, after the modules have been implemented (final outcome)

## Using this framework? 
Trying it out? decided not to? We'd like to hear. Please fill in [this (very) short survey](https://forms.office.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR9LagCGNW01LpMix2pnFWFJUOE1ROElWTldGWTNGRVU4WkM0NE9GOTYzNi4u)

## Why use this framework

Machine learning experimentation is a complex process which can take many months, 
and involve multiple team members and stakeholders. 
Once a good model is developed, the process of operationalizing the new model begins, in which we 
develop the model as a software module, maintain it over time 
and constantly re-evaluate it against newly developed versions.

This package structures the experimentation process differently 
by standardizing model development across experiments and team members, 
tracking model results over time, and by turning research outputs into software components 
capable of being integrated into a production system. 

More specifically, here are the main benefits you get by using this framework:

### Clear structure for the main building blocks

- **DataLoader**: Unified code for getting data
- **DataProcessor**: For data preprocessing, feature engineering and postprocessing
- **BaseModel**: scikit-learn like API wrapper for all models
- **Evaluator**: Define your metrics in an explicit and traceable way
- **ExperimentRunner**: Runs an entire cycle, from data loading,
through preprocessing, model fit, predict and evaluate on new data.
In addition, it takes parameters and metrics from all of the above
and stores a unified and reproducible representation of your experiment.

### Project template for all Jupyter notebooks

All notebooks look the same: Get the data, preprocess it, fit a model, evaluate.
Why not make this standardized so you won't have any errors?

### Implicit and full experiment logging

The experiments are structured in a way, that you will always be able to reproduce any experiment,
as code, hyperparams, data versions and metrics are stored using your favorite experiment logger (mlflow or others)

### Production ready project and package structure
This framework is built on top the CookieCutter Data-Science template: 
- docs: <https://drivendata.github.io/cookiecutter-data-science/>
- folder structure: <https://github.com/drivendata/cookiecutter-data-science/#the-resulting-directory-structure>


## Getting started with experimentation

The easiest way to get started is to generate a notebook from a template,
and implement the basic building blocks (loading data, preprocessing, modeling).

The notebook template is a markdown file which can be used to set up new experiments easier. 
The template is a text file so maintaining it on source control is easy.

For an example template, [see this template](<{{ cookiecutter.repo_name }}/notebook_templates/example_template.md>).

To generate a new notebook:

```sh
python generate_notebook.py --name my_new_notebook.ipynb
```


## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit <https://cla.microsoft.com.>

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.
