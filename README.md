# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity.
The project contains code to analyze and model a bank's customer churn

## Project Description
The project focused on predicting customer churn using clean code. That means parametrizing 
as much as possible and storing relevant insights, models and performance metrics for continuous monitoring in production.
The main code is contained in a single file and can be run with minimal effort, helping deployment to a production environment.

## Files and data description
The package contains the following files:

- data: folder containing the data to be processed
    - bank_data.csv

- images: folder to store images results
    - eda: folder containing plots made in exploratory data analysis
    - results: folder storing model results and performance metrics
        - feature_importance
        - logistic regression results
        - random forest results
        - roc curve results
        - shap summary plot

- logs: folder storing logs from the code runs
    - churn_library.log

- models: folder storing the models trained for production deployment
    - logistic_model.pkl
    - rfc_model.pkl

- churn_library.py: main file containing the code to run data analysis and modelling

- churn_notebook.ipynb: Jupyter notebook exemplifying how the code should be used

- churn_script_logging_and_tests.py: file testing and logging the code from churn_library.py

- config.yaml: configuration file storing all constants needed to run the code, avoiding hard-coding

- poetry.lock: file managing dependencies. Should NOT be updated by hand

- pyproject.toml: file managing main requirements versions. Should NOT be updated by hand

- test_config.yaml: config file used for testing. The only difference with config.yaml is that it uses
a smaller grid for the models parameter gridsearch. It makes sure that the tests run in a shorter amount of time
than the actual code. Also, one can updated it as needed to test specific combinations of parameters in the 
complete pipeline without breaking production pipelines

## Running Files
This project uses poetry to manage dependencies. 
To run it in your environment, first run `pip install poetry` to be sure it is installed.
The development environment should first be locked by running `poetry shell`.
Dependencies are installed by running `poetry install`.
In case any dependency needs to be updated, one only needs to run `poetry update` or `poetry update <package>`.
These will automatically update `poetry.lock` and `pyproject.toml`.

The project requires python version 3.8

An example (sort of documentation) of how the code should be used is in `churn_notebook.ipynb`. 
`churn_script_logging_and_tests` can give insights on this as well. In general, initializing the `ChurnPipeline`
coded in `churn_library.py` already performs exploratory data analysis, feature encoding, feature engineering as well
as training and storing the models.
Performance metrics (ROC curves, Feature importances and model performance metrics) need to be called by calling the relevant
methods from `ChurnPipeline`

### Config file
An example on how to run the whole pipeline can be found in `churn_notebook.ipynb`; the 
pipeline defined in `churn_library.py` requires the path to the .csv data and the config file.
The config is a .yaml file that should be rather self-explanatory. It contains:

- column_types: dict separating the raw columns stored in data between categorical & quant columns;
this is needed for encoding columns and finalizing the features for the model
- target column name: dict giving the name for the target column; needed to name the target and encoded columns consistently
- plot dictionary for eda: dict keeping track of the features to be explored during eda; each feature (dict keys) has a value 
corresponding to the function used to make the relevant plot (_histogram, _value_counts, _distribution).
- parameters grid for gridsearch: dict with the parameters values to be explored during gridsearch; when testing the function,
a `test_config.yaml` file is used which holds a simplified version of this dict in order to save time
- test/train set split: dict giving the fraction of data used to make the model's test set; should usually be set to .2-.3

## Testing
Tests can be run by calling `python churn_script_logging_and_tests.py`; results will be stored in folder `logs`



