# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project focused on predicting customer churn using clean code. That means parametrizing 
as much as possible and storing relevant insights for continuous monitoring

## Files and data description
Overview of the files and data present in the root directory. 

## Running Files
How do you run your files? What should happen when you run your files?
This project uses poetry to manage dependencies. 
To run it in your environment, first run `pip install poetry` to be sure it is installed.
The development environment should first be locked by running `poetry shell`.
Dependencies are installed by running `poetry install`.
In case any dependency needs to be updated, one only needs to run `poetry update` or `poetry update <package>`.

The project requires python version 3.8

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
Tests can be run by calling `python churn_script_logging_and_tests.py`



