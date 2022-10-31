import os
import logging
from churn_library import ChurnPipeline
import pandas as pd
import yaml

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

with open("test_config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

cpl = ChurnPipeline("./data/bank_data.csv", config)


# test target column
# test eda stores right plot
# test encoder helper
# test feature engineering
# test model training
# test model storage
# test classification report images
# test roc storage
# test feature importances
def test_eda():
    '''
    test perform_eda class
    '''
    df = cls.import_data("./data/bank_data.csv")
    eda = cls.perform_eda(df)
    try:
        assert type(eda.shape()) is tuple
        logging.info("SUCCESS: perform_eda.shape type check")
    except AssertionError as err:
        logging.error("Performing eda - shape: the dataframe's shape is not a tuple")

    try:
        assert eda.shape()[0] > 0
        assert eda.shape()[1] > 0
        logging.info("SUCCESS: perform_eda.shape shape check")
    except AssertionError as err:
        logging.error("Performing eda - shape: the dataframe doesn't appear to have rows and columns")
        
    try:
        assert type(eda.check_nulls()) is pd.core.series.Series
        logging.info("SUCCESS: perform_eda.check_nulls type check")
    except AssertionError as err:
        logging.error(f"Performing eda - check_nulls: check_nulls should return a pandas.series, type {type(eda.check_nulls())} was returned instead")
    
    try:
        assert len(eda.check_nulls())==len(df.columns)
        logging.info("SUCCESS: perform_eda.check_nulls column number check")
    except AssertionError as err:
        logging.error(f"Performing eda - check_nulls: check_nulls should return as many values as there are columns in df, {len(eda.check_nulls())} was returned instead")
        
    description = eda.describe()
    try:
        assert len(description.index)==8
        logging.info("SUCCESS: perform_eda.describe row number check")
    except AssertionError as err:
        logging.error(f"Performing eda - describe: should return a dataframe with 8 rows, {len(description.index)} was returned instead")
        
    try:
        assert all(
            description.index == ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        logging.info("SUCCESS: perform_eda.describe row number check")
    except AssertionError as err:
        logging.error(f"Performing eda - describe: should return a dataframe with rows ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], {description.index} were returned instead")
        
    try:
        assert all(
            description.columns == df.select_dtypes(include=['int', 'float']).columns
        )
        logging.info("SUCCESS: perform_eda.describe: column names")
    except AssertionError as err:
        logging.error(f"Performing eda - describe: should return a dataframe with the same column names as df, {description.columns} were returned instead")
    
    # FIX QFontDatabase: Cannot find font directory /root/.hunter/_Base/371aef6/51136b1/f0b9fa5/Install/lib/fonts - is Qt installed correctly? ERROR BEFORE TESTING PLOTTING

def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = cls.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    encoded_df = cls.encoder_helper(df, config['cat_columns'], config['target'])
        
    try:
        # check that the right column names are created
        assert all([col+'_'+config['target'] in encoded_df.columns for col in config['cat_columns']])
        logging.info("SUCCESS: encoded column names created by function encoder_helper")
    except AssertionError as err:
        logging.error(f"FAILED: encoder_helper did not create the expected columns, expected {[col+'_'+config['target'] for col in config['cat_columns']]}, got {[col for col in encoded_df.columns if '_'+config['target'] in col]}")
        
    try:
        # check that output columns have same amount of values as their categories
        assert all([len(encoded_df[col+'_'+config['target']].unique()) == len(
    df[col].unique()) for col in config['cat_columns']])
        logging.info("SUCCESS: encoder helper created the correct amount of values")
    except AssertionError as err:
        logging.error(f"FAILED: a different amount of values was created compared to category")    


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    pass


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
    test_eda()
    test_encoder_helper()








