"""
test and log ChurnPipeline class
"""
import logging
import math
import os

import pandas as pd
import yaml

from churn_library import ChurnPipeline

logging.basicConfig(
    filename="./logs/testing_logs.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_data_import():
    """
    test that data is imported correctly
    """

    try:
        df = pd.read_csv("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_target_column_creation(cpl, config):
    """
    test that the target column is created and computed correctly
    """

    try:
        assert config["target"] in cpl.df.columns
        logging.info(f"column {config['target']} created: SUCCESS")
    except KeyError as err:
        logging.error(
            f"test_target_column_creation: column {config['target']} not created"
        )
        raise err

    try:
        assert all(
            cpl.df[config["target"]]
            == cpl.df["Attrition_Flag"].apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
        )
        logging.info(f"column {config['target']} computed correctly: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"seems like {config['target']} wasn't computed correctly: {cpl.df[[config['target'], 'Attrition_Flag']].head()}"
        )
        raise err


def test_eda_plots_stored(config):
    """
    test that eda stores the right plots in the correct folder
    """
    expected_files = [key + val + ".pdf" for key, val in config["plot_dict"].items()]
    try:
        assert all((os.path.isfile("./images/eda/" + file) for file in expected_files))
        logging.info("plots_stored: SUCCESS")
    except AssertionError as err:
        logging.error(
            "expected to find {expected_files}, got {os.listdir('./images/eda/')} instead"
        )
        raise err


def test_encoder_helper(cpl, config):
    """
    test that encoder helper creates the correct columns with the correct amount of values
    """

    try:
        encoded_columns = [
            col + "_" + config["target"] for col in config["cat_columns"]
        ]
        assert all((col in cpl.df.columns for col in encoded_columns))
        logging.info("correct encoded columns created: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"columns {encoded_columns} were expected in df; df contains {cpl.df.columns} though"
        )

    try:
        actual_n_val = {col: cpl.df[col].nunique() for col in encoded_columns}
        expected_n_val = {
            col + "_" + config["target"]: cpl.df[col].nunique()
            for col in config["cat_columns"]
        }
        assert all(
            (actual_n_val[col] == expected_n_val[col] for col in actual_n_val.keys())
        )
        logging.info("encoded columns have the correct amount of values: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"encoded columns have wrong amount of values: expected {expected_n_val}, got {actual_n_val}"
        )
        raise err


def test_features_and_target_split(cpl, config):
    """
    test that features (ChurnPipeline.X) and target varible are split correctly
    """
    try:
        assert all(cpl.y == cpl.df[config["target"]])
        logging.info("target feature split: SUCCESS")
    except AssertionError as err:
        logging.error("target feature was not generated correctly")
        raise err

    try:
        keep_cols = [
            *config["quant_columns"],
            *[col for col in cpl.df if "_" + config["target"] in col],
        ]
        pd.testing.assert_frame_equal(cpl.df[keep_cols], cpl.X[keep_cols])
        logging.info("features dataframe generated correctly: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"features dataframe not generated correctly: expected {cpl.df[keep_cols].head()}, got {cpl.X[keep_cols].head()}"
        )
        raise err


def test_feature_engineering(cpl, config):
    """
    test that feature_engineering returns the correct dataframes, with the correct splits and columns
    """
    try:  # check test/train split is done correctly
        assert math.isclose(
            config["split"], len(cpl.X_test.index) / len(cpl.df.index), abs_tol=0.01
        )
        assert math.isclose(
            (1 - config["split"]),
            len(cpl.X_train.index) / len(cpl.df.index),
            abs_tol=0.01,
        )
        assert math.isclose(config["split"], len(cpl.y_test) / len(cpl.y), abs_tol=0.01)
        assert math.isclose(
            (1 - config["split"]), len(cpl.y_train) / len(cpl.y), abs_tol=0.01
        )
        logging.info("train/test split performed according to defined split: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"train/test split not performed correctly; expected {config['split']}, got {len(cpl.X_test.index)/len(cpl.df.index)} for features and {len(cpl.y_test)/len(cpl.y)}"
        )

    try:  # check column names in train/test sets
        assert all(cpl.X_test.columns == cpl.X.columns)
        assert all(cpl.X_train.columns == cpl.X.columns)
        logging.info("train/test set have correct columns: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"train/test sets have the wrong columns expected {cpl.X.columns}, got train: {cpl.X_train.columns}; test: {cpl.X_test.columns}"
        )
        raise err


def test_model_training(cpl, config):
    """
    test that models are trained and therefore are class methods
    """
    try:
        assert hasattr(cpl, "lrc")
        logging.info("logistic regression model trained: SUCCESS")
    except AssertionError as err:
        logging.error("logistic regression model not found")
        raise err

    try:
        assert hasattr(cpl, "rfc")
        logging.info("random forest model trained: SUCCESS")
    except AssertionError as err:
        logging.error("random forest model not found")
        raise err

    try:  # check random forest parameters are in fact from the grid
        assert all(
            (
                cpl.rfc.get_params()[key] in values
                for key, values in config["param_grid"].items()
            )
        )
        logging.info("random forest parameters determined correctly: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"random forest parameters seem to have been chosen out of range. Expected {config['param_grid']}, got {cpl.rfc.get_params()}"
        )
        raise err


def test_model_storage():
    """
    test that models are stored correctly
    """
    try:
        assert os.path.isfile("./models/logistic_model.pkl")
        logging.info("logistic regression model stored: SUCCESS")
    except AssertionError as err:
        logging.error("logistic regression model not stored correctly")
        raise err

    try:
        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info("random forest model stored: SUCCESS")
    except AssertionError as err:
        logging.error("random forest model not stored correctly")
        raise err


def test_classification_report_storage(cpl):
    """
    test that classification report is made and stored correctly
    """
    cpl.classification_report_image()

    try:
        assert os.path.isfile("./images/results/logistic_regression_results.png")
        logging.info("logistic regression results stored: SUCCESS")
    except AssertionError as err:
        logging.error(
            "logistic_regression_results.png not found in folder images/results"
        )
        raise err

    try:
        assert os.path.isfile("./images/results/random_forest_results.png")
        logging.info("random forest results stored: SUCCESS")
    except AssertionError as err:
        logging.error("random_forest_results.png not found in folder images/results")
        raise err


def test_roc_curves_storage(cpl):
    """
    test that roc curves are made and stored correctly
    """
    cpl.plot_roc_curves()
    try:
        assert os.path.isfile("./images/results/ROC_curves.png")
        logging.info("ROC curves stored correctly: SUCCESS")
    except AssertionError as err:
        logging.error("ROC_curves.png does not seem to be in folder images/results")
        raise err


def test_feature_importance_plot_storage(cpl):
    """
    test that feature importance plot is made and stored
    """
    cpl.feature_importance_plot()

    try:
        assert os.path.isfile("./images/results/feature_importance.png")
        logging.info("feature imprtance plot correctly stored: SUCCESS")
    except AssertionError as err:
        logging.error(
            "feature_importance.png does not seem to be a file in folder images/results"
        )
        raise err


if __name__ == "__main__":

    with open("test_config.yaml", "r") as stream:
        CONFIG = yaml.safe_load(stream)

    CPL = ChurnPipeline("./data/bank_data.csv", CONFIG)

    test_data_import()
    test_target_column_creation(CPL, CONFIG)
    test_eda_plots_stored(CONFIG)
    test_encoder_helper(CPL, CONFIG)
    test_features_and_target_split(CPL, CONFIG)
    test_feature_engineering(CPL, CONFIG)
    test_model_training(CPL, CONFIG)
    test_model_storage()
    test_classification_report_storage(CPL)
    test_classification_report_storage(CPL)
    test_roc_curves_storage(CPL)
    test_feature_importance_plot_storage(CPL)
