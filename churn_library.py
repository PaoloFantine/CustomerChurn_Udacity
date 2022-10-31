'''
class to process and model customer churn data within udacity MLengineer
course
'''

# import libraries
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

os.environ["QT_QPA_PLATFORM"] = "offscreen"


class ChurnPipeline:
    '''
    Class to process data, model customer churn and store relevant
    plots and results

    '''

    def __init__(self, pth, config, split):
        self.df = self.import_data(pth)

        self.df[config['target']] = self.df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        # encode columns
        self.df = self._encoder_helper(config["cat_columns"], config["target"])

        # features/target split
        self.y = self.df[config["target"]]
        self.X = pd.DataFrame()

        self.X_train, self.X_test, self.y_train, self.y_test = self.perform_feature_engineering(
            config, split)

    def import_data(self, pth):
        """
        returns dataframe for the csv found at pth

        input:
            pth: a path to the csv
        output:
            df: pandas dataframe
        """
        return pd.read_csv(pth)

    def histogram(self, feature):
        """
        plot histogram of the desired feature
        """
        plt.figure(figsize=(20, 10))
        self.df[feature].hist()
        plt.savefig(f"images/eda/{feature}_histogram.pdf")

    def value_counts(self, feature):
        '''
        plot value counts for (categorical) feature

        '''
        plt.figure(figsize=(20, 10))
        self.df[feature].value_counts("normalize").plot(kind="bar")
        plt.savefig(f"images/eda/{feature}_val_counts.pdf")

    def distribution(self, feature):
        '''
        plot feature distribution together with continuous probability distribution
        '''
        plt.figure(figsize=(20, 10))
        sns.histplot(self.df[feature], stat="density", kde=True)
        plt.savefig(f"images/eda/{feature}_distribution.pdf")

    def _corr_heatmap(self):
        '''
        plot correlation heatmap of features stored in self.df

        '''
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig("images/eda/correlation_heatmap.pdf")

    def _plot_eda(self, plot_dict):
        """
        perform eda on df and save figures to images folder

        Parameters
            plot_dict: dict of shape {feature:self.plot_function};
            allowed plot functions are: self.histogram, self.value_counts
            and self.distribution
        Returns:
            None
        """
        for feature in plot_dict.keys():
            plot_dict.get(feature)(feature)

        self._corr_heatmap()

    def perform_eda(self, plot_dict):
        """
        perform eda and save plots to images/eda folder

        Parameters:
            df: pandas.DataFrame to explore
            plot_dict: dict of shape {feature:self.plot_function};
            allowed plot functions are: self.histogram, self.value_counts
            and self.distribution

        Returns:
            None

        """

        print(self.df.shape)
        print(self.df.isnull().sum())
        print(self.df.describe())

        self._plot_eda(plot_dict)

    def _encoder_helper(self, category_lst, target):
        """
        helper function to turn each categorical column into a new column with
        proportion of churn for each category - associated with cell 15 from the
        notebook

        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            target: string identifying the target column (churn)

        output:
            df: pandas dataframe with new columns for feature engineering
        """

        for cat in category_lst:
            grouped_cat = dict(self.df.groupby(cat).mean()[target])
            self.df[cat + "_" + target] = [grouped_cat[label]
                                           for label in self.df[cat]]

        return self.df

    def perform_feature_engineering(self, config, split):
        """
        input:
            config: dict containing keys 'cat_columns':(list of str) holding the names
            of categorical columns, 'target':(str) name of the target column to be
            used for modelling
            split: (float <1) split fraction to be used in defining model test set

        output:
            None
        """

        keep_cols = [
            *config["quant_columns"],
            *[col for col in self.df if "_" + config["target"] in col],
        ]

        self.X[keep_cols] = self.df[keep_cols]

        return train_test_split(
            self.X,
            self.y,
            test_size=split,
            random_state=42)

    def classification_report_image(self):
        """
        produces classification report for training and testing results
        and stores report as image in images folder

         output:
            None
        """

        y_train_preds_rf = self.rfc.predict(self.X_train)
        y_test_preds_rf = self.rfc.predict(self.X_test)

        y_train_preds_lr = self.lrc.predict(self.X_train)
        y_test_preds_lr = self.lrc.predict(self.X_test)

        with open("images/results/random_forest_results.txt", "w") as rf_result_file:
            rf_result_file.write("random forest results \n")
            rf_result_file.write("test results \n")
            rf_result_file.write(
                classification_report(
                    self.y_test, y_test_preds_rf))
            rf_result_file.write("train results \n")
            rf_result_file.write(
                classification_report(
                    self.y_train,
                    y_train_preds_rf))

        with open("images/results/logistic_regression_results.txt", "w") as lr_result_file:
            lr_result_file.write("logistic regression results \n")
            lr_result_file.write("test results \n")
            lr_result_file.write(
                classification_report(
                    self.y_test, y_test_preds_lr))
            lr_result_file.write("train results \n")
            lr_result_file.write(
                classification_report(
                    self.y_train,
                    y_train_preds_lr))

    def train_models(self, param_grid):
        """
        train, store model results: images + scores, and store models

        param_grid: dict giving a grid for gridsearch

        output:
              None
        """
        # grid search
        rfc = RandomForestClassifier(random_state=42)

        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikitlearn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.X_train, self.y_train)

        lrc.fit(self.X_train, self.y_train)

        self.lrc = lrc
        self.rfc = cv_rfc.best_estimator_

        # store models
        joblib.dump(self.rfc, "./models/rfc_model.pkl")
        joblib.dump(self.lrc, "./models/logistic_model.pkl")

    def plot_roc_curves(self):
        """
        function to plot roc curves of the tested models

        """

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        plot_roc_curve(
            self.rfc,
            self.X_test,
            self.y_test,
            ax=ax,
            alpha=0.8)
        plot_roc_curve(self.lrc, self.X_test, self.y_test, ax=ax, alpha=0.8)

        plt.savefig("images/results/ROC_curves.png")

    def feature_importance_plot(self):
        """
        creates and stores the feature importances in pth
        input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

        output:
             None
        """

        # Shap importances
        explainer = shap.TreeExplainer(self.rfc)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test, plot_type="bar")

        # Calculate feature importances
        importances = self.rfc.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self.X.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel("Importance")

        # Add bars
        plt.bar(range(self.X.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(self.X.shape[1]), names, rotation=90)