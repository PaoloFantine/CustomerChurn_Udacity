# library doc string


# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#import shap
import joblib

from scikitplot.metrics import plot_roc_curve

os.environ['QT_QPA_PLATFORM']='offscreen'


class churn_pipeline:
        
    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth
        
        input:
            pth: a path to the csv
        output:
            df: pandas dataframe
        '''
        self.df = pd.read_csv(pth)
        return self.df
    '''
    
    
    def describe(self):
        return self.df.describe()
    '''
    def histogram(self, feature):
        '''
        plot histogram of the desired feature
        '''
        plt.figure(figsize=(20,10))
        self.df[feature].hist()
        plt.savefig(f"images/eda/{feature}_histogram.pdf")
    
        
    def value_counts(self, feature):
        plt.figure(figsize=(20,10))
        self.df[feature].value_counts(
            'normalize'
        ).plot(kind='bar')
        plt.savefig(f'images/eda/{feature}_val_counts.pdf')
        
    def distribution(self, feature):
        plt.figure(figsize=(20,10)) 
        sns.histplot(self.df[feature], 
                     stat='density', 
                     kde=True)
        plt.savefig(f'images/eda/{feature}_distribution.pdf')
        
    def _corr_heatmap(self):
        plt.figure(figsize=(20,10)) 
        sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig('images/eda/correlation_heatmap.pdf')
        
    def _plot_eda(self, plot_dict):
        '''
        perform eda on df and save figures to images folder
        
        Parameters
            plot_dict: dict of shape {feature:self.plot_function}; allowed plot functions are: self.histogram, self.value_counts and self.distribution
        Returns:
            None        
        '''
        for feature in plot_dict.keys():
            plot_dict.get(feature)(feature)
               
        self._corr_heatmap()
        
    def perform_eda(self, plot_dict):
        '''
        perform eda and save plots to images/eda folder
        
        Parameters:
            df: pandas.DataFrame to explore
            plot_dict: dict of shape {feature:self.plot_function}; allowed plot functions are: self.histogram, self.value_counts and self.distribution
            
        Returns:
            None
        
        '''
        
        print(self.df.shape)
        print(self.df.isnull().sum())
        print(self.df.describe())
        
        self._plot_eda(plot_dict)



    def _encoder_helper(self, category_lst, target):
        '''
        helper function to turn each categorical column into a new column with
        proportion of churn for each category - associated with cell 15 from the
        notebook
        
        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            target: string identifying the target column (churn)

        output:
            df: pandas dataframe with new columns for
        '''
        
        for cat in category_lst:
            grouped_cat = dict(self.df.groupby(cat).mean()[target])
            self.df[cat+'_'+ target] = [grouped_cat[label] for label in self.df[cat]]
        
        return self.df
    
    def perform_feature_engineering(self, config, split):
        '''
        input:
            config: dict containing keys 'cat_columns':(list of str) holding the names of categorical columns, 'target':(str) name of the target column to be used for modelling
            split: (float <1) split fraction to be used in defining model test set
             
        output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
        '''
        
        # encode columns
        self.df = self._encoder_helper(config['cat_columns'], config['target'])
    
        # features/target split
        y = self.df[config['target']]
        X = pd.DataFrame()
        
        keep_cols = [*config['quant_columns'],
                     *[col for col in self.df if '_'+config['target'] in col]]
    
        X[keep_cols] = self.df[keep_cols]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = split, random_state=42)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def classification_report_image(self,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image in images folder
        input:
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

         output:
            None
         '''
        with open('images/results/results.txt', 'w') as f:
            f.write('random forest results')
            f.write('test results')
            f.write(classification_report(self.y_test, y_test_preds_rf))
            f.write('train results')
            f.write(classification_report(self.y_train, y_train_preds_rf))
            
            f.write('logistic regression results')
            f.write('test results')
            f.write(classification_report(self.y_test, y_test_preds_lr))
            f.write('train results')
            f.write(classification_report(self.y_train, y_train_preds_lr))

    def train_models(self, param_grid):
        '''
        train, store model results: images + scores, and store models
        
        param_grid: dict giving a grid for gridsearch
        
        output:
              None
        '''
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference: https://scikitlearn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.X_train, self.y_train)
        
        lrc.fit(self.X_train, self.y_train)
        
        # store models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        rfc_model = joblib.load('./models/rfc_model.pkl')
        lr_model = joblib.load('./models/logistic_model.pkl')
        
        # linear regression roc plot
        lrc_plot = plot_roc_curve(lr_model, self.X_test, self.y_test)
        
        # random forest roc plot
        #plt.figure(figsize=(15, 8))
        #ax = plt.gca()
        #rfc_disp = plot_roc_curve(rfc_model, self.X_test, self.y_test, ax=ax, alpha=0.8)
        #plt.savefig(f'images/eda/{feature}_val_counts.pdf')
        #lrc_plot.plot(ax=ax, alpha=0.8)
        #plt.show()
        
        #y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
        #y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)
        #y_train_preds_lr = lrc.predict(self.X_train)
        #y_test_preds_lr = lrc.predict(self.X_test)
        
    def feature_importance_plot(self, model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
            
        output:
             None
        '''
        pass

