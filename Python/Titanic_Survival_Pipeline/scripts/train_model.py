import pandas as pd
from pathlib import Path
from pull_data import get_url_csv
from pull_data import train
from pull_data import test
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import (precision_recall_fscore_support as score,
                             confusion_matrix, accuracy_score,
                             classification_report)
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                     KFold, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.cross_validation import train_test_split, cross_val_predict
import scikitplot as skplt
import random
import pickle
pd.options.mode.chained_assignment = None

# Build transformers


class BinarizeColumn(TransformerMixin, BaseException):

    def __init__(self, field):
        self.field = field

    def transform(self, data):
        """(Convert Male/Female to 1/0)"""

        data[self.field] = (data[self.field] == 'male').astype(np.int)
        return data

    def fit(self, *_):
        return self


class ImputeColumn(TransformerMixin, BaseException):
    
    def __init__(self, field):
        self.field = field

    def transform(self, data):
        """(Impute -1 for missing values)"""
        data[self.field] = data[self.field].fillna(-1)
        return data

    def fit(self, *_):
        return self


class ImputeColumn_df_specific(TransformerMixin, BaseException):
    
    def __init__(self, field):
        self.field = field

    def transform(self, data):
        """Two-step imputation with imputation printout summary
        Imputes special Median Age value for subset (3+ siblings)
        Imputes -1 for leftover NA
        """
        print("NA before imputation:", data[self.field].isna().sum())
        missing_age_df = data[data[self.field].isna()]
        missing_and_na_age_df = missing_age_df[missing_age_df['SibSp'] >= 3]
        data.loc[missing_and_na_age_df.index, "Age"] = 11.7
        data[self.field] = data[self.field].fillna(-1)
        print("special case imputation:", len(
            data.loc[missing_and_na_age_df.index, "Age"]))
        print("-1 after imputation:", sum(data[self.field] == -1))
        return data

    def fit(self, *_):
        return self

# Build Custom function
def classification_report_csv(report):
    """
    Code credit with link is in section3.53 of EDA file
    Function saves classification report as csv file
    """
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report2.csv', index=False)

def Create_X_y_cv_models(df):
    """
    Creates X & y for cross validated models
    """
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
    label = 'Survived'
    X = df[features]
    y = df[label].ravel()
    return(X, y)



# Load dataset
df = train.set_index('PassengerId')

# Initiate class objects
dummy = DummyClassifier(strategy='stratified', random_state=None, constant=None)
binarize = BinarizeColumn('Sex')
rf = RandomForestClassifier()
age_col_special_impute = ImputeColumn_df_specific("Age")

#  pipeline 
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
label = 'Survived'
np.random.seed(7)
X = df[features]
y = df[label].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

my_pipe_4 = Pipeline([
    ('binarize', BinarizeColumn("Sex")),
    ('impute_column', age_col_special_impute),
    ('rf', rf)
]).fit(X_train, y_train)

joblib.dump(my_pipe_4, 'my_pipeline.pkl', compress=1)
print('Saved my_pipe_4 pipeline to file in cwd as my_pipeline.pkl')


# Run classification report testing model on train/test split
# look to EDA section 3.52 for better explanation
np.random.seed(7)
X, y = Create_X_y_cv_models(df)
y_pred = cross_val_predict(my_pipe_4, X, y, cv=10)
report = classification_report(y, y_pred, target_names=['class 0', 'class 1'])
classification_report_csv(report)
file=Path("classification_report2.csv")
if file.is_file():
    print("Classification report is saved as classification_report2.csv and can be safely loaded in score_model.py")
else: print("File did not save, and will not load in score_model.py")
