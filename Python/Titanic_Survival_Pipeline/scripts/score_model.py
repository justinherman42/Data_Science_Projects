
# coding: utf-8

# In[68]:


# import modules
import pickle
import pandas as pd
from pathlib import Path
from pull_data import get_url_csv
from pull_data import train
from pull_data import test
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cross_validation import train_test_split, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_fscore_support as score,
                             confusion_matrix, accuracy_score,
                             classification_report)
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                     KFold, cross_val_score)
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline, make_union
import scikitplot as skplt
import random
from train_model import BinarizeColumn
pd.options.mode.chained_assignment = None

# Function to rebuild model features for train dataset
def Create_train_model():
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
    label = 'Survived'
    X = df[features]
    y = df[label].ravel()
    return(features, label, X, y)


# Load pickle
my_pipeline = joblib.load('my_pipeline.pkl')
print("proof of pipeline:", my_pipeline)

# Load test csv
df = test
if len(test) > 0:
    print("test_df loaded")

# create empty target column
df = df.assign(Survived="NA")

# Set seed
np.random.seed(7)

# Create model features
features, label, X, y = Create_train_model()

# Special case imputation for test[Fair]
med_fair = X["Fare"].median()
X["Fare"] = X["Fare"].fillna(med_fair)

# Check for NA
print("total NA's:", X.isna().sum(), "\n Now display pipeline imputation:")

# Predict
predicted = pd.DataFrame(my_pipeline.predict_proba(X))

# Predict survival by > .5 proba
predicted_survival = pd.DataFrame(np.where(predicted.iloc[:, 1] > 0.5, 1, 0))
predicted_survival["PassengerId"] = df[["PassengerId"]]

# Write to csv
predicted_survival.to_csv("predicted_survival.csv", encoding='utf-8')

# Check if file saved
file=Path("predicted_survival.csv")
if file.is_file():
    print("Prediction csv is saved as predicted_survival.csv, the predictions are based on a probability threshold of .5 and are accompanied by passenger id/original index")
else: print("File did not save")

## Import classification report and display 
my_classification_report=pd.read_csv('classification_report2.csv')
my_classification_report

