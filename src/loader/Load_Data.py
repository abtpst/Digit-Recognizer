'''
Created on Nov 2, 2016

@author: abhijit.tomar

Script to load the training and test data
'''
import pandas as pd

def load_data():
    
    df_train = pd.read_csv('../../resources/data/train/train.csv')
    
    df_test = pd.read_csv('../../resources/data/test/test.csv')
    # Isolate the column that has prediction values
    y_train = df_train['label'].tolist()
    # Everything else is what we must train on
    X_train = df_train.drop(['label'], axis=1)
    # Test data does not require splicing
    X_test = df_test
    return X_train,y_train,X_test
