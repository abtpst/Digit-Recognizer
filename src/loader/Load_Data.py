'''
Created on Nov 2, 2016

@author: abhijit.tomar
'''
import pandas as pd

def load_data():
    
    df_train = pd.read_csv('../../resources/data/train/train.csv')
    
    df_test = pd.read_csv('../../resources/data/test/test.csv')

    y_train = df_train['label'].tolist()
    
    X_train = df_train.drop(['label'], axis=1)
    X_test = df_test
    return X_train,y_train,X_test
