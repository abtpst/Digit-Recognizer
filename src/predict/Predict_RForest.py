'''
Created on Nov 3, 2016

@author: abhijit.tomar

Module for predicting digits with RandomForestClassifier
'''
import Load_Data
import pickle
import pandas as pd
from predict import Get_Optimal
import json

from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    # Load the training and test data
    X_train,y_train,X_test = Load_Data.load_data()
    # Initialize classifier
    clf = RandomForestClassifier()
    # Set up possible values for hyper-parameters. These would be used by GridSearch to derive optimal set of hyper-parameters
    tuned_parameters = [{'n_estimators': [10,20,30,40,50,60,70,80,90,100,150,200], 
                         'bootstrap': [True],'warm_start': [True,False],
                         'oob_score': [True,False],'verbose': [100]}]
    # Initialize scoring metric for this problem space
    scores = ['accuracy']
    # Generate optimal set of hyper-parameters for the above classifier
    Get_Optimal.generate_optimal(X_train, y_train, clf, tuned_parameters, scores)
    # Load the optimal hyper-parameters
    param_map = json.load(open('../../resources/params/'+type(clf).__name__+'_for_accuracy_params.json'))
    # Reinitialize the classifier but now with the optimal hyper-parameters
    clf = RandomForestClassifier(**param_map)
    # Train the classifier
    clf.fit(X_train, y_train)
    # Optionally, save the trained classifier 
    with open('../../resources/models/'+type(clf).__name__+'_Model.pickle','wb') as fileName:
        pickle.dump(clf,fileName)
    # Predict on the test data
    y_pred = clf.predict(X_test)
    # Optionally, find out the prediction probabilities
    y_pred_prob = clf.predict_proba(X_test)
    # Optionally, save the predictions
    with open('../../resources/predictions/'+type(clf).__name__+'_y_pred.pickle','wb') as fileName:
        pickle.dump(y_pred,fileName)
    # Optionally, save the prediction probabilities
    with open('../../resources/predictions/'+type(clf).__name__+'_y_pred_prob.pickle','wb') as fileName:
        pickle.dump(y_pred_prob,fileName)
    # Create the image ids column for submission 
    image_ids=[]
    for i in range(1,len(X_test)+1):
        image_ids.append(i)
    # Save the submission as CSV
    pd.DataFrame({'ImageId': image_ids, 'Label': y_pred}).to_csv('../../resources/results/'+type(clf).__name__+'_Pred.csv', index=False)
    