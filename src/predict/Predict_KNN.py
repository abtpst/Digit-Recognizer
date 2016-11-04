'''
Created on Nov 2, 2016

@author: abhijit.tomar
'''
import Load_Data
import pickle
import pandas as pd
from predict import Get_Optimal
import json

from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    
    X_train,y_train,X_test = Load_Data.load_data()
    
    clf = KNeighborsClassifier()
    tuned_parameters = [{'n_neighbors': [1,2,3], 
                         'algorithm': ['kd_tree','ball_tree']}
                        ]
    
    scores = ['accuracy']
    
    Get_Optimal.generate_optimal(X_train, y_train,clf,tuned_parameters,scores)
    
    param_map = json.load(open('../../resources/params/'+type(clf).__name__+'_for_accuracy_params.json'))
    
    clf = KNeighborsClassifier(**param_map)
    
    clf.fit(X_train, y_train) 
    with open('../../resources/models/'+type(clf).__name__+'_Model.pickle','wb') as fileName:
        pickle.dump(clf,fileName)
    
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    
    with open('../../resources/predictions/'+type(clf).__name__+'_y_pred.pickle','wb') as fileName:
        pickle.dump(y_pred,fileName)
    
    with open('../../resources/predictions/'+type(clf).__name__+'_y_pred_prob.pickle','wb') as fileName:
        pickle.dump(y_pred_prob,fileName)
    
    y_pred = pickle.load(open('../../resources/predictions/'+type(clf).__name__+'_y_pred.pickle','rb'))
    image_ids=[]
    for i in range(1,len(X_test)+1):
        image_ids.append(i)
    
    pd.DataFrame({'ImageId': image_ids, 'Label': y_pred}).to_csv('../../resources/results/'+type(clf).__name__+'_Pred.csv', index=False)
    