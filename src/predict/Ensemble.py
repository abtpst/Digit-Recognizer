'''
Created on Nov 3, 2016

@author: abhijit.tomar
'''
import Load_Data
import pickle
import pandas as pd
import json
from predict import Get_Optimal
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    
    X_train,y_train,X_test = Load_Data.load_data()
    
    rf_clf=RandomForestClassifier()
    rf_param_map = json.load(open('../../resources/params/'+type(rf_clf).__name__+'_for_accuracy_params.json'))
    rf_clf = RandomForestClassifier(**rf_param_map)
    
    knn_clf=KNeighborsClassifier()
    knn_param_map = json.load(open('../../resources/params/'+type(knn_clf).__name__+'_for_accuracy_params.json'))
    knn_clf = KNeighborsClassifier(**knn_param_map)
    
    ensemble_clf = VotingClassifier(estimators=[('rf',rf_clf),('knn',knn_clf)])
    
    tuned_parameters = [{'voting': ['hard','soft'],'weights':[[1,2],[2,1]]}]
    
    scores = ['accuracy']
    
    Get_Optimal.generate_optimal(X_train, y_train, ensemble_clf, tuned_parameters, scores)
    
    param_map = json.load(open('../../resources/params/'+type(ensemble_clf).__name__+'_for_accuracy_params.json'))
    param_map['estimators']=[('rf',rf_clf),('knn',knn_clf)]
    
    ensemble_clf = VotingClassifier(**param_map)
    
    ensemble_clf.fit(X_train, y_train) 
    with open('../../resources/models/'+type(ensemble_clf).__name__+'_Model.pickle','wb') as fileName:
        pickle.dump(ensemble_clf,fileName)
    
    y_pred = ensemble_clf.predict(X_test)
    y_pred_prob = ensemble_clf.predict_proba(X_test)
    
    with open('../../resources/predictions/'+type(ensemble_clf).__name__+'_y_pred.pickle','wb') as fileName:
        pickle.dump(y_pred,fileName)
    
    with open('../../resources/predictions/'+type(ensemble_clf).__name__+'_y_pred_prob.pickle','wb') as fileName:
        pickle.dump(y_pred_prob,fileName)

    image_ids=[]
    for i in range(1,len(X_test)+1):
        image_ids.append(i)
    
    pd.DataFrame({'ImageId': image_ids, 'Label': y_pred}).to_csv('../../resources/results/'+type(ensemble_clf).__name__+'_Pred.csv', index=False)
    