'''
Created on Nov 2, 2016

@author: abhijit.tomar
'''
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

def generate_optimal(X, y, clf, tuned_parameters, scores):
    
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=0)
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        model = GridSearchCV(clf, tuned_parameters, cv=5,
                           scoring='%s' % score, verbose=100000000)
        model.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(model.best_params_)
        with open('../../resources/params/'+type(clf).__name__+'_for_'+score+'_params.json', 'w') as outfile:
            json.dump(model.best_params_, outfile)
        print()
        print("Grid scores on development set:")
        print()
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, model.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()