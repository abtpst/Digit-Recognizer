# Digit Recognizer

Python based approach for solving the Digit Recognizer Kaggle challenge

https://www.kaggle.com/c/digit-recognizer

### Main Idea

1. Use the following classifiers

      - `RandomForestClassifier`
      - `KNeighborsClassifier`

    With the aim of getting the maximum `accuracy`

2. Also, we will use our old friend `GridSearchCV` for getting the optimal values of 
parameters for initializing the above classifiers.

3. Lastly, we will combine both of these classifiers to create an **ensemble**. We will be using the `VotingClassifier`

### Set up

1. Anaconda for python 3.5
2. Required packages can be found in requirements.txt

### Layout

1. `src` folder contains all of the packages
2. `resources` folder contains
    1. `data` :-> This should have two folders. `train` for training data and `test` for test data.
    2. `models` :-> Store your models and classifers here, if needed.
    3. `params` :-> The optimal values of parameters after `GridSearchCV` are stored here.
    4. `results` :-> Store prediction results here.
             
### Packages

***plots***

This has a single module named `Visualize`, which can be used for looking into the data via `seaborn`

***loader***

This has a single module named `Load_Data` that reads the `csv` files and returns `pandas` `DataFrames`

***predict***

This has four modules

1. `Predict_KNN` :-> Predict using `KNeighborsClassifier`
2. `Predict_RForest` :-> Predict using `RandomForestClassifier`
3. `Ensemble` :-> Predict using `VotingClassifier` that combines the other two classifiers
4. `Get_Optimal` :-> Helper module for generating the optimal set of parameters for a classifier using `GridSearchCV`

### Results 

1. `RandomForestClassifier` with {"warm_start": true, "n_estimators": 150, "verbose": 100, "oob_score": false, "bootstrap": true} gets **0.96500**
2. `KNeighborsClassifier` with {'algorithm': 'kd_tree', 'n_neighbors': 1} gets **0.96900**
3. Ensemble `VotingClassifier` with {"voting": "soft", "weights": [2, 1]} and the above optimal classifiers, gets **0.97286** 