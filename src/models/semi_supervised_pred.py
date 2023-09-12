# Data manipulation
import numpy as np
import pandas as pd
from sklearn import preprocessing
# Sklearn
from sklearn.model_selection import train_test_split # for splitting data into train and test samples
from sklearn.svm import SVC # for Support Vector Classification baseline model
from sklearn.semi_supervised import SelfTrainingClassifier # for Semi-Supervised learning
from sklearn.metrics import classification_report # for model evaluation metrics

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier 
from sklearn.ensemble import VotingClassifier


######## READ DATASET ######################
dataset = pd.read_csv("../../data/processed/dataset_mylabels_2020.csv")

dataset_lab = dataset[dataset.my_labels != -1]
dataset_lab.reset_index(drop=True, inplace=True)

X_lab = dataset_lab.iloc[:,[1,3,4,5,6,7]]
y_lab = dataset_lab.iloc[:, 10]

dataset_unlab = dataset[dataset.my_labels == -1]
dataset_unlab.reset_index(drop=True, inplace=True)

X_unlab = dataset_unlab.iloc[:,[1,3,4,5,6,7]]
y_unlab = dataset_unlab.iloc[:, 10]

######## NORMALIZE  DATASET ######################
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_lab = scaler.fit_transform(X_lab)
X_unlab = scaler.fit_transform(X_unlab)



X_train, X_test, y_train, y_test = train_test_split(X_lab, y_lab, test_size=0.25, random_state=0)
X_train = np.concatenate((X_train, X_unlab))
y_train = np.concatenate((y_train, y_unlab))


knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(max_depth=5)
random_forest = RandomForestClassifier(n_estimators=100, random_state=2)
logistic_regression = LogisticRegression()
linear_svm = SVC(kernel='linear', probability=True)
gaussian_svm = SVC(kernel='rbf', probability=True)
sigmoid_svm = SVC(kernel='sigmoid', probability=True)
estimators = [('KNN', knn), ('Decision Tree', decision_tree), \
                ('Random Forest', random_forest), ('Logistic Regression', logistic_regression), \
                ('Linear SVM' ,linear_svm), ('Gaussian SVM', gaussian_svm), ('Sigmoid SVM', sigmoid_svm)]

hard_voting_classifier = VotingClassifier(estimators = estimators, voting = "hard")
soft_voting_classifier = VotingClassifier(estimators = estimators, voting = "soft")

results = pd.DataFrame()

for classifier, model in estimators:
    self_training_model = SelfTrainingClassifier(base_estimator=model, # An estimator object implementing fit and predict_proba.
                                             threshold=0.7, # default=0.75, The decision threshold for use with criterion='threshold'. Should be in [0, 1).
                                             criterion='threshold', # {‘threshold’, ‘k_best’}, default=’threshold’, The selection criterion used to select which labels to add to the training set. If 'threshold', pseudo-labels with prediction probabilities above threshold are added to the dataset. If 'k_best', the k_best pseudo-labels with highest prediction probabilities are added to the dataset.
                                             #k_best=50, # default=10, The amount of samples to add in each iteration. Only used when criterion='k_best'.
                                             max_iter=100, # default=10, Maximum number of iterations allowed. Should be greater than or equal to 0. If it is None, the classifier will continue to predict labels until no new pseudo-labels are added, or all unlabeled samples have been labeled.
                                             verbose=True # default=False, Verbosity prints some information after each iteration
                                            )

    # Fit the model
    clf_ST = self_training_model.fit(X_train, y_train)
    series = pd.DataFrame(clf_ST.transduction_, columns=[classifier])
    results = pd.concat([results, series], axis=1)



    ########## Step 3 - Model Evaluation ########## 
    print('')
    print('---------- Self Training Model - Summary ----------')
    print('Base Estimator: ', clf_ST.base_estimator_)
    print('Classes: ', clf_ST.classes_)
    print('Transduction Labels: ', clf_ST.transduction_, "Len: ", len(clf_ST.transduction_))
    #print('Iteration When Sample Was Labeled: ', clf_ST.labeled_iter_)
    print('Number of Features: ', clf_ST.n_features_in_)
    #print('Feature Names: ', clf_ST.feature_names_in_)
    print('Number of Iterations: ', clf_ST.n_iter_)
    print('Termination Condition: ', clf_ST.termination_condition_)
    print('')

    print('---------- Self Training Model - Evaluation on Test Data ----------')
    accuracy_score_ST = clf_ST.score(X_test, y_test)
    print('Accuracy Score: ', accuracy_score_ST)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, clf_ST.predict(X_test)))

results.to_csv("../../reports/semi_supervised_results_2023.csv", index=False)