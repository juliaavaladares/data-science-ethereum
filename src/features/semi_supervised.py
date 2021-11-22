from numpy import concatenate
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.semi_supervised import LabelPropagation

np.random.seed(0)

# =========================== COLUNAS ===================================
# 'user_account', 'balance_ether', 'balance_value', 'total_transactions', 'sent', 'received', 
# 'n_contracts_sent', 'n_contracts_received', 'classification', 'labels'

data = pd.read_csv("../../data/processed/final_dataset_2020.csv")
data2 = pd.read_csv("../../data/raw/new_accounts_features2020.csv")
data.drop(columns=["Unnamed: 0"], inplace=True)

X = data.iloc[:, [1,3,4,5,6,7]]
X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
y = data.iloc[:, 8]

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab))

# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))]
# recombine training dataset labels
y_train_mixed = concatenate((y_train_lab, nolabel))
# define model
model = LabelPropagation()
# fit model on training dataset
model.fit(X_train_mixed, y_train_mixed)
# make predictions on hold out test set
yhat = model.predict(X_test)
# calculate score for test set
accuracy = metrics.accuracy_score(y_test, yhat)
precision = metrics.precision_score(y_test, yhat)
recall = metrics.recall_score(y_test, yhat)
f_score = metrics.f1_score(y_test, yhat)
f_beta = metrics.fbeta_score(y_test, yhat, beta=2)
ROC = metrics.roc_auc_score(y_test, yhat)
MCC = metrics.matthews_corrcoef(y_test, yhat)

# summarize score
print('Accuracy: %.3f' % (accuracy*100))
print('Precision: %.3f' % (precision*100))
print('Recall: %.3f' % (recall*100))
print('F-score: %.3f' % (f_score*100))
print('F-beta: %.3f' % (f_beta*100))
print('MCC: %.3f' % (MCC*100))
print('ROC: %.3f' % (ROC*100))
