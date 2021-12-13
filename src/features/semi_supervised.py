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
data_kmeans = pd.read_csv("../../data/processed/kmeans_clusters.csv")
data.drop(columns=["Unnamed: 0"], inplace=True)

#Pega as contas que foram classificadas como 0 em todos os agrupamentos
mask_normal_users =  (data_kmeans.cluster_2 == 0) & (data_kmeans.cluster_3 == 0) \
                                & (data_kmeans.cluster_4 == 0)\
                                & (data_kmeans.cluster_5 == 0)\
                                & (data_kmeans.cluster_6 == 0)\
                                & (data_kmeans.cluster_7 == 0)\
                                & (data_kmeans.cluster_8 == 0)\
                                & (data_kmeans.cluster_9 == 0)\
                                & (data_kmeans.cluster_10 == 0)

normal_users = data_kmeans["user_account"].where(mask_normal_users).values
my_labels = np.full(len(data), -1)

# Verifica se a conta dentro do dataset final está entre as 16878 contas
# classificadas como normais 
counter = 0
for account in data.user_account:
    if (account in normal_users):
        my_labels[counter] = 0
        counter += 1
        continue
    counter += 1

# Classifica os dados:
# 0 = normal users | 1 = professional users | -1 = unknown users 
data["my_labels"] = my_labels
data.loc[(data.my_labels==-1) & (data.classification ==1), "my_labels"] = 1 

labeled_data = data[(data.my_labels == 0) | (data.my_labels == 1)]
unlabeled_data = data[data.my_labels == -1]

data.to_csv("../../data/processed/dataset_mylabels_2020.csv", index=False)

### X Labeled Data
X_lab = labeled_data.iloc[:, [1,3,4,5,6,7]]
X_lab = MinMaxScaler(feature_range=(0,1)).fit_transform(X_lab)
y_lab = labeled_data.iloc[:, 10]

X_unlab = unlabeled_data.iloc[:, [1,3,4,5,6,7]]
X_unlab = MinMaxScaler(feature_range=(0,1)).fit_transform(X_unlab)
y_unlab = unlabeled_data.iloc[:, 10]

# First step: train the model with the labeled data 
# Regular supervised learning
X_lab_train, X_lab_test, y_lab_train, y_lab_test = train_test_split(X_lab, y_lab, test_size=0.50, random_state=1, stratify=y_lab)
model = LabelPropagation()
model.fit(X_lab_train, y_lab_train)

# Second step: Predict unlabeled data
y_unlabeled_predict = model.predict(X_unlab)

X_labeled_and_unlabelad = concatenate([X_lab_test, X_unlab])
y_labeled_and_unlabelad = concatenate([y_lab_test, y_unlabeled_predict])


yhat = model.predict(X_labeled_and_unlabelad)


#Third step: train the model with the full datset 
#X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
## create the training dataset input
#X_train_mixed = concatenate((X_train_lab, X_test_unlab))
#
## create "no label" for unlabeled data
#nolabel = [-1 for _ in range(len(y_test_unlab))]
## recombine training dataset labels
#y_train_mixed = concatenate((y_train_lab, nolabel))
## define model
#
## fit model on training dataset
#
## make predictions on hold out test set
#yhat = model.predict(X_test)
## calculate score for test set
accuracy = metrics.accuracy_score(y_labeled_and_unlabelad, yhat)
precision = metrics.precision_score(y_labeled_and_unlabelad, yhat)
recall = metrics.recall_score(y_labeled_and_unlabelad, yhat)
f_score = metrics.f1_score(y_labeled_and_unlabelad, yhat)
f_beta = metrics.fbeta_score(y_labeled_and_unlabelad, yhat, beta=2)
ROC = metrics.roc_auc_score(y_labeled_and_unlabelad, yhat)
MCC = metrics.matthews_corrcoef(y_labeled_and_unlabelad, yhat)
# 
##summarize score
print('Accuracy: %.3f' % (accuracy*100))
print('Precision: %.3f' % (precision*100))
print('Recall: %.3f' % (recall*100))
print('F-score: %.3f' % (f_score*100))
print('F-beta: %.3f' % (f_beta*100))
print('MCC: %.3f' % (MCC*100))
print('ROC: %.3f' % (ROC*100))
