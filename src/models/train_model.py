import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')
np.random.seed(0)

def write_file(line):
    with open("../../reports/train_model_results.txt", "a") as file:
        file.write(line+"\n")

def fit_predict_models(model, X, y, X_new_dataset, y_new_dataset):
    kfold_count = StratifiedKFold(n_splits=10)

    accuracy = 0
    precision = 0
    recall = 0
    f_score = 0
    f_beta = 0
    ROC_auc_curve = 0
    MCC = 0
    tp = 0
    tn = 0 

    count = 0

    for train_index, _ in kfold_count.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]

        #X_test = X[test_index]
        #y_test = y[test_index]

        model.fit(X_train, y_train)
        y_predict = model.predict(X_new_dataset)

        accuracy += metrics.accuracy_score(y_new_dataset, y_predict)
        precision += metrics.precision_score(y_new_dataset, y_predict)
        recall += metrics.recall_score(y_new_dataset, y_predict)
        f_score += metrics.f1_score(y_new_dataset, y_predict)
        f_beta += metrics.fbeta_score(y_new_dataset, y_predict, beta=2)
        ROC_auc_curve += metrics.roc_auc_score(y_new_dataset, y_predict)
        MCC += metrics.matthews_corrcoef(y_new_dataset, y_predict)
        tp += y_new_dataset[(y_new_dataset==1) & (y_predict==1)].count()
        tn += y_new_dataset[(y_new_dataset==0) & (y_predict==0)].count()

        count += 1
    
    result_metrics = np.array([accuracy, precision, recall, f_score, f_beta,
            ROC_auc_curve, MCC, tp, tn])/count

    return result_metrics


######## READ DATASET ######################
old_dataset = pd.read_csv("../../data/raw/new_accounts_features2020.csv")
new_dataset = pd.read_csv("../../data/processed/final_dataset_2021.csv")
# user_account, balance_ether,balance_value,total_transactions,sent,received,n_contracts_sent,n_contracts_received,labels,is_professional

X_new_dataset = new_dataset.iloc[:,[1,3,4,5,6,7]]
y_new_dataset = new_dataset.iloc[:, 9]


X_old_dataset = old_dataset.iloc[:,[1,3,4,5,6,7]]
Y_old_dataset = old_dataset.iloc[:, 8]

######## NORMALIZE  DATASET ######################
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_old_dataset = scaler.fit_transform(X_old_dataset)
X_new_dataset = scaler.fit_transform(X_new_dataset)

######## MODELS ######################
models = [ KNeighborsClassifier(), DecisionTreeClassifier(max_depth=5), 
            RandomForestClassifier(n_estimators=10, random_state=2), LogisticRegression() ]

write_file("accuracy, precision, recall, f_score, f_beta, ROC_auc_curve, MCC, tp, tn")

for model in tqdm(models):
    accuracy, precision, recall, f_score, f_beta, ROC_auc_curve, MCC, tp, tn = fit_predict_models(model, X_old_dataset, Y_old_dataset, X_new_dataset, y_new_dataset)
    line = f"{accuracy}, {precision}, {recall} ,{f_score} , {f_beta}, {ROC_auc_curve}, {MCC}, {tp}, {tn}"
    write_file(line)




