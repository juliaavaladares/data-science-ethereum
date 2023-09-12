#!/usr/bin/env python
# coding: utf-8

from os import path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import itertools

#Preprocessamento
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


#Avaliacao do Modelo
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, make_scorer

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from thundersvm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

#np.random.seed(0)
path = "../../data/processed/"
figures_path = '../../reports/figures/'
reports_path = '../../reports/'


# ### Funções a serem usadas
# In[106]:



def metrics_structure():
    results = pd.DataFrame({'Acuracy':[],
                            'Precision':[],
                            'Recall':[],
                            'F1-score':[],
                            'F-beta':[],
                            'MCC':[],
                            'TP':[],
                            'TN':[],
                            'ROC Curve':[]}, )
    results.index.names = ['Algoritmos']
    return results

def get_metrics(y_test, y_predict):

    acuracy = metrics.accuracy_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict)
    recall= metrics.recall_score(y_test, y_predict)
    f_score = metrics.f1_score(y_test, y_predict)
    f_beta = metrics.fbeta_score(y_test, y_predict, beta=2)
    ROC_auc_curve = metrics.roc_auc_score(y_test, y_predict)
    MCC = metrics.matthews_corrcoef(y_test, y_predict)

    tp = len(y_test[(y_test==1) & (y_predict==1)])
    tn = len(y_test[(y_test==0) & (y_predict==0)])
    fp = len(y_test[(y_test==0) & (y_predict==1)])
    fn = len(y_test[(y_test==1) & (y_predict==0)])  

    return [acuracy, precision, recall, f_score, f_beta, MCC, tp, tn, ROC_auc_curve]


def analysis(X_test, y_test, y_predict, model,text):
    print(text, end='\n\n')

    print(classification_report(y_test, y_predict))
    
    results_metrics = metrics_structure()
    results = get_metrics(y_test, y_predict)
    results_metrics = results_metrics.append(pd.DataFrame([results], index=[text], columns=results_metrics.columns))
    print(results_metrics)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.close()


def feature_importance(model, text, feature,plot=True):
    '''
    Calcula a feature importance da ávore de decisão
    '''
    importances = pd.DataFrame({'feature':feature,
                            'importance':np.round(model.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False)
    
    
    values = importances.importance.values
    labels = importances.feature.values
    y_pos = np.arange(len(labels))

    if plot:
        plt.bar([x for x in range(len(values))], values)
        plt.xticks(y_pos, labels, rotation = 45, ha='right')
        plt.savefig(figures_path+"feature_importance_"+text+".pdf")
        plt.close()
    

# In[114]:

def plot_dct(dct,string,features):
    #features = ['balance_ether','total_transactions',
    #            'sent','received','n_contracts_sent','n_contracts_received']

    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(dct, feature_names=features, filled=True)
    fig.savefig(figures_path+"decistion_tree"+string+".pdf")
    plt.close()
    
    export_graphviz(dct, out_file=figures_path+"mytree"+string+".dot",  
                     feature_names=features, filled=True,rounded=True, special_characters=True)


# In[137]:
def grid_search_plot_results(gs, params):
    cv_results = gs.cv_results_

    for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        print(params, mean_score)

    print("Best score: %0.3f" % gs.best_score_)
    print("Best parameters set:")
    best_parameters = gs.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def cv(X, y, model, name, balancer = None, params=None):

    results_cv = metrics_structure()
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    #auc_media = 0
    
    skfold = StratifiedKFold(n_splits=10)
    
    for fold, (train_index, test_index) in tqdm(enumerate(skfold.split(X, y), 1)):
        start_time = time.time()
        X_train = X[train_index]
        y_train = y[train_index] 

        #if len(X_new_dataset) > 0 and len(y_new_dataset) > 0:
        #    X_test = X_new_dataset
        #    y_test = y_new_dataset
        #else:
        X_test = X[test_index]
        y_test = y[test_index]  
        
        if balancer is not None:
            X_train, y_train = balancer.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        

        y_predict = model.predict(X_test)

        print(f'For fold {fold}:')
        acuracy = metrics.accuracy_score(y_test, y_predict)
        precision = metrics.precision_score(y_test, y_predict)
        recall = metrics.recall_score(y_test, y_predict)
        f_score = metrics.f1_score(y_test, y_predict)
        f_beta = metrics.fbeta_score(y_test, y_predict, beta=2)
        ROC = metrics.roc_auc_score(y_test, y_predict)
        MCC = metrics.matthews_corrcoef(y_test, y_predict)
        TP = len(y_test[(y_test==1) & (y_predict==1)])
        TN = len(y_test[(y_test==0) & (y_predict==0)])
        
        tp += y_test[(y_test==1) & (y_predict==1)].count()
        tn += y_test[(y_test==0) & (y_predict==0)].count()
        fp += y_test[(y_test==0) & (y_predict==1)].count()
        fn += y_test[(y_test==1) & (y_predict==0)].count() 
        

        
        result_fold = [acuracy, precision, recall, f_score, f_beta, MCC, tp, tn, ROC]
        results_cv = results_cv.append(pd.DataFrame([result_fold], index=[('Fold' + str(fold))], columns=results_cv.columns))
        
        #analysis(X_test, y_test, y_predict, model, name)
        #
        if fold == 1:
            print('best fold', str(fold))
            bestROC = ROC
            bestModel = model
            analysis(X_test, y_test, y_predict, model, name)
        elif ROC > bestROC:
            print('best fold', str(fold))
            bestROC = ROC
            bestModel = model
            analysis(X_test, y_test, y_predict, model, name)
            
        
        elapsed_time = time.time() - start_time
        print('Tempo gasto fold', fold, '-', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    
    acur = (tp+tn)/(tp+tn+fp+fn)
    prec = 0 if tp + fp == 0 else tp/(tp+fp)
    reca = 0 if tp + fn == 0 else tp/(tp+fn)
    fbeta = 0 if reca + prec == 0 else (2*reca*prec)/(reca+prec)
    fbeta2 = (0 if reca + prec == 0 else (5*reca*prec)/((4*prec)+reca))
    MCC = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    auc = (tp/((tp+fn)*2)) + (tn/((tn+fp)*2))


    results_metrics = metrics_structure()
    results = [acur, prec, reca, fbeta, fbeta2, MCC, int(tp), int(tn), auc]
    results_metrics = results_metrics.append(pd.DataFrame([results], index=[name], columns=results_metrics.columns))
    

    
    return results_metrics, results_cv, bestModel

def generate_models():
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

    return estimators


######## READ DATASET ######################
dataset = pd.read_csv("../../data/processed/data_unsupervised_supervised_2023.csv")
#new_dataset = pd.read_csv("../../data/processed/final_dataset_2021.csv")

dataset = dataset[dataset.label != -1]
dataset.reset_index(drop=True, inplace=True)

X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]
atributes = ['balance','active_duration','in_degree','out_degree','unique_in_degree','unique_out_degree','min_in_tx_value','max_in_tx_value','avg_in_tx_value','std_in_tx_value','total_in_tx_value','min_out_tx_value','max_out_tx_value','avg_out_tx_value','std_out_tx_value','total_out_tx_value','min_in_tx_gas','max_in_tx_gas','avg_in_tx_gas','std_in_tx_gas','total_in_tx_gas','min_out_tx_gas','max_out_tx_gas','avg_out_tx_gas','std_out_tx_gas','total_out_tx_gas','min_in_tx_gasPrice','max_in_tx_gasPrice','avg_in_tx_gasPrice','std_in_tx_gasPrice','total_in_tx_gasPrice','min_out_tx_gasPrice','max_out_tx_gasPrice','avg_out_tx_gasPrice','std_out_tx_gasPrice','total_out_tx_gasPrice','min_in_tx_gasUsed','max_in_tx_gasUsed','avg_in_tx_gasUsed','std_in_tx_gasUsed','total_in_tx_gasUsed','min_out_tx_gasUsed','max_out_tx_gasUsed','avg_out_tx_gasUsed','std_out_tx_gasUsed','total_out_tx_gasUsed','min_interval_in_tx','max_interval_in_tx','avg_interval_in_tx','std_interval_in_tx','min_interval_out_tx','max_interval_out_tx','avg_interval_out_tx','std_interval_out_tx']

######## NORMALIZE  DATASET ######################
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

ros = RandomOverSampler()
rus = RandomUnderSampler()
smt = SMOTE()


all_results = metrics_structure()  #DataFrame com os resultados finais
estimators = generate_models()  #Lista com os classificadores


def store_results(results_metrics_no_balance, results_metrics_undersample, results_metrics_oversample, results_metrics_smote ):
    results = metrics_structure()

    results = results.append([results_metrics_no_balance,
                                results_metrics_undersample,
                                results_metrics_oversample,
                                results_metrics_smote ])
    
    return results

def calculates_results(X,y,model, text, best_model=False):

    results_metrics_no_balance, _, best_model_no_balance = cv(X,y,model, text +' No Balance')
    results_metrics_undersample, _, best_model_under = cv(X,y,model, text +' UnderSample', balancer=rus)
    results_metrics_oversample, _, best_model_over = cv(X,y,model,text + ' OverSample', balancer=ros)
    results_metrics_smote, _, best_model_smote = cv(X,y,model,text + ' Smote', balancer=smt)

    if best_model:
        return [results_metrics_no_balance, best_model_no_balance], [results_metrics_undersample, best_model_under],\
            [results_metrics_oversample, best_model_over], [results_metrics_smote, best_model_smote]

    return results_metrics_no_balance, results_metrics_undersample, results_metrics_oversample, results_metrics_smote



def cross_validation(X,y,model, text, importance = False, plot_decision_tree=False, feature=""):
    results_metrics = calculates_results(X,y,model, text, importance)
    
    if importance:
        results = store_results(results_metrics[0][0], results_metrics[1][0], results_metrics[2][0], results_metrics[3][0])

        feature_importance(results_metrics[0][1], 'No_balance', feature)
        feature_importance(results_metrics[1][1], 'Undersample', feature)
        feature_importance(results_metrics[2][1], 'Oversample', feature)
        feature_importance(results_metrics[3][1], 'Smote', feature)

        if plot_decision_tree:
            plot_dct(results_metrics[0][1], 'No_balance', feature)
            plot_dct(results_metrics[1][1], 'Undersample', feature)
            plot_dct(results_metrics[2][1], 'Oversample', feature)
            plot_dct(results_metrics[3][1], 'Smote', feature)

        return results

    results = store_results(results_metrics[0], results_metrics[1], results_metrics[2], results_metrics[3])
    return results

def ensamble_method(X, y, estimators, voting, text):
    voting_classifier = VotingClassifier(estimators = estimators, voting = voting)
    X = X.astype(float)
    y = y.astype(float)
    results = cross_validation(X, y, voting_classifier, text)
    return results, voting_classifier

def generate_params():
    combination = itertools.product([0,0.5,1], repeat=7)

    params = []
    for i in list(combination): 
        params.append(list(i))
        
    del params[0]
    params = {'weights': params }
    return params

def grid_search(X, y, estimator):
    roc_score = make_scorer(roc_auc_score)
    params = generate_params()
    grid_search = GridSearchCV(estimator = estimator, param_grid = params, scoring = roc_score, verbose = 3)

    results = cross_validation(X, y, grid_search,'Grid Search')
    return results

def write_file(all_results):
    with open(reports_path+"output_results_2023.txt", 'a') as outfile:
        all_results.to_string(outfile)


#Classifiers
knn_results = cross_validation(X,y,estimators[0][1], estimators[0][0])
decision_tree_results = cross_validation(X,y,estimators[1][1], estimators[1][0], True, True, atributes)
random_forest_results = cross_validation(X,y,estimators[2][1], estimators[2][0], True, feature = atributes)
logistic_regression_results = cross_validation(X,y,estimators[3][1], estimators[3][0])
all_results = all_results.append([knn_results, decision_tree_results, random_forest_results, 
                                logistic_regression_results,])
write_file(all_results)

linear_svm_results = cross_validation(X,y,estimators[4][1], estimators[4][0])
gaussian_svm_results = cross_validation(X,y,estimators[5][1], estimators[5][0])
sigmoid_svm_results = cross_validation(X,y,estimators[6][1], estimators[6][0])

all_results = all_results.append([linear_svm_results, gaussian_svm_results,sigmoid_svm_results])
write_file(all_results)

#Voting Classifiers
results_voting_hard_classifier, voting_hard_classifier = ensamble_method(X,y,estimators, 'hard', 'Voting Hard')
results_voting_soft_classifier, voting_soft_classifier = ensamble_method(X,y,estimators, 'soft', 'Voting Soft')
#
#
all_results = all_results.append([results_voting_soft_classifier, results_voting_hard_classifier])
write_file(all_results)


#Grid Search
#results_grid = grid_search(X,y, voting_soft_classifier)
#write_file(results_grid)
#all_results = all_results.append([results_grid])

all_results.to_csv(reports_path+'resultados_classificadores_2023.csv', index=False)

