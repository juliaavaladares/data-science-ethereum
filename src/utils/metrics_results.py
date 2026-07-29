import pandas as pd
from sklearn import metrics
from collections import Counter


class MetricsDataFrame:
    def __init__(self, y_true, y_pred, classifier_name):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classifier_name = classifier_name
        self.results = pd.DataFrame({'Acuracy':[],
                                'Precision':[],
                                'Recall':[],
                                'F1-score':[],
                                'F-beta':[],
                                'MCC':[],
                                'TP':[],
                                'TP (%)':[],
                                'TN':[],
                                'TN (%)':[],
                                'ROC Curve':[]}, )
        self.results.index.names = ['Algoritmos']

        self.generate_dataframe()
        self.export_results('results/new_data/wright/results_df_'+classifier_name+'.csv')

    def generate_dataframe(self):

        acuracy = metrics.accuracy_score(self.y_true, self.y_pred)
        precision = metrics.precision_score(self.y_true, self.y_pred)
        recall= metrics.recall_score(self.y_true, self.y_pred)
        f_score = metrics.f1_score(self.y_true, self.y_pred)
        f_beta = metrics.fbeta_score(self.y_true, self.y_pred, beta=2)
        ROC_auc_curve = metrics.roc_auc_score(self.y_true, self.y_pred)
        MCC = metrics.matthews_corrcoef(self.y_true, self.y_pred)

        print(f'generate data frame: {Counter((self.y_true==1) & (self.y_pred==1))}')
        
        tp = len(self.y_true[(self.y_true==1) & (self.y_pred==1)])
        tn = len(self.y_true[(self.y_true==0) & (self.y_pred==0)]) 

        tp_percentage = (tp/len(self.y_true == 1))*100
        tn_percentage = (tn/len(self.y_true == 0))*100

        result_metrics = [acuracy, precision, recall, f_score, f_beta, MCC, tp, tp_percentage, tn, tn_percentage, ROC_auc_curve]
        self.results = self.results._append(pd.DataFrame([result_metrics], index=[self.classifier_name], columns=self.results.columns))

        print(f"Metrics result: {self.results}")

        with open("results.txt", 'a') as arquivo:
            arquivo.write(str(result_metrics)+self.classifier_name)
            arquivo.write("\n")


    def export_results(self, file_path=None, return_df=False):
        if file_path:
            self.results.to_csv(file_path)
        if return_df:
            return self.results
        