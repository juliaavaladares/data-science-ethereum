from typing import final
import numpy as np
import pandas as pd

#'user_account', 'balance_ether', 'balance_value', 'total_transactions',
#       'sent', 'received', 'n_contracts_sent', 'n_contracts_received',
#       'classification', 'labels', 'cluster_2', 'cluster_3', 'cluster_4',
#       'cluster_5', 'cluster_6', 'cluster_7', 'cluster_8', 'cluster_9',
#       'cluster_10'
data_kmeans = pd.read_csv("../../data/processed/kmeans_clusters.csv")


columns_features = list(data_kmeans.iloc[:, [0,1,2,3,4,5,6,7]].columns.values)
columns_clusters = list(data_kmeans.iloc[:, [10,11,12,13,14,15,16,17,18]].columns.values)

final_cluster_info = pd.DataFrame()

for cluster in columns_clusters:
    columns_features.append(cluster)

    cluster_info = data_kmeans[columns_features].groupby(cluster).agg(['min', 'max', 'mean'])
    final_cluster_info = final_cluster_info.append(cluster_info)

    del columns_features[-1]

final_cluster_info.to_csv("../../reports/clusters_info.csv")


