import pandas as pd

data = pd.read_csv('../../data/external/eth_addrs_classes.csv')
print('Size dt_kmeans_cluster: ', data['label'].value_counts())

#dt_kmeans_data = pd.read_csv('../../data/processed/kmeans_data_unsupervised_2023.csv')
#print('Size dt_kmeans_data: ', dt_kmeans_cluster.shape)