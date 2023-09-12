# Cria base de dados para o aprendizado não supervisionado

import pandas as pd

data_features = pd.read_csv("../../data/external/eth_addrs_features.csv")
data_classes = pd.read_csv("../../data/external/eth_addrs_classes.csv", usecols=["label"])

concat_df_classes = pd.concat([data_features, data_classes], axis=1)

# concat_df_classes[concat_df_classes.label == 0].reset_index(drop=True).to_csv("../../data/processed/kmeans_data_unsupervised_2023.csv")

## After that, run:
## $ python3 unsupervised.py 
## $ python3 unsupervised_information.py > kmeans_cluster__info_2023.txt
kmeans_clusters = pd.read_csv("../../data/processed/kmeans_clusters_2023.csv", usecols=['cluster_10'])
kmeans_data_unsupervised = pd.read_csv("../../data/processed/kmeans_data_unsupervised_2023.csv")
kmeans_data_unsupervised = kmeans_data_unsupervised.drop(columns='label')

#####
### Como no resultado nao supervisionado, o grupo majoritário ficou com o grupo 3
### Alterar base para considerar 0 os que estão no grupo 3 e -1 no grupo != 3
### Considerando o cluster_10.
###

##### Criando novo dataset 

concat_df_clusters = pd.concat([kmeans_data_unsupervised, kmeans_clusters], axis=1)
concat_df_clusters = concat_df_clusters.drop(columns='Unnamed: 0')

# Aplicar a condição e criar a nova coluna
concat_df_clusters['label'] = concat_df_clusters['cluster_10'].apply(lambda x: 0 if x == 3 else -1)
concat_df_clusters = concat_df_clusters.drop(columns='cluster_10')

new_data_supervised_unsupervised = pd.concat([concat_df_clusters, concat_df_classes[concat_df_classes.label == 1]])

# Misturar aleatoriamente as linhas do DataFrame
new_data_supervised_unsupervised = new_data_supervised_unsupervised.sample(frac=1, random_state=42)  # random_state é opcional, usado para reprodutibilidade

# Redefinir o índice do DataFrame
new_data_supervised_unsupervised.reset_index(drop=True, inplace=True)


new_data_supervised_unsupervised.to_csv("../../data/processed/data_unsupervised_supervised_2023.csv", index=False)
