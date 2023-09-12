import numpy as np
from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_kmeans(X, y_kmeans, kmeans, title):
    plt.scatter(X[:,1], X[:,2], c=y_kmeans, s=50, cmap='viridis')
    plt.grid()
    plt.xlabel("Balance Ether")
    plt.ylabel("Total transactions")
    plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2], c='red', s=100,)
    plt.savefig("../../reports/2023/kmeans_unlabeled"+title+".pdf")
    plt.close()

def plot_graph(x, y, title):
    plt.plot(x, y, 'bx-')
    plt.title(title)
    plt.show()
    plt.savefig("../../reports/2023/"+title+".pdf")
    plt.close()



#################### Import data 
#data = pd.read_csv("../../data/processed/final_dataset_2020.csv")
#data.drop(columns=["Unnamed: 0"], inplace=True)

data = pd.read_csv("../../data/processed/kmeans_data_unsupervised_2023.csv")
#data_features = pd.read_csv("../../data/external/eth_addrs_features.csv")


#data = data_classes[data_classes.label == 0]
#print("Tamanho dos dados sem labels:", len(data))

#################### Preprocess data 
# X = data.iloc[:, [1,3,4,5,6,7]]

X = data.iloc[:, 2:-2]
X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
#y = data.iloc[:, -1]
#print(y)

outpu_dataset = pd.DataFrame()

################# Elbow method 
wcss = [] 
silhouette = []
for n_cluster in range(2, 11): 
    kmeans = KMeans(n_clusters = n_cluster, init = 'k-means++', n_init = 10, max_iter = 300)
    kmeans.fit(X) 
    y_kmeans = kmeans.predict(X)
    
    silhouette_avg = silhouette_score(X, y_kmeans)
    
    title = f"cluster_{n_cluster}"
    plot_kmeans(X, y_kmeans, kmeans, title)

    wcss.append(kmeans.inertia_)
    print(f"For n_cluster = {n_cluster}, silhouette_score = {silhouette_avg}")
    silhouette.append(silhouette)
        
    outpu_dataset[title] = kmeans.labels_


outpu_dataset.to_csv("../../data/processed/kmeans_clusters_2023.csv", index=False)

plt.plot(range(2, 11), wcss, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig("../../reports/2023/elbow_method.pdf")
plt.close()