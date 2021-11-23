import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

def enconde_labels(labels_list):
    labels_encoded = preprocessing.LabelEncoder().fit_transform(labels_list)
    return labels_encoded

def normalize_data(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    return scaler.fit_transform(data) 

def create_3d_graphic(x, y, z, name, labels=None):
    path_figures = "../../reports/figures/"
    
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    scatter = ax.scatter3D(x, y, z, c=labels, s=100)
    ax.set_xlabel('Balance Ether')
    ax.set_ylabel('Total transactions')
    ax.set_zlabel('Transactions sent')
    
    plt.savefig(path_figures+"3D_"+name+".png")
    plt.close()



# Colunas 'user_account', 'balance_ether', 'balance_value', 'total_transactions', 'sent', 'received', 'n_contracts_sent', 'n_contracts_received','labels', 'is_professional'
data = pd.read_csv("../../data/raw/new_accounts_features2020.csv")
data_labels = pd.read_csv("../../data/processed/accounts_labels_2020.txt")
data["labels"] = data_labels["labels"]

labeled_accounts = data[data["labels"] != "No label"]

# ======================= ALLL ACCOUNTS 3D PLLOT ===========================
data_normalized = normalize_data(data[["balance_ether", "total_transactions", "sent"]])
axis_x = data_normalized[:,0] #Balance Ether
axis_y = data_normalized[:,1] #Total_transactions
axis_z = data_normalized[:,2] #Sent

labels = enconde_labels(data["labels"].values)
create_3d_graphic(axis_x, axis_y, axis_z, "all_accounts_2020")

# ======================= ALLL DATA 3D PLLOT ===========================

data_normalized = normalize_data(labeled_accounts[["balance_ether", "total_transactions", "sent"]])
axis_x = data_normalized[:,0] #Balance Ether
axis_y = data_normalized[:,1] #Total_transactions
axis_z = data_normalized[:,2] #Sent

labels = enconde_labels(labeled_accounts["labels"].values)
create_3d_graphic(axis_x, axis_y, axis_z, "labeled_accounts_2020", labels)