import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def get_triangle_inferior(distance_matrix, n_rows, n_cols):
    array_triangle_inferior = np.array([])
    for i in range(n_rows):
        for j in range(i+1, n_cols):
            array_triangle_inferior = np.append(array_triangle_inferior, distance_matrix[i][j])
    
    return array_triangle_inferior

##### COLUMNS = ['user_account', 'balance_ether', 'balance_value', 'total_transactions', 'sent', 'received', 'n_contracts_sent', 'n_contracts_received','labels', 'is_professional']
data_frame = pd.read_csv("../../data/processed/final_dataset_2020.csv", 
                        usecols=["user_account", "balance_ether", "total_transactions", "labels"])

groups = data_frame["labels"].value_counts() > 1
groups = groups[groups.values == True].index

for group in groups:
    if group == 'No label':
        continue
    
    group_info = data_frame[["total_transactions", "balance_ether"]][data_frame.labels == group].values
    group_distance_matrix = cdist(group_info, group_info)
    n_rows, n_cols = group_distance_matrix.shape
    
    distances = get_triangle_inferior(group_distance_matrix, n_rows, n_cols)

    print(f"Group: {group}, mean: {distances.mean(): .2f}, std: {distances.std(): .2f}, min: {distances.min(): .2f}, max: {distances.max(): .2f}")
    
    
    
