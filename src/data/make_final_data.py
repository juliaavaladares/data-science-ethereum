from os import path
import pandas as pd

path = "../../data/processed/"

accounts_features = pd.read_csv(path+"accounts_features_2021.txt")
accounts_created_features = pd.read_csv(path+"Accounts2021_Created_Features.csv", nrows=37083)
accounts_labels = pd.read_csv(path+"accounts_labels_2021.txt", nrows=37083)

accounts_features = pd.concat([accounts_features, \
                                accounts_created_features[["sent", "received", "n_contracts_sent", "n_contracts_received"]], \
                                accounts_labels[["labels", "is_professional"]]], axis=1)

accounts_features.to_csv(path+"final_dataset_2021.csv", index=False)