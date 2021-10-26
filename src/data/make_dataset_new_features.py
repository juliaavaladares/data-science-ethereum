import pandas as pd
import numpy as np
from tqdm import tqdm 

def count_sent_recived(accounts, user_to, user_from):
    '''
    Calcula a quantidade de vezes que uma conta recebeu e enviou transações.
    Receber como parâmetro a lista de contas, a lista de user_to e user_from
    '''
    sent = []
    received = []
    for account in tqdm(accounts): 
        s = np.count_nonzero(user_to == account)
        r = np.count_nonzero(user_from == account)
        
        sent.append(s)
        received.append(r)
    
    sent, received = pd.Series(sent), pd.Series(received)

    return sent, received


def count_contracts(df, accounts):
    '''
    Função para calcular a quantidade de contratos que uma conta envia e quantos
    contratos ela recebe. Para isso a função recebe a base de dados das transações
    e as contas presentes na base
    '''
    contracts_sent = []
    contracts_received = []
    
    for a in tqdm(accounts):
        
        c_s = sum(df.value[df.user_from == a] == 0)
        contracts_sent.append(c_s)
        
        c_r = sum(df.value[df.user_to == a] == 0)
        contracts_received.append(c_r)
        
    return contracts_sent, contracts_received

def main():
    path = "../../data/raw/"
    eth1 = pd.read_csv(path+"eth-new-dataset0.txt", sep=",", usecols=["hash", "from", "to", "value"])
    eth2 = pd.read_csv(path+"eth-new-dataset1.txt", sep=",", usecols=["hash", "from", "to", "value"])
    
    df_transactions = pd.concat([eth1, eth2], ignore_index=True)
    accounts = pd.read_csv(path+"Accounts2021.csv")

    sent, received = count_sent_recived(accounts.accounts, df_transactions.to, df_transactions["from"])
    contracts_sent, contracts_received = count_contracts(df_transactions, accounts.accounts)

    accounts['sent'] = sent
    accounts['received'] = received
    accounts['n_contracts_sent'] = contracts_sent
    accounts['n_contracts_received'] = contracts_received

    accounts.to_csv('accounts_features.csv', index=False)