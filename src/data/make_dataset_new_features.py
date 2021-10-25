import pandas as pd
import numpy as np
import tqdm 

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
    df_transactions = pd.read_csv('dataset_fix_20200810.csv')
    df = pd.read_csv('accounts_features.csv')

    sent, received = count_sent_recived(df.user_account, df_transactions.user_to, df_transactions.user_from)
    contracts_sent, contracts_received = count_contracts(df_transactions, df.user_account)

    df['sent'] = sent
    df['received'] = received
    df['n_contracts_sent'] = contracts_sent
    df['n_contracts_received'] = contracts_received

    df.to_csv('accounts_features.csv', index=False)