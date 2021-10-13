import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
etherscan_token = os.getenv("TOKEN")
etherscan_url=os.getenv("URL")

#Dados
accounts_dataset = pd.read_csv("../../data/raw/Accounts2021.csv", sep=",")
contas = accounts_dataset["accounts"].values

def write_file(line):
    with open("new_features_request_2021.tx", "a") as file:
        file.write(line+"\n")

#Funcoes de requests
def get_ether_balance(conta):
    try:
        url = etherscan_url + "?module=account&action=balance&address="+conta+"&tag=latest&apikey="+etherscan_token

        response = requests.get(url)
        result = response.json()["result"]
        balance_ether = float(result)

        return balance_ether

    except Exception as ex:
        print('Error:', ex)
        print('Conta:', conta)
        print('Try again')
        get_normal_transactions(conta)

def get_normal_transactions(conta):
    try:
        url = etherscan_url + "?module=account&action=txlist&address="+conta+"&startblock=0&endblock=99999999&page=1&offset=10&sort=asc&apikey="+etherscan_token

        response = requests.get(url)
        result = response.json()["result"]
        number_normal_transactions = len(result)

        return number_normal_transactions

    except Exception as ex:
        print('Error:', ex)
        print('Conta:', conta)
        print('Try again')
        get_normal_transactions(conta)
    


def get_internal_transactions(conta):
    try:
        url = etherscan_url + "?module=account&action=txlistinternal&address="+conta+"&startblock=0&endblock=2702578&page=1&offset=10&sort=asc&apikey="+etherscan_token

        response = requests.get(url)
        result = response.json()["result"]
        number_internal_transactions = len(result)

        return number_internal_transactions
    
    except Exception as ex:
        print('Error:', ex)
        print('Conta:', conta)
        print('Try again')
        get_internal_transactions(conta)
    
        

write_file("user_account,balance_ether,normal_transactions,internal_transactions")

for conta in tqdm(contas): 
    ether, normal_transactions, internal_transactions = get_ether_balance(conta), get_normal_transactions(conta), get_internal_transactions(conta)

    linha = f"{conta},{ether},{normal_transactions},{internal_transactions}"
    write_file(linha)
    