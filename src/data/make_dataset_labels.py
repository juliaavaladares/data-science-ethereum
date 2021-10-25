import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pandas as pd
import re

def write_file(line):
    with open("../../data/processed/accounts_labels_2021.txt", "a") as file:
        file.write(line+"\n")

def getLabels(address):
    url = 'https://etherscan.io/address/'+address
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    html = requests.get(url, headers = headers).content
    soup = BeautifulSoup(html, 'html.parser')

    aLabels = soup.find_all("a", {"href" : re.compile('/accounts/label')})
    labels = [l.get('href') for l in aLabels]
    categories = []

    for l in labels:
        split = l.split(sep='/')
        if len(split) > 1:
            categories.append(split[3])
    
    if categories:
        return address, categories, 1
        
    
    return address, ["No label"], 0
    
    
#write_file("user_account,labels,is_professional")

contas = pd.read_csv('../../data/raw/Accounts2021.csv', skiprows=(1,2888))
contas= contas["accounts"].values

for conta in tqdm(contas):
    account, labels, is_professional = getLabels(conta)
    time.sleep(0.5)
    
    if len(labels) > 1:
        labels = ' '.join(labels)
        line = account + "," + labels + "," + str(is_professional)
        write_file(line)
        continue
    
    line = account + "," + labels[0] + "," + str(is_professional)
    write_file(line)