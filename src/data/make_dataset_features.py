import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from urllib.request import Request, urlopen


path = '../../data/raw/'

def write_file(line):
    with open("../../data/processed/accounts_features_2021.txt", "a") as file:
        file.write(line+"\n")

# read the dataset with the data collection by transactions, that contain users hash
accounts_data_frame = pd.read_csv(path + 'Accounts2021.csv', nrows=10000)

url_default = 'https://etherscan.io/address/'
hdr = {'User-Agent': 'Mozilla/5.0'}

s = time.time()


def get_info_account(account_hash):
    global url_default
    global urlopen
    global hdr
    global df

    url = url_default + account_hash

    try:
        req = Request(url,headers=hdr)
        page = urlopen(req)
        soup = BeautifulSoup(page, 'html.parser')


        # Scrap
        balance_ether = soup.select_one('#ContentPlaceHolder1_divSummary > div.row.mb-4 > div.col-md-6.mb-3.mb-md-0 > div > div.card-body > div:nth-child(1) > div.col-md-8')
        balance_value = soup.select_one('#ContentPlaceHolder1_divSummary > div.row.mb-4 > div.col-md-6.mb-3.mb-md-0 > div > div.card-body > div:nth-child(3) > div.col-md-8')
        total_transactions = soup.select_one('#transactions > div.d-md-flex.align-items-center.mb-3 > p > a')


        # Tratamento
        balance_ether = balance_ether.text.strip().split(' ', 1)[0].replace(',', '')
        if(balance_value.text[:4] == 'Less'):
            balance_value = balance_value.text.split(' ', 3)[2].strip()[1:].replace(',', '')
        else:
            balance_value = balance_value.text.strip()[1:].split(' ', 1)[0].replace(',', '')
        total_transactions = total_transactions.text.strip().replace(',', '')

        return (account_hash, balance_ether, balance_value, total_transactions)

    except Exception as ex:
        print('Error:', ex)
        print('Conta:', account_hash)
        print('Try again')
        get_info_account(account_hash)



#write_file("user_account,balance_ether,balance_value,total_transactions")
print("------------------START COLLECTING FEATURES--------------------")

for account_hash in tqdm(accounts_data_frame["accounts"].values):
    time.sleep(4)
    account, balance_ether, balance_value, total_transactions = get_info_account(account_hash)
    line = account + "," + str(balance_ether) + "," + str(balance_value)+ "," + str(total_transactions)
    write_file(line)

print("\n------------------END IDENTIFY CONTRACTS--------------------")
print('\nTempo', (time.time() - s))
