# -*- coding: utf-8 -*-
import pymongo
import time
import shelve
import numpy as np
from tqdm import tqdm


myclient = pymongo.MongoClient('mongodb://192.168.1.19:12345/')
db_file = shelve.open('raw_data.db', flag='n')

mydb = myclient["record"]
btc = mydb['btc']

# mg_data = btc.find().batch_size(5000)

mg_data = list(btc.find())

x = btc.find()[0]


bids = x['data_id']['bids']
asks = x['data_id']['asks']
ts = x['data_id']['ts']

data_len = btc.find().count()

# print(ts)
# print(len(bids))
# print(asks)

def return_one_number(bids, asks):
    bids = np.array(bids)
    asks = np.array(asks)
    bid_price_sum = np.sum(bids[:,0] * bids[:,1])
    bid_quantity_sum = np.sum(bids[:, 1])
    bid = bid_price_sum/bid_quantity_sum
    
    ask_price_sum = np.sum(asks[:,0] * asks[:,1])
    ask_quantity_sum = np.sum(asks[:, 1])
    ask = ask_price_sum/ask_quantity_sum
    return bid, ask, bid_quantity_sum, ask_quantity_sum


# print(return_one_number(bids, asks))


start_time = time.asctime(time.localtime(int(ts/1000)))

print('Start time: ',start_time, '\t', data_len)

db_file['total'] = data_len

for i in tqdm(range(data_len), ncols=75):
    data = mg_data[i]['data_id']

    bids = data['bids']
    asks = data['asks']
    ts = data['ts']
    bid, ask, bid_quantity_sum, ask_quantity_sum = return_one_number(bids, asks)
    db_file[str(i)] = [int(ts/1000), bid, ask, bid_quantity_sum, ask_quantity_sum]


db_file.close()
myclient.close()
