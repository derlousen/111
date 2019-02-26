# -*- coding: utf-8 -*-
import pymongo
from tqdm import tqdm

myclient = pymongo.MongoClient('mongodb://192.168.1.19:12345/')


mydb = myclient["record"]
btc = mydb['btc']


data_len = btc.find().count()

x = list(btc.find().batch_size(50000))


for i in tqdm(range(10000)):
    x[i]

    bids = x['data_id']['bids']
    asks = x['data_id']['asks']
    ts = x['data_id']['ts']

    # print(ts, len(bids), len(asks))
    # print(len(bids))
    # print(asks)

