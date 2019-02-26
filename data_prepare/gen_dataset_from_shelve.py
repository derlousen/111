# -*- coding: utf-8 -*-
import shelve
import numpy as np
import time
from tqdm import tqdm


db = shelve.open('data_id.db', 'r')

processed_data = shelve.open('processed_data_btc.db', 'n')

data_len = db['total']

# for i in range(data_len):
#     ts, bid, ask, bid_quantity_sum, ask_quantity_sum = db[str(i)]
#     # print(ts)
#     ts = time.asctime(time.localtime(ts))
#
#     # print(ts, bid, ask)

time_interval_s = 600

ts_init, bid_init, ask_init, bid_quantity_sum_init, ask_quantity_sum_init = db[str(0)]

time_fragment = []
# bid_fragment = []
# ask_fragment = []


start_time = ts_init
counter = 0
for i in tqdm(range(data_len), ncols=75):
    ts, bid, ask, bid_quantity_sum, ask_quantity_sum = db[str(i)]
    if (ts-start_time)>time_interval_s:
        start_time = ts
        time_fragment = np.array(time_fragment)

        time_fragment = np.mean(time_fragment, axis=0)
        # print(time_fragment.shape)
        processed_data[str(counter)] = time_fragment
        time_fragment = []
        counter+=1
    time_fragment.append([bid,ask])

db.close()
processed_data.close()
