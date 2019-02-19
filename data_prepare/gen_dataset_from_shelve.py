# -*- coding: utf-8 -*-
import shelve
import numpy as np
import time
from tqdm import tqdm


db = shelve.open('data.db', 'r')

processed_data = shelve.open('processed_data.db', 'r')

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
for i in range(data_len):
    ts, bid, ask, bid_quantity_sum, ask_quantity_sum = db[str(i)]
    if (ts-start_time)>time_interval_s:
        time_fragment = np.array(time_fragment)

