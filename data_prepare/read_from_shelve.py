# -*- coding: utf-8 -*-
import shelve
import numpy as np
import time
from tqdm import tqdm


db = shelve.open('processed_data_btc.db', 'r')

# data_len = db['total']
data_len = len(db)

for i in range(data_len):
    # ts, bid, ask, bid_quantity_sum, ask_quantity_sum = db[str(i)]
    # ts = time.asctime(time.localtime(ts))
    #
    # print(ts, bid, ask)
    a = db[str(i)]
    print(a)

# time_interval = 5