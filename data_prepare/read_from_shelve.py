# -*- coding: utf-8 -*-
import shelve
import numpy as np
import time
from tqdm import tqdm


db = shelve.open('raw_data.db', 'r')

data_len = db['total']

for i in range(data_len):
    ts, bid, ask, bid_quantity_sum, ask_quantity_sum = db[str(i)]
    ts = time.asctime(time.localtime(ts))

    print(ts, bid, ask)

# time_interval = 5