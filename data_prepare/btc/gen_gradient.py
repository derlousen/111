# -*- coding: utf-8 -*-

import shelve
import numpy as np
import time
from tqdm import tqdm


db = shelve.open('time_avg_data.db', 'r')
processed_data = shelve.open('gradient_btc.db', 'n')

data_len = len(db)
print(data_len)

for i in tqdm(range(1,data_len), ncols=75):
    diff = db[str(i)]-db[str(i-1)]
    processed_data[str(i-1)] = diff

db.close()
processed_data.close()