# -*- coding: utf-8 -*-
import shelve
import numpy as np
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

time_db = shelve.open('time_avg_data.db', 'r')
grad_db = shelve.open('gradient_btc.db', 'r')


feed_in_data = grad_db

data_len = len(feed_in_data)
x = np.arange(data_len)
y1,y2 = [],[]

with tqdm(range(data_len), ncols=75) as t:
    for data_id in t:
        y1.append(feed_in_data[str(data_id)][0])
        y2.append(feed_in_data[str(data_id)][1])
t.close()

y1 = np.array(y1)/50.
y2 = np.array(y2)

np.save('bid.npy', y1)
np.save('ask.npy', y2)

plt.figure()
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()