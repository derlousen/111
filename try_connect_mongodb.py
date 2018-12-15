# -*- coding: utf-8 -*-
import pymongo

myclient = pymongo.MongoClient('mongodb://192.168.1.19:12345/')

# dblist = myclient.list_database_names()
# # dblist = myclient.database_names()
# if "runoobdb" in dblist:
#     print("数据库已存在！")

mydb = myclient["record"]

btc = mydb['btc']


x = btc.find()[2]

bids = x['data']['bids']
asks = x['data']['asks']
ts = x['data']['ts']

print(ts)
print(bids)
print(asks)

