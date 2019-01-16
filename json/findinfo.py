# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 22:49:49 2019

@author: tarena
"""

import json
import pymongo
#
L = ["fanyu","tianhe","baiyun","haizhu","zengcheng","huadu","yuexiu","huangpua","nansha","liwan","conghua"]
#L = ["foshanf","dongguanz","qingyuanq","zhongshanz","zhaoqingz"]

MONGO_URl = 'localhost'
MONGO_DB = 'houseinfo' 
MONGO_TABLE = 'houseurl' 
client = pymongo.MongoClient(MONGO_URl)
db = client[MONGO_DB] 

List = []
for num in L:
    name = num
    with open("{}.json".format(name),'r') as f:
        infos = f.readlines()
        for i in infos:
            info = json.loads(i)
            if int(info["number"]) >= 50:
                text = {"url":info["houseurl"],"name":info["xiaoqu"]}
                print(text)            
                List.append(info["houseurl"])


#num = db.houseurl.find().count()
with open('houseurl.txt', "ab") as f:
    for i in List:
        text = i +'\n'
        f.write(text.encode('utf-8'))
    print("write ok")




print(num)
print(len(List))
        
def writedb(info):
    try:
                    
        db[MONGO_TABLE].insert(info)
        print('存储到MongoDb  OK')
    except Exception:
        print('存储到MongoDb失败')