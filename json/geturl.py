# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:18:01 2019

@author: tarena
"""
from selenium import webdriver
import re
import time
import random
import json


def auto():
    html = driver.page_source
    xiaoqu = re.findall(r'<h1>([\s\S]*?)</h1>',html)[0]
    infos = re.findall(r'<li class="m-rent-house"([\s\S]*?)</li>',html)
    for info in infos:
        item={}
        imageurl = re.findall(r'<img src="([\s\S]*?)"',info)[0]
        title = re.findall(r'class="title-info"[\s\S]*?title="([\s\S]*?)"',info)[0]
        houseurl = re.findall(r'class="title-info"[\s\S]*?href="([\s\S]*?)"',info)[0]
        res = re.findall(r'<p class="para"([\s\S]*?)</p>',info)[0]
        mianji = re.findall(r'<span>([\s\S]*?)</span>',res)[0]
        huxing = re.findall(r'<span>([\s\S]*?)</span>',res)[1]
        duotu = re.findall(r'<i class="icon-tag">([\s\S]*?)</i>',info)
        anxuan = re.findall(r'<i class="house-icon house-icon-anxuan">([\s\S]*?)</i>',info)
        item["xiaoqu"]=xiaoqu    
        item["imageurl"]=imageurl[:-4]
        item["title"]=title
        item["houseurl"]=houseurl
        item["mianji"]=mianji
        item["huxing"]=huxing
        item["duotu"]=len(duotu)
        item["anxuan"]=len(anxuan)
        with open("everyhouseinfo.json", "ab") as f:
            text = json.dumps(dict(item),ensure_ascii=False)+'\n'
            f.write(text.encode('utf-8'))
            print("writeOK")
    driver.find_element_by_xpath('//a[@class="aNxt"]').click()

    
L = []
with open("/home/tarena/myproject/myproject/DYProject/json/nexthouseurl.txt",'r') as f:
    text = f.readlines()
    for i in text:
       url = i[:-1]
       L.append(url)
       

driver = webdriver.Firefox(executable_path='/home/tarena/anaconda3/geckodriver')
for ourl in L: 
    driver.get(ourl) 
    try:
        for i in range(1,100):
            time.sleep(random.randint(3,5))
            try:
                auto()
                print("第",i,"页")
            except Exception as e:
                print(e)
                break
    except Exception as e:
        print(e)
        continue
    with open("overhouseurl.txt", "ab") as f:
        text = ourl+'\n'
        f.write(text.encode('utf-8'))
        print("writeurlOK")
    time.sleep(random.randint(3,10))
