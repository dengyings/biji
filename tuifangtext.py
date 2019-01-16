# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 15:08:24 2019

@author: tarena
"""
from selenium import webdriver
import json
import requests
from lxml import etree
import time
import re

#driver = webdriver.PhantomJS()
driver = webdriver.Firefox(executable_path='/home/tarena/anaconda3/geckodriver')

#L = ["fanyu","tianhe","baiyun","haizhu","zengcheng","huadu","yuexiu","huangpua","nansha","liwan","conghua","guangzhouzhoubian"]
L = ["foshanf","dongguanz","qingyuanq","zhongshanz","zhaoqingz"]
for i in L:
    name = i
    driver.get("https://guangzhou.anjuke.com/community/{}".format(name))
    begin(name)
    time.sleep(10)
#name = "fanyu"
#driver.get("https://guangzhou.anjuke.com/community/{}".format(name))
#begin(name)

def auto(name):
    name = name
    text = driver.page_source
    Item = {}
    houseinfos = re.findall(r'class="li-itemmod"[\s\S]*?class="price-txt price-down"',text)
    for houseinfo in houseinfos:
        html = etree.HTML(houseinfo)
        xiaoqu = re.findall(r'title="([\s\S]*?)"',houseinfo)[0]
        price = re.findall(r'<strong>([\s\S]*?)</strong>',houseinfo)[0]
        time = re.findall(r'date">.{5}([\s\S]*?)<',houseinfo)[0]
        number = re.findall(r'target="_blank">.(\d+).</a>',houseinfo)[0]
        houseurl = re.findall(r'href="(.*?)".target="_blank">.\d+.</a>',houseinfo)[0]
        Item["xiaoqu"]=xiaoqu
        Item["price"]=price
        Item["time"]=time.strip()
        Item["number"]=number
        Item["houseurl"]=houseurl
        with open("{}.json".format(name), "ab") as f:
            text = json.dumps(dict(Item),ensure_ascii=False)+'\n'
            f.write(text.encode('utf-8'))
    driver.find_element_by_xpath('//a[@class="aNxt"]').click()

def begin(name):
    name = name
    for i in range(1,300):
        time.sleep(5)
        try:
            auto(name)
            print(name,"第",i,"页")
        except:
            break
    print(name,"共爬取了",i,"页")
