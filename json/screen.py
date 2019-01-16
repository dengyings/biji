# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:21:41 2019

@author: tarena
"""

import re
from selenium import webdriver
import requests
import json
import time


L = []
with open("/home/tarena/myproject/myproject/DYProject/json/houseurl.txt",'r') as f:
    text1 = f.readlines()
    for i in text1:
       url1 = i[:-2]
       L.append(url1)
     
OverL = []    
with open("overhouseurl.txt", "r") as f:
    text2 = f.readlines()
    for y in text2:        
        url2 = y[:-1]
        OverL.append(url2)
Z = []
for z in L:
    if z not in OverL:
        Z.append(z)
        with open("nexthouseurl.txt", "ab") as f:
            text = z +'\n'
            f.write(text.encode('utf-8'))
    print("writeurlOK")
       
print(len(L))
print(len(OverL))
print(len(Z))
        
        
        
