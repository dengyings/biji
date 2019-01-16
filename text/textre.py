# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:36:36 2019

@author: tarena
"""
import json

with open("/home/tarena/DYProject/text/ank.text", "r") as f:
    text = f.read()
    print('rideOK')
    
page = re.findall(r'<tr data-unityinfoid[\s\S]*?</tr>', text)
print(len(pageo))

abc = 0
abd = 0 
ALLINFO = []
change = []
for esc in page:
    item = {}
    number = re.findall(r'data-unityinfoid="(\d+)"', esc)[0]
    days = re.findall(r'<span class="left-time">[\u4e00-\u9fa5]{2}.?(\d+)[\u4e00-\u9fa5]{1}', esc)[0]
    fb = re.findall(r'<td class="platform[\s\S]*?</td>', esc)[0]
    #print(fb)
    fb1 = re.findall(r'[\u4e00-\u9fa5]{3}', fb)   
    if len(fb1) == 0:
        fb = "0"
        abc += 1
        
    else:
        fb = "1"
        abd += 1
    everchick = re.findall(r'<p class="click-detail\d+">(\d+)/(\d+)</p>', esc)[0][0]
    newchick = re.findall(r'<p class="click-detail\d+">(\d+)/(\d+)</p>', esc)[0][1]
    chicks = int(everchick) + int(newchick)
    info = re.findall(r'list-comm-name">[\s\S]*?</p>', esc)[0]
    title = re.findall(r'>([\s\S]*?)</span>', info)[0]
    acreage = re.findall(r'</span>.?([\s\S]*?[\u4e00-\u9fa5])[\s\S]*?</p>', info)[0]
    price = re.findall(r'span>[\s\S]*?(\d+.?\d{2}[\u4e00-\u9fa5]{1})</p>', info)[0]
    item["房源编号"]=number
    item["发布天数"]=days
    item["是否发布"]=fb
    item["累计点击"]=chicks
    item["小区"]=title
    item["面积"]=acreage
    item["价格"]=price
    ALLINFO.append(item)
    if int(days) >= 30 and int(fb) == 1:
        change.append(number)
    if int(days) >= 15 and int(chicks) <= 5 and int(fb) == 1:
        change.append(number)
    with open("aje.json", "ab") as f:
        text = json.dumps(dict(item),ensure_ascii=False)+'\n'
        f.write(text.encode('utf-8'))
#print(ALLINFO)
print(change)
print(len(change))
    
   # print("房源编号：", number,"发布天数：", days,fb,"累计点击：", chicks,title,acreage,price)
print(abc,abd)


myl = ["asdnfinsif","asdnifnsiadf","asdfnisdnf","asdnfisnaif"]
sht =""
for i in myl:
    sht += i
print(sht)
    





