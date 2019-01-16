# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:01:09 2019

@author: tarena
"""

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import UnexpectedAlertPresentException
import time,unittest, re

html = ""
driver = webdriver.Firefox(executable_path='/home/tarena/anaconda3/geckodriver')
driver.get("https://vip.anjuke.com/login/?errmsg=%E7%94%A8%E6%88%B7%E5%90%8D%E5%AF%86%E7%A0%81%E9%94%99%E8%AF%AF")
time.sleep(2)
driver.find_element_by_id('loginName').click()
driver.find_element_by_id('loginName').clear()
driver.find_element_by_id('loginName').send_keys('15902047726')
driver.find_element_by_id("loginPwd").click()
driver.find_element_by_id('loginPwd').clear()
driver.find_element_by_id("loginPwd").send_keys('y237454222')
driver.find_element_by_id("loginSubmit").click()#登录摁扭
time.sleep(2)
driver.find_element_by_xpath("/html/body/div[10]/div[1]/a").click()#关闭弹窗
driver.find_element_by_xpath("/html/body/div[4]/div[1]/ul/li[5]/a").click()
time.sleep(0.5)
driver.find_element_by_xpath("/html/body/div[4]/div[1]/ul/li[5]/dl/dt[2]/a").click()#进入房源库
time.sleep(1)
#driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
html1 = driver.page_source
page = re.findall(r'data-page="4">(\d)</a><a href="javascript:;" class="next"', html1)[0]
html += html1
time.sleep(3)

#driver.find_element_by_class_name("next").click()
#htmlnew = driver.page_source
for i in range(1,int(page)):
    driver.find_element_by_class_name("next").click()
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")#下拉到底部
    htmlnew = driver.page_source
    html += htmlnew
    time.sleep(1)

page = re.findall(r'<tr data-unityinfoid[\s\S]*?</tr>', html)
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
        if number not in change:
            change.append(number)
    if int(days) >= 15 and int(chicks) <= 5 and int(fb) == 1:
        if number not in change:
            change.append(number)

#print(ALLINFO)
print(change)
print(len(change))
print(abc,abd)
ids = "592265088975872"
driver.find_element_by_xpath('.//tr[@data-unityinfoid={}]/td[2]/div[2]/p[4]/a[3]/i'.format(ids)).click()
driver.find_element_by_xpath('.//div[@class="off-shelf-result"]/div[@class="page-title"]/input').click()


