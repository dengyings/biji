# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:20:27 2019

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
time.sleep(5)
driver.find_element_by_xpath('/html/body/div[10]/div[1]/a').click()#关闭弹窗
driver.find_element_by_xpath("/html/body/div[4]/div[1]/ul/li[5]/a").click()
time.sleep(0.5)
driver.find_element_by_xpath("/html/body/div[4]/div[1]/ul/li[5]/dl/dt[2]/a").click()#进入房源库
time.sleep(1)
ids = "754910394106881"
driver.find_element_by_xpath('.//tr[@data-unityinfoid={}]/td[2]/div[2]/p[4]/a[2]/i'.format(ids)).click()
window_1 = driver.current_window_handle
windows = driver.window_handles
for current_window in windows:
     if current_window != window_1:
         driver.switch_to.window(current_window)
time.sleep(10)
html = driver.page_source
with open("ajk.txt", "ab") as f:
    text = html
    f.write(text.encode('utf-8'))
    print('writeOK')
driver.close()
driver.find_element_by_class_name("next").click()#点击下一页
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
ibs = "754626417068032"
driver.find_element_by_xpath('.//tr[@data-unityinfoid={}]/td[2]/div[2]/p[4]/a[2]/i'.format(ibs)).click()
time.sleep(10)
driver.close()








