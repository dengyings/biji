# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 19:42:31 2018

@author: tarena
"""

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import UnexpectedAlertPresentException
import time,unittest, re

#driver = webdriver.PhantomJS()
#driver = webdriver.Firefox(executable_path='/home/tarena/anaconda3/geckodriver')
#encoding=utf-8
 
driver = webdriver.Firefox(executable_path='/home/tarena/anaconda3/geckodriver')
driver.get("http://j.esf.leju.com/ucenter/login?curcity=suzhou")
time.sleep(2)
driver.find_element_by_id('username').click()
driver.find_element_by_id('username').clear()
driver.find_element_by_id('username').send_keys('15902047726')
driver.find_element_by_id("password").click()
driver.find_element_by_id('password').clear()
driver.find_element_by_id("password").send_keys('15902047726')
driver.find_element_by_id("btn_login").click()#登录摁扭
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
try:
    driver.find_element_by_xpath("/html/body/div[5]/div[2]/div").click()#关闭弹窗
except:
    pass
driver.find_element_by_xpath('//div[@id="scrollList"]/dl[9]/dd[1]/a').click()#出售房源
driver.find_element_by_xpath("/html/body/div[2]/div/div[1]/div/div/div[1]/a[1]/span").click()#发布房源
driver.find_element_by_id("commName").click()
driver.find_element_by_id('commName').clear()
driver.find_element_by_id("commName").send_keys('星汇海珠湾')
time.sleep(10)
driver.find_element_by_xpath('//div[@class="drop-box dw_352"]/ul/li/span').click()
driver.find_element_by_id('tPrice').click()
driver.find_element_by_id('tPrice').send_keys('520')
driver.find_element_by_xpath('//div[@class="fill mb20"]/i').click()
driver.find_element_by_id('s').send_keys('4')
driver.find_element_by_id('t').send_keys('2')
driver.find_element_by_id('w').send_keys('2')
driver.find_element_by_id('area').send_keys('104')
driver.find_element_by_id('selectFloor').click()
driver.find_element_by_id("ptzz").click()
driver.find_element_by_xpath('//select[@id="ptzz"]/option[1]').click()
driver.find_element_by_id('zsqk').click()#装修情况
driver.find_element_by_xpath('//select[@id="zsqk"]/option[6]').click()
driver.find_element_by_id('sx').click()
driver.find_element_by_xpath('//select[@id="sx"]/option[5]').click()
driver.find_element_by_id('selectFloor').click()
driver.find_element_by_xpath('//select[@id="selectFloor"]/option[4]').click()
driver.find_element_by_id('totalfloor').send_keys('32')
driver.find_element_by_id('buildyear').click()
driver.find_element_by_id('buildyear').clear()
driver.find_element_by_id('buildyear').send_keys('2017')
driver.find_element_by_xpath('//select[@name="property_rights"]').click()
driver.find_element_by_xpath('//select[@name="property_rights"]/option[2]').click()
driver.find_element_by_id('tag_24').click()
driver.find_element_by_id('tag_21').click()
driver.find_element_by_id('tag_19').click()
driver.find_element_by_id('htit').send_keys('星汇海珠湾　广纸新城　实用四房　进地铁')
driver.find_element_by_xpath('//table[@class="ke-textarea-table"]/tbody/tr/td/iframe').send_keys('sdafaeags gqweg sdfgweg asgrwegasdgrwgasg ag')
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/1.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/2.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/3.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/4.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/5.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/6.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/7.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/8.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/9.jpg")
time.sleep(60)
driver.find_element_by_xpath('//div[@id="upload_fx"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/10.jpg")
time.sleep(60)
driver.find_element_by_id('xqdatabase').click()
try:
    driver.find_element_by_xpath('//div[@class="choosePic"]/ul/li[1]/div[2]').click()
    driver.find_element_by_xpath('//div[@class="choosePic"]/ul/li[2]/div[2]').click()
    driver.find_element_by_xpath('//div[@class="choosePic"]/ul/li[3]/div[2]').click()
except:
    pass

driver.find_element_by_id('pubBtn').click()
driver.find_element_by_xpath('//button[@class="ui-dialog-autofocus"').click()

