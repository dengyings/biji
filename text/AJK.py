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
driver.get("https://vip.anjuke.com/login/?errmsg=%E7%94%A8%E6%88%B7%E5%90%8D%E5%AF%86%E7%A0%81%E9%94%99%E8%AF%AF")
time.sleep(2)
#driver.find_element_by_xpath('/html/body/div[4]/div[2]/div/div[1]/span[2]').click()
driver.find_element_by_id('loginName').click()
driver.find_element_by_id('loginName').clear()
driver.find_element_by_id('loginName').send_keys('15902047726')
driver.find_element_by_id("loginPwd").click()
driver.find_element_by_id('loginPwd').clear()
driver.find_element_by_id("loginPwd").send_keys('y237454222')
driver.find_element_by_id("loginSubmit").click()#登录摁扭
#driver.find_element_by_id("TPL_password_1").click()
#driver.find_element_by_id("TPL_password_1").send_keys('0.0123456')
time.sleep(2)
driver.find_element_by_xpath("/html/body/div[10]/div[1]/a").click()
#driver.find_element_by_id('q').send_keys('华为mate20')
#driver.find_element_by_css_selector("/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/form/div[1]/button").click()
#time.sleep(10)
#dragger=driver.find_element_by_id('nc_1_n1z')#.滑块定位
driver.find_element_by_xpath("/html/body/div[4]/div[1]/ul/li[5]/a").click()
time.sleep(2)
driver.find_element_by_xpath("/html/body/div[4]/div[1]/ul/li[5]/dl/dt[1]/a").click()
window_1 = driver.current_window_handle
windows = driver.window_handles
for current_window in windows:
     if current_window != window_1:
         driver.switch_to.window(current_window)
time.sleep(2)
driver.find_element_by_xpath('//*[@id="community_unite"]').click()
driver.find_element_by_xpath('//*[@id="community_unite"]').send_keys('星汇海珠湾')
time.sleep(0.5)
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[2]/div/ul/li[1]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[3]/input[1]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[3]/input[1]").send_keys('4')
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[3]/input[2]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[3]/input[2]").send_keys('2')
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[3]/input[3]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[3]/input[3]").send_keys('2')
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[4]/div[1]/div/div").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[4]/div[1]/div/ul/li[2]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[4]/div[2]/div/div").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[4]/div[2]/div/ul/li[5]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[4]/div[3]/div/div").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[4]/div[3]/div/ul/li[2]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[5]/input[1]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[5]/input[1]").send_keys('16')
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[5]/input[2]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[5]/input[2]").send_keys('55')
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[7]/div[1]/div[1]/input").click()
time.sleep(0.5)
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[7]/div[1]/div[1]/div/ul/li[8]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[7]/div[1]/div[2]/input").click()
time.sleep(0.5)
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[7]/div[1]/div[2]/div/ul/li").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[7]/div[1]/div[3]/input").click()
time.sleep(0.5)
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[7]/div[1]/div[3]/div/ul/li[1]").click()

#driver.find_element_by_xpath("/html/body/div[4]/div/form/div[6]/input[3]").click()
#driver.find_element_by_xpath("/html/body/div[4]/div/form/div[6]/input[3]").send_keys('3')
#driver.find_element_by_xpath("/html/body/div[4]/div/form/div[8]/input[2]").click()
#driver.find_element_by_xpath("/html/body/div[4]/div/form/div[8]/input[2]").send_keys('104')
#driver.find_element_by_xpath("/html/body/div[4]/div/form/div[8]/input[3]").click()
#driver.find_element_by_xpath("/html/body/div[4]/div/form/div[8]/input[3]").send_keys('90')
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[9]/div[1]/div/div").click()
time.sleep(0.5)
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[9]/div[1]/div/ul/li[2]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[9]/div[2]/div/div").click()
time.sleep(0.5)
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[9]/div[2]/div/ul/li[2]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[10]/input").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[10]/input").send_keys('2016')
#全部选择商品房
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[11]/label[2]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[13]/label[4]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[14]/label[3]").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[15]/label[2]").click()
driver.find_element_by_xpath('//*[@id="totalPrice"]').click()
driver.find_element_by_xpath('//*[@id="totalPrice"]').send_keys('488')
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[19]/input").click()
driver.find_element_by_xpath("/html/body/div[4]/div/form/div[19]/input").send_keys('广纸新城　星汇海珠湾　地铁口上盖　实用四房　单合同')
driver.find_element_by_xpath('//*[@id="txt_1"]').click()
driver.find_element_by_xpath('//*[@id="txt_1"]').send_keys('wasingaskldmfowenmas,ldnfiwdagda')

driver.find_element_by_xpath('//*[@id="txt_2"]').click()
driver.find_element_by_xpath('//*[@id="txt_2"]').send_keys('adafksnduakdsnigqnkdaingeagadg')
driver.find_element_by_xpath('//*[@id="txt_4"]').click()
driver.find_element_by_xpath('//*[@id="txt_4"]').send_keys('sdaiognoisadnmogeiognoisadngiweg')
driver.find_element_by_xpath('/html/body/div[4]/div/form/div[25]/div/div[1]/a[2]').click()
driver.find_element_by_xpath('/html/body/div[4]/div/form/div[25]/div/div[1]/a[9]').click()
driver.find_element_by_xpath('/html/body/div[4]/div/form/div[25]/div/div[1]/a[8]').click()
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/1.jpg")
time.sleep(120)
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/2.jpg")
time.sleep(120)
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/3.jpg")
time.sleep(120)
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/4.jpg")
time.sleep(120)
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/5.jpg")
time.sleep(120)
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/6.jpg")
time.sleep(120)
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/7.jpg")
time.sleep(120)
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/8.jpg")
time.sleep(120)
driver.find_element_by_id("room_fileupload").send_keys("/home/tarena/picture/9.jpg")
time.sleep(120)
driver.find_element_by_xpath('/html/body/div[4]/div/form/div[28]/div/div[1]/div/i[1]').click()
driver.find_element_by_id("model-fileupload").send_keys("/home/tarena/picture/10.jpg")
time.sleep(120)
driver.find_element_by_xpath('/html/body/div[4]/div/form/div[30]/div/div[1]/div/i[1]').click()
#driver.find_element_by_id("publish-ershou-add").click()


#driver.find_element_by_xpath("/html/body/div[10]/div[1]/a").click()
#action=ActionChains(driver)
#for index in range(500):

 #   try:

#        action.drag_and_drop_by_offset(dragger, 500, 0).perform()#平行移动鼠标，此处直接设一个超出范围的值，这样拉到头后会报错从而结束这个动作

#    except UnexpectedAlertPresentException:

 #       break

 #   time.sleep(11)  #等待停顿时间

 

#driver.find_element_by_id('J_SubmitStatic').click()#重新摁登录摁扭

#print("finish")
#elem4 = driver.find_element_by_partial_link_text("#J_Reviews")
#elem4.click()

#time.sleep(10)
#red = driver.page_source # 查看网页源码

#with open("toutiao.html", "ab") as f:
#    text = red
#    f.write(text.encode('utf-8'))
#    print('writeOK')
#driver.get_cookies() ## 获取当前浏览器的全部cookies
#driver.current_url # 获取当前页面的url




#driver.close() #退出当前页面， 但浏览器还在
#driver.quit()