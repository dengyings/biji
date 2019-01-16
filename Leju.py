# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:26:46 2019

@author: tarena
"""
import requests
import os
import time
from selenium import webdriver
from urllib.request import urlretrieve
#import pymongo
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor, as_completed, wait
import re
import json

class Leju(object):
    def __init__(self,name,posswd,url):
        self.name = name
        self.posswd = posswd
        self.url = url
        self.s = requests.session()
        logger = logging.getLogger("AJK")
         # 指定logger的输出格式
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
         # 文件日志，终端日志
        file_handler = logging.FileHandler("AJK.txt")
        file_handler.setFormatter(formatter)
        consle_handler = logging.StreamHandler(sys.stdout)
        consle_handler.setFormatter(formatter)
         # 设置默认的级别
        logger.setLevel(logging.INFO)
         # 把文件日志和终端日志添加到日志处理器中
        logger.addHandler(file_handler)
        logger.addHandler(consle_handler)
    
    def AutoCopy(self,name,posswd,url,num_retries=5):
        if num_retries == 0:
            logger.error("Server error")
            return None
        info = self.s.get(url)
        if info.status_code == 200:
            info.encoding='urf-8'
            houseinfos = info.text
            return Analyinfo(houseinfos)
        elif 400 <= info.status_code < 500:
            if response.status_code == 404:
                    logger.error("Page not found") 
            elif response.status_code == 403:
                    logger.error("Have no rights")
            else:
                pass
            return None
        elif 500 <= response.status_code < 600:
            sleeptime = (6-num_retries) ** 2 * 20
            time.sleep(sleeptime)
            self.AutoCopy(self,url,num_retries-1)
            logger.error("server erroe")
            return None
             
    def Analyinfo(self,houseinfos):
        L = []
        Info = {}
        info = {}
        info["xiaoqu"] = re.findall(r'<div class="ui-form publish-auto-wrap" [\s\S]*?<span>([\s\S]*?)</span>', text)[0]
        info["hushi"] = re.findall(r'checkname="room" name="shi" value="(\d+)"',text)[0]
        info["ting"] = re.findall(r'checkname="hall" name="ting" value="(\d+)"',text)[0]
        info["wei"] = re.findall(r'checkname="bathroom" name="wei" value="(\d+)"',text)[0]
        info["fangwuleixing"] = re.findall(r'ui-select-label"><span data-val="1" data-title=[\s\S]*?>([\s\S]*?)</span>',text)[0]
        info["zhuangxiu"] = re.findall(r'span data-val="4"[\s\S]*?[\u4e00-\u9fa5]+">([\s\S]*?)<',text)[0]
        info["chaoxiang"] = re.findall(r'span data-val="9"[\s\S]*?[\u4e00-\u9fa5]+">([\s\S]*?)<',text)[0]
        info["loucheng"] = re.findall(r'checkname="floor" name="suoZaiLouCeng" value="(\d+)"',text)[0]
        info["zonggao"] = re.findall(r'checkname="allFloor" name="zongLouCeng" value="(\d+)"',text)[0]
        info["mianji"] = re.findall(r'mianJi" value="([\s\S]*?)"',text)[0]
        info["nianxian"] = re.findall(r'span data-val="1" data-title="[\u4e00-\u9fa5]{4}">(\d{2}[\u4e00-\u9fa5])<',text)[0]
        info["niandai"] = re.findall(r'name="params_14" maxlength="4" value="(\d+)" ',text)[0]
        info["dianti"] = re.findall(r'ui-radio ui-radio-checked"><i>.</i>([\u4e00-\u9fa5]+)<',text)[0]
        info["hushi"] = re.findall(r'ui-radio ui-radio-checked"><i>.</i>([\u4e00-\u9fa5]+)<',text)[1]
        info["weiyi"] = re.findall(r'ui-radio ui-radio-checked"><i>.</i>([\u4e00-\u9fa5]+)<',text)[2]
        info["yishou"] = re.findall(r'ui-radio ui-radio-checked"><i>.</i>([\u4e00-\u9fa5]+)<',text)[3]
        info["jiage"] = re.findall(r'jiaGe[\s\S]*?value="([\s\S]*?)"',text)[0]
        info["title"] = re.findall(r'name="title"[\s\S]*?value="([\s\S]*?)"',text)[0]
        info["fangyuanxinxi"] = re.findall(r'txt_1[\s\S]*?>([\s\S]*?)</textarea>',text)[0]
        info["yezhuxintai"] = re.findall(r'txt_2[\s\S]*?>([\s\S]*?)</textarea>',text)[0]
        info["fuwujieshao"] = re.findall(r'txt_4[\s\S]*?>([\s\S]*?)</textarea>',text)[0]
        imageurls = re.findall(r'img style="margin[\s\S]*?src="([\s\S]*?)"',text) 
        L.extend([info["xiaoqu"],info["mianji"],info["louceng"]])
        res = "_".join(L)
        Info[res] = info
        with open("/home/tarena/DYProject/text/info.txt", "r+") as f:
            text = f.readlines()
            if L not in text:
                f.write(res+'\n')
                self.__Write(info,res)
                self.__image_download(imageurls,res)
        return Release(self.name,self.posswd,res)
        
    def __image_download(self,imageurls,res):
        self.__verify_url(imageurls)
        filerPath = '/home/tarena/DYProject/Image/{}'.format(res)
        isExists=os.path.exists(filerPath)
        if not isExists:
            os.makedirs(filerPath)
        i = 1
        New = []
        for url in L:
            newurl = url + str(i)
            i +=1        
            New.append(newurl)
        with ThreadPoolExecutor(5) as pool:
            pool.map(self.__downloadImg,New)
    
    def __downloadImg(self,imgUrl):
        ImgUrl = imgUrl[:-1]
        imgName = re.findall(r'jpg(\d+)',imgUrl)[0]
        urlretrieve(ImgUrl, './image/{}'.format(imgName))

               
    def __Write(self,info,res):
        with open('/home/tarena/DYProject/INFO/{}.json'.format(res), "ab") as f:
            text = json.dumps(dict(info),ensure_ascii=False)+'\n'
            f.write(text.encode('utf-8'))

    
    def __verify_url(self, article_url):    
        verify_lst = ["imp", "esfimg", "jpg"]
        for string in verify_lst:
            if string not in article_url:
                logger.error("image download error",article_url)
            
        
        
        