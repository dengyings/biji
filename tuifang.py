# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 14:51:56 2019

@author: tarena
"""
from Leju import Leju





class CopyHouseInfo(object):
    def __init__(self,name,posswd,url):
        self.name = name
        self.posswd = posswd
        self.url = url
        
    def Begin(self,url):
        newurl = url.split("/")
        leju = "fs.esf.leju.com"
        ZY = "fs.centanet.com"
        TC = "fs.58.com"
        Fang = "fs.esf.fang.com"
        AJK = "foshan.anjuke.com"
        GJ = "fs.ganji.com"
        if leju in newurl:
            Leju.AutoCopy(self.name, self.passwd, self.url)
        elif ZY in newurl:
            DoZongyuan(self.name, self.passwd, self.url)
        elif TC in newurl:
            Do58tongcheng(self.name, self.passwd, self.url)
        elif Fang in newurl:
            DoFangtianxia(self.name, self.passwd, self.url)
        elif AJK in newurl:
            DoAjk(self.name, self.passwd, self.url)
        elif GJ in newurl:
            DoGangji(self.name, self.passwd, self.url)
        else:
            return "您输入的ＵＲＬ有误，请重新输入．"
            
            
           


url = "https://gz.esf.leju.com/detail/13954083/#zn=pc-house-zbfy"
newurl = url.split("/")
a = "gz.esf.leju.com"
if a in newurl:
    print("newur")