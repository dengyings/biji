# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:48:47 2019

@author: tarena
"""
from selenium import webdriver
import json
import codecs


class Showinfo(object):
    def __init__(self,name,Num=0):
        self.name = name
        self.Num = Num
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
        
    def showinfo(self):
        page = re.findall(r'data-page="4">(\d)</a><a href="javascript:;" class="next"', html1)[0]
        for i in range(1,int(page)):
            time.sleep(1)
            number = re.findall(r'data-unityinfoid="(\d+)"', esc)[0]
            days = re.findall(r'<span class="left-time">[\u4e00-\u9fa5]{2}.?(\d+)[\u4e00-\u9fa5]{1}', esc)[0]
            everchick = re.findall(r'<p class="click-detail\d+">(\d+)/(\d+)</p>', esc)[0][0]
            newchick = re.findall(r'<p class="click-detail\d+">(\d+)/(\d+)</p>', esc)[0][1]
            chicks = int(everchick) + int(newchick)
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
            try:
                driver.find_element_by_class_name("next").click()               
            except:
                pass
        self.writeinfo(ALLINFO)           
            return ALLINFO
            
    def writeinfo(self,ALLINFO):
        fout = codecs.open('/home/tarena/myproject/myproject/DYProject/user/{}/show.html'.format(self.name),'w',encoding='utf-8')
        fout.write("<html>")
        fout.write("<head><meta charset='utf-8'/><head>")
        fout.write("<body>")
        fout.write("<table>")
        for info in ALLINFO:
            fout.write("<tr>")
            fout.write("<td>%s</td>"%info["小区"])
            fout.write("<td>%s</td>"%info["面积"])
            fout.write("<td>%s</td>"%info["价格"])
            fout.write("<td>%s</td>"%info["累计点击"])
        fout.write("</table>")
        fout.write("</body>")
        fout.write("</html>")
        fout.close()
            
    def judge(self,Num=self.Num):
        if Num == 0:
            self.Again()
        else:
            infos = self.showinfo()
            Name = []
            house = []
            for info in infos:
                L = ([info["xiaoqu"],info["mianji"],info["louceng"]])
                res = "_".join(L)
                Name.append(res)
            with open("/home/tarena/myproject/myproject/DYProject/user/{}/house.txt".format(self.name),r) as f:
                text = f.readlines()
                for i in text:
                    if i not in Name:
                        house.append(i)
            Newlist = random.sample(house, self.Num)
            for firename in Newlist:
                Doleju.doleju(firename)
            self.judge(0)
     
    def Again(self):
        infos = self.showinfo()
        hothouse = []
        Number = []
        for info in infos:
            if int(item["累计点击"]) > 30:
                L = ([info["xiaoqu"],info["mianji"],info["louceng"]])
                res = "_".join(L)
                hothouse.append(res)
            if int(item["发布天数"]) > 15 and int(item["累计点击"]) < 3:
                Number.append(item["房源编号"])
                if int(item["发布天数"]) > 30:
                    Number.append(item["房源编号"])
        self.Lower(Number)#下架房源
            
            
            
                        
                        
                        
                
                
                
            
                
            
           