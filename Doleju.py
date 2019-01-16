# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:04:41 2019

@author: tarena
"""

from selenium import webdriver
import json
import requests


class Doleju(object):
    def __init__(self,name,posswd,firename):
        self.name = name
        self.posswd = posswd
        self.firename = firename
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
        
    def login(self):
        driver = webdriver.Firefox(executable_path='/home/tarena/anaconda3/geckodriver')
        driver.get("http://j.esf.leju.com/ucenter/login?curcity=suzhou")
        time.sleep(2)
        driver.find_element_by_id('username').click()
        driver.find_element_by_id('username').clear()
        driver.find_element_by_id('username').send_keys(self.name)
        driver.find_element_by_id("password").click()
        driver.find_element_by_id('password').clear()
        driver.find_element_by_id("password").send_keys(self.posswd)
        driver.find_element_by_id("btn_login").click()#登录摁扭
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            driver.find_element_by_xpath("/html/body/div[5]/div[2]/div").click()#关闭弹窗
        except:
            pass
        driver.find_element_by_xpath('//div[@id="scrollList"]/dl[9]/dd[1]/a').click()
    
            
    def begin(self):
        hothouse = self.Lowerhouse()
        html = driver.page_source
        Num = re.findall(r'sj_number pa[\s\S]*?<span>([\s\S]*?)</span>',html)[0]
        if Num == 0:
            pass
        else:
            self.Release(Num,hothouse)#发布房源
        self.writeinfo(infos)
        self.writedays(infos)
    
    def writedays(self,infos):
        infos = self.showinfo()
        self.writeinfo(infos)
        L = []
        for info in infos:
            if int(item["累计点击"]) >= 5:
                day = 30 - int(item["发布天数"])
            else:
                day = 15 - int(item["发布天数"])
            L.append(day)
        mintime = min(L)
        with open('/home/tarena/DYProject/Table/startdays.json', "ab") as f:
            text = json.dumps(dict(info),ensure_ascii=False)+'\n'
            f.write(text.encode('utf-8'))

            
        
            
    def Release(self,Num,hothouse):
        if len(hothouse) == int(Num):
            for firename in hothouse:
                self.doleju(firename)
        elif len(hothouse) < int(Num):
            infos = self.showinfo()#已发布房源信息
            Name = []#已发布房源信息
            house = []#用户房源列表
            for info in infos:
                L = ([info["xiaoqu"],info["mianji"]]])
                res = "_".join(L)
                Name.append(res)
            a = int(Num) - len(hothouse)
            with open("/home/tarena/myproject/myproject/DYProject/user/{}/house.txt".format(self.name),r) as f:
                text = f.readlines()
                for i in text:
                    if i not in Name and i not in hothouse:
                        house.append(i)
            Newlist = random.sample(house, a)
            for firename in Newlist:
                self.doleju(firename)
        else:
            pass

    
    
    

    def doleju(self,firename):
        resttime = 60#图片上传时间
        self.verify_firename(firename)
        driver.find_element_by_xpath('//div[@id="scrollList"]/dl[9]/dd[1]/a').click()#出售房源
        driver.find_element_by_xpath("/html/body/div[2]/div/div[1]/div/div/div[1]/a[1]/span").click()#发布房源
        driver.find_element_by_id("commName").click()
        driver.find_element_by_id('commName').clear()
        driver.find_element_by_id("commName").send_keys(info["xiaoqu"])
        time.sleep(10)
        driver.find_element_by_xpath('//div[@class="drop-box dw_352"]/ul/li/span').click()
        driver.find_element_by_id('tPrice').click()
        driver.find_element_by_id('tPrice').send_keys('520')
        driver.find_element_by_xpath('//div[@class="fill mb20"]/i').click()
        driver.find_element_by_id('s').send_keys(info["hushi"])
        driver.find_element_by_id('t').send_keys(info["ting"])
        driver.find_element_by_id('w').send_keys(info["wei"])
        driver.find_element_by_id('area').send_keys(info["mianji"])
        driver.find_element_by_id('selectFloor').click()
        driver.find_element_by_id("ptzz").click()
        driver.find_element_by_xpath('//select[@id="ptzz"]/option[1]').click()
        driver.find_element_by_id('zsqk').click()#装修情况
        driver.find_element_by_xpath('//select[@id="zsqk"]/option[6]').click()
        driver.find_element_by_id('sx').click()
        driver.find_element_by_xpath('//select[@id="sx"]/option[5]').click()
        driver.find_element_by_id('selectFloor').click()
        driver.find_element_by_xpath('//select[@id="selectFloor"]/option[4]').click()
        driver.find_element_by_id('totalfloor').send_keys(info["zonggao"])
        driver.find_element_by_id('buildyear').click()
        driver.find_element_by_id('buildyear').clear()
        driver.find_element_by_id('buildyear').send_keys(info["niandai"])
        driver.find_element_by_xpath('//select[@name="property_rights"]').click()
        driver.find_element_by_xpath('//select[@name="property_rights"]/option[2]').click()
        driver.find_element_by_id('tag_24').click()
        driver.find_element_by_id('tag_21').click()
        driver.find_element_by_id('tag_19').click()
        driver.find_element_by_id('htit').send_keys(info["title"])
        driver.find_element_by_xpath('//table[@class="ke-textarea-table"]/tbody/tr/td/iframe').send_keys(info["fangyuanxinxi"])
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/1.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/2.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/3.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/4.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/5.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/6.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/7.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/8.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_sn"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/9.jpg")
        time.sleep(resttime)
        driver.find_element_by_xpath('//div[@id="upload_fx"]/div[2]/input').send_keys("/home/tarena/myproject/myproject/picture/10.jpg")
        time.sleep(resttime)
        driver.find_element_by_id('xqdatabase').click()
        try:
            driver.find_element_by_xpath('//div[@class="choosePic"]/ul/li[1]/div[2]').click()
            driver.find_element_by_xpath('//div[@class="choosePic"]/ul/li[2]/div[2]').click()
            driver.find_element_by_xpath('//div[@class="choosePic"]/ul/li[3]/div[2]').click()
        except:
            pass

        driver.find_element_by_id('pubBtn').click()
        driver.find_element_by_xpath('//button[@class="ui-dialog-autofocus"').click()
        
                
       
            
    def verify_firename(self, firename):
        with open("/home/tarena/myproject/myproject/DYProject/houseinfo/houseinfo.json",'r', encoding='utf-8') as f:
            load_dict = json.load(f)
            if firename in load_dict.keys():
                info = load_dict[firename]
                return info
            else:
                Leju.AutoCopy(firename)
                info = load_dict[firename]
                return info
                
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
        
    def __Change(self):
        infos = self.showinfo()
        hothouse = []
        Number = []
        for info in infos:
            if int(item["累计点击"]) > 30:
                L = ([info["xiaoqu"],info["mianji"]])
                res = "_".join(L)
                hothouse.append(res)
            if int(item["发布天数"]) >= 15 and int(item["累计点击"]) < 5:
                Number.append(item["房源编号"])
                if int(item["发布天数"]) > 30:
                    Number.append(item["房源编号"])
        return hothouse, Number
        
    def Lowerhouse(self):#下架房源
        html = driver.page_source
        hothouse, Number = self.__Change()
        pages = re.findall(r'agent_page clearfix[\s\S]*?<span>([\s\S]*?)</span>',html)[0]
        page = re.findall(r'\d+')[1]#页数
        numb = re.findall(r'\d+')[0]#房源数量
        driver.find_element_by_xpath('//div[@class="choosePic"]/ul/li[last()]').click()#尾页
        for i in range(int(page)-1):        
            self.__low(Number)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")#下拉到底部
            time.sleep(0.5)
            driver.find_element_by_xpath('//div[@class="choosePic"]/ul/li[2]').click()#上一页
        return hothouse
            

    def __low(self,Number):
        for num in Number:
            try:
                driver.find_element_by_xpath('//div[@data-id="{}"]/div[1]/div[1]\div[1]\div[2]\i]'.format(num)).click()
            except:
                pass
        driver.find_element_by_xpath('//div[@class="sec-2 clearfix"]/div[1]/a[1]').click()
        driver.find_element_by_xpath('//button[@class="ui-dialog-autofocus"]').click()#确定
        
        
            
        
        