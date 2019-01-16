# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 18:45:03 2019

@author: tarena
"""

import re
import requests
import random
import time
import json

headers = {
            "User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 MicroMessenger/6.5.2.501 NetType/WIFI WindowsWechat QBCore/3.43.901.400 QQBrowser/9.0.2524.400",
            }
def getHTMLText():
    for i in range(13,3553):
        time.sleep(10)
        url = "https://www.xicidaili.com/nn/{}".format(i)
        get_ip_list(url)
        print("正在爬第",i,"页!")
        time.sleep(random.randint(1,10))
    

def get_random_ip():
    proxy_list = ['https://47.95.9.128:8118', 'https://218.76.253.201:61310', 'https://47.97.8.244:3128', 'https://59.172.27.6:38380', 'https://180.164.24.165:53281', 'https://47.99.61.236:9999', 'https://125.46.0.62:53281', 'https://113.200.214.164:9999', 'https://115.233.210.218:808', 'https://221.178.176.25:3128', 'https://124.152.32.140:53281', 'https://112.12.37.196:53281', 'https://58.249.55.222:9797', 'https://106.15.42.179:33543', 'https://27.155.83.126:8081', 'https://183.63.123.3:56489', 'https://118.187.58.34:53281', 'https://118.190.94.224:9001', 'https://121.69.13.242:53281', 'https://162.105.87.211:8118', 'https://124.250.70.76:8123', 'https://58.247.127.145:53281', 'https://60.191.134.164:9999', 'https://114.116.10.21:3128', 'https://59.44.247.194:9797', 'https://101.37.79.125:3128', 'https://219.246.90.204:3128', 'https://58.58.48.138:53281', 'https://61.145.182.27:53281', 'https://180.76.111.69:3128', 'https://113.200.56.13:8010', 'https://218.60.8.99:3129', 'https://60.205.204.160:3128', 'https://221.6.201.18:9999', 'https://218.60.8.83:3129', 'https://111.230.183.90:8000', 'https://203.130.46.108:9090', 'https://106.14.197.219:8118']
    proxy_ip = random.choice(proxy_list)
    proxies = {'https': proxy_ip}
    return proxies


def get_ip_list(url):
    proxy_temp=get_random_ip()
    print(proxy_temp)
    try:    
        text = requests.get(url,headers=headers,proxies=proxy_temp).text
        ips = re.findall('<tr([\s\S]*?)</tr>',text)
        List = []
        for i in ips:
            ip = re.findall('<td>(.*?)</td>',i)
            if len(ip) > 1:
                thisip = ip[2].lower() + "://" + ip[0] + ":" + ip[1]
                List.append(thisip)
        print(List)
        with open('/home/tarena/myproject/myproject/DYProject/Ips.json', "ab") as f:
            for ip in List:
                pro = ip.split(':')
                proxy_temp = {pro[0]: ip}
                texts = json.dumps(dict(proxy_temp),ensure_ascii=False)+'\n'
                f.write(texts.encode('utf-8'))
            print("写入成功")
    except Exception as e:
        print(e)
        pass


if __name__ == '__main__':
    starttime = time.time()
    getHTMLText()
    endtime = time.time()
    print(endtime-starttime)