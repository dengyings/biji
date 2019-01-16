# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 19:49:59 2018

@author: tarena
"""
import re
import requests


url = 'http://www.xicidaili.com/nn/'
#getcomment_url = "https://mp.weixin.qq.com/mp/appmsg_comment?action=getcomment&__biz={}&idx={}&comment_id={}&limit=100"
#s = requests.session()
headers = {
            "User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 MicroMessenger/6.5.2.501 NetType/WIFI WindowsWechat QBCore/3.43.901.400 QQBrowser/9.0.2524.400",
            #"Cookie":"rewardsn=; wxuin=1723124627; devicetype=Windows7; version=62060619; lang=zh_CN; pass_ticket=bSSOOBYMttQ3rethJqnlrF4ZhCy3grtSZOYYfVEf8qkYUQqFzi9tnGr07GGF+Rgm; wap_sid2=CJOX07UGElxSLXFSX0tZSjJueE8tRjQ2N0w0VFVsa2MzdVdKUFNCWEJtSmZta2Z6ZFZHZk5ELUpvZlBTa3Vxc0lmZWxHVlNwMWl2WGNWUzh2UHhaSWxpZjRqOWJCdHdEQUFBfjDRt+7gBTgNQAE=; wxtokenkey=777"
        }
def get_ip_list():
    ip_list = []
    with open("/home/tarena/myproject/myproject/IPProxyPool-master/data/proxys.txt", "r") as f:
        Iproxies = f.readlines()
        for i in Iproxies:
            ip_list.append(i[:-1])
    print(ip_list)
    return ip_list

def get():
    ip_list = get_ip_list()
    #print(ip_list)
    for ip in ip_list:
        try:
            proxy_temp = {"https": ip}
            print(proxy_temp)
            res = requests.get("https://www.xicidaili.com/nn/", proxies=proxy_temp).text
            #res = requests.get("https://www.xicidaili.com/nn").text
            print(type(res))
            with open("/home/tarena/myproject/myproject/IPProxyPool-master/data/ipproxys.txt", "a") as f:
                text = ip +'\n'
                f.write(text)
        except Exception as e:
            ip_list.remove(ip)
            print(e)
            continue

    
    
    print(len(ip_list))
    
    
if __name__ == '__main__':
    get()

