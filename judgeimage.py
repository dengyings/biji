# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:07:46 2019

@author: tarena
"""

import cv2
import copy
import numpy as np
import os
import re
from urllib.request import urlretrieve
import urllib.request as urlrequest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor, as_completed, wait

L=[{"mianji": "90平方米", "houseurl": "https://guangzhou.anjuke.com/prop/view/A1548441010?spread=commprop_p&amp;position=4", "title": "仅95万首付 学 位房 中高层精装三房 精装 诚售 随时看", "imageurl": "https://pic1.ajkimg.com/display/hj/e5743c1b6a50961c8e8822d88ccd34c6/240x180c.jpg", "xiaoqu": "\n\t\t\t锦绣香江山水华府二期\t\t\t\t\t", "huxing": "3室2厅", "anxuan": 0, "duotu": 1},
{"mianji": "91平方米", "houseurl": "https://guangzhou.anjuke.com/prop/view/A1503707738?spread=commprop_p&amp;position=5", "title": "锦绣香江山水华府二期 精装三房 南北对流 过2年 随时看房", "imageurl": "https://pic1.ajkimg.com/display/hj/e0aa6583dcb4ba6b2c70f47f118b1bb9/240x180c.jpg", "xiaoqu": "\n\t\t\t锦绣香江山水华府二期\t\t\t\t\t", "huxing": "3室2厅", "anxuan": 0, "duotu": 1},
{"mianji": "91平方米", "houseurl": "https://guangzhou.anjuke.com/prop/view/A1551162793?spread=commprop_p&amp;position=6", "title": "高楼层望园林 南北通透 高使用率 装修保养好 户型方正", "imageurl": "https://pic1.ajkimg.com/display/hj/e262860bafeed6cd61afdba4b3aa7a6c/240x180c.jpg", "xiaoqu": "\n\t\t\t锦绣香江山水华府二期\t\t\t\t\t", "huxing": "4室2厅", "anxuan": 0, "duotu": 1},
{"mianji": "87平方米", "houseurl": "https://guangzhou.anjuke.com/prop/view/A1549324173?spread=commprop_p&amp;position=7", "title": "豪华装修 实用率高 环境优雅 比市场价便宜", "imageurl": "https://pic1.ajkimg.com/display/hj/f437ea2ead20731e7ad4c44a6afff8ab/240x180c.jpg", "xiaoqu": "\n\t\t\t锦绣香江山水华府二期\t\t\t\t\t", "huxing": "2室2厅", "anxuan": 0, "duotu": 1},
{"mianji": "639平方米", "houseurl": "https://guangzhou.anjuke.com/prop/view/A1532704645?spread=commprop_p&amp;position=8", "title": "番禺华南锦绣香江一楼复式带花园，毛坯，可省6.6的豪宅税", "imageurl": "https://pic1.ajkimg.com/display/hj/aa9ebb73e1a96918b6d2197f5e8ce66b/240x180c.jpg", "xiaoqu": "\n\t\t\t锦绣香江山水华府二期\t\t\t\t\t", "huxing": "6室2厅", "anxuan": 0, "duotu": 1},
{"mianji": "381平方米", "houseurl": "https://guangzhou.anjuke.com/prop/view/A1561745713?spread=commprop_p&amp;position=9", "title": "二期 喽王大平层 入户80方空中大花园 360度无敌景观", "imageurl": "https://pic1.ajkimg.com/display/hj/e724c283829a8156983e5168f9299d4e/240x180c.jpg", "xiaoqu": "\n\t\t\t锦绣香江山水华府二期\t\t\t\t\t", "huxing": "5室2厅", "anxuan": 0, "duotu": 1},
{"mianji": "700平方米", "houseurl": "https://guangzhou.anjuke.com/prop/view/A1556074475?spread=commprop_p&amp;position=10", "title": "内部转名一楼带400方大花园640方中空复式 提前约看", "imageurl": "https://pic1.ajkimg.com/display/hj/8fbe6478cbbbb71b079765c1776882bc/240x180c.jpg", "xiaoqu": "\n\t\t\t锦绣香江山水华府\t\t\t\t\t", "huxing": "6室2厅", "anxuan": 0, "duotu": 1},
{"mianji": "266平方米", "houseurl": "https://guangzhou.anjuke.com/prop/view/A1563730602?spread=commprop_p&amp;position=11", "title": "锦绣香江华府二期精装修5房60方大气阳台送50方入户花园", "imageurl": "https://pic1.ajkimg.com/display/hj/dec0cee00ff99f8d3da28a02d53d5811/240x180c.jpg", "xiaoqu": "\n\t\t\t锦绣香江山水华府\t\t\t\t\t", "huxing": "4室2厅", "anxuan": 0, "duotu": 1}]

def getdict():
    '''过滤url建立文件夹'''
    info = {}
    for i in L:
        if i["duotu"] == 1:            
            mianji = re.findall(r'\d+',i['mianji'])[0]
            xiaoqu = i["xiaoqu"].strip()
            imageurl = i["imageurl"]
            houseurl = i["houseurl"].split('?')[0].split('/')[-1]
            huxing = i["huxing"]
            anxuan = i["anxuan"]
            imagename = xiaoqu + "_" + mianji +"_" + huxing +"_" + houseurl + "_" + str(anxuan)
            info[imageurl] = imagename
            
            xiaoqupath = '/home/tarena/myproject/image/{}'.format(xiaoqu)
            if os.path.exists(xiaoqupath):
                pass
            else:
                os.mkdir(xiaoqupath)
                
            huxingpath = '/home/tarena/myproject/image/{x}/{y}'.format(x=xiaoqu, y=huxing)
            if os.path.exists(huxingpath):
                pass
            else:
                os.mkdir(huxingpath) 
        else:
            continue
    return info

def imagedownload():
    '''使用线程池下载图片'''
    infos = getdict()
    New = []
    for k,v in infos.items():
        url = k + '_' + v
        New.append(url)      
    #print(v)
    with ThreadPoolExecutor(5) as pool:
        pool.map(downloadImg,New)
        
       # print(k)
def downloadImg(imgUrl):
    '''下载单张图片'''
    ImgUrl = imgUrl.split('_')[0]
    xiaoqu = imgUrl.split('_')[1]
    huxing = imgUrl.split('_')[3]
    Name = imgUrl.split('_')[1:]
    imgName = '_'.join(Name)
    urlretrieve(ImgUrl, './image/{x}/{y}/{z}.jpg'.format(x=xiaoqu, y=huxing,z=imgName))

 
def randomdownloadImg(imgUrl):
    '''代理下载图片'''
    proxy = urlrequest.ProxyHandler({'https': '47.91.78.201:3128'})
    opener = urlrequest.build_opener(proxy)
    
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30')]
    urlrequest.install_opener(opener)
    
    ImgUrl = imgUrl.split('_')[0]
    Name = imgUrl.split('_')[1:]
    imgName = ''.join(Name)
    urlrequest.urlretrieve(ImgUrl, './image/{}'.format(imgName))



def getimgpath():
    '''拼接每个文件夹下的图片路径'''
    path = '/home/tarena/myproject/image'
    xiaoqulist = os.listdir(path)
    for xiaoqu in xiaoqulist:
        xiaoqupath = path + '/{}'.format(xiaoqu) 
        huxinglist = os.listdir(xiaoqupath)
        for huxing in huxinglist:
            img = []
            huxingpath = xiaoqupath + '/{}'.format(huxing) 
            inglist = os.listdir(huxingpath)
            for i in inglist:
                imgpath = huxingpath + '/{}'.format(i)
                img.append(imgpath)          
            juege(img)



def change():
    path = '/home/tarena/myproject/image'
    xiaoqulist = os.listdir(path)
    for xiaoqu in xiaoqulist:
        xiaoqupath = path + '/{}'.format(xiaoqu) 
        huxinglist = os.listdir(xiaoqupath)
        for huxing in huxinglist:
            Imgs = []
            huxingpath = xiaoqupath + '/{}'.format(huxing) 
            inglist = os.listdir(huxingpath)
            for i in inglist:
                Imgs.append(i)
            changeimg(Imgs)
    
def changeimg(Imgs):
    name = '0_1_0.jpg'
    for i in Imgs:
        if int(i[0]) > int(name[0]):
            name = i
        elif int(i[0]) == int(name[0]):
            if int(i[-5]) > int(name[-5]):
                name = i
        else:
            continue
    print(name)
        

def juege(img):
    '''循环判断图片，并改变名字'''
    t = 0
    for x in img:
        i = 0
        for y in img:
            if judgeimage(x,y):                
                i += 1
            else:
                continue
        oldname = x
        neS = x.split('/')
        neS[7] = str(i) + '_' + x.split('/')[7]                
        newname = '/'.join(neS)
        #print(newname)
        os.rename(oldname,newname)
        print(x)
        img[t] = newname
        t += 1

                                    
    

def judgeimage(paht1,path2):
    '''判断两张图片的相似行，相似返回ＴＲＵＥ'''    
    file1=cv2.imread(paht1)
    file2=cv2.imread(path2)
    stdimg = np.float32(file1)
    ocimg = np.float32(file2)
    stdimg = np.ndarray.flatten(stdimg)
    ocimg = np.ndarray.flatten(ocimg)
    imgocr = np.corrcoef(stdimg, ocimg)
    a = imgocr[0, 1]
    if a > 0.9:
        #print("一样")
        return True
    else:
       # print("不一样")
        return False


change()
#getimgpath() 
#imagedownload()
 
 

