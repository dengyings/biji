# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:51:52 2019

@author: tarena
"""

import json

with open("/home/tarena/DYProject/text/ajk.txt", "r") as f:
    text = f.read()
    print('rideOK')
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
#栋，层，单元，先点击后查找\\div[@class="auto-wrap"]/ul[1]/li[1]/span/text()
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
#hushi = re.findall(r'',text)[0]

#print(xiaoqu,hushi,ting,fangwuleixing,zhuangxiu,chaoxiang,loucheng,zonggao,mianji,nianxian)
#print(dianti,hushi,weiyi,yishou,jiage,title)
#print(fangyuanxinxi)
#print(yezhuxintai)
#print(fuwujieshao)
print(info)