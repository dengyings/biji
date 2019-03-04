from multiprocessing import  Pool
import  os , time, random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor, as_completed, wait
import logging
import asyncio
import functools
from pandas import json
import requests
from lxml import etree
import pymysql
from gevent import monkey
monkey.patch_all()
from gevent.pool import Pool
from queue import Queue


def run_time(func):
    '''运行时间装饰器'''
    def call_fun(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print('程序用时：%s秒' % int(end_time - start_time))
    return call_fun

@run_time
def Mymultiprocessing():
    '''进程池'''
    p = Pool(4)
    for i in range(10):
        p.apply_async(run_tack, args=(i,))
    #p.map(run_tack,[i for i in range(20)])
    p.close()
    p.join()


def run_tack(name=1):
    for i in range(3):
        logger.info('run', name)
        time.sleep(2)

@run_time
def Mythreding():
    '''线程池'''
    with ThreadPoolExecutor(4) as pool:
        for pair in range(10):
            pool.submit(run_tack, pair)
        #pool.map(run_tack, [i for i in range(10)])

@run_time
def Mygevent():
    '''协程'''
    pool = Pool(4)
    pool.map(run_tack, [i for i in range(10)])

def downloadImg(imgUrl,num_retries=10):
    '''下载单张图片'''
    s = requests.session()
    if num_retries == 0:
        logger.error(imgUrl)
        return None
    ImgName = imgUrl.split('_')[-1]
    url = imgUrl.split('_')[:-1]
    Imgurl = '_'.join(url)
    r = s.get(Imgurl)
    if r.status_code == 200:
        with open('./image/{}.png'.format(ImgName), 'wb') as f:
            f.write(r.content)
    else:
        logger.error("server erroe")
        logger.warning(imgUrl)
        time.sleep(random.randint(1,2))
        downloadImg(imgUrl,num_retries-1)
    return None

def get_logger():
    """
    创建日志实例
    """
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    logger = logging.getLogger("monitor")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = get_logger()

@staticmethod
def insert_into_sql():
    """
    插入数据到数据库
    create table jobpost(
        j_salary float(3, 1),
        j_locate text,
        j_post text
        );
        """
    conn = pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        passwd="0303",
        db="chenx",
        charset="utf8",
    )
    cur = conn.cursor()
    with open(os.path.join("data", "post_salary.csv"), "r", encoding="utf-8") as f:
        f_csv = csv.reader(f)
        sql = "insert into jobpost(j_salary, j_locate, j_post) values(%s, %s, %s)"
        for row in f_csv:
            value = (row[0], row[1], row[2])
            try:
                cur.execute(sql, value)
                conn.commit()
            except Exception as e:
                logger.error(e)
    cur.close()

def insert_into_mogp():
    import pymongo
    # 配置数据库信息
    MONGO_URl = 'localhost'
    MONGO_DB = 'taobao'  # 数据库名
    MONGO_TABLE = 'iphonex_url'  # 表名
    # 连接数据库
    client = pymongo.MongoClient(MONGO_URl)
    db = client[MONGO_DB]
    # 存入数据库
    def save_url_to_Mongo(result):
        try:
            if db[MONGO_TABLE].insert(result):
                print('存储到MongoDB成功', result)
        except Exception:
            print('存储到MongoDb失败', result)

desc_url_queue = Queue()

def job_spider():
    company = []
    urls = ['{}'.format(p) for p in range(1, 16)]
    for url in urls:
        logger.info("爬取第 {} 页".format(urls.index(url) + 1))
        html = requests.get(url, headers='').content.decode("gbk")
        html = etree.HTML(html).xpath('/html/body/div/ul/li/a')
        result = etree.tostring(html).decode('utf-8')#转换为字符串
        for b in html:
            try:
                href = b.xpath(".//body")
                item = {"href": href,}
                desc_url_queue.put(href)  # 岗位详情链接加入队列
                company.append(item)
            except Exception:
                pass
    # 打印队列长度,即多少条岗位详情 url
    logger.info("队列长度为 {} ".format(desc_url_queue.qsize()))

def post_require():
    ua = get_random_UA()
    HEADERS = {"User-Agent": ua}
    ip = get_random_ip()#proxies={'https':'101.236.54.97:8866'}
    proxies = {'https': ip}
    count = 1
    while True:
        # 从队列中取 url
        url = desc_url_queue.get()
        resp = requests.get(url, headers=HEADERS,proxies=proxies,timeout=10)
        if resp.status_code == 200:
            logger.info("爬取第 {} 条岗位详情".format(self.count))
            html = resp.content.decode("gbk")
            desc_url_queue.task_done()
            count += 1
        else:
            desc_url_queue.put(url)
            continue
        try:
            bs = ''
            s = bs.replace("微信", "").replace("分享", "").replace("邮件", "").replace(
                "\t", ""
            ).strip()
            with open("aje.json", "ab") as f:
                text = json.dumps(dict(), ensure_ascii=False) + '\n'
                f.write(text.encode('utf-8'))

            data_path = os.path.join(os.getcwd(), "data")
            if os.path.isdir(data_path):
                pass
            os.mkdir('data')

            with open(
                    os.path.join(data_path, "post_pre_desc_counter.csv"), "r", encoding="utf-8"
            ) as f:
                f.write(s)
        except Exception as e:
            logger.error(e)
            logger.warning(url)

def get_random_ip():
    proxy_list = []
    proxy_ip = random.choice(proxy_list)
    print(proxy_ip)
    return proxy_ip

def get_random_UA():
    UA_list = ["Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",]
    UA = random.choice(UA_list)
    print(UA)
    return UA

def text_ip_list():
    HEADERS = {"User-Agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",}
    ip_list=["47.95.9.128:8118",
    "218.76.253.201:61310",
    "139.196.125.96:8088",
    "61.184.109.33:50371",
    "47.97.8.244:3128",
    "221.224.136.211:35101",
    "60.191.201.38:45461",
    "101.76.209.69:1080",
    "111.59.114.46:8118",
    "59.172.27.6:38380",
    "222.171.251.43:40149",
    "59.45.168.235:39856",
    "219.238.186.188:8118",
    "112.86.0.106:53281",
    "180.164.24.165:53281",
    "101.132.174.110:8088",
    "116.228.53.234:43792",
    "47.99.61.236:9999",
    "117.90.252.26:9000",
    "106.2.1.5:3128",
    "125.46.0.62:53281",
    "113.121.242.49:808",
    "223.85.196.75:9797",
    "182.18.13.149:53281",
    "112.115.57.20:3128",
    "113.200.214.164:9999",
    "115.233.210.218:808",
    "221.178.176.25:3128",
    "124.152.32.140:53281",
    "112.12.37.196:53281",
    "58.249.55.222:9797",
    "112.95.206.176:8888",
    "119.57.108.109:53281",
    "119.57.108.53:53281",
    "36.45.160.132:8118",
    "115.55.168.57:8080",
    "221.7.255.168:80",
    "59.42.42.93:8118",
    "219.234.5.128:3128",
    "42.55.94.226:1133",
    "175.165.128.214:1133",
    "61.176.223.7:58822",
    "113.70.78.103:8118",
    "106.15.42.179:33543",
    "175.175.219.179:1133",
    "144.123.94.73:8118",
    "218.73.110.254:8118",
    "121.21.208.112:80",
    "58.118.228.7:1080",
    "180.104.107.46:45700",
    "58.215.140.6:8080",
    "123.7.61.8:53281",
    "61.135.155.82:443",
    "183.163.41.33:808",
    "183.3.150.210:41258",
    "58.218.201.188:58093",
    "116.7.176.75:8118",
    "14.118.135.10:808",
    "27.155.83.126:8081",
    "91.217.42.2:8080",
    "101.236.55.145:8866",
    "61.178.149.237:59042",
    "222.74.237.246:808",
    "183.63.123.3:56489",
    "118.187.58.34:53281",
    "121.33.220.158:808",
    "61.183.233.6:54896",
    "139.159.7.150:52908",
    "118.24.156.214:8118",
    "139.199.38.182:8118",
    "101.132.142.124:8080",
    "118.190.94.224:9001",
    "121.10.71.82:8118",
    "58.53.128.83:3128",
    "121.69.13.242:53281",
    "119.98.171.141:1080",
    "14.115.107.36:9797",
    "123.58.10.9:8080",
    "123.58.10.89:8080",
    "183.129.207.73:14823",
    "123.58.10.26:8080",
    "162.105.87.211:8118",
    "124.250.70.76:8123",
    "223.199.156.26:9797",
    "123.58.10.92:8080",
    "123.58.10.87:8080",
    "123.58.10.79:8080",
    "182.131.18.203:80",
    "123.58.10.90:8080",
    "123.58.10.12:8080",
    "123.58.10.41:8080",
    "123.58.10.96:8080",
    "123.58.10.72:8080",
    "123.58.10.7:8080",
    "123.58.10.82:8080",
    "163.125.67.242:9797",
    "123.58.10.60:8080",
    "123.58.10.54:8080",
    "123.58.10.8:8080",
    "123.58.10.39:8080",
    "123.58.10.113:8080",
    "123.58.10.94:8080",
    "123.58.10.123:8080",
    "123.58.10.51:8080",
    "123.58.10.104:8080",
    "114.217.10.214:3128",
    "124.235.135.210:80",
    "14.115.107.166:808",
    "101.132.122.230:3128",
    "110.52.8.198:53281",
    "121.40.27.50:8118",
    "123.58.10.99:8080",
    "123.58.10.80:8080",
    "123.58.10.95:8080",
    "123.58.10.120:8080",
    "123.58.10.101:8080",
    "123.58.10.38:8080",
    "123.58.10.127:8080",
    "123.58.10.117:8080",
    "123.58.10.121:8080",
    "203.86.26.9:3128",
    "123.58.10.91:8080",
    "123.58.10.22:8080",
    "123.58.10.11:8080",
    "123.58.10.77:8080",
    "58.247.127.145:53281",
    "123.58.10.84:8080",
    "210.77.23.114:1080",
    "60.191.134.164:9999",
    "182.88.197.23:9797",
    "123.58.10.119:8080",
    "114.116.10.21:3128",
    "59.44.247.194:9797",
    "119.3.13.36:8010",
    "113.108.192.74:80",
    "112.95.16.195:8088",
    "221.7.255.168:8080",
    "218.241.234.48:8080",
    "112.250.109.173:53281",
    "183.136.165.7:8080",
    "101.251.216.103:8080",
    "183.136.165.11:8080",
    "221.205.88.4:9797",
    "101.37.79.125:3128",
    "123.161.18.117:9797",
    "124.237.83.14:53281",
    "219.246.90.204:3128",
    "115.28.209.249:3128",
    "119.122.213.38:9000",
    "140.143.170.222:8118",
    "59.78.35.129:1080",
    "59.34.2.92:3128",
    "123.138.89.132:9999",
    "203.93.209.163:53281",
    "220.160.163.49:1080",
    "58.58.48.138:53281",
    "211.83.77.51:3128",
    "36.110.14.186:3128",
    "61.145.182.27:53281",
    "180.76.111.69:3128",
    "113.200.56.13:8010",
    "218.60.8.99:3129",
    "60.205.204.160:3128",
    "27.191.234.69:9999",
    "218.60.8.98:3129",
    "221.6.201.18:9999",
    "218.60.8.83:3129",
    "36.110.101.104:8088",
    "118.31.79.226:3128",
    "101.236.32.119:3128",
    "202.112.237.102:3128",
    "111.230.183.90:8000",
    "218.14.115.211:3128",
    "222.74.61.98:53281",
    "58.38.95.223:9000",
    "101.76.211.214:1080",
    "221.7.255.167:8080",
    "123.172.82.117:53281",
    "203.130.46.108:9090",
    "106.14.197.219:8118"]
    url = 'https://www.sina.com.cn'
    for ip in ip_list:
        try:
            proxy_host = "https://" + ip
            proxy_temp = {"https": proxy_host}
            res = requests.get(url, headers=HEADERS,proxies=proxy_temp,timeout=10)
            if res.status_code == 200:
                print(ip)
        except Exception as e:
            continue

def ancynio_requests():
    urllist=[]
    for i in range(1, 1001):
        urllist.append('http://some.m3u8.play.list/{}.ts'.format(i))
    #  创建一个事件loop
    loop = asyncio.get_event_loop()
    tasks = [crawler(url) for url in urllist]
    # 将协程加入到事件循环loop
    loop.run_untill_complete(asyncio.wait(tasks))
    loop.close()
    # URL响应文件的后续处理
    insert_into_sql()

async def crawler(url):
    print('Start crawling:', url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}

    # 利用BaseEventLoop.run_in_executor()可以在coroutine中执行第三方的命令，例如requests.get()
    # 第三方命令的参数与关键字利用functools.partial传入
    future = asyncio.get_event_loop().run_in_executor(None, functools.partial(requests.get, url, headers=headers))

    response = await future

    print('Response received:', url)
    # 处理获取到的URL响应（在这个例子中我直接将他们保存到硬盘）
    with open(os.path.join('.', 'tmp', url.split('/')[-1]), 'wb') as output:
        output.write(response.content)

def main():
    print('1')


if __name__ == "__main__":
    main()