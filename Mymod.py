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

def text_ip_list(ip_list):
    for ip in ip_list:
        try:
          proxy_host = "https://" + ip
          proxy_temp = {"https": proxy_host}
          res = urllib.urlopen(url, proxies=proxy_temp).read()
        except Exception as e:
          ip_list.remove(ip)
          continue
    return ip_list

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
    print(os.listdir())

if __name__ == "__main__":
    main()