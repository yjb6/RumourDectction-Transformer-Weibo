import requests
import json
import re
import time
from datetime import datetime
import os
import random
# coding:utf-8
url='https://m.weibo.cn/api/comments/show?id='+'&page={}'

# headers = {
#     'cookie':'WEIBOCN_FROM=1110006030; SUB=_2A25PM579DeRhGeBJ41MV-SfNyz-IHXVs3yK1rDV6PUJbkdCOLRbYkW1NQzMnYDYumbBqsqsBsNNe45YNpqxlx4ir; _T_WM=34384633378; MLOGIN=1; XSRF-TOKEN=b97add; loginScene=102003; M_WEIBOCN_PARAMS=luicode%3D10000011%26lfid%3D1076031499104401%26uicode%3D20000061%26fid%3D4749488709439561%26oid%3D4749488709439561',
#    # 'cookie': 'WEIBOCN_FROM=1110006030; SUB=_2A25PMfZTDeRhGeBJ41MV-SfNyz-IHXVs3ZobrDV6PUJbkdCOLRjxkW1NQzMnYGkAnDXDyceAuLx63HB1sfL4duJ7; MLOGIN=1; _T_WM=85563697150; M_WEIBOCN_PARAMS=oid%3D4748092441891315%26luicode%3D20000061%26lfid%3D4748092441891315; XSRF-TOKEN=90adc4',
#     'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
#  }
headers={
    'Accept': 'application/json, text/plain, */*',
    'MWeibo-Pwa': '1',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': "Windows",
    'X-Requested-With': 'XMLHttpRequest',
    'X-XSRF-TOKEN': 'b86196',
    'cookie':'WEIBOCN_FROM=1110006030; loginScene=102003; SUB=_2A25PTSkhDeRhGeBJ41MV-SfNyz-IHXVssbdprDV6PUJbkdAKLRf8kW1NQzMnYEoS9f7sXjcsYru7pmHr3xXEi9Np; MLOGIN=1; _T_WM=20411521923; XSRF-TOKEN=b86196; M_WEIBOCN_PARAMS=oid%3D4713907509792628%26luicode%3D20000061%26lfid%3D4713907509792628%26uicode%3D20000061%26fid%3D4713907509792628; mweibo_short_token=6e0fd4523b',
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'
}

def getsource(wid):
    print("爬取原微博")
    url='https://m.weibo.cn/statuses/extend?id='+str(wid)
    response=requests.get(url=url,headers=headers,timeout=5)
    data=json.loads(response.text)['data']
    # text=data['data']['longTextContent']
    # comments_count=data['data']['comments_count']
    # reposts_count=data['data']['reposts_count']
    text=data['longTextContent']
    pat1 = r"[^((\u4e00-\u9fa5)|(？|，|。|\s|@))]+"
    text= re.sub(pat1, '', text)
    data['longTextContent']=text
    return data
def getlikelist(wid):
    print("爬取likelist")
    starturl = r'https://m.weibo.cn/api/attitudes/show?id={}&page=1'.format(str(wid))
    response = requests.get(url=starturl, headers=headers, timeout=10)
    data = json.loads(response.text)
    flag = data['ok']
    if (flag == 0):
        print("{}的点赞列表读取失败".format(str(wid)))
        return
    datas = []
    data = data['data']
    datas.extend(data['data'])
    page = 2
    flag=1
    while(flag):
        url = r'https://m.weibo.cn/api/attitudes/show?id={0}&page={1}'.format(str(wid), str(page))
        response = requests.get(url=url, headers=headers, timeout=10)
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            if (re.search(r'请求过于频繁', response.text)):
                print("请求过于频繁！")
                print('page：', page, "url:", url)
                break
            elif (re.search((r'出错了'), response.text)):
                print("微博出错,应为系统屏蔽")
                print('page：', page, "url:", url)
                break
            else:
                print(response.text)
                print(page)
                print(url)
                raise Exception("未知的错误！")
        flag = data['ok']
        if (flag == 0):
            print("{0}的评论第{1}页评论读取失败".format(str(wid), str(page)))
            print(url)
            break
        elif(flag==-100):
            raise Exception("没有登陆")
        data = data['data']
        if(data['data']==None):
            break
        page+=1
        datas.extend(data['data'])
        if(len(datas)>100):
            break
        time.sleep(random.randint(1,9))
    likelist = []
    for item in datas:
        com = {}
        com['time'] =item['created_at']
        com['id'] = item['id']
        com['source']=item['source']
        com['username'] = item['user']['screen_name']
        com['followers_count']=item['user']['followers_count']
        likelist.append(com)
    return likelist
# getlikelist(4751279786297162)
def getrepost(wid):
    print("爬取转载信息")
    starturl = 'https://m.weibo.cn/api/statuses/repostTimeline?id={0}&page=1'.format(str(wid))
    response = requests.get(url=starturl, headers=headers, timeout=10)
    data = json.loads(response.text)
    flag = data['ok']
    if (flag == 0):
        print("{}的转发读取失败".format(str(wid)))
        return
    datas = []
    data = data['data']
    datas.extend(data['data'])
    max = data['max']
    page = 2
    for page in range(2,max+1):
        url = 'https://m.weibo.cn/api/statuses/repostTimeline?id={0}&page={1}'.format(str(wid),str(page))
        response = requests.get(url=url, headers=headers, timeout=10)
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            if (re.search(r'请求过于频繁', response.text)):
                print("请求过于频繁！")
                print('page：', page, "url:", url)
                break
            elif (re.search((r'出错了'), response.text)):
                print("微博出错,应为系统屏蔽")
                print('page：', page, "url:", url)
                break
            else:
                print(response.text)
                print(page)
                print(url)
                raise Exception("未知的错误！")
        flag = data['ok']
        if (flag == 0):
            print("{0}的评论第{1}页评论读取失败".format(str(wid), str(page)))
            print(url)
            break
        data = data['data']
        datas.extend(data['data'])
        time.sleep(random.randint(1,9))
    reposts = []
    for item in datas:
        com = {}
        com['time'] = datetime.strptime(item['created_at'], "%a %b %d %H:%M:%S +0800 %Y")
        com['id'] = item['id']
        com['gender'] = item['user']['gender']
        com['username'] = item['user']['screen_name']
        reposts.append(com)
    return reposts
# getrepost(4750771667861902)
def getcom2com(cid):
    '''此函数为获取评论的评论的信息'''
    starturl='https://m.weibo.cn/comments/hotFlowChild?cid={}&max_id=0&max_id_type=0'.format(str(cid))
    response = requests.get(url=starturl, headers=headers, timeout=5)
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        print("相关子评论已被系统屏蔽")
        return
        # raise Exception("出错了！")
    flag=data['ok']
    if(flag==0):
        print("{}的子评论读取失败".format(str(cid)))
        return
    datas = []
    max_id = data['max_id']
    max_id_type=data['max_id_type']
    data=data['data']
    datas.extend(data)
    page=2
    time.sleep(random.randint(5,9))
    while (max_id != 0):
        url = 'https://m.weibo.cn/comments/hotFlowChild?cid={0}&max_id={1}&max_id_type={2}'.format(str(cid),str(max_id),str(max_id_type))
        # print(url)
        response = requests.get(url=url, headers=headers, timeout=5)
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            if(re.search(r'请求过于频繁',response.text)):
                print("请求过于频繁！")
                print('page：',page,"url:",url)
                break
            elif(re.search((r'出错了'),response.text)):
                print("微博出错,应为系统屏蔽")
                print('page：', page, "url:", url)
                break
            else:
                print(response.text)
                print(page)
                print(url)
                raise Exception("未知的错误！")

        flag=data['ok']
        if (flag == 0):
            print("{0}的第{1}页子评论读取失败".format(str(cid),str(page)))
            break
        max_id = data['max_id']
        max_id_type = data['max_id_type']
        data=data['data']
        datas.extend(data)
        page+=1
        time.sleep(random.randint(5,9))
        # print(data)
        # break
    # print(len(datas))
    comments = []
    for item in datas:
        com = {}
        text = item['text']
        '''正则表达式清洗text'''
        pattren='<span class=[\s\S]*<img alt=|src=[\s\S]*</span>|<a href=[\s\S]*@|</a>'
        pat1=r"[^(\u4e00-\u9fa5)]+"
        com['text'] = re.sub(pat1, '', text)
        com['time'] = datetime.strptime(item['created_at'], "%a %b %d %H:%M:%S +0800 %Y")
        com['id'] = item['id']
        com['rootid'] = item['rootid']
        com['like'] = item['like_count']
        com['gender'] = item['user']['gender']
        com['username'] = item['user']['screen_name']
        comments.append(com)

    return comments
def getcomments(wid):
    print("爬取评论")
    starturl="https://m.weibo.cn/comments/hotflow?id={0}&mid={0}&max_id_type=0".format(str(wid))
    response=requests.get(url=starturl,headers=headers,timeout=10)
    data=json.loads(response.text)
    flag=data['ok']
    if(flag==0):
        print("{}的评论读取失败".format(str(wid)))
        return
    datas=[]
    data=data['data']
    datas.extend(data['data'])
    max_id=data['max_id']
    max_id_type=data['max_id_type']
    page=2
    time.sleep(random.randint(5,9))
    while(max_id!=0):
        url='https://m.weibo.cn/comments/hotflow?id={0}&mid={0}&max_id={1}&max_id_type={2}'.format(str(wid),str(max_id),str(max_id_type))
        response = requests.get(url=url, headers=headers, timeout=10)
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            if(re.search(r'请求过于频繁',response.text)):
                print("请求过于频繁！")
                print('page：',page,"url:",url)
                break
            elif (re.search((r'出错了'), response.text)):
                print("微博出错,应为系统屏蔽")
                print('page：', page, "url:", url)
                break
            else:
                print(response.text)
                print(page)
                print(url)
                raise Exception("未知的错误！")
        flag=data['ok']
        if (flag == 0):
            print("{0}的评论第{1}页评论读取失败".format(str(wid),str(page)))
            print(url)
            break
        try:
            data=data['data']
        except:
            print(data)
            raise Exception("未知的错误！")
        max_id=data['max_id']
        max_id_type = data['max_id_type']
        datas.extend(data['data'])
        page+=1
        time.sleep(random.randint(5,9))
    comments=[]
    for item in datas:
        com={}
        text=item['text']
        '''正则表达式清洗text'''
        '<span class=[\s\S]*<img alt=[|] src=[\s\S]*</span>|<a href=[\s\S]*@|</a>'
        pattren='<span class=[\s\S]*<img alt=|src=[\s\S]*</span>|<a href=[\s\S]*@|</a>'
        pat1=r"[^(\u4e00-\u9fa5)]+"
        com['text']=re.sub(pat1,'',text)
        com['time']= datetime.strptime(item['created_at'], "%a %b %d %H:%M:%S +0800 %Y")
        com['id']=item['id']
        com['rootid']=item['rootid']
        com['like']=item['like_count']
        com['gender']=item['user']['gender']
        com['username']=item['user']['screen_name']
        comments.append(com)
        if(item['total_number']>0):
                child=getcom2com(item['id'])
                if (child):

                    comments.extend(child)
    return comments
# getsource('4748781737740015')
# getcomments('4748110616857173')
def getstructure(structure):
    ''' res[i,j] 0:i是j儿子 1:i是j爹 2:i在j前面 3:i在j后面 4:i=j'''
    res = [[-1 for i in range(len(structure) )] for i in range(len(structure) )]
    # print(res)
    for index, row in enumerate(res):
        for j, v in enumerate(row):
            if (index< j):
                row[j] = 2
            else:
                row[j] = 3
        row[index] = 4
        res[index] = row
    for index, father in enumerate(structure):
        if (father != -1):
            res[index][father] = 0
            res[father][index] = 1
    return res
def gettimedlay(times):
  time_dlay=[]
  if (len(times) == 0):
    return time_dlay
  sourcetime=min(times)
  for time in times:
    dlay = (time - sourcetime).total_seconds()
    # print(dlay)
    dlay = dlay / 60
    timebin=dlay/10
    if(timebin>99):
      timebin=99
    time_dlay.append(round(timebin))
  return time_dlay
def main(id):
    likelist = getlikelist(str(id))
    source=getsource(str(id))
    comments=getcomments(str(id))
    reposts=getrepost(str(id))
    print("所有信息爬取完毕")
    # 微博界面显示的评论数和转发数
    comments_count=source['comments_count']
    reposts_count=source['reposts_count']
    #将评论和转发信息根据时间排序
    comments= sorted(comments, key=lambda e: e['time'], reverse=False)
    reposts=  sorted(reposts, key=lambda e: e['time'], reverse=False)

    #构建原文的数据，并将原文信息加入到comments中
    source_dict={}
    source_dict['text']=source['longTextContent']
    source_dict['time']=comments[0]['time']
    source_dict['id']=str(id)
    source_dict['like']=source['attitudes_count']
    source_dict['username']=''
    source_dict['gender']='m'
    source_dict['rootid']=''
    #将原文的字典加入到comments开头
    comments.insert(0,source_dict)

    #构建评论的结构信息和时间信息
    map={}
    comments_times = []
    tweets=[]
    for index,tweet in enumerate(comments):
        map[tweet['id']]=index
        comments_times.append(tweet['time'])
        tweets.append(tweet['text'])
    tweets_structure=[]
    for index,tweet in enumerate(comments):
        if(tweet['rootid']!=''):
            if(tweet['rootid']==tweet['id']):
                tweets_structure.append(0)
            else:
                tweets_structure.append(map[tweet['rootid']])
        else:
            tweets_structure.append(-1)
    structure=getstructure(tweets_structure)
    #构建评论时间信息
    comments_times=gettimedlay(comments_times)
    assert len(comments_times)==len(structure)
    assert len(comments)==len(comments_times)

    #构建转发的时间信息
    reposts_times=[]
    for repost in reposts:
        reposts_times.append(repost['time'])
    reposts_times=gettimedlay(reposts_times)

    #将comments的结构写入json文件
    res={}
    res['id_']=0
    res['label']=-1
    res['tweets']=tweets
    res['time_delay']=comments_times
    res['structure']=structure

    out_path=r"rumoure_dection\data\test"
    if not os.path.exists(out_path):
        # print(os.curdir+out_path)
        os.mkdir(out_path)
    with open(r"rumoure_dection\data\test"+r'/'+str(id)+r'_test.json', 'w', encoding='utf8') as fp:
        json.dump(res, fp, ensure_ascii=False)
        fp.write('\n')
    print("写模型检测数据文件完毕")
    #返回原文、评论、转发的结果
    data={}
    data['source']=source
    for i in range(len(comments)):
        comments[i].pop('time')
    data['comments']=comments
    data['comments_times']=comments_times
    for i in range(len(reposts)):
        reposts[i].pop('time')
    data['reposts']=reposts
    data['reposts_times']=reposts_times
    data['likelist']=likelist
    data['comments_structure']=structure
    with open(r"rumoure_dection\data\test"+r'/'+str(id)+r'.json', 'w', encoding='utf8') as fp:
        json.dump(data, fp, ensure_ascii=False)
        fp.write('\n')
    return data
# main(4756620066425085)
# url='https://m.weibo.cn/comments/hotflow?id=4753064223048549&mid=4753064223048549&max_id=138583824465712&max_id_type=0'
# response=requests.get(url=url,headers=headers,timeout=10)
# print(response.text)
# data=json.loads(response.text)
# print(data)
#假:4713907509792628 4750035658736123 4753064223048549
#真实：4751565107234760 4738263266364252

