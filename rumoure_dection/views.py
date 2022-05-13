import json

from django.http import HttpResponse
from django.shortcuts import render
from . import crawl
from . import process
import os
import sys
# sys.path.append('/codes/')
from rumoure_dection.codes import test
def hello(request):
    context={}
    context['hello']="Hello World!"
    return render(request,'index.html',context)
    # return HttpResponse("Hello world ! ")
def testt(request):
    '''仅供测试使用'''
    data = []
    print (os.getcwd())
    with open(r"rumoure_dection\data\test\test2_1.json", 'r', encoding='utf8') as fp:
        print(111)
    print(data[0])
    mes=process.main(data[0])
    return render(request,'main.html',mes)

def search_form(request):
    return render(request, 'search_form.html')


# 接收请求数据
def search(request):
    print(request)
    request.encoding = 'utf-8'
    if 'wbid' in request.GET and request.GET['wbid']:
        wbid = request.GET['wbid']
        datapath = r'rumoure_dection\data\test' + r'/' + wbid + '.json'
        if  os.path.exists(datapath):
            data = []
            with open(datapath, 'r',
                      encoding='utf8') as fp:
                for line in fp:
                    data.append(json.loads(line))
            data=data[0]
        else:
            data = crawl.main(wbid)

        print("爬取数据及数据预处理完毕")
        label=test.main(wbid)
        # label=0
        print("模型判断完毕")
        mes=process.main(data)
        print("微博信息分析完毕")
        mes['label']=label
        return render(request, 'main.html', mes)
    else:
        message = '你提交了空表单'
    return HttpResponse(message)