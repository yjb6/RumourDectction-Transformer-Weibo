import re
import jieba
import pandas as pd
from gensim import corpora, models
import json
import numpy as np
def getstopwords():
    stopwords = pd.read_csv(r"C:\Users\闫金博\PycharmProjects\rumoure_dection\rumoure_dection\data\stopwords.txt", index_col=False, sep="\t", quoting=3,
                            names=['stopword'],
                            encoding='utf-8')
    return set(stopwords['stopword'].values.tolist())
def wordcould(all_text):
    dict={}
    for text in all_text:
        seq_list=jieba.lcut(text)
        for word in seq_list:
            if word in dict.keys():
                dict[word]+=1
            else:
                dict[word]=1
    stopwords=getstopwords()
    temp=dict.keys()
    for k in list(temp):
        if k in stopwords:
            dict.pop(k)
    dict = sorted(dict.items(), key=lambda e: e[1], reverse=True)
    res=[]
    for i in range(min(40,len(dict))):
        temp={}
        temp['name']=dict[i][0]
        temp['value']=dict[i][1]
        res.append(temp)
    return res
def get_time(timedlay):
    '''分为1-12h，12+，这13个区间'''
    res=[0 for i in range(13)]
    for timebin in timedlay:
        if(timebin<72):
            res[round(timebin/6)]+=1
        else:
            res[-1]+=1
    return res
def getword_len(texts):
    res=[0 for i in range(11)]
    for text in texts:
        l=len(text)
        if(l<100):
            res[round(l/10)]+=1
        else:
            res[-1]+=1
    return res
def getuser(comments,reposts):
    s=set()
    for item in comments:
        s.add(item['username'])
    for item in reposts:
        s.add(item['username'])
    return s
def getlikelist(likelist):
    res=[]
    likelist=likelist[:100]
    for item in likelist:
        dict={}
        dict['username']=item['username'][0]+"**"+item['username'][-1]
        dict['time']=item['time']
        dict['source']=item['source'][:6]
        dict['fscount']=item['followers_count']
        res.append(dict)
    return res
def gettotallike(comments):
    res=0
    for item in comments:
        res+=item['like']
    return res
def get_structureinfo(structure):
    ''' res[i,j] 0:i是j儿子 1:i是j爹 2:i在j前面 3:i在j后面 4:i=j'''
    max_depth=dfs_depth(structure,0)
    leaf_count=0
    for i in range(len(structure)):
        flag=0
        for j in range(len(structure[i])):
            if(structure[i][j]==1):
                flag=1
                break
        if(flag==0):
            leaf_count+=1
    max_outdegree=0
    for row in structure:
        outdegree=0
        for j in row:
            if (j == 1):
                outdegree+=1
        max_outdegree=max(max_outdegree,outdegree)
    return max_depth,leaf_count,max_outdegree

def dfs_depth(structure,i):
    max_depth=1
    for j in range(len(structure[i])):
        if(structure[i][j]==1):
            depth=dfs_depth(structure,j)
            max_depth=max(depth+1,max_depth)
    return max_depth
def lda(texts):
    dict={}
    stopwords = getstopwords()
    stopwords.add(" ")
    data=[[word for word in jieba.lcut(text) if word not in stopwords]for text in texts]
    dictionary = corpora.Dictionary(data)
    V = len(dictionary)

    # 转换文本数据为索引，并计数
    corpus = [dictionary.doc2bow(text) for text in data]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    num_topics=10
    lda = models.LdaModel(corpus_tfidf, num_topics=10, id2word=dictionary,
                          alpha=0.01, eta=0.01, minimum_probability=0.001,
                          update_every=1, chunksize=100, passes=1)
    # # 随机打印某10个文档的主题
    # num_show_topic = 10  # 每个文档显示前几个主题
    # print('7.结果：10个文档的主题分布：--')
    # doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    # idx = np.arange(M)
    # np.random.shuffle(idx)
    # idx = idx[:10]
    # for i in idx:
    #     topic = np.array(doc_topics[i])
    #     topic_distribute = np.array(topic[:, 1])
    #     # print topic_distribute
    #     topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
    #     print('第%d个文档的前%d个主题：' % (i, num_show_topic)), topic_idx
    #     print(topic_distribute[topic_idx])

    num_show_term = 10  # 每个主题显示几个词
    print('8.结果：每个主题的词分布：--')
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        print('词：\t', )
        for t in term_id:
            print(dictionary.id2token[t], )
        print('\n概率：\t', term_distribute[:, 1])
def main(data):
    texts=[]
    gender=[{"value":0, "name": '男性评论'},{"value": 0, "name": '女性评论'},{"value": 0, "name": '男性转发'},{"value": 0, "name": '女性转发'}]
    source=data['source']
    comments=data['comments']
    comments_times=data['comments_times']
    reposts_times=data['reposts_times']
    reposts=data['reposts']
    likelist=data['likelist']
    structure=data['comments_structure']
    for item in comments:
        if(item['gender']=='m'):
            gender[0]["value"]+=1
        else:
            gender[1]["value"]+=1
        texts.append(item['text'])
    for item in reposts:
        if(item['gender']=='m'):
            gender[2]["value"]+=1
        else:
            gender[3]["value"]+=1
    wc=wordcould(texts)#词云
    comments_times=get_time(comments_times)
    reposts_times=get_time(reposts_times)
    word_len=getword_len(texts)
    alluser=getuser(comments,reposts)
    likelist=getlikelist(likelist)
    totallike=gettotallike(comments)
    max_depth,leaf_count,max_outdegree=get_structureinfo(structure)
    return {"wc":wc,"comments_times":comments_times,"reposts_times":reposts_times,
            "word_len":word_len,"gender":gender,"totaluser":len(alluser),
            "totalpost":source['comments_count']+source['reposts_count'],'likecount':source['attitudes_count'],
            'commentscount':source['comments_count'],'repostscount':source['reposts_count'],
            'likelist':likelist,'totallike':totallike,'max_depth':max_depth,'leaf_count':leaf_count,
            'max_outdegree':max_outdegree}

# data = []
# with open(r"C:\Users\闫金博\PycharmProjects\rumoure_dection\rumoure_dection\data\test\4713907509792628.json", 'r',
#           encoding='utf8') as fp:
#     for line in fp:
#         data.append(json.loads(line))
# data=data[0]
# texts=[com['text'] for com in data['comments']]
# lda(texts)