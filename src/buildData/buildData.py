# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from model import setting
from pymongo import *
import re


# 从MongoDB中获取Java和C#的数据,提取NL和Code的Token,过滤所有\t的字符(将\t替换为\space);
# 生成的数据集每一行表示一个数据Code NL,分为 data.txt train.txt test.txt valid.txt,用\t进行划分
# 将数据过滤停用词等,构造为Token,将清理后的数据放到workdir目录下,命名为
#      (github|stackoverflow).(java|csharp).(data|train|test|valid|vocab)

def download_data(plat,lang):
    client = MongoClient("192.168.7.113",30000)
    db = client.code2text
    collection = db[plat+"_"+lang]

    dataset = []
    id = 0
    for doc in collection.find():
        text = doc["title"].replace('\r','') # MongoDB的字段
        code = doc["code"].replace('\r','')
        docEntry = {"id": id, "text": text, "code":code }
        dataset.append(docEntry)
        id = id + 1
        # 加入下载进度条显示

    f = open(setting.HOME_DIR + '/data/'+plat+'/'+lang+'/'+"data.txt", 'w')
    for dataline in dataset:
        line = ""+str(dataline["id"])+"\t"+dataline["text"].replace('\t','\\t').strip() + \
                "\t"+dataline["code"].replace('\t','\\t').strip()
        f.write(re.sub(r'\n',r'\\n',line.strip()).encode('utf-8')+'\n')
    f.close()

    return

# 获取GitHub和StackOverflow的数据,并将处理后的数据放到workdir目录下
if __name__ == '__main__':
     
    print(setting.HOME_DIR)
    download_data("stackoverflow","java")
    download_data("stackoverflow","csharp")
